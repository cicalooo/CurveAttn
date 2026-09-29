from __future__ import annotations
import argparse
import importlib
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path


def aggregate(times, peaks):
    if not times or len(times) != len(peaks):
        raise ValueError("matching nonempty timing/peak lists required")
    ordered = sorted(times)
    return dict(median_ms=statistics.median(times),
                p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1],
                peak_allocated_mib=max(peaks))


def timed(fn, warmup, repeats, torch):
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 required")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times, wall, peaks, reserved = [], [], [], []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - before) * 1000)
        times.append(start.elapsed_time(end))
        peaks.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        reserved.append(torch.cuda.max_memory_reserved() / (1024 * 1024))
        del out
    result = aggregate(times, peaks)
    result["median_wall_ms"] = statistics.median(wall)
    result["peak_reserved_mib"] = max(reserved)
    result["latencies_ms"] = times
    return result


def error_metrics(actual, reference, torch):
    a, b = actual.float().flatten(), reference.float().flatten()
    if not bool(torch.isfinite(a).all().item()):
        raise RuntimeError("nonfinite benchmark output")
    delta = a - b
    return dict(relative_l2=float((delta.norm() / b.norm().clamp_min(1e-12)).item()),
                max_abs=float(delta.abs().max().item()),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0).item()))


def load_modules():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parents[1]))
    from comfy.cli_args import args
    args.cpu = False
    name = "curve_benchmark_package"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    package.__version__ = "2.3"
    sys.modules[name] = package
    return (importlib.import_module(name + ".curve_attention"),
            importlib.import_module(name + ".chunked_attention"))


def run(args):
    import torch
    import comfy_kitchen as ck
    if not torch.cuda.is_available() or not ck.sol_attn_is_available():
        raise RuntimeError("requested GPU Sol benchmark unsupported on this device")
    ca, ch = load_modules()
    from comfy.ldm.minimax.model import Attention
    rows = []
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    operations = types.SimpleNamespace(Linear=torch.nn.Linear, RMSNorm=torch.nn.RMSNorm)
    attn = Attention(128, args.heads, 128, 1e-6, dtype=torch.bfloat16,
                     device=torch.device("cuda"), operations=operations).eval()
    cases = [("dense_pytorch", "pytorch", "native", True),
             ("dense_kitchen", "comfy_kitchen_int8", "native", True),
             ("curve_native", "pytorch", "native", False),
             ("curve_owned", "pytorch", "owned_bf16", False),
             ("curve_chunked_current_stats", "pytorch", "chunked", False)]
    with torch.inference_mode():
        for seq in args.seq:
            x = torch.randn(seq, 128, device="cuda", dtype=torch.bfloat16)
            rope = torch.eye(2, device="cuda", dtype=torch.bfloat16).view(
                1, 1, 1, 1, 2, 2).expand(1, seq, 1, 32, 2, 2).contiguous()
            reference = None
            for name, backend_name, mode, force_dense in cases:
                backend = ca.get_attention_function(backend_name, None)
                if backend is None:
                    raise RuntimeError("requested dense backend missing: " + backend_name)
                state = ca.CurveAttnState(
                    keep_ratio=.2, min_tokens=0, dense_steps=frozenset(),
                    dense_last_steps=0, tail=False, protect_audio=False, verbose=False,
                    dense_attention=backend, prepared_available=mode == "owned_bf16",
                    dense_backend=backend_name, qkv_mode=mode)
                options = {"_curve_dense": force_dense}
                ca.install_override(state, options)
                target = ch.make_chunked_attention(attn, state) if mode == "chunked" else attn

                def fn(target=target, x=x, rope=rope, options=options):
                    return target(x, rope_freqs=rope, transformer_options=options)

                actual = fn()
                if reference is None:
                    reference = actual.detach().clone()
                metrics = error_metrics(actual, reference, torch)
                del actual
                timing = timed(fn, args.warmup, args.repeats, torch)
                expected = (state.dense_calls > 0 if force_dense else
                            state.chunked_calls > 0 if mode == "chunked" else state.sparse_calls > 0)
                if not expected or (not force_dense and state.dense_calls):
                    raise RuntimeError("requested benchmark path did not execute: " + name)
                rows.append(dict(seq=seq, heads=args.heads, hidden=128, case=name,
                                 phase="projection_inclusive", **timing, **metrics,
                                 dense_calls=state.dense_calls, sparse_calls=state.sparse_calls,
                                 chunked_calls=state.chunked_calls))
            del reference, x, rope
            packed = torch.randn(1, seq, args.heads * 128 * 3,
                                 device="cuda", dtype=torch.bfloat16)
            q, k, v = [t.view(1, seq, args.heads, 128) for t in packed.chunk(3, dim=-1)]

            def kernel(q=q, k=k, v=v):
                return ck.sol_attn(q, k, v, topk_ratio=.2, tau=0., tail=False,
                                   sink_blocks=[0, 0], sink_q=[0, 0], token_aug=0)

            dense = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            metrics = error_metrics(kernel(), dense, torch)
            del dense
            rows.append(dict(seq=seq, heads=args.heads, case="kitchen_kernel_only",
                             phase="kernel_only_packed_owner_retained",
                             **timed(kernel, args.warmup, args.repeats, torch), **metrics))
            del packed, q, k, v
    return dict(seed=args.seed, torch_version=torch.__version__,
                kitchen_version=importlib.import_module("importlib.metadata").version("comfy-kitchen"),
                rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicitly authorized synthetic CurveAttention benchmark")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--seq", type=int, nargs="+", default=[65, 257, 4096])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if not args.allow_gpu:
        print("GPU permission required; no GPU work performed.")
        return 2
    if args.warmup < 0 or args.repeats < 1 or args.heads < 1 or any(s < 1 for s in args.seq):
        parser.error("invalid benchmark dimensions/repeat counts")
    sys.argv = [sys.argv[0]]
    report = run(args)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
