"""CurveAttn Sparse: MiniMax H3 sparse attention on stock Comfy Kitchen sol_attn.

One engine, three selection rules. ``sla`` keeps a fixed top-k fraction of
key blocks, ``sol-attn`` uses the adaptive tau threshold, ``vsa`` reorders
the sequence into 4x4x4 video cubes and adds VSA's coarse branch when the
checkpoint has ``to_gate_compress`` (fine stage only otherwise). All three
call ck.sol_attn (or ck.sol_attn_chunked for qkv_mode=chunked).

Stacking contract
-----------------
* The attention override is installed on top of whatever CurveAttn override
  (or Model Attention Backend pass-through) is already there and keeps it as
  ``previous``. Foreign attention overrides are refused.
* ON_PREPARE_STATE reinstalls idempotently: if this node's override is
  anywhere in the chain, nothing happens.
* Every node publishes its per-step decision under
  ``transformer_options["curve_attn"]["sparse"][uid]``. The sparse owner of a
  step is the most recently applied node whose decision is sparse. Every
  other CurveAttn override passes the call down the chain.
* A static policy fingerprint is published for the residual cache.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from functools import partial
from math import isfinite
from threading import Lock
from typing import NamedTuple

import comfy_kitchen as ck
import torch

from comfy.ldm.modules.attention import AttentionTensorContainer, get_attention_function
from comfy.model_prefetch import pause_malloc_graph
from comfy.patcher_extension import CallbacksMP, WrappersMP

from . import blocks
from ._version import VERSION
from .common import (
    LOG, NS, OVERRIDE_KEY, PREFIX, check_object_patches, curve_states,
    h3_sinks, install_cond_batch_limiter, lerp_schedule, next_seq, outside_sigma_window,
    override_chain, parse_steps, require_h3, resolve_sigma_bounds, resolve_step, scalar,
    schedule_progress, schedule_tuple,
)
from .sol_attn import check_prepared_available, consume_sla, prepare_sla, sol_attn_zerocopy
from .reblock import (SUMMARIES, build_reblock_plan, gather_heads, reweighted_attention,
                      scatter_heads, video_span)
from .vsa import (build_vsa_plan, gather_qkv, has_gate, make_vsa_gate_attention, scatter_rows,
                  take_gate)

WHO = f"{PREFIX} Sparse"
_HEAD_DIM = 128
SELECTIONS = ("sla", "sol-attn", "vsa", "reblock")
DENSE_BACKENDS = ("comfy_kitchen_int8", "pytorch")
QKV_MODES = ("native", "owned_bf16", "chunked")
VSA_PLAN_CACHE = 4


# ---------------------------------------------------------------- validation

def validate_sparse(*, selection, keep_percent, keep_percent_end, tau, tau_end,
                    min_tokens, dense_last_steps, start_percent, end_percent,
                    extra_tokens, dense_backend, qkv_mode, reblock_summary="pooled",
                    reblock_updates=1, reblock_refresh=2):
    def real(name, value, low, high):
        if isinstance(value, bool) or not isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError(f"{WHO}: invalid {name}")

    def integer(name, value, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{WHO}: invalid {name}")

    if selection not in SELECTIONS:
        raise ValueError(f"{WHO}: invalid selection")
    real("keep_percent", keep_percent, .5, 95)
    if keep_percent_end != -1:
        real("keep_percent_end", keep_percent_end, .5, 95)
    real("tau", tau, 0, 4)
    if tau_end != -1:
        real("tau_end", tau_end, 0, 4)
    integer("min_tokens", min_tokens, 0, 1 << 20)
    integer("dense_last_steps", dense_last_steps, 0, 16)
    real("start_percent", start_percent, 0, 1)
    real("end_percent", end_percent, 0, 1)
    if start_percent > end_percent:
        raise ValueError(f"{WHO}: start_percent exceeds end_percent")
    integer("extra_tokens", extra_tokens, 0, 256)
    if extra_tokens % 64:
        raise ValueError(f"{WHO}: extra_tokens must be a multiple of 64")
    if dense_backend not in DENSE_BACKENDS:
        raise ValueError(f"{WHO}: invalid dense_backend")
    if qkv_mode not in QKV_MODES:
        raise ValueError(f"{WHO}: invalid qkv_mode")
    if reblock_summary not in SUMMARIES:
        raise ValueError(f"{WHO}: invalid reblock summary")
    integer("reblock_updates", reblock_updates, 0, 4)
    integer("reblock_refresh", reblock_refresh, 1, 8)


# ---------------------------------------------------------------- state

class StepDecision(NamedTuple):
    dense: bool
    reason: str | None
    topk: float
    tau: float
    step: int | None = None


def is_fatal_device_error(exc) -> bool:
    """True for CUDA/device-side failures that must not fall through to dense."""
    cuda_error = getattr(torch.cuda, "CudaError", None)
    if cuda_error is not None and isinstance(exc, cuda_error):
        return True
    type_name = type(exc).__name__
    if "Cuda" in type_name or "CUDA" in type_name:
        return True
    message = str(exc).lower()
    return "cuda error" in message or "illegal memory access" in message


class SparseState:
    def __init__(self, *, dense_attention, selection="sla", keep_ratio=.2,
                 keep_ratio_end=None, tau=1.3, tau_end=None, min_tokens=12288,
                 dense_steps=frozenset(), dense_last_steps=0, tail=False, token_aug=0,
                 protect_audio=False, protect_prefix=False, disable_fp16_accum=True,
                 sigma_start=None, sigma_end=None, dense_backend="comfy_kitchen_int8",
                 qkv_mode="native", prepared_available=False, verbose=False,
                 reblock_summary="pooled", reblock_updates=1, reblock_refresh=2):
        self.seq = next_seq()
        self.uid = f"sparse{self.seq}"
        self.dense_attention = dense_attention
        self.selection = selection
        self.keep_ratio = float(keep_ratio)
        self.keep_ratio_end = keep_ratio_end
        self.tau = float(tau)
        self.tau_end = tau_end
        self.min_tokens = int(min_tokens)
        self.dense_steps = frozenset(dense_steps)
        self.dense_last_steps = int(dense_last_steps)
        self.tail = bool(tail)
        self.token_aug = int(token_aug)
        self.protect_audio = bool(protect_audio)
        self.protect_prefix = bool(protect_prefix)
        self.disable_fp16_accum = bool(disable_fp16_accum)
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.dense_backend = dense_backend
        self.qkv_mode = qkv_mode
        self.prepared_available = bool(prepared_available)
        self.verbose = bool(verbose)
        self.schedule = ()
        self.chunked_stats = {}
        self._vsa_plans = OrderedDict()   # LRU, kept across runs, cleared in cleanup()
        self._vsa_last = None             # (weakref(layout), device, plan) fast path
        self._vsa_rope = None             # (weakref(rope_freqs), plan, permuted)
        self.vsa_gate = False
        self.reblock_summary = reblock_summary
        self.reblock_updates = int(reblock_updates)
        self.reblock_refresh = int(reblock_refresh)
        self.reblock_plans = {}
        self.lock = Lock()
        self.reset()

    def reset(self):
        self._logged = set()
        self.sparse_calls = 0
        self.dense_calls = 0
        self.chunked_calls = 0
        self.failure_reason = None
        self.reweight_failed = None
        self.reblock_paths = {}
        self.reblock_timing = {}
        self.reblock_plans = {}
        self.prepared_enabled = self.prepared_available
        self.last_reason = None

    def log_once(self, key, message):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            LOG.info("%s [%s]: %s", WHO, self.uid, message)

    def warn_once(self, key, message):
        if key not in self._logged:
            self._logged.add(key)
            LOG.warning("%s [%s]: %s", WHO, self.uid, message)

    def info_once(self, key, message):
        if key not in self._logged:
            self._logged.add(key)
            LOG.info("%s [%s]: %s", WHO, self.uid, message)

    def vsa_plan(self, options, tokens, device):
        """Cached cube plan for this forward, or None when VSA must stay dense."""
        layout = (options or {}).get("minimax_h3_layout")
        if layout is None or getattr(layout, "seq_len", None) != tokens:
            return None
        device_key = str(device)
        last = self._vsa_last
        if last is not None and last[0]() is layout and last[1] == device_key:
            return last[2]
        key = (tuple(getattr(layout, "signature", ()) or ()),
               tuple(getattr(layout, "segments", ()) or ()), device_key)
        plans = self._vsa_plans
        if key in plans:
            plans.move_to_end(key)
            plan = plans[key]
        else:
            plan = plans[key] = build_vsa_plan(layout, tokens, device)
            while len(plans) > VSA_PLAN_CACHE:
                plans.popitem(last=False)
        try:
            self._vsa_last = (weakref.ref(layout), device_key, plan)
        except TypeError:
            self._vsa_last = None
        return plan

    def vsa_rope_freqs(self, rope_freqs, plan):
        """RoPE in tile order (pad rows zero), once per plan and rope tensor."""
        hit = self._vsa_rope
        if hit is not None and hit[0]() is rope_freqs and hit[1] is plan:
            return hit[2]
        padded = rope_freqs.new_zeros((1, plan["n"]) + tuple(rope_freqs.shape[2:]))
        padded[0, plan["inv"]] = rope_freqs[0]
        self._vsa_rope = (weakref.ref(rope_freqs), plan, padded)
        return padded

    # -- policy ------------------------------------------------------------

    def static_policy(self):
        """Residual identity of this node. Excludes per-step lerped keep/tau:
        those change every step and would delete the cache anchor each step."""
        return (self.seq, self.selection, self.tail, self.token_aug, self.protect_prefix,
                self.protect_audio, self.dense_backend, self.qkv_mode,
                self.disable_fp16_accum)

    def base_decision(self) -> StepDecision:
        return StepDecision(False, None, self.keep_ratio, self.tau)

    def decide(self, options) -> StepDecision:
        step0, n_steps = resolve_step(options, self.schedule)
        t = schedule_progress(step0, n_steps)
        topk = (lerp_schedule(self.keep_ratio, self.keep_ratio_end, t)
                if self.keep_ratio_end is not None and self.selection == "sla" else self.keep_ratio)
        tau = (lerp_schedule(self.tau, self.tau_end, t)
               if self.tau_end is not None and self.selection == "sol-attn" else self.tau)
        dense, reason = False, None
        if step0 is None and (self.dense_steps or self.dense_last_steps):
            dense, reason = True, "sigma schedule unavailable"
        elif step0 is not None:
            if step0 in self.dense_steps:
                dense, reason = True, f"dense_steps contains {step0}"
            elif self.dense_last_steps > 0 and step0 >= n_steps - self.dense_last_steps:
                dense, reason = True, f"dense_last_steps ({step0}/{n_steps})"
        if not dense and outside_sigma_window(
                scalar((options or {}).get("sigmas")), self.sigma_start, self.sigma_end):
            dense, reason = True, "outside start/end_percent"
        return StepDecision(dense, reason, topk, tau, step0)

    def resolve(self, options):
        """(decision, reason, passthrough) for one attention call.

        decision is set only when this node owns the step. passthrough means
        another CurveAttn node owns it, so the call is not counted here.
        """
        ns = namespace_of(options)
        if "owner" not in ns:
            # No CurveAttn step wrapper ran (direct call): act as sole owner.
            return self.base_decision(), None, False
        owner = ns.get("owner")
        own = (ns.get("sparse") or {}).get(self.uid)
        if owner == self.uid and own is not None:
            return own, None, False
        if owner is not None:
            return None, "another CurveAttn Sparse node owns this step", True
        if own is None:
            return None, "step wrapper did not run", False
        return None, own.reason or "scheduled dense step", False

    def sparse_kwargs(self, scale, sink, sink_q, decision=None):
        decision = decision or self.base_decision()
        if self.selection == "sol-attn":
            topk_ratio, tau = 0.0, float(decision.tau)
            tail, token_aug = self.tail, int(self.token_aug)
        elif self.selection == "vsa":
            # Fine stage only. Prefix sinks are supplied by the caller.
            topk_ratio, tau = float(decision.topk), 0.0
            tail, token_aug = False, 0
        else:
            topk_ratio, tau = float(decision.topk), 0.0
            tail, token_aug = self.tail, int(self.token_aug)
        return dict(topk_ratio=topk_ratio, tau=tau, scale=scale,
                    sink_blocks=list(sink), sink_q=list(sink_q),
                    tail=tail, token_aug=token_aug)

    # -- run lifecycle -------------------------------------------------------

    def outer(self, executor, *args, **kwargs):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(f"{WHO}: concurrent use of one patched model is unsupported")
        try:
            sigmas = kwargs.get("sigmas", args[3] if len(args) > 3 else None)
            self.reset()
            self.schedule = schedule_tuple(sigmas)
            self.chunked_stats = {}
            try:
                return executor(*args, **kwargs)
            finally:
                self.report()
                self.schedule = ()
                self.chunked_stats = {}
        finally:
            self.lock.release()

    def report(self):
        if self.verbose or self.sparse_calls or self.dense_calls:
            LOG.info("%s [%s]: selection=%s sparse_calls=%d dense_calls=%d chunked_calls=%d "
                     "dense_backend=%s last_dense=%s", WHO, self.uid, self.selection,
                     self.sparse_calls, self.dense_calls, self.chunked_calls,
                     self.dense_backend, self.last_reason)
        if self.sparse_calls == 0 and self.dense_calls and not self.failure_reason:
            LOG.warning("%s [%s]: sparse was requested but sparse_calls=0 (last_dense=%s)",
                        WHO, self.uid, self.last_reason)
        if self.reblock_paths:
            LOG.info("%s [%s]: reblock paths %s", WHO, self.uid,
                     " ".join(f"{k}={v}" for k, v in sorted(self.reblock_paths.items())))
        for step in sorted(self.reblock_timing):
            self._log_reblock_step(step)

    def _log_reblock_step(self, step):
        calls, plan_ms, attn_ms, paths = self.reblock_timing.pop(step)
        LOG.info("%s [%s]: reblock step %d: %d calls plan %.0f ms attention %.0f ms path %s",
                 WHO, self.uid, step, calls, plan_ms, attn_ms, "/".join(sorted(paths)))

    def time_reblock(self, step, plan_ms, attn_ms, path):
        for old in [s for s in self.reblock_timing if s != step]:
            self._log_reblock_step(old)
        row = self.reblock_timing.setdefault(step, [0, 0.0, 0.0, set()])
        row[0] += 1
        row[1] += plan_ms
        row[2] += attn_ms
        row[3].add(path)

    def cleanup(self):
        self.schedule = ()
        self.chunked_stats = {}
        self._vsa_plans.clear()
        self._vsa_last = None
        self._vsa_rope = None
        # ComfyUI's outer_sample runs model_patcher.cleanup() (ON_CLEANUP) *inside* our
        # OUTER_SAMPLE wrapper, before report(); resetting then zeroed every count.
        # outer() resets at the start of each run anyway.
        if not self.lock.locked():
            self.reset()


def namespace_of(options) -> dict:
    return (options or {}).get(NS) or {}


# ---------------------------------------------------------------- attention

def _ineligible(q, k, v, dim_head):
    if q.device.type != "cuda":
        return "not on CUDA"
    if not ck.sol_attn_is_available(q.device):
        return "no compiled sol_attn kernel for this GPU"
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {q.dtype}"
    if dim_head != _HEAD_DIM:
        return f"head_dim {dim_head} != {_HEAD_DIM}"
    if q.shape != k.shape or q.shape != v.shape:
        return "cross-attention or GQA"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "mixed dtypes"
    return None


def _prepare_owned(q, k, v, heads, skip_reshape, **options):
    """Consume the H3 containers in a separate frame from attention output.

    Restore ownership on preparation failure so the native sparse fallback
    can still use the original inputs.
    """
    tensors = [container.take() for container in (q, k, v)]
    try:
        if skip_reshape:
            views = [tensor.transpose(1, 2) for tensor in tensors]
        else:
            views = [tensor.view(tensor.shape[0], -1, heads, tensor.shape[-1] // heads)
                     for tensor in tensors]
        views = [tensor.to(torch.bfloat16) for tensor in views]
        return prepare_sla(*views, **options)
    except Exception:
        for container, tensor in zip((q, k, v), tensors):
            container.tensor = tensor
        raise


def _run_dense(state, previous, q, k, v, heads, mask, attn_precision, skip_reshape,
               skip_output_reshape, kwargs, reason=None, count=True):
    """Hand the call down the chain (or to this node's dense backend)."""
    if count:
        state.dense_calls += 1
        if reason:
            state.last_reason = reason
            state.log_once(("dense", reason), f"dense: {reason}")
    args = (q, k, v, heads)
    kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
              skip_output_reshape=skip_output_reshape, **kwargs)
    # Bypass override recursion but keep the container handoff so scheduled
    # dense steps can release float QKV early.
    kw["_inside_attn_wrapper"] = True
    if previous is not None:
        return previous(state.dense_attention, *args, **kw)
    return state.dense_attention(*args, **kw)


def _scatter(out, plan):
    return out if plan is None else scatter_rows(out, plan["inv"])


def _consume_prepared(state, prepared, input_dtype, skip_output_reshape,
                      b, heads, dim_head, tokens, sink, sink_q, plan=None):
    # QKV is gone now; a consume failure cannot fall back to dense.
    out = _scatter(consume_sla(prepared), plan)
    selector = prepared.selector
    del prepared
    if input_dtype != torch.bfloat16:
        out = out.to(input_dtype)
    state.sparse_calls += 1
    state.log_once(("prepared", tokens, sink, sink_q),
                   f"sparse ({b}, {tokens}, {heads}, {dim_head}) sinks {sink}/{sink_q} "
                   f"selector={selector} prepared_qkv=True block=64")
    return out.transpose(1, 2) if skip_output_reshape else out.reshape(b, -1, heads * dim_head)


def _prepare_gathered(state, q, k, v, qs, ks, vs, options):
    """Prepare already-tiled VSA views. Original containers stay until success."""
    try:
        prepared = prepare_sla(qs, ks, vs, **options)
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:
        if is_fatal_device_error(exc):
            raise
        state.prepared_enabled = False
        LOG.warning("%s: prepared backend unavailable (%s); native sol_attn for this run", WHO, exc)
        return None, qs, ks, vs
    for container in (q, k, v):
        container.take()
    return prepared, None, None, None


def _prepare_or_disable(state, q, k, v, heads, skip_reshape, b, dim_head, options):
    try:
        return _prepare_owned(q, k, v, heads, skip_reshape, **options), None, None, None
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:
        if is_fatal_device_error(exc):
            raise
        state.prepared_enabled = False
        LOG.warning("%s: prepared backend unavailable (%s); native sol_attn for this run", WHO, exc)
        inputs = [tensor.peek() for tensor in (q, k, v)]
        if skip_reshape:
            qs, ks, vs = (tensor.transpose(1, 2) for tensor in inputs)
        else:
            qs, ks, vs = (tensor.view(b, -1, heads, dim_head) for tensor in inputs)
        return None, qs, ks, vs


def _run_native_sparse(state, q, k, v, qs, ks, vs, heads, skip_output_reshape,
                       b, dim_head, sink, sink_q, options, input_dtype, owned, dense,
                       plan=None):
    if input_dtype != torch.bfloat16:
        qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
    try:
        out = _scatter(sol_attn_zerocopy(qs, ks, vs, **options), plan)
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        if is_fatal_device_error(exc):
            raise
        state.failure_reason = f"sol_attn failed: {exc}"
        LOG.warning("%s: %s; using dense for this run", WHO, state.failure_reason)
        return dense(state.failure_reason)
    if owned:
        for container in (q, k, v):
            container.take()
    if input_dtype != torch.bfloat16:
        out = out.to(input_dtype)
    state.sparse_calls += 1
    state.log_once(("sparse", tuple(qs.shape), sink, sink_q),
                   f"sparse {tuple(qs.shape)} sinks {sink}/{sink_q} selector=native block=64")
    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(b, -1, heads * dim_head)


def _step_of(decision):
    step = getattr(decision, "step", None)
    return -1 if step is None else int(step)


def _reblock_plan(state, qs, ks, span, decision, options):
    """Build the plan, or reuse this layer's plan for ``reblock_refresh`` steps.

    Keyed by block index, cond/uncond mix, shape and span; ``reset()`` (new run)
    drops the cache. Routing still runs per call inside sol_attn."""
    refresh = state.reblock_refresh
    block = options.get("block_index")
    step = _step_of(decision)
    if refresh <= 1 or block is None or step < 0:
        return build_reblock_plan(qs, ks, span, updates=state.reblock_updates)
    key = (block, tuple(options.get("cond_or_uncond") or ()), tuple(qs.shape), span,
           str(qs.device))
    hit = state.reblock_plans.get(key)
    if hit is not None and 0 <= step - hit[0] < refresh:
        return hit[1]
    plan = build_reblock_plan(qs, ks, span, updates=state.reblock_updates)
    state.reblock_plans[key] = (step, plan)
    return plan


def _run_reblock(state, q, k, v, qs, ks, vs, heads, skip_output_reshape, b, dim_head,
                 decision, scale, layout, owned, dense, options=None):
    options = options or {}
    """Reblocked selection: per-head clustered order, then flex (reweighted tail)
    or Kitchen sol_attn (pooled / no tail) on that order."""
    span = video_span(layout, qs.shape[1])
    if span is None:
        return dense("H3 layout unavailable for reblock")
    input_dtype = qs.dtype
    if input_dtype != torch.bfloat16:
        qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
    timing = state.verbose and qs.is_cuda
    if timing:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record()
    plan = _reblock_plan(state, qs, ks, span, decision, options)
    if timing:
        events[1].record()
    if plan is None:
        return dense("target video too short for reblock")
    qs, ks, vs = gather_heads(qs, ks, vs, plan["src"], plan["dead"])
    ratio = float(decision.topk)
    summary = state.reblock_summary
    if summary == "reweighted" and state.reweight_failed is not None:
        summary = "pooled"
    out = None
    try:
        if summary == "reweighted":
            try:
                out = reweighted_attention(qs, ks, vs, plan, ratio, scale)
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:  # flex/compile failures are not device faults
                if is_fatal_device_error(exc):
                    raise
                state.reweight_failed = f"{type(exc).__name__}: {exc}"
                LOG.warning("%s: reweighted tail unavailable (%s); using the pooled "
                            "sol_attn tail for this run", WHO, state.reweight_failed)
                summary = "pooled"
        if out is None:
            sink = [0, plan["n_prefix"]]
            out = sol_attn_zerocopy(qs, ks, vs, topk_ratio=ratio, tau=0.0, scale=scale,
                                    sink_blocks=sink, sink_q=sink, tail=summary == "pooled",
                                    token_aug=0, block_len=plan["block_len"])
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        if is_fatal_device_error(exc):
            raise
        state.failure_reason = f"sol_attn failed: {exc}"
        LOG.warning("%s: %s; using dense for this run", WHO, state.failure_reason)
        return dense(state.failure_reason)
    del qs, ks, vs
    out = scatter_heads(out, plan["inv"])
    state.reblock_paths[summary] = state.reblock_paths.get(summary, 0) + 1
    if timing:
        events[2].record()
        events[2].synchronize()
        state.time_reblock(_step_of(decision), events[0].elapsed_time(events[1]),
                           events[1].elapsed_time(events[2]),
                           "flex" if summary == "reweighted" else f"sol-{summary}")
    if owned:
        for container in (q, k, v):
            container.take()
    if input_dtype != torch.bfloat16:
        out = out.to(input_dtype)
    state.sparse_calls += 1
    state.info_once(("reblock", plan["n"], plan["n_prefix"], summary),
                    "reblock: %d rows, %d prefix tiles, %d video blocks, keep %.1f%%, tail %s"
                    % (plan["n"], plan["n_prefix"], plan["n_video_blocks"], 100.0 * ratio,
                       summary))
    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(b, -1, heads * dim_head)


def make_attention_override(state: SparseState, previous=None):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}

        def dense(reason=None, count=True):
            return _run_dense(state, previous, q, k, v, heads, mask, attn_precision,
                              skip_reshape, skip_output_reshape, kwargs,
                              reason=reason, count=count)

        decision, reason, passthrough = state.resolve(transformer_options)
        if decision is None:
            return dense(reason, count=not passthrough)
        if mask is not None:
            return dense("attention mask present")
        if state.failure_reason is not None:
            return dense(state.failure_reason)
        owned = isinstance(q, AttentionTensorContainer)
        inputs = [tensor.peek() for tensor in (q, k, v)] if owned else [q, k, v]
        tokens = inputs[0].shape[2] if skip_reshape else inputs[0].shape[1]
        if tokens < state.min_tokens:
            del inputs
            return dense(f"{tokens} tokens < min_tokens {state.min_tokens}")
        if skip_reshape:
            b, _, _, dim_head = inputs[0].shape
            qs, ks, vs = (t.transpose(1, 2) for t in inputs)
        else:
            b, _, dim_head = inputs[0].shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in inputs)
        reason = _ineligible(qs, ks, vs, dim_head)
        if reason is not None:
            return dense(reason)
        if state.selection == "reblock":
            return _run_reblock(state, q, k, v, qs, ks, vs, heads, skip_output_reshape, b,
                                dim_head, decision, kwargs.get("scale"),
                                transformer_options.get("minimax_h3_layout"), owned, dense,
                                transformer_options)
        sinks = h3_sinks(transformer_options, tokens, state.protect_audio, state.protect_prefix)
        if sinks is None:
            return dense("H3 layout unavailable for requested protection")
        sink, sink_q = sinks
        plan = None
        if state.selection == "vsa":
            plan = state.vsa_plan(transformer_options, tokens, inputs[0].device)
            if plan is None:
                return dense("H3 layout unavailable for VSA")
            gate = take_gate(transformer_options, state, plan)
            qs, ks, vs = gather_qkv(qs, ks, vs, plan["src"])
            sink = sink_q = (0, plan["n_prefix"])
            state.info_once(
                ("vsa", plan["n"], plan["n_prefix"], gate is not None),
                "VSA: %d padded rows, %d prefix tiles, keep %.1f%%, %s"
                % (plan["n"], plan["n_prefix"], 100.0 * float(decision.topk),
                   "fine + coarse gate" if gate is not None else "fine stage only (no coarse gate)"))
        options = state.sparse_kwargs(kwargs.get("scale"), sink, sink_q, decision)
        if plan is not None:
            options["block_len"] = plan["block_len"]
            if gate is not None:
                options["coarse_gate"] = gate if gate.dtype == torch.bfloat16 else gate.to(torch.bfloat16)
                del gate
        # The accepted SLA Kitchen path uses BF16 even for FP16 inputs.
        input_dtype = qs.dtype
        prepared = None
        if owned and state.prepared_enabled:
            if plan is None:
                # Do not keep borrowed views alive while the owner frame consumes QKV.
                del inputs, qs, ks, vs
                prepared, qs, ks, vs = _prepare_or_disable(
                    state, q, k, v, heads, skip_reshape, b, dim_head, options)
            else:
                prepared, qs, ks, vs = _prepare_gathered(
                    state, q, k, v, qs, ks, vs, options)
        if prepared is not None:
            return _consume_prepared(state, prepared, input_dtype, skip_output_reshape,
                                     b, heads, dim_head, tokens, sink, sink_q, plan)
        return _run_native_sparse(state, q, k, v, qs, ks, vs, heads, skip_output_reshape,
                                  b, dim_head, sink, sink_q, options, input_dtype, owned, dense,
                                  plan)

    override.container_function = partial(override, None)
    for fn in (override, override.container_function):
        fn._curve_state = state
        fn._curve_previous = previous
    return override


def is_curve_override(fn) -> bool:
    return getattr(fn, "_curve_state", None) is not None


def is_dense_backend_override(fn) -> bool:
    """True when the override is Model Attention Backend's one-line forwarder,
    identified by the captured attention function, not by __qualname__."""
    closure = getattr(fn, "__closure__", None)
    if not closure or len(closure) != 1:
        return False
    try:
        captured = closure[0].cell_contents
    except ValueError:
        return False
    from comfy.ldm.modules.attention import get_attention_function as real_get
    for name in DENSE_BACKENDS:
        backend = real_get(name, None)
        if backend is not None and captured is backend:
            return True
    return False


def check_override_slot(transformer_options):
    current = (transformer_options or {}).get(OVERRIDE_KEY)
    if current is not None and not (is_curve_override(current) or is_dense_backend_override(current)):
        raise ValueError(
            f"{WHO}: MODEL already has a non-CurveAttn attention patch. Unwire Model Sparse "
            "Attention, Kitchen Sparse, Sage, etc. Model Attention Backend can stay.")


def install_override(state: SparseState, transformer_options):
    """Put this node's override on top unless it is already in the chain."""
    current = transformer_options.get(OVERRIDE_KEY)
    if any(getattr(fn, "_curve_state", None) is state for fn in override_chain(current)):
        return
    check_override_slot(transformer_options)
    transformer_options[OVERRIDE_KEY] = make_attention_override(state, current)


@contextmanager
def reduction_policy(disable):
    """Match SLA's reduction flags and restore them even after cancellation."""
    if not disable:
        yield
        return
    backend = torch.backends.cuda.matmul
    names = ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
    saved = {}
    try:
        for name in names:
            value = getattr(backend, name)
            split_k = getattr(backend, name + "_split_k", None)
            saved[name] = value if split_k is None else (value, split_k)
            setattr(backend, name, False)
        yield
    finally:
        for name, value in saved.items():
            setattr(backend, name, value)


def publish(options, state: SparseState, decision: StepDecision):
    """Copy options and record this node's decision, ownership and policy."""
    out = dict(options or {})
    ns = dict(out.get(NS) or {})
    sparse = dict(ns.get("sparse") or {})
    sparse[state.uid] = decision
    ns["sparse"] = sparse
    ns.setdefault("owner", None)
    if not decision.dense and state.seq > ns.get("owner_seq", 0):
        ns["owner"], ns["owner_seq"] = state.uid, state.seq
    policy = dict(ns.get("policy_parts") or {})
    policy[state.seq] = state.static_policy() + (decision.dense,)
    ns["policy_parts"] = policy
    ns["policy"] = tuple(policy[key] for key in sorted(policy))
    out[NS] = ns
    return out


def make_step_wrapper(state: SparseState):
    def wrapper(executor, x, timestep, context, transformer_options=None,
                minimax_payload=None, **kwargs):
        options = transformer_options or {}
        to = publish(options, state, state.decide(options))
        if state.qkv_mode == "chunked" and "multigpu_thread_device" in to:
            raise ValueError(f"{WHO}: chunked qkv_mode is not validated for multigpu")
        pause = pause_malloc_graph() if state.qkv_mode == "chunked" else nullcontext()
        with reduction_policy(state.disable_fp16_accum), pause:
            return executor(x, timestep, context, transformer_options=to,
                            minimax_payload=minimax_payload, **kwargs)

    return wrapper


# ---------------------------------------------------------------- apply

def apply_sparse(model, *, selection="sla", keep_percent=20.0, keep_percent_end=-1.0,
                 tau=1.3, tau_end=-1.0, min_tokens=12288, dense_steps="1",
                 dense_last_steps=0, start_percent=0.0, end_percent=1.0, extra_tokens=0,
                 tail=False, protect_audio=False, protect_prefix=False,
                 disable_fp16_accum=True, dense_backend="comfy_kitchen_int8",
                 qkv_mode="native", verbose=False, reblock_summary="pooled",
                 reblock_updates=1, reblock_refresh=2):
    selection = str(selection or "sla")
    validate_sparse(selection=selection, keep_percent=keep_percent,
                    keep_percent_end=keep_percent_end, tau=tau, tau_end=tau_end,
                    min_tokens=min_tokens, dense_last_steps=dense_last_steps,
                    start_percent=start_percent, end_percent=end_percent,
                    extra_tokens=extra_tokens, dense_backend=dense_backend, qkv_mode=qkv_mode,
                    reblock_summary=reblock_summary, reblock_updates=reblock_updates,
                    reblock_refresh=reblock_refresh)
    steps = parse_steps(dense_steps)
    diffusion_model = require_h3(model, WHO)
    check_object_patches(model, WHO)
    blocks.check_dit(model.model_options, WHO)
    check_override_slot(model.model_options.get("transformer_options"))
    cache = blocks.current_cache(model.model_options)
    if cache is not None and cache.span == "whole" and qkv_mode != "native" \
            and selection != "reblock":
        raise ValueError(f"{WHO}: whole-span CurveAttn Cache requires native qkv_mode")
    if not ck.sol_attn_is_available():
        raise RuntimeError(f"{WHO}: Comfy Kitchen sol_attn is not available on this GPU.")
    dense_attention = get_attention_function(dense_backend, None)
    if dense_attention is None:
        raise RuntimeError(f"{WHO}: dense backend {dense_backend} is unavailable.")
    prepared_available = False
    if qkv_mode == "owned_bf16":
        prepared_available = True
        try:
            check_prepared_available()
        except (RuntimeError, OSError) as exc:
            prepared_available = False
            LOG.warning("%s: %s; using native ck.sol_attn without owned BF16 copies", WHO, exc)
    vsa_gate = selection == "vsa" and has_gate(diffusion_model)
    if selection == "vsa" and (tail or extra_tokens or protect_audio or protect_prefix):
        LOG.info("%s: vsa ignores tail, extra tokens, and protect toggles; "
                 "prefix tiles stay dense", WHO)
    if selection == "vsa" and not vsa_gate:
        LOG.warning("%s: model has no to_gate_compress layers; VSA runs the fine stage "
                    "without the coarse branch", WHO)
    if selection == "vsa":
        tail = False
        extra_tokens = 0
        protect_audio = False
        protect_prefix = False
    if selection == "reblock":
        if tail or extra_tokens or protect_audio or protect_prefix:
            LOG.info("%s: reblock ignores tail, extra tokens, and protect toggles; "
                     "conditioning rows stay dense and exact", WHO)
        if qkv_mode != "native":
            LOG.info("%s: reblock orders rows from the whole K; qkv_mode=%s runs as native",
                     WHO, qkv_mode)
        tail = False
        extra_tokens = 0
        protect_audio = False
        protect_prefix = False
        qkv_mode = "native"
        prepared_available = False
    keep_ratio_end = (float(keep_percent_end) / 100.0
                      if float(keep_percent_end) >= 0.5 and selection == "sla" else None)
    scheduled_tau_end = (float(tau_end)
                         if float(tau_end) >= 0.0 and selection == "sol-attn" else None)
    sigma_start, sigma_end = resolve_sigma_bounds(model, start_percent, end_percent)
    state = SparseState(
        dense_attention=dense_attention, selection=selection,
        keep_ratio=float(keep_percent) / 100.0, keep_ratio_end=keep_ratio_end,
        tau=float(tau), tau_end=scheduled_tau_end, min_tokens=int(min_tokens),
        dense_steps=steps, dense_last_steps=int(dense_last_steps), tail=bool(tail),
        token_aug=int(extra_tokens), protect_audio=bool(protect_audio),
        protect_prefix=bool(protect_prefix), disable_fp16_accum=bool(disable_fp16_accum),
        sigma_start=sigma_start, sigma_end=sigma_end, dense_backend=dense_backend,
        qkv_mode=qkv_mode, prepared_available=prepared_available, verbose=bool(verbose),
        reblock_summary=reblock_summary, reblock_updates=int(reblock_updates),
        reblock_refresh=int(reblock_refresh))

    state.vsa_gate = vsa_gate
    m = model.clone()
    install_override(state, m.model_options["transformer_options"])
    block_provider = qkv_mode == "chunked" or vsa_gate
    if qkv_mode == "chunked":
        blocks.install(m, diffusion_model, provider=state)
    elif vsa_gate:
        blocks.install(m, diffusion_model, provider=state,
                       provider_factory=make_vsa_gate_attention)

    def _prepare(model_patcher, timestep, model_options):
        install_override(state, model_options["transformer_options"])
        blocks.verify(model_options, WHO, provider=state if block_provider else None)

    key = f"curvattn_{state.uid}"
    m.add_callback_with_key(CallbacksMP.ON_PREPARE_STATE, key, _prepare)
    m.add_callback_with_key(CallbacksMP.ON_CLEANUP, key, lambda model_patcher: state.cleanup())
    m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, state.outer)
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, make_step_wrapper(state))
    install_cond_batch_limiter(m)
    stacked = [s.uid for s in curve_states(m.model_options["transformer_options"])]
    LOG.info(
        "%s %s [%s]: selection=%s keep=%.1f%% keep_end=%.1f tau=%.3f tau_end=%.3f "
        "extra_tokens=%d start_percent=%.3f end_percent=%.3f sigma_start=%s sigma_end=%s "
        "qkv_mode=%s prepared_qkv=%s dense_backend=%s dense_steps=%s dense_last_steps=%d verbose=%s "
        "protect_prefix=%s protect_audio=%s tail=%s disable_fp16_accum=%s vsa_gate=%s stack=%s",
        WHO, VERSION, state.uid, selection, float(keep_percent), float(keep_percent_end),
        float(tau), float(tau_end), int(extra_tokens), float(start_percent), float(end_percent),
        None if sigma_start is None else f"{sigma_start:.6g}",
        None if sigma_end is None else f"{sigma_end:.6g}",
        qkv_mode, state.prepared_enabled, dense_backend, sorted(steps), state.dense_last_steps, state.verbose,
        state.protect_prefix, state.protect_audio, state.tail, state.disable_fp16_accum, vsa_gate,
        stacked)
    return m
