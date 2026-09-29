from __future__ import annotations

import torch
import comfy_kitchen as ck
from comfy.model_management import cast_to


def projection_chunks(attention, x, chunk_tokens=2048):
    if chunk_tokens <= 0 or chunk_tokens % 64:
        raise ValueError("CurveAttention: chunk_tokens must be positive and 64-aligned")

    def factory():
        for start in range(0, x.shape[0], chunk_tokens):
            yield attention.qkv_proj(x[start:start + chunk_tokens]).to(torch.bfloat16)

    return factory


def chunked_reason(attention, x, rope_freqs, state, options):
    if options.get("_curve_dense"):
        return "scheduled dense"
    if "multigpu_thread_device" in options:
        return "multigpu unsupported"
    if x.device.type != "cuda":
        return "not CUDA"
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        return "requires BF16 packed rows"
    if torch.is_grad_enabled() or x.requires_grad:
        return "inference only"
    if state.failure_reason is not None:
        return "run disabled sparse"
    if x.shape[0] < state.min_tokens or attention.head_dim != 128:
        return "shape below sparse contract"
    if rope_freqs is None or rope_freqs.ndim != 6 or rope_freqs.device != x.device:
        return "rope unavailable"
    rot = rope_freqs.shape[-3] * 2
    if rot <= 0 or rot > 128 or rot % 8:
        return "rope dimension unsupported"
    if attention.q_norm.eps != attention.k_norm.eps:
        return "different Q/K norm epsilons"
    if not callable(getattr(ck, "sol_attn_chunked", None)):
        return "Kitchen chunked API unavailable"
    if not ck.sol_attn_is_available(x.device):
        return "Kitchen compiled Sol unavailable"
    return None


def run_chunked(attention, x, rope_freqs, options, *, kernel=None, cast=None,
                kmean=None, vscale=None, on_stats=None):
    kernel = ck.sol_attn_chunked if kernel is None else kernel
    cast = (lambda weight: cast_to(weight, device=x.device)) if cast is None else cast
    out, kmean_next, vscale_next = kernel(
        projection_chunks(attention, x), t=x.shape[0], h=attention.heads,
        rope_freqs=rope_freqs,
        qk_norm_weights=(cast(attention.q_norm.weight), cast(attention.k_norm.weight)),
        kmean=kmean, vscale=vscale, rope_eps=attention.q_norm.eps, **options)
    if on_stats is not None:
        on_stats(kmean_next, vscale_next)
    return attention.out_proj(out.reshape(x.shape[0], attention.heads * attention.head_dim))


def make_chunked_attention(attention, state, index=0):
    from .curve_attention import h3_sinks, _sparse_options

    def call(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        reason = chunked_reason(attention, x, rope_freqs, state, options)
        sinks = h3_sinks(options, x.shape[0], state.protect_audio, state.protect_prefix)
        if sinks is None:
            reason = reason or "layout unavailable for protection"
        if reason is not None:
            state.log_once(("chunked_fallback", reason), "chunked bypass: " + reason)
            return attention(x, rope_freqs=rope_freqs, transformer_options=options)
        sparse_options = _sparse_options(state, None, sinks[0], sinks[1])
        run = state.run_scope.current() if state.run_scope is not None else None
        prev = run.chunked.get(index, (None, None)) if run is not None else (None, None)

        def keep(kmean_next, vscale_next):
            if run is not None and kmean_next is not None:
                run.chunked[index] = (kmean_next, vscale_next)

        result = run_chunked(
            attention, x, rope_freqs, sparse_options,
            kmean=prev[0], vscale=prev[1], on_stats=keep)
        state.chunked_calls += 1
        state.sparse_calls += 1
        state.log_once(("chunked", index), f"selector=chunked block={index}")
        return result

    return call
