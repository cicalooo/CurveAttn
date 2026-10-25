"""qkv_mode=chunked: stream QKV projection chunks into ck.sol_attn_chunked.

Works on any object with SparseState's interface (resolve, sparse_kwargs,
chunked_stats, counters); it does not import the sparse module.
"""
from __future__ import annotations

import torch
import comfy_kitchen as ck
from comfy.model_management import cast_to

from .common import h3_sinks
from .vsa import permute_rows


def projection_chunks(attention, x, chunk_tokens=2048):
    if chunk_tokens <= 0 or chunk_tokens % 64:
        raise ValueError("CurveAttn: chunk_tokens must be positive and 64-aligned")

    def factory():
        for start in range(0, x.shape[0], chunk_tokens):
            yield attention.qkv_proj(x[start:start + chunk_tokens]).to(torch.bfloat16)

    return factory


def chunked_reason(attention, x, rope_freqs, state, options):
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


def run_chunked_vsa(attention, x, rope_freqs, options, plan, *, kernel=None, cast=None,
                    kmean=None, vscale=None, on_stats=None, chunk_tokens=2048):
    """VSA through the chunked producer, as core Model Sparse Attention does:
    x rows are permuted per chunk, RoPE is pre-permuted once per plan, and the
    coarse gate (when the checkpoint has to_gate_compress) is filled per chunk."""
    if chunk_tokens <= 0 or chunk_tokens % 64:
        raise ValueError("CurveAttn: chunk_tokens must be positive and 64-aligned")
    kernel = ck.sol_attn_chunked if kernel is None else kernel
    cast = (lambda weight: cast_to(weight, device=x.device)) if cast is None else cast
    heads, head_dim = attention.heads, attention.head_dim
    n, src = plan["n"], plan["src"]
    gate_layer = getattr(attention, "to_gate_compress", None)
    kw = dict(options)
    kw.update(tail=False, block_len=plan["block_len"], token_aug=0,
              sink_blocks=[0, plan["n_prefix"]], sink_q=[0, plan["n_prefix"]])
    gate = None
    if gate_layer is not None:
        gate = x.new_empty(n, heads * head_dim)
        kw["coarse_gate"] = gate.view(1, n, heads, head_dim)

    def factory():
        for start in range(0, n, chunk_tokens):
            xc = permute_rows(x, src[start:start + chunk_tokens])
            if gate is not None:
                gate[start:start + xc.shape[0]] = gate_layer(xc)
            yield attention.qkv_proj(xc).to(torch.bfloat16)

    out, kmean_next, vscale_next = kernel(
        factory, t=n, h=heads, rope_freqs=rope_freqs,
        qk_norm_weights=(cast(attention.q_norm.weight), cast(attention.k_norm.weight)),
        kmean=kmean, vscale=vscale, rope_eps=attention.q_norm.eps, **kw)
    if on_stats is not None:
        on_stats(kmean_next, vscale_next)
    out = out.reshape(n, heads * head_dim)[plan["inv"]]
    return attention.out_proj(out)


def make_chunked_attention(attention, state, index=0):
    def call(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        decision, reason, _ = state.resolve(options)
        if decision is None:
            reason = reason or "not this node's sparse step"
        else:
            reason = chunked_reason(attention, x, rope_freqs, state, options)
        vsa = getattr(state, "selection", None) == "vsa"
        plan = None
        if vsa:
            sinks = ((0, 0), (0, 0))
            if reason is None:
                plan = state.vsa_plan(options, x.shape[0], x.device)
                if plan is None:
                    reason = "H3 layout unavailable for VSA"
        else:
            sinks = h3_sinks(options, x.shape[0], state.protect_audio, state.protect_prefix)
        if sinks is None:
            reason = reason or "layout unavailable for protection"
        if reason is not None:
            state.log_once(("chunked_fallback", reason), "chunked bypass: " + reason)
            # Native attention goes through the override chain as usual.
            return attention(x, rope_freqs=rope_freqs, transformer_options=options)
        sparse_options = state.sparse_kwargs(None, sinks[0], sinks[1], decision)
        stats = state.chunked_stats
        key = index if plan is None else (index, plan["n"])
        prev = stats.get(key, (None, None))

        def keep(kmean_next, vscale_next):
            if kmean_next is not None:
                stats[key] = (kmean_next, vscale_next)

        if plan is None:
            result = run_chunked(attention, x, rope_freqs, sparse_options,
                                 kmean=prev[0], vscale=prev[1], on_stats=keep)
        else:
            result = run_chunked_vsa(attention, x, state.vsa_rope_freqs(rope_freqs, plan),
                                     sparse_options, plan,
                                     kmean=prev[0], vscale=prev[1], on_stats=keep)
        state.chunked_calls += 1
        state.sparse_calls += 1
        state.log_once(("chunked", index), f"selector=chunked block={index}"
                       + ("" if plan is None else
                          f" vsa rows={plan['n']} prefix_tiles={plan['n_prefix']} "
                          f"coarse_gate={getattr(attention, 'to_gate_compress', None) is not None}"))
        return result

    return call
