"""Shared CurveAttn dit dispatcher.

Both nodes need the per-block ``("dit", "double_block", i)`` replacement
slot: the residual cache wraps the middle stack and chunked qkv swaps the
block's attention. A slot holds one callable, so CurveAttn installs a single
immutable ``CurveBlockDispatch`` per block and every node rebuilds it with
its own layer added. The layer order is fixed here (cache outside, chunked
attention or VSA gate inside), so node order in the graph does not matter. A fresh
dispatcher is built per apply, so the input MODEL is never mutated.
"""
from __future__ import annotations

from functools import partial

from .common import PREFIX


class CurveBlockDispatch:
    __slots__ = ("index", "count", "attention", "cache", "providers")

    def __init__(self, index, count, attention, cache=None, providers=()):
        self.index = index
        self.count = count
        self.attention = attention
        self.cache = cache
        self.providers = tuple(providers)

    def block_attention(self, options):
        """Chunked attention of the provider that owns this step, if any."""
        for state, fn in reversed(self.providers):
            decision, _, _ = state.resolve(options)
            if decision is not None:
                return fn
        return None

    def __call__(self, args, extra):
        options = args["transformer_options"]

        def fresh(block_args, block_extra):
            attention = self.block_attention(options)
            if attention is None:
                return block_extra["original_block"](block_args)
            local = dict(block_args)
            local["attention"] = attention
            return block_extra["original_block"](local)

        cache = self.cache
        run = cache.run_scope.current() if cache is not None else None
        if run is None:
            return fresh(args, extra)
        return run.cache.call(self.index, self.count, args, extra,
                              partial(cache.metadata, options, run), fresh)


def _dit(model_options) -> dict:
    to = (model_options or {}).get("transformer_options") or {}
    return (to.get("patches_replace") or {}).get("dit") or {}


def check_dit(model_options, who):
    """Refuse foreign dit block replacements (e.g. Model Sparse Attention)."""
    if any(not isinstance(v, CurveBlockDispatch) for v in _dit(model_options).values()):
        raise ValueError(
            f"{who}: model already has dit block replacements from another node "
            "(for example Model Sparse Attention or another block cache).")


def dispatchers(model_options) -> list[CurveBlockDispatch]:
    return [v for v in _dit(model_options).values() if isinstance(v, CurveBlockDispatch)]


def current_cache(model_options):
    for dispatch in dispatchers(model_options):
        if dispatch.cache is not None:
            return dispatch.cache
    return None


def install(patcher, diffusion_model, *, cache=None, provider=None, provider_factory=None):
    """Rebuild every block dispatcher on ``patcher`` (a clone) with a new layer."""
    check_dit(patcher.model_options, PREFIX)
    from .chunked_attention import make_chunked_attention
    factory = provider_factory or make_chunked_attention

    existing = _dit(patcher.model_options)
    if cache is not None and current_cache(patcher.model_options) is not None:
        raise ValueError(f"{PREFIX} Cache: this MODEL already has a CurveAttn residual cache.")
    model_blocks = diffusion_model.blocks
    count = len(model_blocks)
    for index, block in enumerate(model_blocks):
        old = existing.get(("double_block", index))
        layer_cache = old.cache if old is not None else None
        providers = old.providers if old is not None else ()
        if cache is not None:
            layer_cache = cache
        if provider is not None:
            providers = providers + ((provider, factory(block.attn, provider, index)),)
        patcher.set_model_patch_replace(
            CurveBlockDispatch(index, count, block.attn, layer_cache, providers),
            "dit", "double_block", index)


def verify(model_options, who, *, cache=None, provider=None):
    """ON_PREPARE_STATE check that nobody replaced or dropped our layer."""
    dit = _dit(model_options)
    if any(not isinstance(v, CurveBlockDispatch) for v in dit.values()):
        raise ValueError(f"{who}: another node replaced CurveAttn dit patches")
    if cache is not None and (not dit or any(v.cache is not cache for v in dit.values())):
        raise ValueError(f"{who}: CurveAttn cache layer missing from dit patches")
    if provider is not None and (not dit or any(
            all(state is not provider for state, _ in v.providers) for v in dit.values())):
        raise ValueError(f"{who}: CurveAttn chunked layer missing from dit patches")
