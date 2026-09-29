from __future__ import annotations

from .chunked_attention import make_chunked_attention


def make_block_dispatch(index, count, attention, state):
    chunked = (make_chunked_attention(attention, state, index)
               if state.qkv_mode == "chunked" else None)

    def dispatch(args, extra):
        options = args["transformer_options"]

        def fresh(block_args, block_extra):
            if chunked is None:
                return block_extra["original_block"](block_args)
            local = dict(block_args)
            local["attention"] = chunked
            return block_extra["original_block"](local)

        run = state.run_scope.current() if state.run_scope is not None else None
        if run is None or not state.cache_enabled:
            return fresh(args, extra)
        meta = options.get("_curve_meta", {"eligible": False})
        return run.cache.call(index, count, args, extra, meta, fresh)

    return dispatch
