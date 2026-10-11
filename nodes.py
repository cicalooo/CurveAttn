"""CurveAttn V3 nodes. Registered only through comfy_entrypoint."""
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .cache import CACHE_PRESETS, MANUAL_DEFAULTS, apply_cache
from .sparse import DENSE_BACKENDS, QKV_MODES, apply_sparse

CATEGORY = "model/patch/CurveAttn"


class CurveAttnSparse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CurveAttnSparse",
            display_name="CurveAttn Sparse",
            category=CATEGORY,
            is_experimental=True,
            search_aliases=["SLA", "Sol-Attn", "sparse attention"],
            description=(
                "MiniMax H3 sparse attention on stock Comfy Kitchen sol_attn. "
                "sla keeps a fixed top-k fraction of key blocks; sol-attn uses an adaptive "
                "tau threshold. Stacks with CurveAttn Residual Cache in either order and "
                "with another CurveAttn Sparse (the later one wins overlapping steps). "
                "Model Attention Backend may sit in front; other attention patches are refused."
            ),
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 model to patch."),
                io.DynamicCombo.Input("selection", display_name="Selection", options=[
                    io.DynamicCombo.Option("sla", [
                        io.Float.Input("keep_percent", display_name="Keep %", default=20.0,
                                       min=0.5, max=95.0, step=0.5,
                                       tooltip="Percent of key blocks each query keeps. 20 is the "
                                               "SLA default (~80% skipped). Higher is denser and slower."),
                        io.Float.Input("keep_percent_end", display_name="Keep % end", default=-1.0,
                                       min=-1.0, max=95.0, step=0.5,
                                       tooltip="If 0.5 or higher, lerp Keep % toward this over the "
                                               "run. -1 disables."),
                    ]),
                    io.DynamicCombo.Option("sol-attn", [
                        io.Float.Input("tau", display_name="Tau", default=1.3, min=0.0, max=4.0,
                                       step=0.05, tooltip="Adaptive threshold. Higher is sparser."),
                        io.Float.Input("tau_end", display_name="Tau end", default=-1.0, min=-1.0,
                                       max=4.0, step=0.05,
                                       tooltip="If 0 or higher, lerp Tau toward this over the run. "
                                               "-1 disables."),
                    ]),
                ], tooltip="sla: fixed keep %. sol-attn: adaptive tau threshold. Same Kitchen kernel."),
                io.Int.Input("min_tokens", display_name="Min tokens", default=12288, min=0,
                             max=1 << 20, step=512,
                             tooltip="Sequences shorter than this stay fully dense."),
                io.String.Input("dense_steps", display_name="Dense steps", default="1",
                                tooltip="0-based steps that always run dense. 1 means the second "
                                        "step (SLA default). Examples: 1, 0-1, 0,2."),
                io.Int.Input("dense_last_steps", display_name="Dense last steps", default=0,
                             min=0, max=16, tooltip="Force the last N sampler steps dense."),
                io.Float.Input("start_percent", display_name="Sparse start", default=0.0,
                               min=0.0, max=1.0, step=0.01,
                               tooltip="Sampling progress where sparse starts."),
                io.Float.Input("end_percent", display_name="Sparse end", default=1.0,
                               min=0.0, max=1.0, step=0.01,
                               tooltip="Sampling progress where sparse stops."),
                io.Boolean.Input("protect_audio", display_name="Protect audio", default=False,
                                 tooltip="Keep target-audio query rows dense. Does not affect "
                                         "the residual cache (use its audio guard)."),
                io.Boolean.Input("protect_prefix", display_name="Protect prefix", default=False,
                                 advanced=True,
                                 tooltip="Force every key block before target video exact, "
                                         "including references. Extra attention work."),
                io.Int.Input("extra_tokens", display_name="Extra tokens", default=0, min=0,
                             max=256, step=64, advanced=True,
                             tooltip="Extra top-scoring tokens on top of selected blocks. 0 matches SLA."),
                io.Boolean.Input("tail", display_name="Tail", default=False, advanced=True,
                                 tooltip="Extra pooled-tail correction. Off matches SLA."),
                io.Boolean.Input("disable_fp16_accum", display_name="Disable FP16 accum",
                                 default=True, advanced=True,
                                 tooltip="Full-precision matmul reductions during sampling steps, "
                                         "matching SLA. Restored after each step."),
                io.Combo.Input("dense_backend", display_name="Dense backend",
                               options=list(DENSE_BACKENDS), default="comfy_kitchen_int8",
                               advanced=True,
                               tooltip="Dense refresh/fallback when no Model Attention Backend "
                                       "or earlier CurveAttn Sparse sits below this node."),
                io.Combo.Input("qkv_mode", display_name="QKV mode", options=list(QKV_MODES),
                               default="native", advanced=True,
                               tooltip="native keeps aligned views. owned_bf16 copies only when "
                                       "QKV does not own its storage. chunked streams QKV "
                                       "projection chunks (whole-span cache needs native)."),
                io.Boolean.Input("verbose", display_name="Verbose", default=False, advanced=True,
                                 tooltip="Log sparse vs dense once per shape."),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
        )

    @classmethod
    def execute(cls, model, selection, min_tokens=12288, dense_steps="1",
                dense_last_steps=0, start_percent=0.0, end_percent=1.0,
                protect_audio=False, protect_prefix=False, extra_tokens=0, tail=False,
                disable_fp16_accum=True, dense_backend="comfy_kitchen_int8",
                qkv_mode="native", verbose=False) -> io.NodeOutput:
        mode = selection["selection"]
        return io.NodeOutput(apply_sparse(
            model, selection=mode,
            keep_percent=selection.get("keep_percent", 20.0),
            keep_percent_end=selection.get("keep_percent_end", -1.0),
            tau=selection.get("tau", 1.3), tau_end=selection.get("tau_end", -1.0),
            min_tokens=min_tokens, dense_steps=dense_steps, dense_last_steps=dense_last_steps,
            start_percent=start_percent, end_percent=end_percent, extra_tokens=extra_tokens,
            tail=tail, protect_audio=protect_audio, protect_prefix=protect_prefix,
            disable_fp16_accum=disable_fp16_accum, dense_backend=dense_backend,
            qkv_mode=qkv_mode, verbose=verbose))


def _manual_inputs():
    d = MANUAL_DEFAULTS
    return [
        io.Float.Input("threshold", display_name="Threshold", default=d["threshold"], min=0.0,
                       max=0.2, step=0.001,
                       tooltip="Max sampled drift vs the last fresh step. 0 disables reuse. "
                               "Not an output-error bound."),
        io.Int.Input("max_hits", display_name="Max hits", default=d["max_hits"], min=1, max=2,
                     tooltip="Consecutive residual reuses before a fresh stack."),
        io.Combo.Input("span", display_name="Span", options=["middle", "whole"],
                       default=d["span"],
                       tooltip="middle keeps edge blocks fresh. whole skips every block on a hit "
                               "and needs native QKV on stacked sparse nodes."),
        io.Float.Input("start_percent", display_name="Cache start", default=d["start_percent"],
                       min=0.0, max=1.0, step=0.01, tooltip="Earliest step progress that may reuse."),
        io.Float.Input("end_percent", display_name="Cache end", default=d["end_percent"],
                       min=0.0, max=1.0, step=0.01, tooltip="Latest step progress that may reuse."),
        io.Int.Input("min_steps", display_name="Min steps", default=d["min_steps"], min=2, max=64,
                     tooltip="Minimum schedule length before the cache can hit."),
        io.Boolean.Input("temporal_guard", display_name="Temporal guard",
                         default=d["temporal_guard"],
                         tooltip="Also require max video-row-chunk drift under the threshold."),
        io.Boolean.Input("audio_guard", display_name="Audio guard", default=d["audio_guard"],
                         tooltip="Refuse a hit if audio drifted, without clearing the store."),
        io.Combo.Input("mode", display_name="Mode", options=["fingerprint", "first_block"],
                       default=d["mode"],
                       tooltip="fingerprint samples T/A/V drift. first_block uses the input "
                               "delta at the span start."),
    ]


class CurveAttnCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = [io.DynamicCombo.Option(name, _manual_inputs() if mapped is None else [])
                   for name, mapped in CACHE_PRESETS.items()]
        return io.Schema(
            node_id="CurveAttnCache",
            display_name="CurveAttn Residual Cache",
            category=CATEGORY,
            is_experimental=True,
            search_aliases=["residual cache", "block cache"],
            description=(
                "Experimental branch-scoped reuse of the MiniMax H3 middle-block residual. "
                "In the graph means on (bypass the node to turn it off). Works on a dense H3 "
                "model and does not need sol_attn. Stacks with CurveAttn Sparse in either order. "
                "Do not combine with other block caches (FirstBlockCache, TeaCache, ...)."
            ),
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 model to patch."),
                io.DynamicCombo.Input("preset", display_name="Preset", options=options,
                                      tooltip="manual exposes every knob. turbo_safe / native / "
                                              "fast are unvalidated starting points, not card "
                                              "tunes; they do not set Max MiB."),
                io.Int.Input("max_mib", display_name="Max MiB", default=1024, min=0, max=8192,
                             step=64,
                             tooltip="Residual memory budget across at most two branches. "
                                     "0 disables storage."),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
        )

    @classmethod
    def execute(cls, model, preset, max_mib=1024) -> io.NodeOutput:
        name = preset["preset"]
        knobs = {k: v for k, v in preset.items() if k != "preset"}
        return io.NodeOutput(apply_cache(model, preset=name, max_mib=max_mib, **knobs))


class CurveAttnExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CurveAttnSparse, CurveAttnCache]


async def comfy_entrypoint() -> CurveAttnExtension:
    return CurveAttnExtension()
