from typing_extensions import override

from comfy_api.v0_0_2 import ComfyExtension, io

from .config import CACHE_PRESETS
from .curve_attention import apply_curve_attention


class CurveAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CurveAttention",
            display_name="CurveAttention",
            category="model/patch/unet",
            description=(
                "MiniMax H3 sparse attention through stock Comfy Kitchen. "
                "CFG branches run one at a time because H3 rejects batch>1; that is what "
                "lets the experimental residual cache see each branch. "
                "Do not stack with other attention or block-cache patch nodes."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "selection",
                    display_name="Selection",
                    options=["sla", "sol-attn"],
                    default="sla",
                    tooltip="sla uses a fixed keep %. sol-attn uses an adaptive tau threshold and ignores keep %.",
                ),
                io.Float.Input(
                    "keep_percent",
                    display_name="Keep %",
                    default=20.0,
                    min=0.5,
                    max=95.0,
                    step=0.5,
                    tooltip="Percent of key blocks each query keeps. 20 is the SLA default (~80% skipped). Higher is denser and slower.",
                ),
                io.Int.Input(
                    "min_tokens",
                    display_name="Min tokens",
                    default=12288,
                    min=0,
                    max=1 << 20,
                    step=512,
                    tooltip="Sequences shorter than this stay fully dense.",
                ),
                io.String.Input(
                    "dense_steps",
                    display_name="Dense steps",
                    default="1",
                    tooltip="0-based steps that always run dense. 1 means the second step (SLA default). Examples: 1 or 0-1.",
                ),
                io.Int.Input(
                    "dense_last_steps",
                    display_name="Dense last steps",
                    default=0,
                    min=0,
                    max=16,
                    tooltip="Force the last N sampler steps dense. 0 is the SLA default (off).",
                ),
                io.Boolean.Input(
                    "tail",
                    display_name="Tail",
                    default=False,
                    tooltip="Extra pooled-tail correction. Off matches SLA.",
                ),
                io.Boolean.Input(
                    "protect_audio",
                    display_name="Protect audio",
                    default=False,
                    tooltip="Keep target-audio rows dense. Off matches SLA. Also blocks cache.",
                ),
                io.Boolean.Input(
                    "verbose",
                    display_name="Verbose",
                    default=False,
                    tooltip="Log sparse vs dense once per shape.",
                ),
                io.Boolean.Input(
                    "protect_prefix",
                    display_name="Protect prefix",
                    default=False,
                    optional=True,
                    tooltip="Force every key block before target video exact, including references. Extra attention work. Off matches SLA.",
                ),
                io.Boolean.Input(
                    "disable_fp16_accum",
                    display_name="Disable FP16 accum",
                    default=True,
                    optional=True,
                    tooltip="Use full-precision matmul reductions during sparse attention, matching SLA. Restored after each step.",
                ),
                io.Float.Input(
                    "tau",
                    display_name="Tau",
                    default=1.3,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    optional=True,
                    tooltip="Sol-Attn threshold. Higher is sparser. Ignored when selection is sla.",
                ),
                io.Int.Input(
                    "extra_tokens",
                    display_name="Extra tokens",
                    default=0,
                    min=0,
                    max=256,
                    step=64,
                    optional=True,
                    tooltip="Extra top-scoring tokens added on top of selected blocks. 0 matches SLA.",
                ),
                io.Float.Input(
                    "start_percent",
                    display_name="Sparse start",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Sampling progress where sparse starts. 0 with Sparse end 1 keeps sparse on for the whole run.",
                ),
                io.Float.Input(
                    "end_percent",
                    display_name="Sparse end",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Sampling progress where sparse stops. 1 with Sparse start 0 keeps sparse on for the whole run.",
                ),
                io.Float.Input(
                    "keep_percent_end",
                    display_name="Keep % end",
                    default=-1.0,
                    min=-1.0,
                    max=95.0,
                    step=0.5,
                    optional=True,
                    tooltip="If 0.5 or higher and selection is sla, lerp Keep % toward this over the run. -1 disables.",
                ),
                io.Float.Input(
                    "tau_end",
                    display_name="Tau end",
                    default=-1.0,
                    min=-1.0,
                    max=4.0,
                    step=0.05,
                    optional=True,
                    tooltip="If 0 or higher and selection is sol-attn, lerp Tau toward this over the run. -1 disables.",
                ),
                io.Combo.Input(
                    "dense_backend",
                    display_name="Dense backend",
                    options=["comfy_kitchen_int8", "pytorch"],
                    default="comfy_kitchen_int8",
                    optional=True,
                    tooltip="Backend for dense refresh and fallback. pytorch skips Kitchen INT8.",
                ),
                io.Combo.Input(
                    "qkv_mode",
                    display_name="QKV mode",
                    options=["owned_bf16", "native", "chunked"],
                    default="native",
                    optional=True,
                    tooltip="native keeps aligned views. owned_bf16 copies only when QKV does not already own its storage. chunked streams last-step stats after the first call. Whole-cache needs native.",
                ),
                io.Boolean.Input(
                    "cache_enabled",
                    display_name="Enabled",
                    default=False,
                    optional=True,
                    tooltip="Experimental residual reuse. Unmasked T2AV only. No hits means no cache speedup.",
                ),
                io.Float.Input(
                    "cache_threshold",
                    display_name="Threshold",
                    default=.01,
                    min=0.,
                    max=.2,
                    step=.001,
                    optional=True,
                    tooltip="Max sampled drift vs the last fresh step. 0 disables reuse. Not an output-error bound.",
                ),
                io.Int.Input(
                    "cache_max_hits",
                    display_name="Max hits",
                    default=1,
                    min=1,
                    max=2,
                    step=1,
                    optional=True,
                    tooltip="Consecutive residual reuses before a fresh middle stack.",
                ),
                io.Int.Input(
                    "cache_max_mib",
                    display_name="Max MiB",
                    default=1024,
                    min=0,
                    max=8192,
                    step=64,
                    optional=True,
                    tooltip="Residual memory budget across at most two branches. 0 disables storage.",
                ),
                io.Combo.Input(
                    "cache_span",
                    display_name="Span",
                    options=["middle", "whole"],
                    default="middle",
                    optional=True,
                    tooltip="middle keeps edge blocks fresh. whole skips every transformer block on a hit and needs native QKV.",
                ),
                io.Combo.Input(
                    "cache_preset",
                    display_name="Preset",
                    options=list(CACHE_PRESETS),
                    default="manual",
                    optional=True,
                    tooltip=(
                        "Experimental. off disables. manual keeps the knobs. "
                        "turbo_safe / native / fast are unvalidated starting points, not card tunes. "
                        "They do not set Max MiB; raise that widget for a larger card. "
                        "Removed: Native/3090, Fast/3090, Native/4090, Fast/4090, Native/5090, "
                        "Fast/5090, native_safe, native_fast, native_aggressive."
                    ),
                ),
                io.Float.Input(
                    "cache_start_percent",
                    display_name="Cache start",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Earliest step progress that may reuse. Default 0.2.",
                ),
                io.Float.Input(
                    "cache_end_percent",
                    display_name="Cache end",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Latest step progress that may reuse. Default 0.8.",
                ),
                io.Int.Input(
                    "cache_min_steps",
                    display_name="Min steps",
                    default=6,
                    min=2,
                    max=64,
                    step=1,
                    optional=True,
                    tooltip="Minimum schedule length before cache can hit. Manual default 6; turbo_safe uses 4.",
                ),
                io.Boolean.Input(
                    "cache_temporal_guard",
                    display_name="Temporal guard",
                    default=False,
                    optional=True,
                    tooltip="Also require max video-row drift to stay under the threshold.",
                ),
                io.Boolean.Input(
                    "cache_audio_guard",
                    display_name="Audio guard",
                    default=False,
                    optional=True,
                    tooltip="Soft-refuse a hit if audio drifted, without clearing the store. protect_audio still hard-disables cache.",
                ),
                io.Combo.Input(
                    "cache_mode",
                    display_name="Mode",
                    options=["fingerprint", "first_block"],
                    default="fingerprint",
                    optional=True,
                    tooltip="fingerprint samples T/A/V drift. first_block uses input delta at the span start, not a full block-0 cache.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        keep_percent=20.0,
        min_tokens=12288,
        dense_steps="1",
        dense_last_steps=0,
        tail=False,
        protect_audio=False,
        verbose=False,
        protect_prefix=False,
        disable_fp16_accum=True,
        selection="sla",
        tau=1.3,
        extra_tokens=0,
        start_percent=0.0,
        end_percent=1.0,
        keep_percent_end=-1.0,
        tau_end=-1.0,
        dense_backend="comfy_kitchen_int8",
        qkv_mode="native",
        cache_enabled=False,
        cache_threshold=0.01,
        cache_max_hits=1,
        cache_max_mib=1024,
        cache_span="middle",
        cache_preset="manual",
        cache_start_percent=0.2,
        cache_end_percent=0.8,
        cache_min_steps=6,
        cache_temporal_guard=False,
        cache_audio_guard=False,
        cache_mode="fingerprint",
    ) -> io.NodeOutput:
        return io.NodeOutput(
            apply_curve_attention(
                model,
                keep_percent=keep_percent,
                min_tokens=min_tokens,
                dense_steps=dense_steps,
                dense_last_steps=dense_last_steps,
                tail=tail,
                protect_audio=protect_audio,
                verbose=verbose,
                protect_prefix=protect_prefix,
                disable_fp16_accum=disable_fp16_accum,
                selection=selection,
                tau=tau,
                extra_tokens=extra_tokens,
                start_percent=start_percent,
                end_percent=end_percent,
                keep_percent_end=keep_percent_end,
                tau_end=tau_end,
                dense_backend=dense_backend,
                qkv_mode=qkv_mode,
                cache_enabled=cache_enabled,
                cache_threshold=cache_threshold,
                cache_max_hits=cache_max_hits,
                cache_max_mib=cache_max_mib,
                cache_span=cache_span,
                cache_preset=cache_preset,
                cache_start_percent=cache_start_percent,
                cache_end_percent=cache_end_percent,
                cache_min_steps=cache_min_steps,
                cache_temporal_guard=cache_temporal_guard,
                cache_audio_guard=cache_audio_guard,
                cache_mode=cache_mode,
            )
        )


class CurveAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CurveAttention]


async def comfy_entrypoint() -> CurveAttentionExtension:
    return CurveAttentionExtension()
