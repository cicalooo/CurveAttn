# CurveAttention

MiniMax H3 sparse attention for ComfyUI, running on stock Comfy Kitchen
`ck.sol_attn`, with an optional experimental residual cache.

Version 2.5.

## Requirements

- ComfyUI 0.32.0 or newer
- A CUDA GPU with a compiled Comfy Kitchen `sol_attn` kernel
  (`comfy-kitchen` 0.2.35 or newer)
- A MiniMax H3 model

Nothing else is supported. The node checks the model type and the Kitchen
kernel and refuses to load otherwise. No weights and no core forward
implementation are replaced: FFN, AdaLN, LoRA and the sampler stay in ComfyUI.

## Install

Clone into `custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cicalooo/CurveAttention.git CurveAttention
```

Restart ComfyUI. One node is added, **CurveAttention**, in `model/patch/unet`.

## Usage

Wire the loader `MODEL` into CurveAttention, and CurveAttention `MODEL` into
the sampler. Defaults are conservative: SLA top-k at 20% keep, full dense for
sequences under 12288 tokens, and the residual cache off.

Widgets are grouped into collapsible sections — SLA, Protect, Sol-Attn,
Backend, Cache. Closing a section does not silently change what is armed;
the Protect and Sol-Attn sections reset their own optional knobs, and closing
Cache forces the cache off.

Every run logs a line at patch time and a summary at the end. The one to watch
is the cache line: `CurveAttention cache: hits=0` means the cache did no work.

## What it does

- **Sparse path** is Comfy Kitchen Sol, either `sla` (fixed keep %) or
  `sol-attn` (adaptive tau threshold, ignores keep %).
- **QKV mode** — `native` keeps aligned views, `owned_bf16` copies Q/K/V only
  when the tensor does not already own its storage, `chunked` streams last-step
  stats. Chunked is experimental: the first call of a run may project twice, and
  later steps reuse the previous `kmean` / `vscale`.
- **CFG branches run one at a time.** H3 rejects batch > 1, so a batched
  cond+uncond call crashes before any block runs. Splitting the branches is what
  lets the residual cache see each one.
- **Residual cache** reuses transformer-block residuals between steps. It is
  experimental, off by default, and only helps when it actually hits.

## Presets

Cache presets set starting points, not card tunes. None of them set Max MiB.

| Preset | What it sets |
|---|---|
| `off` | cache disabled |
| `manual` | leaves the widgets as they are |
| `turbo_safe` | threshold 0.05, window 0.15–0.85, min steps 4, audio guard |
| `native` | threshold 0.08, window 0.15–0.85, both guards |
| `fast` | threshold 0.10, window 0.10–0.90, both guards |

`turbo_safe` is the short-schedule (4–8 step turbo LoRA) starting point.
`native` and `fast` are the long-schedule starting points and are unvalidated.
On a 32 GB card, raise Max MiB yourself; 2048–3072 is a guess, not a measurement.

## Cache limits

Read these before turning the cache on.

- `protect_audio` hard-disables the cache. `cache_audio_guard` only refuses a hit.
- Masks, refs, keyframes and multigpu bypass the cache entirely.
- Cache and `chunked` mode are rejected on multigpu.
- A keep-% or tau lerp no longer invalidates the residual every step.
- `whole` span skips every transformer block on a hit, disables dynamic block
  prefetch, and requires `qkv_mode=native`. Middle-span residual add/sub is not
  proven compatible with malloc-graph capture. If a cached run is slower than a
  fresh one, turn the cache off.
- A cache threshold is a drift gate, not an output-error bound.
- No hits means no speedup. Check the log before expecting one.

## Do not stack

CurveAttention rejects a MODEL that already carries another attention or
block-cache patch, and a second CurveAttention. Unwire H3 SLA, Model Sparse
Attention, Kitchen Sparse, Sage, and external FirstBlockCache / TeaCache /
SeaCache / MotionCache nodes from the same MODEL wire.

Model Attention Backend can sit in front of it.

## Migrating from 2.4

Card presets are gone. A saved workflow that still names one will error — switch
it to `manual` and set the widgets, or pick `native` / `fast` / `turbo_safe` and
set Max MiB yourself.

Removed: `Native/3090`, `Fast/3090`, `Native/4090`, `Fast/4090`, `Native/5090`,
`Fast/5090`, `native_safe`, `native_fast`, `native_aggressive`.

`selection` moved into the always-applied core, so closing the Sol-Attn section
no longer flips it back to `sla`. Closing Backend no longer reverts
`dense_backend` or `qkv_mode`.

## Tests

CPU only, from the ComfyUI interpreter, no GPU:

```bash
python custom_nodes/CurveAttention/tools/run_cpu_tests.py --self-test
python custom_nodes/CurveAttention/tools/run_cpu_tests.py
```

The self-test proves the CPU guard actually blocks GPU work. GPU kernel tests
need `--allow-gpu`:

```bash
python custom_nodes/CurveAttention/tools/run_gpu_tests.py --allow-gpu
```

Passing either is not a speed or quality measurement.

## Notes

- [Cache speed modes](docs/cache-speed.md) — spans, modes, guards, telemetry.
- [Larger-card notes](docs/rtx5090-settings.md) — setting Max MiB yourself.
