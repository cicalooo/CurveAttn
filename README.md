# CurveAttn

MiniMax H3 sparse attention and an experimental residual cache for ComfyUI,
as two independent V3 nodes. Sparse attention runs on the stock Comfy Kitchen
`ck.sol_attn` kernel; the cache needs no special kernel at all.

Version 3.4.0. Sparse `vsa` (4×4×4 video cubes) now runs fine + coarse gate
(`to_gate_compress`) when the checkpoint has it, works with `chunked` QKV, and
keeps an LRU plan cache. The cache still works with denoise masks, references and keyframes,
and both nodes understand multi-stage samplers (see below).

3.x is a clean break from CurveAttention 2.x: the old `CurveAttention` node,
its widget layout and its legacy cache preset names are gone, and workflows
using it must be rebuilt with the nodes below.

## Requirements

- ComfyUI with the V3 node API (`comfy_api.latest`, `io.DynamicCombo`) and
  MiniMax H3 support.
- **CurveAttn Sparse:** Comfy Kitchen with a compiled `sol_attn` kernel for your
  GPU (CUDA). `qkv_mode=chunked` additionally needs `ck.sol_attn_chunked`.
- **CurveAttn Residual Cache:** any MiniMax H3 model; works on a plain dense model.

## Nodes

Both live in `model/patch/CurveAttn`, take `MODEL` and return a patched `MODEL`.
Neither changes weights or the core forward.

### CurveAttn Sparse (`CurveAttnSparse`)

| Input | Default | Meaning |
|---|---|---|
| Selection | `sla` | `sla`: fixed top-k key blocks. `sol-attn`: adaptive tau threshold. `vsa`: 4×4×4 video cubes. Same kernel. |
| ↳ sla: Keep % / Keep % end | 20 / -1 | Percent of key blocks each query keeps; optional linear schedule toward *end* over the run (-1 off). |
| ↳ sol-attn: Tau / Tau end | 1.3 / -1 | Threshold (higher is sparser); optional schedule (-1 off). |
| ↳ vsa: Keep % | 10 | Percent of video cubes each query cube keeps. Fine stage only (no `to_gate_compress` / coarse gate). Prefix tiles, including text, audio, stills and guides, stay dense. Ignores tail, extra tokens and the protect toggles. `chunked` QKV is refused. |
| Min tokens | 12288 | Shorter sequences stay dense. |
| Dense steps | `1` | 0-based steps that always run dense (`1`, `0-1`, `0,2`). |
| Dense last steps | 0 | Force the last N steps dense. |
| Sparse start / end | 0 / 1 | Sampling-progress window where sparse is allowed. |
| Protect audio | off | Keep target-audio query rows dense. Does **not** affect the cache. |
| Protect prefix *(advanced)* | off | Keep all key blocks before target video exact. |
| Extra tokens *(advanced)* | 0 | Extra top-scoring tokens (multiple of 64). |
| Tail *(advanced)* | off | Pooled-tail correction. |
| Disable FP16 accum *(advanced)* | on | Full-precision matmul reductions during steps (restored after). |
| Dense backend *(advanced)* | `comfy_kitchen_int8` | Dense fallback when nothing CurveAttn-compatible sits below. |
| QKV mode *(advanced)* | `native` | `native` aligned views, `owned_bf16` prepared copies, `chunked` streamed projection. |
| Verbose *(advanced)* | off | Log sparse/dense decisions once per shape. |

### CurveAttn Residual Cache (`CurveAttnCache`)

In the graph means on; bypass the node to turn it off.

| Input | Default | Meaning |
|---|---|---|
| Preset | `manual` | `manual` shows every knob below. `turbo_safe`, `native`, `fast` are unvalidated starting points (no extra widgets). |
| ↳ Threshold | 0.01 | Max sampled drift vs the last fresh step. 0 disables reuse. Not an error bound. |
| ↳ Max hits | 1 | Consecutive reuses before a fresh stack (1–2). |
| ↳ Span | `middle` | `middle` keeps first/last blocks fresh; `whole` skips every block on a hit. |
| ↳ Cache start / end | 0.2 / 0.8 | Step-progress window where reuse may happen. |
| ↳ Min steps | 6 | Shortest schedule that may hit. |
| ↳ Temporal guard | off | Also bound the worst video row-chunk drift. |
| ↳ Audio guard | off | Refuse a hit if audio drifted (keeps the store). |
| ↳ Mode | `fingerprint` | `fingerprint` samples text/audio/video drift; `first_block` uses the span-input delta. |
| Max MiB | 1024 | Residual budget across at most two CFG branches. Presets never set it. |

Preset values: `turbo_safe` threshold 0.05, window 0.15–0.85, min steps 4,
audio guard. `native` 0.08, 0.15–0.85, temporal+audio guard. `fast` 0.10,
0.10–0.90, temporal+audio guard.

The cache is skipped (and logged once) for multi-GPU, batched CFG branches,
sigmas that are not on the schedule, schedules shorter than Min steps, and a
denoise mask that cannot be aligned to the packed video/audio rows. Both nodes
split CFG into one branch per call because H3 rejects batch > 1.

#### References and keyframes

Reference images/audio/video (`ref_img`, `ref_audio` rows) and keyframes
(`cond`, `cond_audio` rows) are supported:

- The cached residual covers the whole packed sequence, conditioning rows
  included, so a hit updates every row exactly as a fresh middle stack would
  (to the same approximation as plain t2v: conditioning rows attend to the
  noisy target, so their per-block output also moves between steps).
- Drift (fingerprint, temporal and audio guards) is measured on the denoised
  text, audio and video rows only. Conditioning rows, and denoise-mask rows
  that are fully preserved, sit at a pinned near-clean timestep and do not
  count toward the drift decision.
- The packed layout is part of the residual signature: a different reference
  or keyframe set (or a different shape) refreshes instead of reusing.
- When a conditioning stream stops being pinned (its target timestep passes
  the cond noise-aug pin, normally only in the last sliver of the schedule),
  the cache refreshes at that step.
- Any segment kind other than text, audio, video, cond, cond_audio, ref_img
  and ref_audio makes the cache bypass for that call.

#### Masked edit and inpaint

`denoise_mask` / `audio_denoise_mask` are supported. That is the path used by
masked-edit, soft-mask, retake, and any other node that sets a latent
`noise_mask` (CF H3 Masked Edit, CF H3 Soft Mask Edit, H3 Retake, and the same
pattern). Add Guide / keyframes do not set a denoise mask; they were already
supported as conditioning rows.

- Preserved rows (mask ≈ 0) are carried in the residual and left out of the
  drift decision, same as keyframes. Drift is sampled from regenerating rows,
  weighted by mask strength, so a small edit region is not hidden by frozen
  background.
- The mask pattern is part of the residual signature. A mid-run change (Soft
  Mask Edit releasing the mask, or any `denoise_mask_function` that changes
  the pattern) refreshes instead of reusing. Per-row timesteps are not in the
  signature; they move every step the same way the global timestep already does.
- A mask that is non-finite or does not line up with the packed video/audio
  rows bypasses for that call.
- Same approximation as plain t2v. Keep the threshold conservative. Temporal
  guard still bounds the worst regenerating chunk.

#### Multi-stage samplers

A call whose sigma lies strictly between two schedule points (heun-style
midpoints, `dpmpp_2s`, `res_2s`, ...) belongs to the enclosing step. The
Sparse node applies that step's schedule (dense steps, keep/tau schedule) to
it. The cache runs such substeps in full and leaves its store alone: it
neither refreshes from nor reuses on a substep, so hits only happen on main
steps and the hit chain is not broken. Sigmas that are off the schedule,
ambiguous, the final sigma, or non-finite still bypass. The run summary
reports `substeps=`.

## Stacking and order

Wire `loader → [Model Attention Backend] → CurveAttn nodes → sampler`. The two
CurveAttn nodes can be chained in either order with the same result:

- **Sparse** owns the attention override. It installs itself on top of an
  existing CurveAttn Sparse or Model Attention Backend and keeps it as the
  dense fallback below. Any other attention patch (Model Sparse Attention,
  Kitchen Sparse, Sage, ...) is refused.
- **Cache** and `qkv_mode=chunked` share one CurveAttn block dispatcher per
  transformer block, with a fixed layer order (cache outside, chunked attention
  inside). Foreign dit block replacements are refused; so is a second cache.
- **Two Sparse nodes** stack: on each step the most recently applied node whose
  schedule says "sparse" handles attention, and the others pass through. For
  example, an `sla` node for 0–0.5 followed by a `sol-attn` node for 0.5–1.
  With overlapping windows the later node wins.
- Each Sparse node publishes a policy fingerprint every step. The cache adds it
  to the residual signature, so a step where no sparse node is active (a dense
  step) refreshes instead of reusing a sparse residual.
- `span=whole` requires `qkv_mode=native` on every stacked Sparse node
  (checked in both orders).

Do not combine the cache with other block caches (FirstBlockCache, TeaCache,
SeaCache, MotionCache).

## Logs

Each node logs a line at patch time and a summary at the end of every run
(prefix `CurveAttn`). For the cache, watch `CurveAttn cache: hits=`: zero hits
means the cache did no work; the warning names the last eligibility blocker
(and the last layout/runtime bypass reason). An off-schedule sigma is logged
once with its value and the schedule range. A masked run that lined up logs
`denoise mask active: drift on regenerating rows` once; `denoise mask unaligned`
means the mask could not be mapped onto the packed rows.

## Tests

CPU only, no GPU or model weights:

```
python custom_nodes/CurveAttn/tools/run_cpu_tests.py            # all
python custom_nodes/CurveAttn/tools/run_cpu_tests.py test_composition
```

GPU kernel checks and the synthetic benchmark need explicit opt-in:
`tools/run_gpu_tests.py --allow-gpu`, `benchmark_harness.py --allow-gpu`.

## Layout

| File | Contents |
|---|---|
| `nodes.py` | The two V3 nodes and `comfy_entrypoint`. |
| `sparse.py` | Sparse engine: step decisions, override chain, apply. |
| `cache.py` | Presets, eligibility metadata, run scope, apply. |
| `residual_cache.py` | `MiddleCache` reuse logic and drift measures. |
| `blocks.py` | Shared per-block dispatcher (cache + chunked layers). |
| `chunked_attention.py` | `qkv_mode=chunked` producer. |
| `sol_attn.py`, `prepared_sol.py` | Kitchen kernel entry points. |
| `common.py` | Step clock, H3 guards, CFG split, shared helpers. |

## License

Apache 2.0, see `LICENSE`.
