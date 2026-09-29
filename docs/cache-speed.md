# Cache speed modes

CurveAttention uses stock Comfy Kitchen Sol attention. New nodes default to native
QKV views. Explicit saved `owned_bf16` settings remain unchanged; the local owned
path copies BF16 Q/K/V and is not a prequantized INT8 Sol backend.

## Dual-track design

Inference is two tracks — pick a preset for the track you run, do not mix assumptions:

1. **Turbo LoRA** (often 4–8 steps): use `cache_preset=turbo_safe` (or `off`).
   Tighter threshold (0.05), window 0.15–0.85, max_hits 1, min_steps 4, audio_guard on.
   Quality is fragile; prove hits with fixed-seed A/B before trusting speedup.
2. **Non-Turbo / native** (often ~20 steps, `res_multistep`): `native` then
   `fast` are unvalidated starting points (threshold 0.08 / 0.10, windows
   0.15–0.85 and 0.10–0.90, both guards, max_hits 1). They do not set the
   residual budget. Raise `cache_max_mib` on a larger card.

`cache_preset=manual` leaves the cache widgets as set. `cache_preset=off`
forces caching disabled. 2.4 card preset names are gone; a saved workflow
that still names one will error.

## Spans and modes

`cache_enabled=false` leaves residual caching off. `cache_span=middle` retains
fresh first/last transformer blocks. Experimental `cache_span=whole` can skip
all transformer blocks on a hit, but still runs native input/output processing
and the native Python block loop. Whole mode requires `qkv_mode=native` and
uses on-demand rather than dynamic-prefetched block weights, including fresh steps.

`cache_mode=fingerprint` (default) uses multi-modality sampled relative drift.
`cache_mode=first_block` uses residual-**input** delta at span begin as an
FBC-style predictor (post edge block 0 for middle; pre-all for whole). This is
**not** a full FirstBlockCache block-0 replace patch.

## Guards and eligibility

Both modes retain branch isolation, signed per-modality drift checks, a bounded
combined residual budget, fresh boundary steps, and T2AV eligibility limits.
Configurable `cache_start_percent` / `cache_end_percent` replace the old hard-coded
20–80% band (manual defaults remain 0.2 / 0.8). `cache_min_steps` replaces the
fixed ≥6 floor (manual default 6).

- `protect_audio=true` still **hard-disables** cache eligibility (existing contract).
- `cache_audio_guard=true` **soft-refuses** a hit when audio fingerprint drift ≥
  threshold without clearing the residual store (SeaCache-style). Presets enable it.
- `cache_temporal_guard=true` takes max(global, max video-row-chunk drift).

Masked/reference/multi-GPU paths and schedules under `cache_min_steps` are not
accelerated. A cache threshold is not an output-error bound.

## Telemetry

When cache is enabled, run end logs one clear line via `format_cache_report`:
hits, refreshes, bypasses, budget_bypasses, bytes, span, mode, start/end percent,
and max drift seen.

Start with threshold 0.01 and one consecutive hit (or a named preset). Keep all
attention, sampler, CFG and content settings fixed when comparing modes. Check
cache hit/admission counters: zero hits means no cache speedup. Choose memory
budget from actual entry bytes and available VRAM; do not assume 512 or 1024 MiB
fits two branches.

Whole mode is experimental until paired video/audio quality and end-to-end
wall-time measurements pass. CPU test success is not GPU performance evidence.
Do not stack with other attention or block-cache patches. No weights or core
forward implementation are replaced.

These widget defaults are **RTX 3090 / 24 GB provisional**, not measured best
settings. Caching stays off by default (`cache_preset=manual`, `cache_enabled=false`)
until target-card quality and timing pass. Existing saved node values are not migrated.
