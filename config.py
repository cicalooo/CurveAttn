from math import isfinite

def _starting_point(threshold, start, end):
    """Unvalidated cache window. Budget stays on the Max MiB widget."""
    return {
        "cache_enabled": True,
        "cache_threshold": threshold,
        "cache_start_percent": start,
        "cache_end_percent": end,
        "cache_max_hits": 1,
        "cache_span": "middle",
        "cache_mode": "fingerprint",
        "cache_min_steps": 6,
        "cache_temporal_guard": True,
        "cache_audio_guard": True,
    }


CACHE_PRESETS = {
    "off": {
        "cache_enabled": False,
    },
    "manual": None,
    "turbo_safe": {
        "cache_enabled": True,
        "cache_threshold": 0.05,
        "cache_start_percent": 0.15,
        "cache_end_percent": 0.85,
        "cache_max_hits": 1,
        "cache_span": "middle",
        "cache_mode": "fingerprint",
        "cache_min_steps": 4,
        "cache_temporal_guard": False,
        "cache_audio_guard": True,
    },
    "native": _starting_point(0.08, 0.15, 0.85),
    "fast": _starting_point(0.10, 0.10, 0.90),
}


def apply_cache_preset(preset, settings):
    """Return a shallow-copied settings dict with preset overrides applied.

    ``manual`` leaves values unchanged. ``off`` forces cache_enabled False.
    Named presets force cache_enabled and overwrite mapped fields.
    """
    out = dict(settings)
    key = str(preset or "manual")
    if key not in CACHE_PRESETS:
        raise ValueError("CurveAttention: invalid cache_preset")
    mapped = CACHE_PRESETS[key]
    if mapped is None:
        return out
    out.update(mapped)
    return out


def validate_options(*, keep_percent, min_tokens, dense_last_steps, selection,
                     tau, extra_tokens, start_percent, end_percent,
                     keep_percent_end, tau_end, dense_backend, qkv_mode,
                     cache_threshold, cache_max_hits, cache_max_mib, cache_span="middle",
                     cache_preset="manual", cache_start_percent=0.2, cache_end_percent=0.8,
                     cache_min_steps=6, cache_temporal_guard=False, cache_audio_guard=False,
                     cache_mode="fingerprint"):
    def real(name, value, low, high):
        if isinstance(value, bool) or not isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError(f"CurveAttention: invalid {name}")

    def integer(name, value, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"CurveAttention: invalid {name}")

    real("keep_percent", keep_percent, .5, 95)
    integer("min_tokens", min_tokens, 0, 1 << 20)
    integer("dense_last_steps", dense_last_steps, 0, 16)
    real("tau", tau, 0, 4)
    integer("extra_tokens", extra_tokens, 0, 256)
    if extra_tokens % 64:
        raise ValueError("CurveAttention: extra_tokens must be a multiple of 64")
    real("start_percent", start_percent, 0, 1)
    real("end_percent", end_percent, 0, 1)
    if start_percent > end_percent:
        raise ValueError("CurveAttention: start_percent exceeds end_percent")
    if keep_percent_end != -1:
        real("keep_percent_end", keep_percent_end, .5, 95)
    if tau_end != -1:
        real("tau_end", tau_end, 0, 4)
    if selection not in ("sla", "sol-attn"):
        raise ValueError("CurveAttention: invalid selection")
    if dense_backend not in ("comfy_kitchen_int8", "pytorch"):
        raise ValueError("CurveAttention: invalid dense_backend")
    if qkv_mode not in ("owned_bf16", "native", "chunked"):
        raise ValueError("CurveAttention: invalid qkv_mode")
    if cache_preset not in CACHE_PRESETS:
        raise ValueError("CurveAttention: invalid cache_preset")
    real("cache_threshold", cache_threshold, 0, .2)
    integer("cache_max_hits", cache_max_hits, 1, 2)
    integer("cache_max_mib", cache_max_mib, 0, 8192)
    if cache_span not in ("middle", "whole"):
        raise ValueError("CurveAttention: invalid cache span")
    real("cache_start_percent", cache_start_percent, 0, 1)
    real("cache_end_percent", cache_end_percent, 0, 1)
    if cache_start_percent > cache_end_percent:
        raise ValueError("CurveAttention: cache_start_percent exceeds cache_end_percent")
    integer("cache_min_steps", cache_min_steps, 2, 64)
    if cache_mode not in ("fingerprint", "first_block"):
        raise ValueError("CurveAttention: invalid cache_mode")
