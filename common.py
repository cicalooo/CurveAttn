"""Shared CurveAttn plumbing: logging, H3 guards, step clock, CFG split.

Nothing here knows about sparse attention or the residual cache; both nodes
build on it so they can be applied independently and in any order.
"""
from __future__ import annotations

import itertools
import logging
import re
from math import isfinite

import torch

from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.patcher_extension import WrappersMP

LOG = logging.getLogger("CurveAttn")
PREFIX = "CurveAttn"
# Per-step namespace inside transformer_options. Each node copies it before
# writing, so a forward never mutates the caller's dicts.
NS = "curve_attn"
BLOCK = 64
COND_BATCH_KEY = "curvattn_cond_batch"
OVERRIDE_KEY = "optimized_attention_override"

_SEQ = itertools.count(1)


def next_seq() -> int:
    """Monotonic apply order. Later-applied sparse nodes win overlapping steps."""
    return next(_SEQ)


def namespace(options) -> dict:
    return (options or {}).get(NS) or {}


# ---------------------------------------------------------------- step clock

_DENSE_STEPS_TOKEN = re.compile(r"\d+\s*-\s*\d+|\d+")
_DENSE_STEPS_SEPARATORS = re.compile(r"[\s,;]+")


def parse_steps(text: str) -> frozenset[int]:
    """Parse dense step specs like '1', '0-1', '0,2'. Reject leftover junk."""
    text = text or ""
    remainder = _DENSE_STEPS_SEPARATORS.sub("", _DENSE_STEPS_TOKEN.sub("", text))
    if remainder:
        raise ValueError(
            f"Invalid dense_steps {text!r}: unexpected characters {remainder!r} "
            "(only digits, ranges like 0-1, whitespace, commas, and semicolons are allowed)"
        )
    steps = set()
    for part in _DENSE_STEPS_TOKEN.findall(text):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            steps.update(range(min(a, b), max(a, b) + 1))
        else:
            steps.add(int(part))
    return frozenset(steps)


def schedule_tuple(value):
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            return ()
        value = value.detach().to(device="cpu", dtype=torch.float64).tolist()
    if not isinstance(value, (tuple, list)):
        return ()
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return ()
    if len(values) < 2 or not all(isfinite(v) for v in values):
        return ()
    if not all(a > b for a, b in zip(values, values[1:])):
        return ()
    return values


def _sigma_value(current):
    if isinstance(current, torch.Tensor):
        if current.numel() != 1:
            return None
        current = float(current.detach().reshape(()).item())
    elif isinstance(current, (tuple, list)):
        if len(current) != 1:
            return None
        current = current[0]
    try:
        current = float(current)
    except (TypeError, ValueError):
        return None
    return current if isfinite(current) else None


def _matches(a, b):
    return abs(a - b) <= max(1e-7, abs(a) * 1e-6)


def schedule_position(schedule, current):
    """Locate a model call on the run schedule: (step, total, substep).

    An exact match of schedule[i] (i < last) is step i. A sigma strictly inside
    (schedule[i], schedule[i+1]) is an intermediate stage of a multi-stage
    sampler (heun midpoints, dpmpp_2s, res_2s, ...) and maps to step i with
    substep=True. Ambiguous, out-of-range, final-sigma or non-finite values give
    step None.
    """
    if len(schedule) < 2:
        return None, 0, False
    total = len(schedule) - 1
    value = _sigma_value(current)
    if value is None:
        return None, total, False
    matches = [i for i, s in enumerate(schedule) if _matches(s, value)]
    if len(matches) > 1 or matches == [total]:
        return None, total, False
    if matches:
        return matches[0], total, False
    for i in range(total):
        if schedule[i] > value > schedule[i + 1]:
            return i, total, True
    return None, total, False


def exact_step(schedule, current):
    """Step index only for an exact schedule sigma (substeps give None)."""
    step, total, substep = schedule_position(schedule, current)
    return (None if substep else step), total


def resolve_step(options, schedule=()) -> tuple[int | None, int]:
    """Step index from the run schedule, else from sample_sigmas.

    Substeps resolve to their enclosing interval, so dense_steps and windows
    apply to every stage of a step.
    """
    schedule = schedule or schedule_tuple((options or {}).get("sample_sigmas"))
    if not schedule:
        return None, 0
    step, total, _ = schedule_position(schedule, (options or {}).get("sigmas"))
    return step, total


def scalar(value) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return scalar(value[0]) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lerp_schedule(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def schedule_progress(step0: int | None, n_steps: int) -> float:
    if step0 is None or n_steps <= 1:
        return 0.0
    return float(step0) / float(n_steps - 1)


def outside_sigma_window(sigma, sigma_start, sigma_end) -> bool:
    """Dense when sigma > sigma_start or sigma < sigma_end; False if unknown."""
    if sigma is None or sigma_start is None or sigma_end is None:
        return False
    return sigma > sigma_start or sigma < sigma_end


def resolve_sigma_bounds(model, start_percent: float, end_percent: float):
    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        return None, None
    if model_sampling is None or not hasattr(model_sampling, "percent_to_sigma"):
        return None, None
    try:
        return (float(model_sampling.percent_to_sigma(float(start_percent))),
                float(model_sampling.percent_to_sigma(float(end_percent))))
    except Exception as exc:
        LOG.warning("%s: percent_to_sigma failed (%s); sigma window disabled", PREFIX, exc)
        return None, None


# ---------------------------------------------------------------- H3 layout

def h3_sinks(transformer_options, tokens, protect_audio: bool, protect_prefix=False):
    """Optional native protections; both off matches the SLA reference run."""
    if not protect_audio and not protect_prefix:
        return (0, 0), (0, 0)
    layout = (transformer_options or {}).get("minimax_h3_layout")
    if layout is None or getattr(layout, "seq_len", None) != tokens:
        return None
    video = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
    kv = (0, (video[0] + BLOCK - 1) // BLOCK) if protect_prefix and video else (0, 0)
    if not protect_audio:
        return kv, (0, 0)
    audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
    rows = (audio[0] // BLOCK, (audio[1] + BLOCK - 1) // BLOCK) if audio else (0, 0)
    return kv, rows


# ---------------------------------------------------------------- guards

def require_h3(model, who: str):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError(f"{who} only supports MiniMax H3 models.")
    return diffusion_model


def conflicting_h3_object_patch(key: str) -> bool:
    """True only for a replaced H3 forward or attention module, not Linear weights."""
    return (key == "diffusion_model._forward" or key.endswith(".attn")
            or key.endswith(".attn.forward"))


def check_object_patches(model, who: str):
    if any(conflicting_h3_object_patch(k) for k in getattr(model, "object_patches", {})):
        raise ValueError(f"{who}: conflicting H3 object patch")


def override_chain(fn):
    """Walk a CurveAttn override chain top-down (stops at the first foreign fn)."""
    while fn is not None:
        yield fn
        fn = getattr(fn, "_curve_previous", None)


def curve_states(transformer_options):
    """Sparse states currently stacked on this MODEL's attention override."""
    top = (transformer_options or {}).get(OVERRIDE_KEY)
    return [fn._curve_state for fn in override_chain(top)
            if getattr(fn, "_curve_state", None) is not None]


# ---------------------------------------------------------------- CFG split

def limit_h3_cond_batch(executor, model, conds, x_in, timestep, model_options):
    """Keep the sampler's own fit loop at one branch per call.

    MiniMax H3 rejects video batch > 1, and the residual cache keys entries
    by branch. Returning inf for a wider batch makes the existing memory
    check split instead of reimplementing it.
    """
    single = int(x_in.shape[0])
    original = model.memory_required

    def limited(input_shape, cond_shapes=None):
        if input_shape and int(input_shape[0]) > single:
            return float("inf")
        if cond_shapes is None:
            return original(input_shape)
        return original(input_shape, cond_shapes)

    model.memory_required = limited
    try:
        return executor(model, conds, x_in, timestep, model_options)
    finally:
        model.memory_required = original


def install_cond_batch_limiter(patcher):
    """Add the CFG split once, however many CurveAttn nodes are stacked."""
    if not patcher.get_wrappers(WrappersMP.CALC_COND_BATCH, COND_BATCH_KEY):
        patcher.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, COND_BATCH_KEY,
                                     limit_h3_cond_batch)
