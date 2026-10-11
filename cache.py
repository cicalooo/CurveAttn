"""CurveAttn Residual Cache: branch-scoped reuse of the H3 middle-block residual.

Independent of sparse attention: it does not need ck.sol_attn and works on a
dense H3 model. When CurveAttn Sparse nodes are stacked on the same MODEL,
their published policy fingerprint joins the residual signature, so a change
of attention policy (e.g. a dense step) refreshes instead of reusing.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from uuid import UUID

from comfy.patcher_extension import CallbacksMP, WrappersMP

from . import blocks
from ._version import VERSION
from .common import (
    LOG, NS, PREFIX, check_object_patches, curve_states, exact_step,
    install_cond_batch_limiter, require_h3, schedule_tuple,
)
from .residual_cache import MiddleCache, format_cache_report

WHO = f"{PREFIX} Cache"
KEY = "curvattn_cache"
SPANS = ("middle", "whole")
MODES = ("fingerprint", "first_block")

MANUAL_DEFAULTS = dict(
    threshold=0.01, max_hits=1, span="middle", start_percent=0.2, end_percent=0.8,
    min_steps=6, temporal_guard=False, audio_guard=False, mode="fingerprint",
)


def _starting_point(threshold, start, end):
    """Unvalidated cache window. Budget stays on the Max MiB widget."""
    return dict(threshold=threshold, start_percent=start, end_percent=end, max_hits=1,
                span="middle", mode="fingerprint", min_steps=6,
                temporal_guard=True, audio_guard=True)


CACHE_PRESETS = {
    "manual": None,
    "turbo_safe": dict(threshold=0.05, start_percent=0.15, end_percent=0.85, max_hits=1,
                       span="middle", mode="fingerprint", min_steps=4,
                       temporal_guard=False, audio_guard=True),
    "native": _starting_point(0.08, 0.15, 0.85),
    "fast": _starting_point(0.10, 0.10, 0.90),
}


def apply_cache_preset(preset, settings):
    """Shallow copy of settings with the preset's fields applied. manual keeps them."""
    key = str(preset or "manual")
    if key not in CACHE_PRESETS:
        raise ValueError(f"{WHO}: invalid preset")
    out = dict(settings)
    if CACHE_PRESETS[key] is not None:
        out.update(CACHE_PRESETS[key])
    return out


def validate_cache(*, preset, threshold, max_hits, max_mib, span, start_percent,
                   end_percent, min_steps, temporal_guard, audio_guard, mode):
    def real(name, value, low, high):
        if isinstance(value, bool) or not isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError(f"{WHO}: invalid {name}")

    def integer(name, value, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{WHO}: invalid {name}")

    if preset not in CACHE_PRESETS:
        raise ValueError(f"{WHO}: invalid preset")
    real("threshold", threshold, 0, .2)
    integer("max_hits", max_hits, 1, 2)
    integer("max_mib", max_mib, 0, 8192)
    if span not in SPANS:
        raise ValueError(f"{WHO}: invalid cache span")
    real("start_percent", start_percent, 0, 1)
    real("end_percent", end_percent, 0, 1)
    if start_percent > end_percent:
        raise ValueError(f"{WHO}: start_percent exceeds end_percent")
    integer("min_steps", min_steps, 2, 64)
    if mode not in MODES:
        raise ValueError(f"{WHO}: invalid cache mode")
    if not isinstance(temporal_guard, bool) or not isinstance(audio_guard, bool):
        raise ValueError(f"{WHO}: invalid guard flag")


# ---------------------------------------------------------------- eligibility

def cache_input_flags(payload, kwargs) -> dict:
    """Forward-level facts the block dispatcher cannot see."""
    payload = payload or {}

    def present(value):
        return value is not None and not (isinstance(value, (list, tuple)) and len(value) == 0)

    return dict(
        denoise_mask=any((kwargs or {}).get(k) is not None
                         for k in ("denoise_mask", "audio_denoise_mask")),
        refs=any(present(payload.get(k)) for k in ("refs", "keyframes")),
    )


def cache_blockers(options, inputs, *, identity_ok, step, total, min_steps):
    """Why this step cannot cache. None when it can."""
    reasons = []
    if inputs is None:
        reasons.append("cache step wrapper did not run")
    else:
        if inputs.get("denoise_mask"):
            reasons.append("denoise mask")
        if inputs.get("refs"):
            reasons.append("refs or keyframes")
    if "multigpu_thread_device" in options:
        reasons.append("multigpu")
    if not identity_ok:
        branches = options.get("cond_or_uncond", ())
        n = len(branches) if isinstance(branches, (list, tuple)) else "missing"
        reasons.append(f"cond_or_uncond length {n}" if n != 1 else "missing branch uuid")
    if step is None:
        reasons.append("sigma did not match the schedule")
    elif total < int(min_steps):
        reasons.append(f"schedule length {total} < min_steps {int(min_steps)}")
    return "; ".join(reasons) if reasons else None


def cache_metadata(options, inputs, *, schedule, policy, min_steps=6,
                   start_percent=0.2, end_percent=0.8):
    step, total = exact_step(schedule, options.get("sigmas"))
    branches = options.get("cond_or_uncond", ())
    uuids = options.get("uuids", ())
    identity_ok = (isinstance(branches, (list, tuple)) and len(branches) == 1
                   and branches[0] in (0, 1)
                   and isinstance(uuids, (list, tuple)) and len(uuids) == 1)
    identity = None
    if identity_ok:
        try:
            identity = uuids[0] if isinstance(uuids[0], UUID) else UUID(uuids[0])
        except (TypeError, ValueError, AttributeError):
            identity_ok = False
    blockers = cache_blockers(options, inputs, identity_ok=identity_ok, step=step,
                              total=total, min_steps=min_steps)
    ns = options.get(NS) or {}
    # A step is dense for the cache only when sparse nodes ran and none owns it.
    dense = "owner" in ns and ns.get("owner") is None
    return dict(eligible=blockers is None, reason=blockers,
                key=(branches[0], identity) if identity_ok else None,
                step=step, total=total, dense=dense, policy=policy,
                min_steps=int(min_steps), start_percent=float(start_percent),
                end_percent=float(end_percent))


# ---------------------------------------------------------------- run scope

@dataclass
class Run:
    cache: MiddleCache
    schedule: tuple
    extra: dict = field(default_factory=dict)


class RunScope:
    def __init__(self, threshold, max_hits, max_mib, span="middle", start_percent=0.2,
                 end_percent=0.8, min_steps=6, temporal_guard=False, audio_guard=False,
                 mode="fingerprint"):
        self.settings = dict(
            threshold=threshold, max_hits=max_hits, max_bytes=max_mib * 1024 * 1024,
            span=span, start_percent=start_percent, end_percent=end_percent,
            min_steps=min_steps, temporal_guard=temporal_guard, audio_guard=audio_guard,
            mode=mode)
        self.slot = ContextVar("curvattn_run", default=None)
        self.lock = Lock()
        self.on_begin = lambda: None
        self.on_end = lambda run: None

    def current(self):
        return self.slot.get()

    def clear_current(self):
        run = self.current()
        if run is not None:
            run.cache.clear()

    def outer(self, executor, *args, **kwargs):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(f"{WHO}: concurrent use of one patched model is unsupported")
        token = None
        run = None
        try:
            sigmas = kwargs.get("sigmas", args[3] if len(args) > 3 else None)
            run = Run(MiddleCache(**self.settings), schedule_tuple(sigmas))
            token = self.slot.set(run)
            self.on_begin()
            return executor(*args, **kwargs)
        finally:
            try:
                if run is not None:
                    try:
                        self.on_end(run)
                    finally:
                        run.cache.clear()
            finally:
                if token is not None:
                    self.slot.reset(token)
                self.lock.release()


# ---------------------------------------------------------------- state

class CacheState:
    def __init__(self, settings: dict, max_mib: int, preset="manual"):
        self.preset = preset
        self.threshold = float(settings["threshold"])
        self.max_hits = int(settings["max_hits"])
        self.max_mib = int(max_mib)
        self.span = str(settings["span"])
        self.start_percent = float(settings["start_percent"])
        self.end_percent = float(settings["end_percent"])
        self.min_steps = int(settings["min_steps"])
        self.temporal_guard = bool(settings["temporal_guard"])
        self.audio_guard = bool(settings["audio_guard"])
        self.mode = str(settings["mode"])
        self.run_scope = RunScope(
            self.threshold, self.max_hits, self.max_mib, span=self.span,
            start_percent=self.start_percent, end_percent=self.end_percent,
            min_steps=self.min_steps, temporal_guard=self.temporal_guard,
            audio_guard=self.audio_guard, mode=self.mode)
        self.run_scope.on_begin = self.reset
        self.run_scope.on_end = self.report
        self.reset()

    def reset(self):
        self._logged = set()
        self.last_block = None

    def warn_once(self, key, message):
        if key not in self._logged:
            self._logged.add(key)
            LOG.warning("%s: %s", WHO, message)

    def policy(self, options):
        ns = options.get(NS) or {}
        return (ns.get("policy", ()),
                options.get("minimax_h3_sigma_shift_video"),
                options.get("minimax_h3_sigma_shift_audio"),
                self.span, self.mode, self.start_percent, self.end_percent,
                self.min_steps, self.temporal_guard, self.audio_guard)

    def metadata(self, options, run):
        meta = cache_metadata(
            options, (options.get(NS) or {}).get("cache_inputs"),
            schedule=run.schedule, policy=self.policy(options),
            min_steps=self.min_steps, start_percent=self.start_percent,
            end_percent=self.end_percent)
        if meta["reason"]:
            self.last_block = meta["reason"]
            self.warn_once(("ineligible", meta["reason"]), "cache ineligible: " + meta["reason"])
        return meta

    def report(self, run):
        cache = run.cache
        LOG.info(format_cache_report(cache, span=self.span, mode=self.mode,
                                     start_percent=self.start_percent,
                                     end_percent=self.end_percent))
        LOG.info("%s admission: entry_bytes=%d budget_bytes=%d storage_bypasses=%d "
                 "entries=%d prefetch=%s", WHO, cache.required_entry_bytes, cache.max_bytes,
                 cache.budget_bypasses, len(cache.entries),
                 "on_demand" if self.span == "whole" else "inherit")
        if cache.hits == 0:
            blocked = f" last_block={self.last_block}" if self.last_block else ""
            LOG.warning("%s had zero hits; inspect eligibility, schedule, threshold and "
                        "storage budget before expecting a cache speedup.%s", WHO, blocked)

    def cleanup(self):
        self.run_scope.clear_current()
        self.reset()


def forward_options(options, span):
    """Copy options; whole span loads block weights on demand instead of prefetching."""
    result = dict(options or {})
    if span == "whole":
        result["prefetch_dynamic_vbars"] = False
    return result


def make_cache_step_wrapper(state: CacheState):
    def wrapper(executor, x, timestep, context, transformer_options=None,
                minimax_payload=None, **kwargs):
        to = forward_options(transformer_options, state.span)
        ns = dict(to.get(NS) or {})
        ns["cache_inputs"] = cache_input_flags(minimax_payload, kwargs)
        to[NS] = ns
        if "multigpu_thread_device" in to:
            raise ValueError(f"{WHO}: residual cache is not validated for multigpu")
        try:
            return executor(x, timestep, context, transformer_options=to,
                            minimax_payload=minimax_payload, **kwargs)
        except BaseException:
            state.run_scope.clear_current()
            raise

    return wrapper


# ---------------------------------------------------------------- apply

def apply_cache(model, *, preset="manual", threshold=0.01, max_hits=1, span="middle",
                start_percent=0.2, end_percent=0.8, min_steps=6, temporal_guard=False,
                audio_guard=False, mode="fingerprint", max_mib=1024):
    preset = str(preset or "manual")
    settings = apply_cache_preset(preset, dict(
        threshold=float(threshold), max_hits=int(max_hits), span=str(span),
        start_percent=float(start_percent), end_percent=float(end_percent),
        min_steps=int(min_steps), temporal_guard=bool(temporal_guard),
        audio_guard=bool(audio_guard), mode=str(mode)))
    validate_cache(preset=preset, max_mib=max_mib, **settings)
    diffusion_model = require_h3(model, WHO)
    check_object_patches(model, WHO)
    blocks.check_dit(model.model_options, WHO)
    if blocks.current_cache(model.model_options) is not None:
        raise ValueError(f"{WHO}: this MODEL already has a CurveAttn residual cache.")
    if len(diffusion_model.blocks) < 3:
        raise ValueError(f"{WHO}: needs at least three transformer blocks")
    if settings["span"] == "whole":
        stacked = curve_states(model.model_options.get("transformer_options"))
        if any(s.qkv_mode != "native" for s in stacked):
            raise ValueError(f"{WHO}: whole span requires native qkv_mode on every "
                             "CurveAttn Sparse node")
    state = CacheState(settings, int(max_mib), preset=preset)
    m = model.clone()
    blocks.install(m, diffusion_model, cache=state)
    m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, state.run_scope.outer)
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, make_cache_step_wrapper(state))
    m.add_callback_with_key(CallbacksMP.ON_PREPARE_STATE, KEY,
                            lambda mp, timestep, model_options: blocks.verify(
                                model_options, WHO, cache=state))
    m.add_callback_with_key(CallbacksMP.ON_CLEANUP, KEY, lambda mp: state.cleanup())
    install_cond_batch_limiter(m)
    LOG.info("%s %s: preset=%s threshold=%.3f max_hits=%d max_mib=%d span=%s mode=%s "
             "window=%.2f-%.2f min_steps=%d temporal_guard=%s audio_guard=%s cfg_branches=serial",
             WHO, VERSION, preset, state.threshold, state.max_hits, state.max_mib, state.span,
             state.mode, state.start_percent, state.end_percent, state.min_steps,
             state.temporal_guard, state.audio_guard)
    return m
