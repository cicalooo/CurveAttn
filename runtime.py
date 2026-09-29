from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from uuid import UUID

import torch

from .residual_cache import MiddleCache


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


def exact_step(schedule, current):
    if len(schedule) < 2:
        return None, 0
    if isinstance(current, torch.Tensor):
        if current.numel() != 1:
            return None, len(schedule) - 1
        current = float(current.detach().reshape(()).item())
    elif isinstance(current, (tuple, list)):
        if len(current) != 1:
            return None, len(schedule) - 1
        current = current[0]
    try:
        current = float(current)
    except (TypeError, ValueError):
        return None, len(schedule) - 1
    if not isfinite(current):
        return None, len(schedule) - 1
    matches = [i for i, s in enumerate(schedule[:-1])
               if abs(s - current) <= max(1e-7, abs(s) * 1e-6)]
    return (matches[0] if len(matches) == 1 else None), len(schedule) - 1


def cache_blockers(options, payload, kwargs, *, enabled, protect_audio,
                   identity_ok, step, total, min_steps):
    """Why this step cannot cache. None when it can. None when cache is off."""
    if not enabled:
        return None
    reasons = []
    if protect_audio:
        reasons.append("protect_audio")
    if "multigpu_thread_device" in options:
        reasons.append("multigpu")
    if any(kwargs.get(k) is not None for k in ("denoise_mask", "audio_denoise_mask")):
        reasons.append("denoise mask")
    payload = payload or {}

    def present(value):
        return value is not None and not (isinstance(value, (list, tuple)) and len(value) == 0)

    if any(present(payload.get(k)) for k in ("refs", "keyframes")):
        reasons.append("refs or keyframes")
    if not identity_ok:
        branches = options.get("cond_or_uncond", ())
        n = len(branches) if isinstance(branches, (list, tuple)) else "missing"
        if n != 1:
            reasons.append(f"cond_or_uncond length {n}")
        else:
            reasons.append("missing branch uuid")
    if step is None:
        reasons.append("sigma did not match the schedule")
    elif total < int(min_steps):
        reasons.append(f"schedule length {total} < min_steps {int(min_steps)}")
    return "; ".join(reasons) if reasons else None


def cache_metadata(options, payload, kwargs, *, schedule, enabled,
                   protect_audio, policy, min_steps=6,
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
    # protect_audio remains a hard eligibility kill-switch (existing contract).
    # Soft audio drift refusal is handled inside MiddleCache via cache_audio_guard.
    # Batched CFG never reaches a successful H3 forward (batch size 1). The
    # CALC_COND_BATCH wrapper splits those calls; a length other than 1 here
    # means that split did not run.
    blockers = cache_blockers(
        options, payload, kwargs, enabled=enabled, protect_audio=protect_audio,
        identity_ok=identity_ok, step=step, total=total, min_steps=min_steps)
    return dict(eligible=bool(enabled and blockers is None),
                reason=blockers,
                key=(branches[0], identity) if identity_ok else None,
                step=step, total=total,
                dense=bool(options.get("_curve_dense", False)), policy=policy,
                min_steps=int(min_steps),
                start_percent=float(start_percent),
                end_percent=float(end_percent))


def limit_h3_cond_batch(executor, model, conds, x_in, timestep, model_options):
    """Keep the sampler's own fit loop at one branch per call.

    MiniMax H3 rejects video batch > 1. Concatenating cond and uncond therefore
    crashes before any block runs, and the experimental cache never sees a
    branch. Returning inf for a wider batch makes the existing memory check
    split instead of reimplementing it.
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


@dataclass
class Run:
    cache: MiddleCache
    schedule: tuple
    chunked: dict = field(default_factory=dict)


class RunScope:
    def __init__(self, threshold, max_hits, max_mib, span="middle",
                 start_percent=0.2, end_percent=0.8, min_steps=6,
                 temporal_guard=False, audio_guard=False, mode="fingerprint"):
        self.settings = dict(
            threshold=threshold, max_hits=max_hits,
            max_bytes=max_mib * 1024 * 1024, span=span,
            start_percent=start_percent, end_percent=end_percent,
            min_steps=min_steps, temporal_guard=temporal_guard,
            audio_guard=audio_guard, mode=mode,
        )
        self.slot = ContextVar("curve_run", default=None)
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
            raise RuntimeError("CurveAttention: concurrent use of one patched model is unsupported")
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
