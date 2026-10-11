from __future__ import annotations

from dataclasses import dataclass

import torch

from comfy.model_prefetch import pause_malloc_graph


@dataclass
class Entry:
    signature: tuple
    buffer: torch.Tensor
    anchor: tuple
    step: int = -1
    hits: int = 0
    valid: bool = False
    pre_middle: tuple = ()


def fingerprint(x, segments):
    result = []
    for start, end, kind in segments:
        if end <= start:
            raise ValueError("CurveAttn Cache: empty cache segment")
        rows = torch.linspace(start, end - 1, min(32, end - start),
                              device=x.device).long()
        cols = torch.linspace(0, x.shape[1] - 1, min(32, x.shape[1]),
                              device=x.device).long()
        result.append(x.detach()[rows[:, None], cols[None, :]].float().clone())
    return tuple(result)


def relative_drift(current, anchor):
    if not current or len(current) != len(anchor):
        return float("inf")
    values = []
    for a, b in zip(current, anchor):
        if a.shape != b.shape:
            return float("inf")
        finite = torch.isfinite(a).all() & torch.isfinite(b).all()
        drift = (a - b).abs().mean() / b.abs().mean().clamp_min(1e-6)
        values.append(torch.where(finite, drift, drift.new_full((), float("inf"))))
    return float(torch.stack(values).amax().item())


def _segment_index(segments, kind):
    for i, (_, _, k) in enumerate(segments):
        if k == kind:
            return i
    return None


def modality_drift(current, anchor, segments, kind):
    """Relative drift for a single modality segment; inf if missing/mismatched."""
    idx = _segment_index(segments, kind)
    if idx is None or not current or not anchor or idx >= len(current) or idx >= len(anchor):
        return float("inf")
    return relative_drift((current[idx],), (anchor[idx],))


def temporal_video_drift(current, anchor, segments, chunks=4):
    """Max relative drift across subsampled row-groups inside the video segment.

    FirstBlockCache-style local motion gate: a quiet global mean can hide a
    hot frame cluster; take the worst chunk drift instead.
    """
    idx = _segment_index(segments, "video")
    if idx is None or not current or not anchor or idx >= len(current) or idx >= len(anchor):
        return float("inf")
    a, b = current[idx], anchor[idx]
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 1:
        return float("inf")
    n = a.shape[0]
    parts = max(1, min(int(chunks), n))
    drifts = []
    for i in range(parts):
        lo = i * n // parts
        hi = (i + 1) * n // parts
        if hi <= lo:
            continue
        drifts.append(relative_drift((a[lo:hi],), (b[lo:hi],)))
    return max(drifts) if drifts else float("inf")


def format_cache_report(cache, **meta) -> str:
    """One-line end-of-run cache telemetry for logs and tests."""
    span = meta.get("span", getattr(cache, "span", "middle"))
    mode = meta.get("mode", getattr(cache, "mode", "fingerprint"))
    start = meta.get("start_percent", getattr(cache, "start_percent", 0.2))
    end = meta.get("end_percent", getattr(cache, "end_percent", 0.8))
    max_drift = getattr(cache, "max_drift", float("nan"))
    if max_drift != max_drift:  # NaN
        drift_s = "n/a"
    elif max_drift == float("inf"):
        drift_s = "inf"
    else:
        drift_s = f"{max_drift:.4g}"
    return (
        f"CurveAttn cache: hits={cache.hits} refreshes={cache.refreshes} "
        f"bypasses={cache.bypasses} budget_bypasses={cache.budget_bypasses} "
        f"bytes={cache.bytes} span={span} mode={mode} "
        f"start={float(start):.2f} end={float(end):.2f} max_drift={drift_s}"
    )


class MiddleCache:
    def __init__(self, threshold=.01, max_hits=1, max_bytes=1024 * 1024 * 1024,
                 span="middle", start_percent=.2, end_percent=.8, min_steps=6,
                 temporal_guard=False, audio_guard=False, mode="fingerprint"):
        self.threshold = threshold
        self.max_hits = max_hits
        self.max_bytes = max_bytes
        if span not in ("middle", "whole"):
            raise ValueError("CurveAttn Cache: invalid cache span")
        if mode not in ("fingerprint", "first_block"):
            raise ValueError("CurveAttn Cache: invalid cache mode")
        self.span = span
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.min_steps = int(min_steps)
        self.temporal_guard = bool(temporal_guard)
        self.audio_guard = bool(audio_guard)
        self.mode = mode
        self.entries = {}
        self.active = None
        self.hits = self.refreshes = self.bypasses = 0
        self.budget_bypasses = 0
        self.required_entry_bytes = 0
        self.max_drift = float("nan")

    @property
    def bytes(self):
        return sum(e.buffer.numel() * e.buffer.element_size()
                   for e in self.entries.values())

    def clear(self):
        self.active = None
        self.entries.clear()

    def _decision_drift(self, current, entry, segments):
        """Return (may_use_drift, drift) using mode + optional guards."""
        if self.mode == "first_block":
            # Residual-input delta gating at span begin (post edge block 0 for
            # middle; pre-all for whole). Not a full FBC block-0 replace patch.
            anchor = entry.pre_middle or entry.anchor
        else:
            anchor = entry.anchor
        drift = relative_drift(current, anchor)
        if self.temporal_guard:
            drift = max(drift, temporal_video_drift(current, anchor, segments))
        if self.audio_guard:
            audio = modality_drift(current, anchor, segments, "audio")
            if audio >= self.threshold:
                return False, max(drift, audio)
        return True, drift

    def _begin(self, args, meta):
        self.active = None
        x = args["img"]
        layout = args.get("layout")
        segments = tuple(getattr(layout, "segments", ()))
        eligible = (meta.get("eligible", False) and self.threshold > 0
                    and self.max_bytes > 0 and x.ndim == 2
                    and not x.requires_grad and not torch.is_grad_enabled()
                    and tuple(s[2] for s in segments) == ("text", "audio", "video")
                    and x.shape[1] > 0 and all(0 <= a < b <= x.shape[0] for a, b, _ in segments)
                    and segments[0][0] == 0 and segments[-1][1] == x.shape[0]
                    and all(a[1] == b[0] for a, b in zip(segments, segments[1:])))
        if not eligible:
            self.clear()
            self.bypasses += 1
            return
        key = meta["key"]
        step, total = meta["step"], meta["total"]
        signature = (tuple(x.shape), x.dtype, x.device, segments, meta["policy"])
        entry = self.entries.get(key)
        if entry is not None and entry.signature != signature:
            del self.entries[key]
            entry = None
        if entry is None:
            needed = x.numel() * x.element_size()
            self.required_entry_bytes = max(self.required_entry_bytes, needed)
            if len(self.entries) >= 2 or self.bytes + needed > self.max_bytes:
                self.bypasses += 1
                self.budget_bypasses += 1
                return
            entry = Entry(signature, torch.empty_like(x), ())
            self.entries[key] = entry
        current = fingerprint(x, segments)
        progress = step / (total - 1) if total > 1 else 0.
        min_steps = int(meta.get("min_steps", self.min_steps))
        start_p = float(meta.get("start_percent", self.start_percent))
        end_p = float(meta.get("end_percent", self.end_percent))
        may_hit = (entry.valid and total >= min_steps and step == entry.step + 1
                   and not meta["dense"] and start_p <= progress <= end_p
                   and entry.hits < self.max_hits)
        if may_hit:
            ok, drift = self._decision_drift(current, entry, segments)
            if drift == drift:  # not NaN
                prev = self.max_drift
                self.max_drift = drift if prev != prev else max(prev, drift)
        else:
            ok, drift = False, float("inf")
        if may_hit and ok and drift < self.threshold:
            entry.step = step
            entry.hits += 1
            self.hits += 1
            self.active = ("hit", entry)
        else:
            entry.valid = False
            entry.buffer.copy_(x)
            entry.anchor = current
            entry.pre_middle = current
            entry.step = step
            entry.hits = 0
            self.active = ("refresh", entry)

    def call(self, index, count, args, extra, meta, run_block):
        if index == 0:
            self.active = None
        if count < 3:
            return run_block(args, extra)
        first, last = (0, count - 1) if self.span == "whole" else (1, count - 2)
        try:
            if index == first:
                # meta may be a callable so it is computed lazily, at the
                # first span block, after every step wrapper has published.
                with pause_malloc_graph():
                    self._begin(args, meta() if callable(meta) else meta)
            if first <= index <= last and self.active is not None:
                mode, entry = self.active
                if mode == "hit":
                    if index == first:
                        args["img"].add_(entry.buffer)
                    result = {"img": args["img"]}
                else:
                    result = run_block(args, extra)
                    if index == last:
                        torch.sub(result["img"], entry.buffer, out=entry.buffer)
                        entry.valid = True
                        self.refreshes += 1
                if index == last:
                    self.active = None
                return result
            return run_block(args, extra)
        except BaseException:
            self.clear()
            raise
