"""Prepared SLA integration and the unchanged native sparse fallback."""

from __future__ import annotations

import comfy_kitchen as ck
import torch

from . import prepared_sol as _local_prepared
from .prepared_sol import _sol_layout_ok, maybe_contiguous  # noqa: F401  re-export for tests


def _kitchen_prepared():
    try:
        from comfy_kitchen import sol_prepared as kitchen
    except ImportError:
        return None
    required = ("prequantize_sol_attn", "sol_attn_from_prequantized", "check_available")
    if any(not hasattr(kitchen, name) for name in required):
        return None
    return kitchen


sol_prepared = _kitchen_prepared() or _local_prepared


def check_prepared_available():
    """Raise RuntimeError when prepared-Sol cannot run on this GPU."""
    sol_prepared.check_available()


def prepare_sla(q, k, v, **kwargs):
    return sol_prepared.prequantize_sol_attn(
        maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v), **kwargs,
    )


def consume_sla(prepared):
    return sol_prepared.sol_attn_from_prequantized(prepared)


def sol_attn_zerocopy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    topk_ratio: float,
    scale: float | None = None,
    sink_blocks=None,
    sink_q=None,
    tail: bool = False,
    token_aug: int = 0,
    tau: float = 0.0,
) -> torch.Tensor:
    """Keep native threshold arithmetic, partial blocks, ties and dispatch.

    A Python tile/cat/top-k loop is not kernel fusion and adds work at
    ordinary H3 lengths. Use the installed operation, including its selector.
    """
    return ck.sol_attn(
        maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v),
        topk_ratio=topk_ratio,
        scale=scale,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
        tail=tail,
        token_aug=int(token_aug),
        tau=float(tau),
    )
