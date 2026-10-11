"""CurveAttn-owned prepare/consume layer over stock Kitchen sol_attn.

selector is owned_bf16. Stock Kitchen has no comfy_kitchen.sol_prepared.
"""
from __future__ import annotations

from dataclasses import dataclass

import comfy_kitchen as ck
import torch

SELECTOR = "owned_bf16"


def check_available() -> None:
    if not ck.sol_attn_is_available():
        raise RuntimeError("Kitchen sol_attn is not available on this GPU.")


def _sol_layout_ok(x: torch.Tensor) -> bool:
    if x.stride(-1) != 1 or x.data_ptr() % 16:
        return False
    return all(x.shape[dim] <= 1 or x.stride(dim) % 8 == 0 for dim in range(3))


def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    if _sol_layout_ok(x):
        return x
    if x.is_contiguous():
        return x.clone(memory_format=torch.contiguous_format)
    return x.contiguous()


def _exclusive_storage(x: torch.Tensor) -> bool:
    """True when x is the sole occupant of its storage, so a clone would only copy."""
    storage = x.untyped_storage()
    return (x.data_ptr() == storage.data_ptr()
            and storage.nbytes() == x.numel() * x.element_size())


def _own(x: torch.Tensor) -> torch.Tensor:
    y = maybe_contiguous(x)
    if _exclusive_storage(y):
        return y
    return y.clone()


@dataclass(slots=True)
class PreparedSol:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kwargs: dict
    selector: str = SELECTOR


def prequantize_sol_attn(q, k, v, **kwargs) -> PreparedSol:
    check_available()
    return PreparedSol(q=_own(q), k=_own(k), v=_own(v), kwargs=dict(kwargs))


def sol_attn_from_prequantized(prepared: PreparedSol) -> torch.Tensor:
    if not isinstance(prepared, PreparedSol):
        raise TypeError(
            "prepared must be returned by prequantize_sol_attn, "
            f"got {type(prepared).__name__}"
        )
    return ck.sol_attn(prepared.q, prepared.k, prepared.v, **prepared.kwargs)
