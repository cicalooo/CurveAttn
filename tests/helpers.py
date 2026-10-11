"""Shared CPU-only fixtures. Run tests through tools/run_cpu_tests.py."""
from contextlib import contextmanager
from unittest.mock import Mock, patch

import torch

from curvattn_test_package import sparse as sp

PKG = "curvattn_test_package"
SCHEDULE = [1., .875, .75, .625, .5, .375, .25, .125, 0.]


def h3_patcher(n_blocks=4):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_patcher import ModelPatcher
    base = torch.nn.Module()
    h3 = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(h3)
    h3.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(n_blocks)])
    for block in h3.blocks:
        block.attn = torch.nn.Identity()
    base.diffusion_model = h3
    return ModelPatcher(base, load_device=torch.device("cpu"),
                        offload_device=torch.device("cpu"))


@contextmanager
def kitchen(available=True, backend=None):
    """Pretend Kitchen sol_attn exists; the dense backend is a Mock."""
    backend = Mock(return_value="dense") if backend is None else backend
    with patch.object(sp.ck, "sol_attn_is_available", return_value=available), \
         patch.object(sp, "get_attention_function", return_value=backend):
        yield backend


def options(model):
    return model.model_options["transformer_options"]


def dit(model):
    return (options(model).get("patches_replace") or {}).get("dit") or {}


def sparse_state(**kwargs):
    defaults = dict(keep_ratio=0.2, min_tokens=0, dense_steps=frozenset({1}),
                    dense_attention=Mock(return_value="dense"))
    return sp.SparseState(**(defaults | kwargs))


def step_options(state, dense=False, reason="scheduled dense step", **extra):
    """transformer_options as the sparse step wrapper would publish them."""
    decision = sp.StepDecision(dense, reason if dense else None, state.keep_ratio, state.tau)
    return sp.publish(extra, state, decision)
