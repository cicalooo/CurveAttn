"""CPU proof of the H3 CFG batching decision. No GPU, no generation."""

import unittest
from uuid import uuid4

import torch

from comfy.conds import CONDRegular
from comfy.model_base import BaseModel
from comfy.patcher_extension import WrapperExecutor
from comfy.samplers import _calc_cond_batch
from curve_test_package.runtime import limit_h3_cond_batch


class _Patcher:
    def __init__(self, free_bytes):
        self.free_bytes = free_bytes

    def get_free_memory(self, device):
        return self.free_bytes

    def apply_hooks(self, hooks=None):
        return {"transformer_options": {}}

    def prepare_state(self, timestep, model_options):
        return None

    def prepare_hook_patches_current_keyframe(self, *args, **kwargs):
        return None


class _Model:
    memory_usage_factor = 0.114
    memory_usage_factor_conds = ()

    def __init__(self, free_bytes):
        self.current_patcher = _Patcher(free_bytes)
        self.batches = []

    def get_dtype_inference(self):
        return torch.bfloat16

    def memory_required(self, input_shape, cond_shapes=None):
        return BaseModel.memory_required(self, input_shape, cond_shapes or {})

    def apply_model(self, x, timestep, **conds):
        branches = tuple(conds["transformer_options"]["cond_or_uncond"])
        self.batches.append(branches)
        return x


def _conds():
    token = torch.zeros(1, 4)
    def one():
        return {"uuid": uuid4(), "model_conds": {"c_crossattn": CONDRegular(token.clone())}}
    return [[one()], [one()]]


class CfgBatchTests(unittest.TestCase):
    def test_sampler_batches_cfg_then_wrapper_splits_it(self):
        # 720p-ish packed latent. Flash-path estimate is well under 24 GiB,
        # which is the free memory a 3090/4090/5090 still has beside the model.
        x = torch.zeros(1, 32, 31, 46, 80)
        timestep = torch.tensor([1.0])
        bare = _Model(24 * 1024 ** 3)
        _calc_cond_batch(bare, _conds(), x, timestep, {})
        self.assertEqual(bare.batches, [(1, 0)])

        split = _Model(24 * 1024 ** 3)
        executor = WrapperExecutor.new_executor(_calc_cond_batch, [limit_h3_cond_batch])
        executor.execute(split, _conds(), x, timestep, {})
        self.assertEqual(split.batches, [(1,), (0,)])
        self.assertIs(split.memory_required.__func__, _Model.memory_required)
