import types
import unittest
from unittest.mock import Mock, patch
import torch
from curve_test_package.block_dispatch import make_block_dispatch
from curve_test_package.runtime import RunScope
from curve_test_package import curve_attention as ca


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.state = types.SimpleNamespace(
            qkv_mode="native", cache_enabled=True, run_scope=RunScope(.01, 1, 1),
            dense_steps=frozenset(), dense_last_steps=0, disable_fp16_accum=False,
            selection="sla", current_topk=.2, current_tau=0., tail=False,
            token_aug=0, protect_prefix=False, protect_audio=False,
            dense_backend="pytorch", sigma_start=None, sigma_end=None)
        self.layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 6, "video")))

    def test_native_fallback(self):
        sentinel = object()
        args = {"transformer_options": {}}
        fn = make_block_dispatch(0, 4, None, self.state)
        result = fn(args, {"original_block": lambda a: {"img": sentinel, "extra": 7}})
        self.assertIs(result["img"], sentinel)
        self.assertEqual(result["extra"], 7)

    def test_middle_skip_keeps_edge_blocks(self):
        calls = []

        @torch.no_grad()
        def sample(*args, **kwargs):
            outputs = []
            for step in (2, 3):
                x = torch.ones(6, 4)
                meta = dict(eligible=True, key=(0,), step=step, total=8,
                            dense=False, policy=(.2,))
                for i in range(4):
                    def original(a, i=i):
                        calls.append(i)
                        return {"img": a["img"].add_(i + 1)}
                    dispatch = make_block_dispatch(i, 4, None, self.state)
                    x = dispatch(dict(img=x, layout=self.layout,
                                      transformer_options={"_curve_meta": meta}),
                                 {"original_block": original})["img"]
                outputs.append(x)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            self.assertEqual(self.state.run_scope.current().cache.hits, 1)

        self.state.run_scope.outer(sample, None, None, None,
                                   [1., .875, .75, .625, .5, .375, .25, .125, 0.])
        self.assertEqual(calls, [0, 1, 2, 3, 0, 3])

    def test_foreign_replacement_rejected(self):
        from comfy.model_patcher import ModelPatcher
        base = torch.nn.Module()
        base.diffusion_model = ca.MiniMaxH3Model.__new__(ca.MiniMaxH3Model)
        torch.nn.Module.__init__(base.diffusion_model)
        model = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        model.set_model_patch_replace(lambda *a, **k: None, "dit", "double_block", 0)
        with patch.object(ca.ck, "sol_attn_is_available") as avail:
            with self.assertRaisesRegex(ValueError, "dit block replacements"):
                ca.apply_curve_attention(model)
            avail.assert_not_called()
        self.assertTrue(("double_block", 0) in
                        model.model_options["transformer_options"]["patches_replace"]["dit"])

    def _cpu_patcher(self):
        from comfy.model_patcher import ModelPatcher
        base = torch.nn.Module()
        base.diffusion_model = ca.MiniMaxH3Model.__new__(ca.MiniMaxH3Model)
        torch.nn.Module.__init__(base.diffusion_model)
        return ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    def test_weight_object_patches_are_not_conflicts(self):
        model = self._cpu_patcher()
        model.add_object_patch("diffusion_model.blocks.0.attn.qkv_proj", object())
        model.add_object_patch("manual_cast_dtype", torch.bfloat16)
        with patch.object(ca.ck, "sol_attn_is_available", return_value=True), patch.object(
            ca, "get_attention_function", return_value=Mock()
        ):
            patched = ca.apply_curve_attention(model)
        self.assertIsNot(patched, model)

    def test_forward_object_patch_rejected(self):
        model = self._cpu_patcher()
        model.add_object_patch("diffusion_model._forward", object())
        with patch.object(ca.ck, "sol_attn_is_available") as avail:
            with self.assertRaisesRegex(ValueError, "conflicting H3 object patch"):
                ca.apply_curve_attention(model)
            avail.assert_not_called()

    def test_forward_options_do_not_leak(self):
        s = ca.CurveAttnState(
            keep_ratio=.2, min_tokens=0, dense_steps=frozenset(), dense_last_steps=0,
            tail=False, protect_audio=False, verbose=False, dense_attention=Mock())
        wrapper = ca.make_step_wrapper(s)
        original = {"sample_sigmas": [1, .5, 0], "sigmas": .5}

        def ok(*args, **kwargs):
            return 1

        wrapper(ok, None, None, None, original, None)
        self.assertNotIn("_curve_dense", original)
        self.assertNotIn("_curve_meta", original)

        def boom(*args, **kwargs):
            raise RuntimeError("fail")

        with self.assertRaisesRegex(RuntimeError, "fail"):
            wrapper(boom, None, None, None, original, None)
        self.assertNotIn("_curve_dense", original)

    def test_masked_call_keeps_native_kwargs(self):
        s = ca.CurveAttnState(
            keep_ratio=.2, min_tokens=0, dense_steps=frozenset(), dense_last_steps=0,
            tail=False, protect_audio=False, verbose=False, dense_attention=Mock(),
            cache_enabled=True, run_scope=self.state.run_scope)
        wrapper = ca.make_step_wrapper(s)
        video, audio = object(), object()
        seen = {}

        def inner(*args, **kwargs):
            seen.update(kwargs)
            return 0

        def sample(*a, **k):
            return wrapper(inner, None, None, None, {"sigmas": .5}, None,
                           denoise_mask=video, audio_denoise_mask=audio)

        self.state.run_scope.outer(sample, None, None, None,
                                   [1., .875, .75, .625, .5, .375, .25, .125, 0.])
        self.assertIs(seen["denoise_mask"], video)
        self.assertIs(seen["audio_denoise_mask"], audio)
        self.assertFalse(seen["transformer_options"]["_curve_meta"]["eligible"])

    def test_exception_drops_snapshot(self):
        calls = []

        def sample(*args, **kwargs):
            x = torch.ones(6, 4)
            meta = dict(eligible=True, key=(0,), step=2, total=8, dense=False, policy=(.2,))
            for i in range(4):
                def original(a, i=i):
                    calls.append(i)
                    if i == 2:
                        raise RuntimeError("mid")
                    return {"img": a["img"].add_(1)}
                dispatch = make_block_dispatch(i, 4, None, self.state)
                x = dispatch(dict(img=x, layout=self.layout,
                                  transformer_options={"_curve_meta": meta}),
                             {"original_block": original})["img"]

        with self.assertRaisesRegex(RuntimeError, "mid"):
            self.state.run_scope.outer(sample, None, None, None,
                                       [1., .875, .75, .625, .5, .375, .25, .125, 0.])
        self.assertEqual(self.state.run_scope.current(), None)

    def test_chunked_injected_only_when_selected(self):
        seen = []

        def original(args):
            seen.append("attention" in args)
            return {"img": args["img"]}

        native = make_block_dispatch(0, 3, None, self.state)
        native({"img": torch.ones(1, 1), "transformer_options": {}}, {"original_block": original})
        chunked_state = types.SimpleNamespace(
            qkv_mode="chunked", cache_enabled=False, run_scope=None)
        chunked = make_block_dispatch(0, 3, Mock(), chunked_state)
        chunked({"img": torch.ones(1, 1), "transformer_options": {}}, {"original_block": original})
        self.assertEqual(seen, [False, True])
