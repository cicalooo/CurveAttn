import ast
import inspect
import math
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from curve_test_package import curve_attention as ca
from curve_test_package import residual_cache as rc
from curve_test_package.nodes import CurveAttention
from curve_test_package.runtime import RunScope

ROOT = Path(__file__).resolve().parents[1]


class SpeedPlanTests(unittest.TestCase):
    @torch.no_grad()
    def chain(self, cache, step=2, x=None, key=(0,), eligible=True):
        layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 6, "video")))
        x = torch.ones(6, 4) if x is None else x.clone()
        called = []
        meta = dict(eligible=eligible, key=key, step=step, total=8,
                    dense=False, policy=(.2,))
        for index in range(4):
            def fresh(args, extra, index=index):
                called.append(index)
                return {"img": args["img"].add_(index + 1)}
            x = cache.call(index, 4, dict(img=x, layout=layout), {},
                           meta, fresh)["img"]
        return x, called

    def test_native_default(self):
        self.assertEqual(inspect.signature(ca.apply_curve_attention)
                         .parameters["qkv_mode"].default, "native")
        self.assertEqual(inspect.signature(CurveAttention.execute)
                         .parameters["qkv_mode"].default, "native")
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        matches = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and n.args
                   and isinstance(n.args[0], ast.Constant)
                   and n.args[0].value == "qkv_mode"]
        self.assertEqual(len(matches), 1)
        defaults = {k.arg: ast.literal_eval(k.value)
                    for k in matches[0].keywords if k.arg == "default"}
        self.assertEqual(defaults, {"default": "native"})

    def test_drift_one_host_read_and_no_python_tensor_bool(self):
        samples = tuple(torch.ones(2, 2) for _ in range(3))
        real_item = torch.Tensor.item
        calls = []
        def item(tensor, *args):
            calls.append(1)
            return real_item(tensor, *args)
        def forbidden_bool(tensor):
            raise AssertionError("tensor truth-value forces another host decision")
        cache = rc.MiddleCache(max_bytes=4096)
        self.chain(cache)
        with patch.object(torch.Tensor, "item", item), \
             patch.object(torch.Tensor, "__bool__", forbidden_bool):
            self.chain(cache, step=3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(rc.relative_drift(samples, samples), 0.0)
        bad = (torch.full((2, 2), float("nan")),) + samples[1:]
        self.assertTrue(math.isinf(rc.relative_drift(bad, samples)))
        self.assertTrue(math.isinf(rc.relative_drift(samples, bad)))

    def test_cache_wrapper_does_not_pause_entire_forward(self):
        state = ca.CurveAttnState(
            .2, 0, frozenset(), 0, False, False, False, Mock(),
            cache_enabled=True, run_scope=RunScope(.01, 1, 1),
            disable_fp16_accum=False)
        with patch.object(ca, "pause_malloc_graph") as pause:
            ca.make_step_wrapper(state)(lambda *a, **k: 7,
                                        None, None, None, {}, None)
        pause.assert_not_called()

    def test_cache_begin_paused_but_block_compute_unpaused(self):
        from contextlib import contextmanager
        events = []
        @contextmanager
        def pause():
            events.append("enter")
            yield
            events.append("exit")
        cache = rc.MiddleCache(max_bytes=4096)
        original_begin = cache._begin
        def begin(*args):
            self.assertEqual(events[-1], "enter")
            return original_begin(*args)
        with patch.object(rc, "pause_malloc_graph", pause, create=True), \
             patch.object(cache, "_begin", begin):
            self.chain(cache)
        self.assertEqual(events, ["enter", "exit"])

    def test_budget_failure_is_explained(self):
        cache = rc.MiddleCache(max_bytes=1)
        self.chain(cache)
        self.assertEqual(getattr(cache, "budget_bypasses", None), 1)
        self.assertEqual(getattr(cache, "required_entry_bytes", None), 96)
        self.assertEqual(cache.bytes, 0)

    def test_whole_span_skips_all_block_compute(self):
        cache = rc.MiddleCache(max_bytes=4096)
        cache.span = "whole"
        fresh, fresh_calls = self.chain(cache)
        reused, hit_calls = self.chain(cache, step=3)
        torch.testing.assert_close(fresh, reused, rtol=0, atol=0)
        self.assertEqual(fresh_calls, [0, 1, 2, 3])
        self.assertEqual(hit_calls, [])
        self.assertEqual(cache.hits, 1)

    def test_whole_span_contract(self):
        cache = rc.MiddleCache(max_bytes=4096, span="whole")
        self.chain(cache)
        _, other = self.chain(cache, step=3, key=(1,))
        self.assertEqual(other, [0, 1, 2, 3])
        _, hit = self.chain(cache, step=3)
        self.assertEqual(hit, [])
        _, capped = self.chain(cache, step=4)
        self.assertEqual(capped, [0, 1, 2, 3])
        cache.clear()
        self.assertEqual(cache.bytes, 0)
        with self.assertRaisesRegex(ValueError, "invalid cache span"):
            rc.MiddleCache(span="unknown")

    def test_scope_constructs_and_clears_whole_cache(self):
        scope = RunScope(.01, 1, 1, span="whole")
        seen = []
        def sample(*args, **kwargs):
            seen.append(scope.current().cache)
            self.assertEqual(seen[-1].span, "whole")
            self.chain(seen[-1])
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            scope.outer(sample, None, None, None, [1., .5, 0.])
        self.assertIsNone(scope.current())
        self.assertEqual(seen[0].entries, {})

    def test_whole_disables_prefetch_without_mutating_input(self):
        original = {"prefetch_dynamic_vbars": True, "other": 7}
        whole = ca.forward_options(original, True, "whole")
        self.assertFalse(whole["prefetch_dynamic_vbars"])
        self.assertTrue(original["prefetch_dynamic_vbars"])
        self.assertEqual(whole["other"], 7)
        self.assertTrue(ca.forward_options(original, True, "middle")
                        ["prefetch_dynamic_vbars"])
        self.assertTrue(ca.forward_options(original, False, "whole")
                        ["prefetch_dynamic_vbars"])

    def test_whole_wrapper_applies_prefetch_policy(self):
        state = ca.CurveAttnState(
            .2, 0, frozenset(), 0, False, False, False, Mock(),
            cache_enabled=True, run_scope=RunScope(.01, 1, 1, span="whole"),
            disable_fp16_accum=False)
        state.cache_span = "whole"
        seen = {}
        def inner(*args, **kwargs):
            seen.update(kwargs["transformer_options"])
            return 7
        original = {"prefetch_dynamic_vbars": True}
        ca.make_step_wrapper(state)(inner, None, None, None, original, None)
        self.assertFalse(seen["prefetch_dynamic_vbars"])
        self.assertTrue(original["prefetch_dynamic_vbars"])

    def test_span_validation(self):
        from curve_test_package.config import validate_options
        from test_config import DEFAULTS
        validate_options(**DEFAULTS, cache_span="middle")
        validate_options(**DEFAULTS, cache_span="whole")
        with self.assertRaisesRegex(ValueError, "cache span"):
            validate_options(**DEFAULTS, cache_span="invalid")

    def test_whole_patcher_wiring(self):
        from comfy.model_patcher import ModelPatcher
        model = torch.nn.Module()
        h3 = ca.MiniMaxH3Model.__new__(ca.MiniMaxH3Model)
        torch.nn.Module.__init__(h3)
        h3.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        for block in h3.blocks:
            block.attn = torch.nn.Identity()
        model.diffusion_model = h3
        patcher = ModelPatcher(model, load_device=torch.device("cpu"),
                               offload_device=torch.device("cpu"))
        with patch.object(ca.ck, "sol_attn_is_available", return_value=True), \
             patch.object(ca, "get_attention_function", return_value=Mock()), \
             patch.object(ca, "RunScope", wraps=RunScope) as scope:
            result = ca.apply_curve_attention(patcher, cache_enabled=True,
                                              cache_span="whole", qkv_mode="native")
        self.assertIsNot(result, patcher)
        scope.assert_called_once()
        args, kwargs = scope.call_args
        self.assertEqual(args, (.01, 1, 1024))
        self.assertEqual(kwargs.get("span"), "whole")
        self.assertEqual(kwargs.get("mode", "fingerprint"), "fingerprint")
        dit = result.model_options["transformer_options"]["patches_replace"]["dit"]
        self.assertEqual(set(dit), {("double_block", i) for i in range(4)})
        self.assertNotIn("diffusion_model._forward", result.object_patches)

    def test_whole_chunked_rejected_before_device_probe(self):
        with patch.object(ca.ck, "sol_attn_is_available") as probe:
            with self.assertRaisesRegex(ValueError, "whole cache requires native"):
                ca.apply_curve_attention(None, cache_enabled=True,
                                          cache_span="whole", qkv_mode="chunked")
            probe.assert_not_called()

    def test_append_only_cache_span_widget(self):
        expected = ["model", "selection", "keep_percent", "min_tokens", "dense_steps",
                    "dense_last_steps", "tail", "protect_audio", "verbose",
                    "protect_prefix", "disable_fp16_accum", "tau",
                    "extra_tokens", "start_percent", "end_percent", "keep_percent_end",
                    "tau_end", "dense_backend", "qkv_mode", "cache_enabled",
                    "cache_threshold", "cache_max_hits", "cache_max_mib", "cache_span",
                    "cache_preset", "cache_start_percent", "cache_end_percent",
                    "cache_min_steps", "cache_temporal_guard", "cache_audio_guard",
                    "cache_mode"]
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        schema = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "Schema")
        inputs = next(k.value for k in schema.keywords if k.arg == "inputs")
        names = [ast.literal_eval(node.args[0]) for node in inputs.elts]
        self.assertEqual(names, expected)
        with patch("curve_test_package.nodes.apply_curve_attention", return_value=object()) as fn:
            CurveAttention.execute(object(), cache_span="whole", qkv_mode="native")
        self.assertEqual(fn.call_args.kwargs["cache_span"], "whole")

    def test_whole_rejects_audio_drift_and_nonfinite(self):
        for bad in ("audio", "nan", "sign"):
            with self.subTest(bad=bad):
                cache = rc.MiddleCache(max_bytes=4096, span="whole")
                self.chain(cache)
                x = torch.ones(6, 4)
                if bad == "audio":
                    x[2:4] += 1
                elif bad == "nan":
                    x[:] = float("nan")
                else:
                    x.neg_()
                _, calls = self.chain(cache, step=3, x=x)
                self.assertEqual(calls, [0, 1, 2, 3])
                self.assertEqual(cache.hits, 0)

    def test_whole_rejects_ineligible_repeat_and_jump(self):
        cache = rc.MiddleCache(max_bytes=4096, span="whole")
        for step in (2, 2, 4):
            _, calls = self.chain(cache, step=step)
            self.assertEqual(calls, [0, 1, 2, 3])
        _, calls = self.chain(cache, step=5, eligible=False)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(cache.entries, {})

    def test_whole_boundary_steps_are_fresh(self):
        cache = rc.MiddleCache(max_bytes=4096, span="whole")
        for step in (0, 1, 6, 7):
            _, calls = self.chain(cache, step=step)
            self.assertEqual(calls, [0, 1, 2, 3])

    def test_rtx3090_widget_and_api_defaults_agree(self):
        expected = {
            "keep_percent": 20.0, "min_tokens": 12288, "dense_steps": "1",
            "dense_last_steps": 0, "tail": False, "protect_audio": False,
            "verbose": False, "protect_prefix": False, "disable_fp16_accum": True,
            "selection": "sla", "tau": 1.3, "extra_tokens": 0,
            "start_percent": 0.0, "end_percent": 1.0,
            "keep_percent_end": -1.0, "tau_end": -1.0,
            "dense_backend": "comfy_kitchen_int8", "qkv_mode": "native",
            "cache_enabled": False, "cache_threshold": 0.01,
            "cache_max_hits": 1, "cache_max_mib": 1024, "cache_span": "middle",
            "cache_preset": "manual", "cache_start_percent": 0.2,
            "cache_end_percent": 0.8, "cache_min_steps": 6,
            "cache_temporal_guard": False, "cache_audio_guard": False,
            "cache_mode": "fingerprint",
        }
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        schema = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "Schema")
        inputs = next(k.value for k in schema.keywords if k.arg == "inputs")
        actual = {}
        for node in inputs.elts:
            name = ast.literal_eval(node.args[0])
            if name == "model":
                continue
            default = next(k.value for k in node.keywords if k.arg == "default")
            actual[name] = ast.literal_eval(default)
        self.assertEqual(actual, expected)
        for function in (CurveAttention.execute, ca.apply_curve_attention):
            signature = inspect.signature(function)
            self.assertEqual({name: signature.parameters[name].default
                              for name in expected}, expected)






