"""Focused policy/ownership checks; no models, samplers or generation runs."""

import importlib
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import torch

NODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_DIR.parents[1]))
package = types.ModuleType("curve_test_package")
package.__path__ = [str(NODE_DIR)]
package.__version__ = "2.5"
sys.modules[package.__name__] = package
ca = importlib.import_module("curve_test_package.curve_attention")
sol = importlib.import_module("curve_test_package.sol_attn")


def state(**kwargs):
    defaults = dict(
        keep_ratio=0.2, min_tokens=0, dense_steps=frozenset({1}),
        dense_last_steps=0, tail=False, protect_audio=False, verbose=False,
        dense_attention=Mock(return_value="dense"),
    )
    return ca.CurveAttnState(**(defaults | kwargs))


class PolicyTests(unittest.TestCase):
    def test_parse_steps_accepts_valid_specs(self):
        self.assertEqual(ca._parse_steps("1"), frozenset({1}))
        self.assertEqual(ca._parse_steps("0-1"), frozenset({0, 1}))
        self.assertEqual(ca._parse_steps("0,2"), frozenset({0, 2}))
        self.assertEqual(ca._parse_steps("0; 2"), frozenset({0, 2}))
        self.assertEqual(ca._parse_steps("  "), frozenset())
        self.assertEqual(ca._parse_steps(""), frozenset())

    def test_parse_steps_rejects_invalid_leftover(self):
        for bad in ("1a", "0-1x", "step:1", "1..2", "0_1", "1/2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ca._parse_steps(bad)

    def test_is_fatal_device_error(self):
        self.assertTrue(ca._is_fatal_device_error(RuntimeError("CUDA error: illegal memory access")))
        self.assertTrue(ca._is_fatal_device_error(RuntimeError("cuda error: device-side assert")))
        self.assertTrue(ca._is_fatal_device_error(RuntimeError("illegal memory access")))
        self.assertFalse(ca._is_fatal_device_error(RuntimeError("unsupported sparse op")))
        class CudaBoom(Exception):
            pass
        self.assertTrue(ca._is_fatal_device_error(CudaBoom("boom")))
        cuda_error = getattr(torch.cuda, "CudaError", None)
        if cuda_error is not None:
            try:
                sample = cuda_error(700)
            except TypeError:
                sample = cuda_error.__new__(cuda_error)
            self.assertTrue(ca._is_fatal_device_error(sample))

    def test_default_sinks_do_not_add_the_69_block_prefix(self):
        layout = types.SimpleNamespace(
            seq_len=46202,
            segments=[(0, 1024, "text"), (1024, 4390, "audio"), (4390, 46202, "video")],
        )
        options = {"minimax_h3_layout": layout, "_curve_sinks": ((0, 69), (0, 0))}
        self.assertEqual(ca.h3_sinks(options, 46202, False), ((0, 0), (0, 0)))
        self.assertEqual(ca.h3_sinks(options, 46202, False, True), ((0, 69), (0, 0)))
        self.assertEqual(ca.h3_sinks(options, 46202, True), ((0, 0), (16, 69)))
        self.assertIsNone(ca.h3_sinks(options, 257, False, True))
        self.assertEqual(ca.h3_sinks({}, 257, False), ((0, 0), (0, 0)))

    def test_schedule_uses_sigma_not_call_count(self):
        s = state()
        wrapper = ca.make_step_wrapper(s)
        schedule = [1.0, 0.75, 0.5, 0.25, 0.0]
        actual = []
        payload = {"sentinel": object()}

        def forward(x, timestep, context, transformer_options, minimax_payload):
            self.assertIs(minimax_payload, payload)
            actual.append(transformer_options["_curve_dense"])

        for sigma in [1.0, 0.75, 0.75, 0.5, 0.25]:
            wrapper(forward, None, None, None,
                    {"sample_sigmas": schedule, "sigmas": torch.tensor([sigma])}, payload)
        self.assertEqual(actual, [False, True, True, False, False])
        s.dense_last_steps = 1
        wrapper(forward, None, None, None,
                {"sample_sigmas": torch.tensor(schedule), "sigmas": 0.25}, payload)
        self.assertTrue(actual[-1])
        wrapper(forward, None, None, None, {}, payload)
        self.assertTrue(actual[-1])
        self.assertEqual(ca.resolve_step({"sample_sigmas": [], "sigmas": 1}), (None, 0))

    def test_reduction_flags_restore_including_split_k_and_cancellation(self):
        backend = torch.backends.cuda.matmul
        names = ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")

        def snapshot():
            return [(getattr(backend, n), getattr(backend, n + "_split_k", None)) for n in names]

        original = snapshot()
        try:
            for name, setting in zip(names, ((False, False), (True, True))):
                setattr(backend, name, setting if hasattr(backend, name + "_split_k") else setting[0])
            before = snapshot()
            with self.assertRaises(KeyboardInterrupt):
                with ca.reduction_policy(True):
                    self.assertTrue(all(not getattr(backend, n) for n in names))
                    raise KeyboardInterrupt
            self.assertEqual(snapshot(), before)
            with ca.reduction_policy(False):
                self.assertEqual(snapshot(), before)
        finally:
            for name, (value, split_k) in zip(names, original):
                setattr(backend, name, value if split_k is None else (value, split_k))

    def test_dense_backend_and_container_handoff(self):
        containers = [ca.AttentionTensorContainer(torch.empty(1, 2, 65, 128)) for _ in range(3)]

        def dense(q, k, v, heads, **kwargs):
            self.assertTrue(kwargs["_inside_attn_wrapper"])
            self.assertTrue(all(isinstance(t, ca.AttentionTensorContainer) for t in (q, k, v)))
            return [t.take() for t in (q, k, v)]

        s = state(dense_attention=dense)
        override = ca.make_attention_override(s)
        result = override.container_function(
            *containers, 2, skip_reshape=True, transformer_options={"_curve_dense": True})
        self.assertEqual(len(result), 3)
        self.assertTrue(all(t.tensor is None for t in containers))
        self.assertEqual(s.dense_calls, 1)
        displaced = Mock(side_effect=AssertionError("must not call displaced backend"))
        with patch.object(s, "dense_attention", return_value="kitchen"):
            self.assertEqual(override(displaced, None, None, None, 2, mask=True), "kitchen")
        displaced.assert_not_called()

    def test_sparse_dtype_policy_and_output_layout(self):
        qkv = [torch.randn(1, 2, 65, 128, dtype=torch.float16) for _ in range(3)]
        seen = []

        def sparse(q, k, v, **kwargs):
            seen.append(kwargs)
            self.assertEqual(q.dtype, torch.bfloat16)
            self.assertEqual(tuple(q.shape), (1, 65, 2, 128))
            return v

        s = state()
        with patch.object(ca, "_ineligible", return_value=None), patch.object(ca, "sol_attn_zerocopy", sparse):
            out = ca.make_attention_override(s)(None, *qkv, 2, skip_reshape=True)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (1, 65, 256))
        self.assertEqual(seen[0]["sink_blocks"], [0, 0])
        self.assertEqual(seen[0]["sink_q"], [0, 0])
        self.assertFalse(seen[0]["tail"])
        self.assertEqual(seen[0]["token_aug"], 0)
        self.assertEqual(seen[0]["tau"], 0.0)

    def test_token_aug_threaded_into_sol_kwargs(self):
        qkv = [torch.randn(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        seen = []

        def sparse(q, k, v, **kwargs):
            seen.append(kwargs)
            return v

        s = state(token_aug=256)
        with patch.object(ca, "_ineligible", return_value=None), patch.object(ca, "sol_attn_zerocopy", sparse):
            ca.make_attention_override(s)(None, *qkv, 2, skip_reshape=True)
        self.assertEqual(seen[0]["token_aug"], 256)

    def test_sol_attn_selection_forces_topk_zero(self):
        qkv = [torch.randn(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        seen = []

        def sparse(q, k, v, **kwargs):
            seen.append(kwargs)
            return v

        s = state(selection="sol-attn", tau=1.3, keep_ratio=0.2)
        s.current_tau = 1.3
        with patch.object(ca, "_ineligible", return_value=None), patch.object(ca, "sol_attn_zerocopy", sparse):
            ca.make_attention_override(s)(None, *qkv, 2, skip_reshape=True)
        self.assertEqual(seen[0]["topk_ratio"], 0.0)
        self.assertEqual(seen[0]["tau"], 1.3)

    def test_lerp_schedule_helper(self):
        self.assertAlmostEqual(ca.lerp_schedule(0.2, 0.4, 0.0), 0.2)
        self.assertAlmostEqual(ca.lerp_schedule(0.2, 0.4, 1.0), 0.4)
        self.assertAlmostEqual(ca.lerp_schedule(0.2, 0.4, 0.5), 0.3)
        self.assertEqual(ca.schedule_progress(None, 4), 0.0)
        self.assertEqual(ca.schedule_progress(0, 1), 0.0)
        self.assertAlmostEqual(ca.schedule_progress(1, 3), 0.5)

    def test_keep_schedule_updates_current_topk(self):
        s = state(selection="sla", keep_ratio=0.2, keep_ratio_end=0.4)
        wrapper = ca.make_step_wrapper(s)
        schedule = [1.0, 0.5, 0.0]
        values = []

        def forward(x, timestep, context, transformer_options, minimax_payload=None):
            values.append(s.current_topk)

        for sigma in [1.0, 0.5]:
            wrapper(forward, None, None, None,
                    {"sample_sigmas": schedule, "sigmas": torch.tensor([sigma])}, None)
        self.assertAlmostEqual(values[0], 0.2)
        self.assertAlmostEqual(values[1], 0.4)

    def test_outside_sigma_window_helper(self):
        self.assertFalse(ca.outside_sigma_window(None, 1.0, 0.0))
        self.assertFalse(ca.outside_sigma_window(0.5, None, 0.0))
        self.assertFalse(ca.outside_sigma_window(0.5, 1.0, 0.0))
        self.assertTrue(ca.outside_sigma_window(1.5, 1.0, 0.0))
        self.assertTrue(ca.outside_sigma_window(-0.1, 1.0, 0.0))

    def test_sigma_window_forces_dense(self):
        s = state(dense_steps=frozenset(), sigma_start=1.0, sigma_end=0.25)
        wrapper = ca.make_step_wrapper(s)
        schedule = [2.0, 1.0, 0.5, 0.1, 0.0]
        reasons = []

        def forward(x, timestep, context, transformer_options, minimax_payload=None):
            reasons.append((
                transformer_options["_curve_dense"],
                transformer_options["_curve_dense_reason"],
            ))

        for sigma in [2.0, 1.0, 0.5, 0.1]:
            wrapper(forward, None, None, None,
                    {"sample_sigmas": schedule, "sigmas": torch.tensor([sigma])}, None)
        self.assertEqual(reasons[0], (True, "outside start/end_percent"))
        self.assertEqual(reasons[1][0], False)
        self.assertEqual(reasons[2][0], False)
        self.assertEqual(reasons[3], (True, "outside start/end_percent"))

    def test_failed_sparse_op_is_not_retried_until_cleanup(self):
        s = state()
        qkv = [torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        override = ca.make_attention_override(s)
        with patch.object(ca, "_ineligible", return_value=None), patch.object(
            ca, "sol_attn_zerocopy", side_effect=RuntimeError("unsupported sparse op")
        ) as sparse:
            with self.assertLogs("CurveAttention", level="WARNING"):
                self.assertEqual(override(None, *qkv, 2, skip_reshape=True), "dense")
            self.assertEqual(override(None, *qkv, 2, skip_reshape=True), "dense")
            sparse.assert_called_once()
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 2))
        ca._cleanup(s)
        self.assertIsNone(s.failure_reason)
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 0))

    def test_fatal_cuda_errors_do_not_launch_dense_fallback(self):
        s = state()
        qkv = [torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        override = ca.make_attention_override(s)
        for error in (torch.OutOfMemoryError("out of memory"), RuntimeError("CUDA error: illegal memory access")):
            with self.subTest(error=type(error)), patch.object(ca, "_ineligible", return_value=None), patch.object(
                ca, "sol_attn_zerocopy", side_effect=error
            ), self.assertRaises(type(error)):
                override(None, *qkv, 2, skip_reshape=True)
        s.dense_attention.assert_not_called()

    def test_preparation_failure_restores_ownership_and_stays_sparse(self):
        s = state(prepared_available=True)
        override = ca.make_attention_override(s)
        with patch.object(ca, "_ineligible", return_value=None), patch.object(
            ca, "prepare_sla", side_effect=RuntimeError("selector unavailable")
        ) as prepare, patch.object(
            ca, "sol_attn_zerocopy", side_effect=lambda q, k, v, **kwargs: v
        ):
            with self.assertLogs("CurveAttention", level="WARNING"):
                for _ in range(2):
                    containers = [ca.AttentionTensorContainer(torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16))
                                  for _ in range(3)]
                    result = override.container_function(*containers, 2, skip_reshape=True)
                    self.assertEqual(result.shape, (1, 65, 256))
                    self.assertTrue(all(t.tensor is None for t in containers))
            prepare.assert_called_once()
        self.assertEqual((s.sparse_calls, s.dense_calls), (2, 0))
        self.assertFalse(s.prepared_enabled)
        ca._cleanup(s)
        self.assertTrue(s.prepared_enabled)

    def test_failed_preparation_returns_original_container_inputs(self):
        tensors = [torch.zeros(1, 2, 65, 128, dtype=torch.float16) for _ in range(3)]
        containers = [ca.AttentionTensorContainer(tensor) for tensor in tensors]
        with patch.object(ca, "prepare_sla", side_effect=torch.OutOfMemoryError("out of memory")):
            with self.assertRaises(torch.OutOfMemoryError):
                ca._prepare_owned(*containers, 2, True)
        self.assertTrue(all(container.peek() is tensor for container, tensor in zip(containers, tensors)))

    def test_patcher_clone_and_sole_override_ownership(self):
        from comfy.model_patcher import ModelPatcher

        base = torch.nn.Module()
        base.diffusion_model = ca.MiniMaxH3Model.__new__(ca.MiniMaxH3Model)
        torch.nn.Module.__init__(base.diffusion_model)
        model = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        with patch.object(ca.ck, "sol_attn_is_available", return_value=True), patch.object(
            ca, "get_attention_function", return_value=Mock()
        ) as backend:
            first = ca.apply_curve_attention(model)
            second = ca.apply_curve_attention(model)
            self.assertNotIn("optimized_attention_override", model.model_options["transformer_options"])
            first_override = first.model_options["transformer_options"]["optimized_attention_override"]
            self.assertIsNot(first_override, second.model_options["transformer_options"]["optimized_attention_override"])
            options = {"transformer_options": {}}
            for callback in first.get_all_callbacks(ca.CallbacksMP.ON_PREPARE_STATE):
                callback(first, None, options)
                callback(first, None, options)
            self.assertIs(options["transformer_options"]["optimized_attention_override"], first_override)
            with self.assertRaisesRegex(ValueError, "already has CurveAttention"):
                ca.apply_curve_attention(first)
            from comfy.ldm.modules.attention import get_attention_function as real_get
            model.set_model_optimized_attention(real_get("pytorch"))
            behind_backend = ca.apply_curve_attention(model)
            self.assertIs(
                behind_backend.model_options["transformer_options"]["optimized_attention_override"]._curve_attention,
                True,
            )
            self.assertTrue(all(call.args[0] == "comfy_kitchen_int8" for call in backend.call_args_list))
            for callback in first.get_all_callbacks(ca.CallbacksMP.ON_CLEANUP):
                callback(first)

    def test_dense_backend_override_matches_captured_function_not_qualname(self):
        from comfy.ldm.modules.attention import get_attention_function
        from comfy.model_patcher import ModelPatcher
        patcher = ModelPatcher(torch.nn.Module(), load_device=torch.device("cpu"),
                                offload_device=torch.device("cpu"))
        patcher.set_model_optimized_attention(get_attention_function("pytorch"))
        installed = patcher.model_options["transformer_options"]["optimized_attention_override"]
        self.assertTrue(ca._is_dense_backend_override(installed))
        self.assertFalse(ca._is_dense_backend_override(lambda *a, **k: None))

        def optimized_attention_override(_, *args, **kwargs):
            return None
        optimized_attention_override.__qualname__ = (
            "set_model_optimized_attention.<locals>.optimized_attention_override")
        self.assertFalse(ca._is_dense_backend_override(optimized_attention_override))

    def test_lerped_keep_is_not_part_of_the_cache_signature(self):
        s = state(selection="sla", keep_ratio=0.2, keep_ratio_end=0.8)
        s.current_topk = 0.2
        first = ca.cache_policy_key(s, False, None, None)
        s.current_topk = 0.8
        s.current_tau = 3.0
        self.assertEqual(ca.cache_policy_key(s, False, None, None), first)

    def test_owned_bf16_does_not_clone_an_exclusive_tensor(self):
        from curve_test_package import prepared_sol as ps
        fresh = torch.randn(1, 65, 2, 128, dtype=torch.bfloat16)
        owned = ps._own(fresh)
        self.assertEqual(owned.data_ptr(), fresh.data_ptr())
        packed = torch.randn(1, 65, 3 * 2 * 128, dtype=torch.bfloat16)
        view = packed.chunk(3, dim=-1)[0].view(1, 65, 2, 128)
        copied = ps._own(view)
        self.assertNotEqual(copied.untyped_storage().data_ptr(),
                            view.untyped_storage().data_ptr())

    def test_schema_keeps_existing_widget_order_and_defaults(self):
        nodes = importlib.import_module("curve_test_package.nodes")
        schema = nodes.CurveAttention.define_schema()
        ids = [entry.id for entry in schema.inputs]
        # 2.5 regroup: selection is in the always-applied core. Closing Sol-Attn
        # must not be able to flip it. Order matches web/curve_attention.js sections.
        expected_prefix = [
            "model", "selection", "keep_percent", "min_tokens", "dense_steps",
            "dense_last_steps", "tail", "protect_audio", "verbose",
            "protect_prefix", "disable_fp16_accum",
        ]
        expected_new = [
            "tau", "extra_tokens", "start_percent", "end_percent",
            "keep_percent_end", "tau_end", "dense_backend", "qkv_mode",
            "cache_enabled", "cache_threshold", "cache_max_hits", "cache_max_mib",
            "cache_span",
            "cache_preset", "cache_start_percent", "cache_end_percent",
            "cache_min_steps", "cache_temporal_guard", "cache_audio_guard",
            "cache_mode",
        ]
        self.assertEqual(ids[:len(expected_prefix)], expected_prefix)
        self.assertEqual(ids[len(expected_prefix):], expected_new)
        self.assertEqual(ids, expected_prefix + expected_new)

    def test_package_version_is_2_5(self):
        self.assertEqual(package.__version__, "2.5")
        real = Path(NODE_DIR) / "__init__.py"
        self.assertIn('__version__ = "2.5"', real.read_text())


class PreparedWiringTests(unittest.TestCase):
    def test_sol_attn_module_uses_local_prepared_when_kitchen_has_none(self):
        self.assertIsNotNone(sol.sol_prepared)
        self.assertTrue(hasattr(sol.sol_prepared, "prequantize_sol_attn"))
        self.assertEqual(sol.sol_prepared.SELECTOR, "owned_bf16")

    def test_check_prepared_available_defers_to_backend(self):
        with patch.object(sol.sol_prepared, "check_available", side_effect=RuntimeError("nope")):
            with self.assertRaises(RuntimeError) as ctx:
                sol.check_prepared_available()
        self.assertEqual(str(ctx.exception), "nope")


class PreparedSolLocalTests(unittest.TestCase):
    def test_check_available_requires_compiled_sol(self):
        from curve_test_package import prepared_sol as ps
        with patch.object(ps.ck, "sol_attn_is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                ps.check_available()
        self.assertIn("sol_attn", str(ctx.exception).lower())

    def test_check_available_passes_when_sol_exists(self):
        from curve_test_package import prepared_sol as ps
        with patch.object(ps.ck, "sol_attn_is_available", return_value=True):
            ps.check_available()

    def test_prequantize_drops_packed_parent_storage(self):
        from curve_test_package import prepared_sol as ps
        packed = torch.randn(1, 65, 3 * 2 * 128, dtype=torch.bfloat16)
        owner = weakref.ref(packed)
        q, k, v = [x.view(1, 65, 2, 128) for x in packed.chunk(3, dim=-1)]
        del packed
        with patch.object(ps.ck, "sol_attn_is_available", return_value=True):
            prepared = ps.prequantize_sol_attn(
                q, k, v, topk_ratio=0.2, tail=False, tau=0.0,
                sink_blocks=[0, 0], sink_q=[0, 0], token_aug=0, scale=None,
            )
        del q, k, v
        self.assertIsNone(owner(), "packed parent still alive; prepare did not take ownership")
        self.assertEqual(prepared.selector, "owned_bf16")
        self.assertEqual(tuple(prepared.q.shape), (1, 65, 2, 128))
        self.assertEqual(prepared.q.dtype, torch.bfloat16)
        self.assertEqual(type(prepared).__name__, "PreparedSol")

    def test_consume_calls_kitchen_sol_with_stored_kwargs(self):
        from curve_test_package import prepared_sol as ps
        q = torch.randn(1, 65, 2, 128, dtype=torch.bfloat16)
        captured = {}

        def fake_sol(aq, ak, av, **kwargs):
            captured["args"] = (aq, ak, av)
            captured["kwargs"] = kwargs
            return av

        with patch.object(ps.ck, "sol_attn_is_available", return_value=True), \
             patch.object(ps.ck, "sol_attn", fake_sol):
            prepared = ps.prequantize_sol_attn(
                q, q, q, topk_ratio=0.2, tail=False, tau=0.0,
                sink_blocks=[0, 0], sink_q=[0, 0], token_aug=0, scale=None,
            )
            out = ps.sol_attn_from_prequantized(prepared)
        self.assertIs(out, captured["args"][2])
        self.assertEqual(captured["kwargs"]["topk_ratio"], 0.2)
        self.assertEqual(captured["kwargs"]["tail"], False)
        self.assertEqual(captured["kwargs"]["token_aug"], 0)

    def test_consume_rejects_wrong_type(self):
        from curve_test_package import prepared_sol as ps
        with self.assertRaises(TypeError):
            ps.sol_attn_from_prequantized(object())


class LayoutTests(unittest.TestCase):
    def test_aligned_fused_qkv_views_are_not_copied(self):
        packed = torch.randn(1, 65, 3 * 2 * 128, dtype=torch.bfloat16)
        for x in packed.chunk(3, dim=-1):
            x = x.view(1, 65, 2, 128)
            self.assertFalse(x.is_contiguous())
            self.assertIs(sol.maybe_contiguous(x), x)

    def test_misaligned_contiguous_offset_is_really_copied(self):
        storage = torch.arange(1 + 65 * 128, dtype=torch.float32)
        x = storage[1:].view(1, 65, 1, 128)
        self.assertTrue(x.is_contiguous())
        y = sol.maybe_contiguous(x)
        self.assertNotEqual(y.data_ptr(), x.data_ptr())
        self.assertTrue(sol._sol_layout_ok(y))
        torch.testing.assert_close(y, x, rtol=0, atol=0)

    def test_noncontiguous_last_dim_is_normalized(self):
        x = torch.randn(1, 65, 2, 256)[..., ::2]
        y = sol.maybe_contiguous(x)
        self.assertTrue(sol._sol_layout_ok(y))
        torch.testing.assert_close(y, x, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and ca.ck.sol_attn_is_available(), "native CUDA Sol required")
class NativeOperatorTests(unittest.TestCase):
    def test_prepared_h3_dispatch_releases_float_owner_before_output(self):
        from comfy.ldm.modules.attention import attention_pytorch

        def inputs():
            packed = torch.randn(1, 257, 768, dtype=torch.bfloat16, device="cuda")
            reference = weakref.ref(packed)
            tensors = [tensor.view(1, 257, 2, 128) for tensor in packed.chunk(3, dim=-1)]
            expected = ca.ck.sol_attn(*tensors, topk_ratio=.2, tail=False).reshape(1, 257, 256)
            containers = [ca.AttentionTensorContainer(tensor.transpose(1, 2)) for tensor in tensors]
            return containers, reference, expected

        containers, owner, expected = inputs()
        s = state(prepared_available=True)
        options = {}
        ca.install_override(s, options)
        native_consume = ca.consume_sla

        def consume(quantized):
            self.assertIsNone(owner(), "float QKV owner retained while allocating attention output")
            self.assertTrue(all(container.tensor is None for container in containers))
            self.assertEqual(quantized.selector, "owned_bf16")
            return native_consume(quantized)

        with patch.object(ca, "consume_sla", consume):
            actual = attention_pytorch(*containers, 2, skip_reshape=True, transformer_options=options)
        self.assertEqual((s.sparse_calls, s.dense_calls), (1, 0))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dense_container_dispatch_matches_native_kitchen(self):
        from comfy.ldm.modules.attention import attention_pytorch

        qkv = [torch.randn(1, 2, 65, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
        backend = ca.get_attention_function("comfy_kitchen_int8")
        s = state(dense_attention=backend)
        options = {"_curve_dense": True}
        ca.install_override(s, options)
        containers = [ca.AttentionTensorContainer(x) for x in qkv]
        actual = attention_pytorch(
            *containers, 2, skip_reshape=True, transformer_options=options)
        expected = backend(*qkv, 2, skip_reshape=True)
        self.assertTrue(all(x.tensor is None for x in containers))
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_strided_attention_matches_contiguous_native_sla(self):
        generator = torch.Generator(device="cuda").manual_seed(7321)
        for tokens, ties, sinks, query_sinks in [
            (65, False, [0, 0], [0, 0]),
            (257, False, [0, 0], [0, 0]),
            (257, True, [0, 0], [0, 0]),
            (257, False, [0, 2], [1, 2]),
        ]:
            with self.subTest(tokens=tokens, ties=ties, sinks=sinks):
                packed = torch.randn(
                    1, tokens, 3 * 2 * 128, dtype=torch.bfloat16, device="cuda", generator=generator)
                q, k, v = [x.view(1, tokens, 2, 128) for x in packed.chunk(3, dim=-1)]
                if ties:
                    q.zero_()
                options = dict(topk_ratio=0.2, tail=False, sink_blocks=sinks, sink_q=query_sinks)
                actual = sol.sol_attn_zerocopy(q, k, v, **options)
                expected = ca.ck.sol_attn(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    token_aug=0, tau=0.0, **options,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class OwnershipRegressionTests(unittest.TestCase):
    def test_container_sparse_failure_can_run_dense(self):
        tensors = [torch.ones(1, 2, 65, 128, dtype=torch.bfloat16)
                   for _ in range(3)]
        containers = [ca.AttentionTensorContainer(t) for t in tensors]
        def dense(q, k, v, heads, **kw):
            self.assertTrue(kw["_inside_attn_wrapper"])
            values = [c.take() for c in (q, k, v)]
            self.assertTrue(all(a is b for a, b in zip(values, tensors)))
            return values[2].transpose(1, 2).reshape(1, 65, 256)
        s = state(dense_attention=dense, prepared_available=False)
        with patch.object(ca, "_ineligible", return_value=None), patch.object(
            ca, "sol_attn_zerocopy", side_effect=RuntimeError("unsupported op")
        ):
            out = ca.make_attention_override(s).container_function(
                *containers, 2, skip_reshape=True)
        self.assertEqual(tuple(out.shape), (1, 65, 256))
        self.assertTrue(all(c.tensor is None for c in containers))
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 1))

    def test_container_fatal_error_does_not_call_dense(self):
        for error in (torch.OutOfMemoryError("test OOM"),
                      RuntimeError("CUDA error: illegal memory access")):
            s = state(prepared_available=False)
            containers = [ca.AttentionTensorContainer(
                torch.ones(1, 2, 65, 128, dtype=torch.bfloat16)) for _ in range(3)]
            with patch.object(ca, "_ineligible", return_value=None), patch.object(
                ca, "sol_attn_zerocopy", side_effect=error
            ), self.assertRaises(type(error)):
                ca.make_attention_override(s).container_function(
                    *containers, 2, skip_reshape=True)
            s.dense_attention.assert_not_called()

    def test_prepared_consume_failure_does_not_run_dense(self):
        s = state(prepared_available=True)
        containers = [ca.AttentionTensorContainer(
            torch.ones(1, 2, 65, 128, dtype=torch.bfloat16)) for _ in range(3)]
        carrier = types.SimpleNamespace(selector="test")
        with patch.object(ca, "_ineligible", return_value=None), patch.object(
            ca, "prepare_sla", return_value=carrier
        ), patch.object(ca, "consume_sla", side_effect=RuntimeError("consume failed")), \
             self.assertRaisesRegex(RuntimeError, "consume failed"):
            ca.make_attention_override(s).container_function(
                *containers, 2, skip_reshape=True)
        s.dense_attention.assert_not_called()
        self.assertTrue(all(c.tensor is None for c in containers))


class DenseChoiceTests(unittest.TestCase):
    def test_pytorch_choice(self):
        from comfy.model_patcher import ModelPatcher
        base = torch.nn.Module()
        base.diffusion_model = ca.MiniMaxH3Model.__new__(ca.MiniMaxH3Model)
        torch.nn.Module.__init__(base.diffusion_model)
        model = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        with patch.object(ca.ck, "sol_attn_is_available", return_value=True), patch.object(
            ca, "get_attention_function", return_value=Mock()
        ) as backend:
            ca.apply_curve_attention(model, dense_backend="pytorch")
        self.assertEqual(backend.call_args.args[:2], ("pytorch", None))
        with patch.object(ca.ck, "sol_attn_is_available", return_value=True), patch.object(
            ca, "get_attention_function", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "pytorch"):
                ca.apply_curve_attention(model, dense_backend="pytorch")


class QkvModeTests(unittest.TestCase):
    def test_native_never_prepares(self):
        qkv = [torch.randn(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        containers = [ca.AttentionTensorContainer(t) for t in qkv]
        s = state(prepared_available=False, qkv_mode="native")
        with patch.object(ca, "_ineligible", return_value=None), patch.object(
            ca, "prepare_sla"
        ) as prepare, patch.object(ca, "sol_attn_zerocopy", side_effect=lambda q, k, v, **kw: v):
            out = ca.make_attention_override(s).container_function(*containers, 2, skip_reshape=True)
        prepare.assert_not_called()
        self.assertEqual(tuple(out.shape), (1, 65, 256))
        self.assertTrue(all(c.tensor is None for c in containers))
        self.assertEqual(s.sparse_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

