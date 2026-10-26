"""CurveAttn Sparse engine: policy, ownership, fallbacks. No models or generation."""
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from curvattn_test_package import common
from curvattn_test_package import sparse as sp
from curvattn_test_package import sol_attn as sol
from curvattn_test_package import prepared_sol as ps
from helpers import h3_patcher, kitchen, sparse_state as state, step_options

ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def test_parse_steps_accepts_valid_specs(self):
        self.assertEqual(common.parse_steps("1"), frozenset({1}))
        self.assertEqual(common.parse_steps("0-1"), frozenset({0, 1}))
        self.assertEqual(common.parse_steps("0,2"), frozenset({0, 2}))
        self.assertEqual(common.parse_steps("0; 2"), frozenset({0, 2}))
        self.assertEqual(common.parse_steps("  "), frozenset())
        self.assertEqual(common.parse_steps(""), frozenset())

    def test_bare_zero_means_no_dense_steps(self):
        self.assertEqual(common.parse_steps("0"), frozenset())
        self.assertEqual(common.parse_steps(" 0 "), frozenset())
        self.assertEqual(common.parse_steps("0-0"), frozenset({0}))
        self.assertEqual(common.parse_steps("0,1"), frozenset({0, 1}))

    def test_cleanup_inside_sampling_keeps_counts_for_report(self):
        s = state()
        s.sparse_calls, s.dense_calls = 5, 2
        with s.lock:
            s.cleanup()  # ComfyUI fires ON_CLEANUP before our outer() reports
        self.assertEqual((s.sparse_calls, s.dense_calls), (5, 2))
        s.cleanup()
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 0))

    def test_parse_steps_rejects_invalid_leftover(self):
        for bad in ("1a", "0-1x", "step:1", "1..2", "0_1", "1/2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                common.parse_steps(bad)

    def test_is_fatal_device_error(self):
        self.assertTrue(sp.is_fatal_device_error(RuntimeError("CUDA error: illegal memory access")))
        self.assertTrue(sp.is_fatal_device_error(RuntimeError("illegal memory access")))
        self.assertFalse(sp.is_fatal_device_error(RuntimeError("unsupported sparse op")))

        class CudaBoom(Exception):
            pass
        self.assertTrue(sp.is_fatal_device_error(CudaBoom("boom")))

    def test_default_sinks_do_not_add_the_69_block_prefix(self):
        layout = types.SimpleNamespace(
            seq_len=46202,
            segments=[(0, 1024, "text"), (1024, 4390, "audio"), (4390, 46202, "video")])
        options = {"minimax_h3_layout": layout}
        self.assertEqual(common.h3_sinks(options, 46202, False), ((0, 0), (0, 0)))
        self.assertEqual(common.h3_sinks(options, 46202, False, True), ((0, 69), (0, 0)))
        self.assertEqual(common.h3_sinks(options, 46202, True), ((0, 0), (16, 69)))
        self.assertIsNone(common.h3_sinks(options, 257, False, True))
        self.assertEqual(common.h3_sinks({}, 257, False), ((0, 0), (0, 0)))

    def wrap(self, s, options, payload=None):
        seen = []

        def forward(x, timestep, context, transformer_options, minimax_payload):
            self.assertIs(minimax_payload, payload)
            seen.append(transformer_options)
        sp.make_step_wrapper(s)(forward, None, None, None, options, payload)
        return seen[0]

    def decision(self, to, s):
        return to[common.NS]["sparse"][s.uid]

    def test_schedule_uses_sigma_not_call_count(self):
        s = state()
        schedule = [1.0, 0.75, 0.5, 0.25, 0.0]
        payload = {"sentinel": object()}
        actual = [self.decision(self.wrap(s, {"sample_sigmas": schedule,
                                              "sigmas": torch.tensor([sigma])}, payload), s).dense
                  for sigma in [1.0, 0.75, 0.75, 0.5, 0.25]]
        self.assertEqual(actual, [False, True, True, False, False])
        s.dense_last_steps = 1
        to = self.wrap(s, {"sample_sigmas": torch.tensor(schedule), "sigmas": 0.25}, payload)
        self.assertTrue(self.decision(to, s).dense)
        self.assertTrue(self.decision(self.wrap(s, {}, payload), s).dense)
        self.assertEqual(common.resolve_step({"sample_sigmas": [], "sigmas": 1}), (None, 0))

    def test_substep_sigmas_follow_the_enclosing_step(self):
        s = state()  # dense_steps={1}
        schedule = [1.0, 0.75, 0.5, 0.25, 0.0]
        actual = [self.decision(self.wrap(s, {"sample_sigmas": schedule, "sigmas": sigma}), s)
                  for sigma in [0.9, 0.75, 0.6, 0.4, 0.1]]
        self.assertEqual([d.dense for d in actual], [False, True, True, False, False])
        self.assertEqual([d.step for d in actual], [0, 1, 1, 2, 3])
        for sigma in [1.5, 0.0, float("nan")]:
            self.assertTrue(self.decision(self.wrap(
                s, {"sample_sigmas": schedule, "sigmas": sigma}), s).dense, sigma)

    def test_owner_published_only_on_sparse_steps(self):
        s = state()
        schedule = [1.0, 0.5, 0.0]
        sparse_to = self.wrap(s, {"sample_sigmas": schedule, "sigmas": 1.0})
        dense_to = self.wrap(s, {"sample_sigmas": schedule, "sigmas": 0.5})
        self.assertEqual(sparse_to[common.NS]["owner"], s.uid)
        self.assertIsNone(dense_to[common.NS]["owner"])
        self.assertEqual(len(sparse_to[common.NS]["policy"]), 1)

    def test_forward_options_do_not_leak(self):
        s = state(dense_steps=frozenset())
        original = {"sample_sigmas": [1, .5, 0], "sigmas": .5}
        sp.make_step_wrapper(s)(lambda *a, **k: 1, None, None, None, original, None)
        self.assertNotIn(common.NS, original)

        def boom(*args, **kwargs):
            raise RuntimeError("fail")
        with self.assertRaisesRegex(RuntimeError, "fail"):
            sp.make_step_wrapper(s)(boom, None, None, None, original, None)
        self.assertNotIn(common.NS, original)

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
                with sp.reduction_policy(True):
                    self.assertTrue(all(not getattr(backend, n) for n in names))
                    raise KeyboardInterrupt
            self.assertEqual(snapshot(), before)
            with sp.reduction_policy(False):
                self.assertEqual(snapshot(), before)
        finally:
            for name, (value, split_k) in zip(names, original):
                setattr(backend, name, value if split_k is None else (value, split_k))

    def test_dense_backend_and_container_handoff(self):
        containers = [sp.AttentionTensorContainer(torch.empty(1, 2, 65, 128)) for _ in range(3)]

        def dense(q, k, v, heads, **kwargs):
            self.assertTrue(kwargs["_inside_attn_wrapper"])
            self.assertTrue(all(isinstance(t, sp.AttentionTensorContainer) for t in (q, k, v)))
            return [t.take() for t in (q, k, v)]

        s = state(dense_attention=dense)
        override = sp.make_attention_override(s)
        result = override.container_function(
            *containers, 2, skip_reshape=True, transformer_options=step_options(s, dense=True))
        self.assertEqual(len(result), 3)
        self.assertTrue(all(t.tensor is None for t in containers))
        self.assertEqual(s.dense_calls, 1)
        displaced = Mock(side_effect=AssertionError("must not call displaced backend"))
        with patch.object(s, "dense_attention", return_value="kitchen"):
            self.assertEqual(override(displaced, None, None, None, 2, mask=True), "kitchen")
        displaced.assert_not_called()

    def run_sparse(self, s, dtype=torch.bfloat16, **call):
        qkv = [torch.randn(1, 2, 65, 128, dtype=dtype) for _ in range(3)]
        seen = []

        def sparse(q, k, v, **kwargs):
            seen.append(kwargs)
            self.assertEqual(q.dtype, torch.bfloat16)
            self.assertEqual(tuple(q.shape), (1, 65, 2, 128))
            return v
        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "sol_attn_zerocopy", sparse):
            out = sp.make_attention_override(s)(None, *qkv, 2, skip_reshape=True, **call)
        return out, seen

    def test_sparse_dtype_policy_and_output_layout(self):
        out, seen = self.run_sparse(state(), dtype=torch.float16)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (1, 65, 256))
        self.assertEqual(seen[0]["sink_blocks"], [0, 0])
        self.assertEqual(seen[0]["sink_q"], [0, 0])
        self.assertFalse(seen[0]["tail"])
        self.assertEqual(seen[0]["token_aug"], 0)
        self.assertEqual(seen[0]["tau"], 0.0)
        self.assertAlmostEqual(seen[0]["topk_ratio"], 0.2)

    def test_token_aug_threaded_into_sol_kwargs(self):
        _, seen = self.run_sparse(state(token_aug=256))
        self.assertEqual(seen[0]["token_aug"], 256)

    def test_sol_attn_selection_forces_topk_zero(self):
        _, seen = self.run_sparse(state(selection="sol-attn", tau=1.3, keep_ratio=0.2))
        self.assertEqual(seen[0]["topk_ratio"], 0.0)
        self.assertEqual(seen[0]["tau"], 1.3)

    def test_scheduled_decision_values_reach_kernel(self):
        s = state(selection="sla", keep_ratio=0.2)
        to = sp.publish({}, s, sp.StepDecision(False, None, 0.4, 0.0))
        _, seen = self.run_sparse(s, transformer_options=to)
        self.assertAlmostEqual(seen[0]["topk_ratio"], 0.4)

    def test_lerp_schedule_helper(self):
        self.assertAlmostEqual(common.lerp_schedule(0.2, 0.4, 0.5), 0.3)
        self.assertEqual(common.schedule_progress(None, 4), 0.0)
        self.assertEqual(common.schedule_progress(0, 1), 0.0)
        self.assertAlmostEqual(common.schedule_progress(1, 3), 0.5)

    def test_keep_and_tau_schedules(self):
        s = state(selection="sla", keep_ratio=0.2, keep_ratio_end=0.4, dense_steps=frozenset())
        opts = {"sample_sigmas": [1.0, 0.5, 0.0]}
        self.assertAlmostEqual(s.decide(opts | {"sigmas": 1.0}).topk, 0.2)
        self.assertAlmostEqual(s.decide(opts | {"sigmas": 0.5}).topk, 0.4)
        t = state(selection="sol-attn", tau=1.0, tau_end=2.0, dense_steps=frozenset())
        self.assertAlmostEqual(t.decide(opts | {"sigmas": 0.5}).tau, 2.0)

    def test_outside_sigma_window(self):
        self.assertFalse(common.outside_sigma_window(None, 1.0, 0.0))
        self.assertFalse(common.outside_sigma_window(0.5, 1.0, 0.0))
        self.assertTrue(common.outside_sigma_window(1.5, 1.0, 0.0))
        s = state(dense_steps=frozenset(), sigma_start=1.0, sigma_end=0.25)
        schedule = [2.0, 1.0, 0.5, 0.1, 0.0]
        got = [s.decide({"sample_sigmas": schedule, "sigmas": torch.tensor([x])})[:2]
               for x in [2.0, 1.0, 0.5, 0.1]]
        self.assertEqual(got[0], (True, "outside start/end_percent"))
        self.assertEqual([g[0] for g in got[1:3]], [False, False])
        self.assertEqual(got[3], (True, "outside start/end_percent"))

    def test_lerped_keep_is_not_part_of_the_policy(self):
        s = state(selection="sla", keep_ratio=0.2, keep_ratio_end=0.8)
        a = sp.publish({}, s, sp.StepDecision(False, None, 0.2, 1.3))
        b = sp.publish({}, s, sp.StepDecision(False, None, 0.8, 3.0))
        self.assertEqual(a[common.NS]["policy"], b[common.NS]["policy"])

    def test_failed_sparse_op_is_not_retried_until_cleanup(self):
        s = state()
        qkv = [torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        override = sp.make_attention_override(s)
        with patch.object(sp, "_ineligible", return_value=None), patch.object(
                sp, "sol_attn_zerocopy", side_effect=RuntimeError("unsupported sparse op")) as sparse:
            with self.assertLogs("CurveAttn", level="WARNING"):
                self.assertEqual(override(None, *qkv, 2, skip_reshape=True), "dense")
            self.assertEqual(override(None, *qkv, 2, skip_reshape=True), "dense")
            sparse.assert_called_once()
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 2))
        s.cleanup()
        self.assertIsNone(s.failure_reason)
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 0))

    def test_fatal_cuda_errors_do_not_launch_dense_fallback(self):
        s = state()
        qkv = [torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        override = sp.make_attention_override(s)
        for error in (torch.OutOfMemoryError("out of memory"),
                      RuntimeError("CUDA error: illegal memory access")):
            with self.subTest(error=type(error)), \
                 patch.object(sp, "_ineligible", return_value=None), \
                 patch.object(sp, "sol_attn_zerocopy", side_effect=error), \
                 self.assertRaises(type(error)):
                override(None, *qkv, 2, skip_reshape=True)
        s.dense_attention.assert_not_called()

    def test_preparation_failure_restores_ownership_and_stays_sparse(self):
        s = state(prepared_available=True)
        override = sp.make_attention_override(s)
        with patch.object(sp, "_ineligible", return_value=None), patch.object(
                sp, "prepare_sla", side_effect=RuntimeError("selector unavailable")) as prepare, \
             patch.object(sp, "sol_attn_zerocopy", side_effect=lambda q, k, v, **kw: v):
            with self.assertLogs("CurveAttn", level="WARNING"):
                for _ in range(2):
                    containers = [sp.AttentionTensorContainer(
                        torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16)) for _ in range(3)]
                    result = override.container_function(*containers, 2, skip_reshape=True)
                    self.assertEqual(result.shape, (1, 65, 256))
                    self.assertTrue(all(t.tensor is None for t in containers))
            prepare.assert_called_once()
        self.assertEqual((s.sparse_calls, s.dense_calls), (2, 0))
        self.assertFalse(s.prepared_enabled)
        s.cleanup()
        self.assertTrue(s.prepared_enabled)

    def test_failed_preparation_returns_original_container_inputs(self):
        tensors = [torch.zeros(1, 2, 65, 128, dtype=torch.float16) for _ in range(3)]
        containers = [sp.AttentionTensorContainer(tensor) for tensor in tensors]
        with patch.object(sp, "prepare_sla", side_effect=torch.OutOfMemoryError("out of memory")):
            with self.assertRaises(torch.OutOfMemoryError):
                sp._prepare_owned(*containers, 2, True)
        self.assertTrue(all(c.peek() is t for c, t in zip(containers, tensors)))

    def test_dense_backend_override_matches_captured_function_not_qualname(self):
        from comfy.ldm.modules.attention import get_attention_function
        from comfy.model_patcher import ModelPatcher
        patcher = ModelPatcher(torch.nn.Module(), load_device=torch.device("cpu"),
                               offload_device=torch.device("cpu"))
        patcher.set_model_optimized_attention(get_attention_function("pytorch"))
        installed = patcher.model_options["transformer_options"]["optimized_attention_override"]
        self.assertTrue(sp.is_dense_backend_override(installed))
        self.assertFalse(sp.is_dense_backend_override(lambda *a, **k: None))

        def optimized_attention_override(_, *args, **kwargs):
            return None
        optimized_attention_override.__qualname__ = (
            "set_model_optimized_attention.<locals>.optimized_attention_override")
        self.assertFalse(sp.is_dense_backend_override(optimized_attention_override))

    def test_owned_bf16_does_not_clone_an_exclusive_tensor(self):
        fresh = torch.randn(1, 65, 2, 128, dtype=torch.bfloat16)
        self.assertEqual(ps._own(fresh).data_ptr(), fresh.data_ptr())
        packed = torch.randn(1, 65, 3 * 2 * 128, dtype=torch.bfloat16)
        view = packed.chunk(3, dim=-1)[0].view(1, 65, 2, 128)
        self.assertNotEqual(ps._own(view).untyped_storage().data_ptr(),
                            view.untyped_storage().data_ptr())

    def test_validation(self):
        good = dict(selection="sla", keep_percent=20., keep_percent_end=-1., tau=1.3,
                    tau_end=-1., min_tokens=12288, dense_last_steps=0, start_percent=0.,
                    end_percent=1., extra_tokens=0, dense_backend="comfy_kitchen_int8",
                    qkv_mode="native")
        sp.validate_sparse(**good)
        sp.validate_sparse(**(good | dict(selection="vsa", keep_percent=10.)))
        sp.validate_sparse(**(good | dict(selection="vsa", qkv_mode="chunked")))
        for change in [dict(tau=float("nan")), dict(keep_percent=float("inf")),
                       dict(extra_tokens=1), dict(extra_tokens=65),
                       dict(start_percent=.8, end_percent=.2), dict(keep_percent_end=-.5),
                       dict(tau_end=-.5), dict(dense_backend="unknown"),
                       dict(qkv_mode="unknown"), dict(selection="nope"), dict(min_tokens=True)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                sp.validate_sparse(**(good | change))


class PreparedWiringTests(unittest.TestCase):
    def test_sol_attn_module_uses_local_prepared_when_kitchen_has_none(self):
        self.assertTrue(hasattr(sol.sol_prepared, "prequantize_sol_attn"))
        self.assertEqual(sol.sol_prepared.SELECTOR, "owned_bf16")

    def test_check_prepared_available_defers_to_backend(self):
        with patch.object(sol.sol_prepared, "check_available", side_effect=RuntimeError("nope")):
            with self.assertRaisesRegex(RuntimeError, "nope"):
                sol.check_prepared_available()

    def test_check_available_requires_compiled_sol(self):
        with patch.object(ps.ck, "sol_attn_is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "sol_attn"):
                ps.check_available()
        with patch.object(ps.ck, "sol_attn_is_available", return_value=True):
            ps.check_available()

    def test_prequantize_drops_packed_parent_storage(self):
        packed = torch.randn(1, 65, 3 * 2 * 128, dtype=torch.bfloat16)
        owner = weakref.ref(packed)
        q, k, v = [x.view(1, 65, 2, 128) for x in packed.chunk(3, dim=-1)]
        del packed
        with patch.object(ps.ck, "sol_attn_is_available", return_value=True):
            prepared = ps.prequantize_sol_attn(q, k, v, topk_ratio=0.2, tail=False, tau=0.0,
                                               sink_blocks=[0, 0], sink_q=[0, 0],
                                               token_aug=0, scale=None)
        del q, k, v
        self.assertIsNone(owner(), "packed parent still alive")
        self.assertEqual(prepared.selector, "owned_bf16")
        self.assertEqual(prepared.q.dtype, torch.bfloat16)

    def test_consume_calls_kitchen_sol_with_stored_kwargs(self):
        q = torch.randn(1, 65, 2, 128, dtype=torch.bfloat16)
        captured = {}

        def fake_sol(aq, ak, av, **kwargs):
            captured.update(args=(aq, ak, av), kwargs=kwargs)
            return av
        with patch.object(ps.ck, "sol_attn_is_available", return_value=True), \
             patch.object(ps.ck, "sol_attn", fake_sol):
            prepared = ps.prequantize_sol_attn(q, q, q, topk_ratio=0.2, tail=False, tau=0.0,
                                               sink_blocks=[0, 0], sink_q=[0, 0],
                                               token_aug=0, scale=None)
            out = ps.sol_attn_from_prequantized(prepared)
        self.assertIs(out, captured["args"][2])
        self.assertEqual(captured["kwargs"]["topk_ratio"], 0.2)
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
        y = sol.maybe_contiguous(x)
        self.assertNotEqual(y.data_ptr(), x.data_ptr())
        self.assertTrue(sol._sol_layout_ok(y))
        torch.testing.assert_close(y, x, rtol=0, atol=0)

    def test_noncontiguous_last_dim_is_normalized(self):
        x = torch.randn(1, 65, 2, 256)[..., ::2]
        y = sol.maybe_contiguous(x)
        self.assertTrue(sol._sol_layout_ok(y))
        torch.testing.assert_close(y, x, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and sp.ck.sol_attn_is_available(),
                     "native CUDA Sol required")
class NativeOperatorTests(unittest.TestCase):
    def test_prepared_h3_dispatch_releases_float_owner_before_output(self):
        from comfy.ldm.modules.attention import attention_pytorch
        packed = torch.randn(1, 257, 768, dtype=torch.bfloat16, device="cuda")
        owner = weakref.ref(packed)
        tensors = [t.view(1, 257, 2, 128) for t in packed.chunk(3, dim=-1)]
        expected = sp.ck.sol_attn(*tensors, topk_ratio=.2, tail=False).reshape(1, 257, 256)
        containers = [sp.AttentionTensorContainer(t.transpose(1, 2)) for t in tensors]
        del packed, tensors
        s = state(prepared_available=True)
        options = {}
        sp.install_override(s, options)
        native_consume = sp.consume_sla

        def consume(quantized):
            self.assertIsNone(owner(), "float QKV owner retained while allocating output")
            return native_consume(quantized)
        with patch.object(sp, "consume_sla", consume):
            actual = attention_pytorch(*containers, 2, skip_reshape=True,
                                       transformer_options=options)
        self.assertEqual((s.sparse_calls, s.dense_calls), (1, 0))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_strided_attention_matches_contiguous_native_sla(self):
        generator = torch.Generator(device="cuda").manual_seed(7321)
        for tokens, sinks, query_sinks in [(65, [0, 0], [0, 0]), (257, [0, 2], [1, 2])]:
            packed = torch.randn(1, tokens, 768, dtype=torch.bfloat16, device="cuda",
                                 generator=generator)
            q, k, v = [x.view(1, tokens, 2, 128) for x in packed.chunk(3, dim=-1)]
            options = dict(topk_ratio=0.2, tail=False, sink_blocks=sinks, sink_q=query_sinks)
            actual = sol.sol_attn_zerocopy(q, k, v, **options)
            expected = sp.ck.sol_attn(q.contiguous(), k.contiguous(), v.contiguous(),
                                      token_aug=0, tau=0.0, **options)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class OwnershipRegressionTests(unittest.TestCase):
    def test_container_sparse_failure_can_run_dense(self):
        tensors = [torch.ones(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        containers = [sp.AttentionTensorContainer(t) for t in tensors]

        def dense(q, k, v, heads, **kw):
            values = [c.take() for c in (q, k, v)]
            self.assertTrue(all(a is b for a, b in zip(values, tensors)))
            return values[2].transpose(1, 2).reshape(1, 65, 256)
        s = state(dense_attention=dense)
        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "sol_attn_zerocopy", side_effect=RuntimeError("unsupported op")):
            out = sp.make_attention_override(s).container_function(*containers, 2, skip_reshape=True)
        self.assertEqual(tuple(out.shape), (1, 65, 256))
        self.assertTrue(all(c.tensor is None for c in containers))
        self.assertEqual((s.sparse_calls, s.dense_calls), (0, 1))

    def test_prepared_consume_failure_does_not_run_dense(self):
        s = state(prepared_available=True)
        containers = [sp.AttentionTensorContainer(
            torch.ones(1, 2, 65, 128, dtype=torch.bfloat16)) for _ in range(3)]
        carrier = types.SimpleNamespace(selector="test")
        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "prepare_sla", return_value=carrier), \
             patch.object(sp, "consume_sla", side_effect=RuntimeError("consume failed")), \
             self.assertRaisesRegex(RuntimeError, "consume failed"):
            sp.make_attention_override(s).container_function(*containers, 2, skip_reshape=True)
        s.dense_attention.assert_not_called()
        self.assertTrue(all(c.tensor is None for c in containers))

    def test_native_never_prepares(self):
        containers = [sp.AttentionTensorContainer(
            torch.randn(1, 2, 65, 128, dtype=torch.bfloat16)) for _ in range(3)]
        s = state(qkv_mode="native")
        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "prepare_sla") as prepare, \
             patch.object(sp, "sol_attn_zerocopy", side_effect=lambda q, k, v, **kw: v):
            out = sp.make_attention_override(s).container_function(*containers, 2, skip_reshape=True)
        prepare.assert_not_called()
        self.assertEqual(tuple(out.shape), (1, 65, 256))
        self.assertEqual(s.sparse_calls, 1)


class ApplyTests(unittest.TestCase):
    def test_dense_backend_choice(self):
        model = h3_patcher()
        with patch.object(sp.ck, "sol_attn_is_available", return_value=True), \
             patch.object(sp, "get_attention_function", return_value=Mock()) as getter:
            sp.apply_sparse(model, dense_backend="pytorch")
        self.assertEqual(getter.call_args.args[:2], ("pytorch", None))
        with patch.object(sp.ck, "sol_attn_is_available", return_value=True), \
             patch.object(sp, "get_attention_function", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "pytorch"):
                sp.apply_sparse(model, dense_backend="pytorch")

    def test_requires_sol_attn(self):
        with kitchen(available=False), self.assertRaisesRegex(RuntimeError, "sol_attn"):
            sp.apply_sparse(h3_patcher())

    def test_rejects_non_h3(self):
        from comfy.model_patcher import ModelPatcher
        base = torch.nn.Module()
        base.diffusion_model = torch.nn.Module()
        model = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        with kitchen(), self.assertRaisesRegex(ValueError, "MiniMax H3"):
            sp.apply_sparse(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
