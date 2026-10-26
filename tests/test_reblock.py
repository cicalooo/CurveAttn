"""Reblock selection: plan, routing, reweighted tail, override wiring. No GPU."""
import types
import unittest
from unittest.mock import patch

import torch

from comfy.ldm.minimax.model import PackedLayout
from curvattn_test_package import blocks
from curvattn_test_package import cache as cc
from curvattn_test_package import reblock as rb
from curvattn_test_package import sparse as sp
from helpers import h3_patcher, kitchen
from helpers import sparse_state as state

B = rb.BLOCK


def _layout(text=37, video_blocks=6, extra=23):
    start = text
    end = start + video_blocks * B + extra
    return types.SimpleNamespace(seq_len=end, segments=((0, start, "text"), (start, end, "video")))


def _qkv(t, h=2, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, t, h, d, generator=g) for _ in range(3)]


def _plan(layout, q, k, updates=2):
    return rb.build_reblock_plan(q, k, rb.video_span(layout, layout.seq_len), updates=updates)


def _dense(q, k, v, scale):
    qh, kh, vh = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    p = torch.softmax(qh @ kh.transpose(-1, -2) * scale, -1)
    return (p @ vh).permute(0, 2, 1, 3)


class ReblockPlanTests(unittest.TestCase):
    def test_permutation_is_invertible_per_head(self):
        layout = _layout()
        q, k, v = _qkv(layout.seq_len, h=3)
        plan = _plan(layout, q, k)
        src, inv = plan["src"], plan["inv"]
        self.assertEqual(plan["n"] % B, 0)
        self.assertEqual(int(plan["block_len"].sum()), layout.seq_len)
        for h in range(3):
            live = src[0, h][src[0, h] >= 0]
            self.assertEqual(sorted(live.tolist()), list(range(layout.seq_len)))
            torch.testing.assert_close(src[0, h, inv[0, h]], torch.arange(layout.seq_len))
        # conditioning rows keep their place at the front of every head
        torch.testing.assert_close(src[0, :, :37], torch.arange(37).expand(3, 37))
        gathered = rb.gather_heads(q, k, v, src)
        self.assertTrue(torch.all(gathered[0][:, ~plan["live"]] == 0))
        torch.testing.assert_close(rb.scatter_heads(gathered[2], inv), v)
        fast = rb.gather_heads(q, k, v, src, plan["dead"])
        for x, y in zip(fast, gathered):
            torch.testing.assert_close(x, y)

    def test_step_timing_logs_when_step_changes_and_at_report(self):
        s = state(selection="reblock", verbose=True)
        with self.assertLogs("CurveAttn", level="INFO") as logs:
            s.time_reblock(0, 1.0, 2.0, "sol-pooled")
            s.time_reblock(0, 1.0, 2.0, "sol-pooled")
            s.time_reblock(1, 3.0, 4.0, "sol-pooled")
            s.report()
        steps = [line for line in logs.output if "reblock step" in line]
        self.assertEqual(len(steps), 2)
        self.assertIn("step 0: 2 calls plan 2 ms attention 4 ms path sol-pooled", steps[0])

    def test_spare_rows_go_to_exact_prefix_and_video_fills_full_blocks(self):
        layout = _layout(text=37, video_blocks=6, extra=23)
        q, k, _ = _qkv(layout.seq_len)
        plan = _plan(layout, q, k)
        self.assertEqual(plan["n_prefix"], 1)          # 37 text + 23 spare = 60 rows
        self.assertEqual(int(plan["block_len"][0]), 60)
        self.assertEqual(plan["n_video_blocks"], 6)
        self.assertTrue(torch.all(plan["src"][..., B:] >= 37))

    def test_clusters_separate_distinct_key_groups(self):
        layout = _layout(text=0, video_blocks=4, extra=0)
        t, d = layout.seq_len, 8
        centers = torch.eye(d)[:4] * 5
        label = torch.arange(t) % 4                       # interleaved groups
        k = (centers[label] + 0.01 * torch.randn(t, d)).view(1, t, 1, d)
        q = torch.randn(1, t, 1, d)
        plan = _plan(layout, q, k)
        order = plan["src"][0, 0]
        for blk in range(4):
            rows = order[blk * B:(blk + 1) * B]
            self.assertEqual(len(set(label[rows].tolist())), 1)

    def test_short_or_missing_video_is_unplanned(self):
        self.assertIsNone(rb.video_span(None, 100))
        self.assertIsNone(rb.video_span(_layout(video_blocks=1, extra=10), 37 + 74))
        bad = types.SimpleNamespace(seq_len=300, segments=((0, 300, "audio"),))
        self.assertIsNone(rb.video_span(bad, 300))


class ReblockRoutingTests(unittest.TestCase):
    def test_topk_picks_highest_scored_blocks_plus_band(self):
        b, h, d, nv, n_prefix = 1, 1, 4, 8, 1
        n = (n_prefix + nv) * B
        q = torch.zeros(b, n, h, d)
        q[..., 0] = 1.0
        k_bar = torch.zeros(b, h, nv, d)
        k_bar[0, 0, :, 0] = torch.arange(nv, dtype=torch.float32)   # block 7 scores highest
        mass = torch.zeros(b, h, nv)
        sel = rb.select_blocks(q, k_bar, mass, n_prefix, 0.25, 1.0)  # keep 2
        self.assertTrue(torch.all(sel[0, 0, :n_prefix]))            # dense prefix rows
        for qi in range(n_prefix, n_prefix + nv):
            want = {6, 7} | {j for j in range(nv) if abs(qi - (j + n_prefix)) <= 1}
            self.assertEqual(set(torch.nonzero(sel[0, 0, qi]).flatten().tolist()), want)

    def test_mass_correction_shifts_routing(self):
        q = torch.zeros(1, 4 * B, 1, 2)
        k_bar = torch.zeros(1, 1, 3, 2)
        mass = torch.tensor([[[0.0, 5.0, 0.0]]])
        sel = rb.select_blocks(q, k_bar, mass, 1, 1 / 3, 1.0)
        self.assertTrue(bool(sel[0, 0, 3, 1]))


class ReweightTests(unittest.TestCase):
    def _setup(self, uniform_q=False, ratio=0.25, seed=1):
        layout = _layout(text=37, video_blocks=8, extra=23)
        q, k, v = _qkv(layout.seq_len, h=2, d=16, seed=seed)
        if uniform_q:
            q = q[:, :1].expand_as(q).contiguous()
        plan = _plan(layout, q, k)
        qp, kp, vp = rb.gather_heads(q, k, v, plan["src"])
        scale = 16 ** -0.5
        k_bar, v_bar, mass = rb.block_summaries(qp, kp, vp, plan["n_prefix"], scale)
        sel = rb.select_blocks(qp, k_bar, mass, plan["n_prefix"], ratio, scale)
        return layout, (q, k, v), plan, (qp, kp, vp), scale, (k_bar, v_bar, mass, sel)

    def test_summary_is_exact_when_every_query_equals_the_anchor(self):
        _, (q, k, v), plan, (qp, kp, vp), scale, (k_bar, v_bar, mass, sel) = self._setup(True)
        out = rb.reweighted_reference(qp, kp, vp, plan["live"], plan["n_prefix"], sel,
                                      k_bar, v_bar, mass, scale)
        got = rb.scatter_heads(out, plan["inv"])
        torch.testing.assert_close(got, _dense(q, k, v, scale), atol=1e-5, rtol=1e-5)

    def test_mass_is_log_effective_size(self):
        qp = torch.zeros(1, 3 * B, 1, 4)
        kp = torch.randn(1, 3 * B, 1, 4)
        _, _, mass = rb.block_summaries(qp, kp, kp, 1, 1.0)
        torch.testing.assert_close(mass, torch.full_like(mass, float(torch.log(torch.tensor(64.)))))

    def test_reweighted_beats_dropping_the_tail(self):
        _, (q, k, v), plan, (qp, kp, vp), scale, (k_bar, v_bar, mass, sel) = self._setup()
        ref = _dense(q, k, v, scale)
        tail = rb.scatter_heads(rb.reweighted_reference(
            qp, kp, vp, plan["live"], plan["n_prefix"], sel, k_bar, v_bar, mass, scale),
            plan["inv"])
        none = rb.scatter_heads(rb.reweighted_reference(
            qp, kp, vp, plan["live"], plan["n_prefix"], sel, k_bar, v_bar,
            torch.full_like(mass, float("-inf")), scale), plan["inv"])
        self.assertLess(float((tail - ref).norm()), float((none - ref).norm()))

    def test_conditioning_rows_are_dense(self):
        layout, (q, k, v), plan, (qp, kp, vp), scale, (k_bar, v_bar, mass, sel) = \
            self._setup(ratio=0.05)
        got = rb.scatter_heads(rb.reweighted_reference(
            qp, kp, vp, plan["live"], plan["n_prefix"], sel, k_bar, v_bar, mass, scale),
            plan["inv"])
        torch.testing.assert_close(got[:, :37], _dense(q, k, v, scale)[:, :37],
                                   atol=1e-5, rtol=1e-5)

    def test_flex_matches_reference(self):
        _, _, plan, (qp, kp, vp), scale, (k_bar, v_bar, mass, sel) = self._setup()
        ref = rb.reweighted_reference(qp, kp, vp, plan["live"], plan["n_prefix"], sel,
                                      k_bar, v_bar, mass, scale)
        try:
            got = rb.reweighted_flex(qp, kp, vp, plan["live"], plan["block_len"],
                                     plan["n_prefix"], sel, k_bar, v_bar, mass, scale)
        except Exception as exc:  # eager flex unsupported on this torch build
            self.skipTest(f"flex_attention unavailable on CPU: {exc}")
        live = plan["live"]
        torch.testing.assert_close(got[:, live], ref[:, live], atol=1e-4, rtol=1e-4)


class ReblockOverrideTests(unittest.TestCase):
    def _call(self, s, layout, kernel=None, flex=None):
        qkv = [t.permute(0, 2, 1, 3).contiguous() for t in _qkv(layout.seq_len, h=2, d=128)]
        patches = [patch.object(sp, "_ineligible", return_value=None)]
        if kernel is not None:
            patches.append(patch.object(sp, "sol_attn_zerocopy", kernel))
        if flex is not None:
            patches.append(patch.object(sp, "reweighted_attention", flex))
        for p in patches:
            p.start()
        try:
            return sp.make_attention_override(s)(
                None, *qkv, 2, skip_reshape=True,
                transformer_options={"minimax_h3_layout": layout})
        finally:
            for p in patches:
                p.stop()

    def test_pooled_uses_sol_attn_with_prefix_sinks(self):
        layout = _layout()
        s = state(selection="reblock", keep_ratio=0.2, reblock_summary="pooled")
        seen = {}

        def kernel(q, k, v, **kwargs):
            seen.update(kwargs, shape=tuple(q.shape))
            return q.clone()

        out = self._call(s, layout, kernel=kernel)
        self.assertEqual(seen["sink_blocks"], [0, 1])
        self.assertEqual(seen["sink_q"], [0, 1])
        self.assertTrue(seen["tail"])
        self.assertEqual(seen["shape"][1] % B, 0)
        self.assertEqual(tuple(out.shape), (1, layout.seq_len, 256))
        self.assertEqual(s.sparse_calls, 1)
        self.assertEqual(s.reblock_paths, {"pooled": 1})

    def test_default_tail_is_pooled(self):
        self.assertEqual(state(selection="reblock").reblock_summary, "pooled")

    def test_reweighted_failure_falls_back_to_pooled(self):
        layout = _layout()
        s = state(selection="reblock", keep_ratio=0.2, reblock_summary="reweighted")
        calls = []

        def broken(*args, **kwargs):
            raise RuntimeError("no triton")

        def kernel(q, k, v, **kwargs):
            calls.append(kwargs["tail"])
            return q.clone()

        self._call(s, layout, kernel=kernel, flex=broken)
        self._call(s, layout, kernel=kernel, flex=broken)
        self.assertEqual(calls, [True, True])
        self.assertIsNotNone(s.reweight_failed)
        self.assertIsNone(s.failure_reason)
        self.assertEqual(s.reblock_paths, {"pooled": 2})

    def test_output_restores_packed_rows(self):
        layout = _layout()
        s = state(selection="reblock", keep_ratio=0.2, reblock_summary="off")
        out = self._call(s, layout, kernel=lambda q, k, v, **kw: q.clone())
        qkv = _qkv(layout.seq_len, h=2, d=128)
        torch.testing.assert_close(
            out, qkv[0].to(torch.bfloat16).to(out.dtype).reshape(1, layout.seq_len, 256))

    def test_missing_layout_stays_dense(self):
        s = state(selection="reblock")
        qkv = [torch.randn(1, 2, 600, 128) for _ in range(3)]
        with patch.object(sp, "_ineligible", return_value=None):
            out = sp.make_attention_override(s)(None, *qkv, 2, skip_reshape=True,
                                                transformer_options={})
        self.assertEqual(out, "dense")
        self.assertIn("unavailable for reblock", s.last_reason)

    def test_refresh_reuses_layer_plan_then_replans(self):
        layout = _layout()
        q, k, _ = _qkv(layout.seq_len)
        span = rb.video_span(layout, layout.seq_len)
        s = state(selection="reblock", reblock_refresh=2)
        opts = {"block_index": 3, "cond_or_uncond": [0]}
        d = lambda step: sp.StepDecision(False, None, 0.2, 1.3, step)
        with patch.object(sp, "build_reblock_plan", wraps=rb.build_reblock_plan) as build:
            p0 = sp._reblock_plan(s, q, k, span, d(0), opts)
            self.assertIs(sp._reblock_plan(s, q, k, span, d(1), opts), p0)
            sp._reblock_plan(s, q, k, span, d(2), opts)
            sp._reblock_plan(s, q, k, span, d(2), dict(opts, block_index=4))
            short = _layout(extra=22)
            sp._reblock_plan(s, q[:, :-1], k[:, :-1],
                             rb.video_span(short, short.seq_len), d(2), opts)
            self.assertEqual(build.call_count, 4)
            s.reset()
            sp._reblock_plan(s, q, k, span, d(2), opts)
            self.assertEqual(build.call_count, 5)

    def test_refresh_one_never_caches(self):
        layout = _layout()
        q, k, _ = _qkv(layout.seq_len)
        s = state(selection="reblock", reblock_refresh=1)
        sp._reblock_plan(s, q, k, rb.video_span(layout, layout.seq_len),
                         sp.StepDecision(False, None, 0.2, 1.3, 0), {"block_index": 0})
        self.assertEqual(s.reblock_plans, {})

    def test_validation(self):
        base = dict(selection="reblock", keep_percent=20, keep_percent_end=-1, tau=1.3,
                    tau_end=-1, min_tokens=0, dense_last_steps=0, start_percent=0,
                    end_percent=1, extra_tokens=0, dense_backend="pytorch", qkv_mode="native")
        sp.validate_sparse(**base)
        with self.assertRaises(ValueError):
            sp.validate_sparse(**base, reblock_summary="mean")
        with self.assertRaises(ValueError):
            sp.validate_sparse(**base, reblock_updates=9)
        with self.assertRaises(ValueError):
            sp.validate_sparse(**base, reblock_refresh=0)


class ReblockCompositionTests(unittest.TestCase):
    def test_whole_cache_accepts_reblock_with_non_native_qkv(self):
        for mode in ("owned_bf16", "chunked"):
            with self.subTest(qkv_mode=mode), kitchen():
                whole = cc.apply_cache(h3_patcher(), span="whole")
                model = sp.apply_sparse(whole, selection="reblock", qkv_mode=mode)
                state = next(s for s in sp.curve_states(model.model_options["transformer_options"])
                             if isinstance(s, sp.SparseState))
                self.assertEqual(state.qkv_mode, "native")
                self.assertEqual(blocks.current_cache(model.model_options).span, "whole")
                with self.assertRaisesRegex(ValueError, "requires native qkv_mode"):
                    sp.apply_sparse(whole, qkv_mode=mode)


if __name__ == "__main__":
    unittest.main()
