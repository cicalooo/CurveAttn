import unittest
from uuid import UUID

from comfy.patcher_extension import WrapperExecutor
import types

import torch

from curvattn_test_package.cache import (
    RunScope, cache_input_flags, cache_metadata, forward_options, make_cache_step_wrapper,
    CacheState, MANUAL_DEFAULTS, resolve_masks, timestep_regime, timestep_values,
)
from curvattn_test_package.common import NS, exact_step, schedule_position, schedule_tuple


class RuntimeTests(unittest.TestCase):
    schedule = (1., .875, .75, .625, .5, .375, .25, .125, 0.)

    def meta(self, options=None, payload=None, kwargs=None, **changes):
        opts = dict(sigmas=.625, cond_or_uncond=[0],
                    uuids=[UUID("00000000-0000-0000-0000-000000000001")])
        opts.update(options or {})
        settings = dict(schedule=self.schedule, policy=(.2,))
        settings.update(changes)
        return cache_metadata(opts, cache_input_flags(payload, kwargs or {}), **settings)

    def test_exact_step(self):
        self.assertEqual(exact_step(self.schedule, .625), (3, 8))
        self.assertEqual(exact_step(self.schedule, .61), (None, 8))
        self.assertEqual(exact_step(self.schedule, float("nan")), (None, 8))
        self.assertEqual(schedule_tuple([1., .5, .5, 0.]), ())
        self.assertEqual(schedule_tuple([1., .7, .8, 0.]), ())

    def test_missing_identity(self):
        self.assertTrue(self.meta()["eligible"])
        self.assertFalse(self.meta(options={"uuids": []})["eligible"])

    def test_unaligned_masks_bypass(self):
        for key in ("denoise_mask", "audio_denoise_mask"):
            meta = self.meta(kwargs={key: object()})
            self.assertFalse(meta["eligible"])
            self.assertIn("denoise mask unaligned", meta["reason"])

    def test_aligned_denoise_mask_is_eligible(self):
        from comfy.ldm.minimax.model import PackedLayout
        layout = PackedLayout(2, 2, 4, 4, 2)
        video = torch.zeros(1, 1, 2, 4, 4)
        video[..., :2] = 1
        audio = torch.zeros(1, 1, 2, 2)
        audio[..., 1] = 1
        meta = self.meta(options={"minimax_h3_layout": layout},
                         kwargs={"denoise_mask": video, "audio_denoise_mask": audio})
        self.assertTrue(meta["eligible"], meta["reason"])
        self.assertEqual(meta["drift_weights"]["video"].numel(), 8)
        self.assertEqual(meta["drift_weights"]["audio"].numel(), 4)
        self.assertGreater(float(meta["drift_weights"]["video"].max()), 0.5)
        self.assertLess(float(meta["drift_weights"]["video"].min()), 0.5)

    def test_uniform_full_mask_matches_no_mask(self):
        from comfy.ldm.minimax.model import PackedLayout
        layout = PackedLayout(2, 2, 4, 4, 2)
        ones = torch.ones(1, 1, 2, 4, 4)
        masked = self.meta(options={"minimax_h3_layout": layout}, kwargs={"denoise_mask": ones})
        plain = self.meta(options={"minimax_h3_layout": layout})
        self.assertTrue(masked["eligible"], masked["reason"])
        self.assertEqual(masked["drift_weights"], {})
        state = CacheState(dict(MANUAL_DEFAULTS), max_mib=1)
        opts = {"minimax_h3_layout": layout, NS: {"cache_inputs": cache_input_flags(None, {})}}
        full = dict(opts)
        full[NS] = {"cache_inputs": cache_input_flags(None, {"denoise_mask": ones})}
        self.assertEqual(state.policy(opts), state.policy(full))
        self.assertEqual(plain["drift_weights"], {})

    def test_mask_pattern_joins_the_policy(self):
        from comfy.ldm.minimax.model import PackedLayout
        layout = PackedLayout(2, 2, 4, 4, 2)
        left = torch.zeros(1, 1, 2, 4, 4)
        left[..., :2] = 1
        right = torch.zeros(1, 1, 2, 4, 4)
        right[..., 2:] = 1
        zeros = torch.zeros(1, 1, 2, 4, 4)
        state = CacheState(dict(MANUAL_DEFAULTS), max_mib=1)

        def opts(mask):
            return {"minimax_h3_layout": layout, NS: {"cache_inputs": cache_input_flags(
                None, {"denoise_mask": mask})}}

        self.assertEqual(state.policy(opts(left)), state.policy(opts(left.clone())))
        self.assertNotEqual(state.policy(opts(left)), state.policy(opts(right)))
        self.assertNotEqual(state.policy(opts(left)), state.policy(opts(None)))
        self.assertNotEqual(state.policy(opts(zeros)), state.policy(opts(None)))
        bad = resolve_masks(opts(object()), opts(object())[NS]["cache_inputs"])
        self.assertEqual(bad.problem, "denoise mask unaligned")
        mismatch = torch.zeros(1, 1, 3, 4, 4)
        meta = self.meta(options={"minimax_h3_layout": layout}, kwargs={"denoise_mask": mismatch})
        self.assertIn("denoise mask unaligned", meta["reason"])
        nan = torch.zeros(1, 1, 2, 4, 4)
        nan[0, 0, 0, 0, 0] = float("nan")
        meta = self.meta(options={"minimax_h3_layout": layout}, kwargs={"denoise_mask": nan})
        self.assertIn("denoise mask unaligned", meta["reason"])

    def test_mask_aligns_to_target_video_not_keyframes(self):
        from comfy.ldm.minimax.model import PackedLayout
        keyframes = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 4)}]
        layout = PackedLayout(2, 2, 4, 4, 2, keyframes=keyframes)
        video = torch.zeros(1, 1, 2, 4, 4)
        video[..., :2] = 1
        meta = self.meta(options={"minimax_h3_layout": layout}, kwargs={"denoise_mask": video})
        self.assertTrue(meta["eligible"], meta["reason"])
        video_len = next(b - a for a, b, k in layout.segments if k == "video")
        self.assertEqual(meta["drift_weights"]["video"].numel(), video_len)
        self.assertTrue(any(k == "cond" for *_, k in layout.segments))

    def test_changed_mask_refreshes_instead_of_hitting(self):
        from comfy.ldm.minimax.model import PackedLayout
        from curvattn_test_package.residual_cache import MiddleCache
        layout = PackedLayout(2, 2, 4, 4, 2)
        state = CacheState(dict(MANUAL_DEFAULTS, min_steps=4, start_percent=0.0, end_percent=1.0),
                           max_mib=64)
        run = types.SimpleNamespace(schedule=self.schedule)
        left = torch.zeros(1, 1, 2, 4, 4)
        left[..., :2] = 1

        def options(mask):
            return {"sigmas": .5, "cond_or_uncond": [0],
                    "uuids": [UUID("00000000-0000-0000-0000-000000000001")],
                    "minimax_h3_layout": layout,
                    NS: {"cache_inputs": cache_input_flags(None, {"denoise_mask": mask})}}

        meta_a = state.metadata(options(left), run)
        meta_b = dict(state.metadata(options(left.flip(-1)), run), step=meta_a["step"] + 1)
        self.assertTrue(meta_a["eligible"], meta_a["reason"])
        self.assertTrue(meta_b["eligible"], meta_b["reason"])
        self.assertNotEqual(meta_a["policy"], meta_b["policy"])
        cache = MiddleCache(threshold=.5, max_hits=1, max_bytes=10 ** 9, min_steps=4,
                            start_percent=0.0, end_percent=1.0)
        x = torch.ones(layout.seq_len, 4)

        def run_step(meta):
            calls = []
            img = x.clone()
            for index in range(4):
                def original(args, extra, index=index):
                    calls.append(index)
                    args["img"].add_(1)
                    return {"img": args["img"]}
                img = cache.call(index, 4, dict(img=img, layout=layout), {}, meta, original)["img"]
            return calls

        with torch.no_grad():
            run_step(meta_a)
            calls = run_step(meta_b)
        self.assertEqual((cache.hits, cache.refreshes, calls), (0, 2, [0, 1, 2, 3]))

    def test_references_and_keyframes_are_eligible(self):
        for payload in ({"refs": [object()]}, {"keyframes": [object()]},
                        {"refs": [object()], "keyframes": [object()]}, {"refs": []}):
            self.assertTrue(self.meta(payload=payload)["eligible"], payload)

    def test_schedule_position(self):
        sch = self.schedule
        self.assertEqual(schedule_position(sch, .625), (3, 8, False))
        self.assertEqual(schedule_position(sch, torch.tensor([.625])), (3, 8, False))
        self.assertEqual(schedule_position(sch, .6), (3, 8, True))
        self.assertEqual(schedule_position(sch, .99), (0, 8, True))
        self.assertEqual(schedule_position(sch, .01), (7, 8, True))
        for bad in (1.5, 0., -.1, float("nan"), float("inf"), torch.tensor([.6, .5]), "x"):
            self.assertIsNone(schedule_position(sch, bad)[0], bad)
        # two schedule points within tolerance of one sigma: ambiguous
        self.assertIsNone(schedule_position((1., .5000001, .5, 0.), .5)[0])
        self.assertEqual(exact_step(sch, .6), (None, 8))

    def test_substep_metadata(self):
        meta = self.meta(options={"sigmas": .6})
        self.assertTrue(meta["eligible"])
        self.assertEqual((meta["step"], meta["substep"]), (3, True))
        self.assertFalse(self.meta()["substep"])
        self.assertIn("sigma did not match", self.meta(options={"sigmas": 2.})["reason"])

    def test_timestep_values_mirror_the_model(self):
        model = types.SimpleNamespace(sigma_shift_video=12.0, sigma_shift_audio=3.0)
        t_v, t_a, vis, aud = timestep_values(torch.tensor([750.]), {}, {}, model)
        self.assertAlmostEqual(t_v, .25, places=6)
        self.assertGreater(t_a, t_v)
        self.assertEqual((vis, aud), (.999, 1.0))
        self.assertEqual(timestep_values(torch.tensor([750.]), {"visual_cond_noise_aug": .5},
                                         {}, model)[2], .5)
        self.assertIsNone(timestep_values(None, {}, {}))

    def test_timestep_regime(self):
        segs = ((0, 1, "text"), (1, 2, "ref_img"), (2, 3, "cond_audio"), (3, 4, "audio"),
                (4, 5, "video"))
        pinned = timestep_regime((.25, .6, .999, 1.0), segs)
        self.assertEqual(pinned, (4, ("cond_audio", True), ("ref_img", True)))
        self.assertEqual(timestep_regime((.4, .7, .999, 1.0), segs), pinned)
        # ref_img starts tracking t_v once t_v passes its aug pin
        self.assertNotEqual(timestep_regime((.9995, .9999, .999, 1.0), segs), pinned)
        self.assertEqual(timestep_regime((.25, .6, .999, 1.0), segs[3:]), (2,))
        self.assertIsNone(timestep_regime(None, segs))

    def test_missing_cache_wrapper_bypasses(self):
        meta = cache_metadata(dict(sigmas=.625, cond_or_uncond=[0], uuids=[
            UUID("00000000-0000-0000-0000-000000000001")]), None,
            schedule=self.schedule, policy=())
        self.assertIn("cache step wrapper did not run", meta["reason"])

    def test_multigpu_bypasses(self):
        self.assertFalse(self.meta(options={"multigpu_thread_device": object()})["eligible"])

    def test_branch_batch_bypasses(self):
        meta = self.meta(options={"cond_or_uncond": [0, 1]})
        self.assertFalse(meta["eligible"])
        self.assertIn("cond_or_uncond length 2", meta["reason"])

    def test_dense_only_when_sparse_ran_and_nobody_owns(self):
        self.assertFalse(self.meta()["dense"])
        self.assertTrue(self.meta(options={NS: {"owner": None}})["dense"])
        self.assertFalse(self.meta(options={NS: {"owner": "sparse1"}})["dense"])

    def test_min_steps(self):
        schedule = (1., .75, .5, .25, 0.)
        self.assertFalse(self.meta(options={"sigmas": .5}, schedule=schedule, min_steps=6)["eligible"])
        self.assertTrue(self.meta(options={"sigmas": .5}, schedule=schedule, min_steps=4)["eligible"])

    def test_scope_clears_on_cancel(self):
        scope = RunScope(.01, 1, 1)
        runs = []

        def cancelled(*args, **kwargs):
            runs.append(scope.current())
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            scope.outer(cancelled, None, None, None, self.schedule)
        self.assertIsNone(scope.current())
        self.assertEqual(runs[0].cache.entries, {})
        self.assertEqual(scope.outer(lambda *a, **k: runs.append(scope.current()) or 7,
                                     None, None, None, self.schedule), 7)
        self.assertIsNot(runs[0], runs[1])

    def test_executor_chain(self):
        scope = RunScope(.01, 1, 1)
        seen = []

        def second(executor, *args, **kwargs):
            seen.append("second")
            return executor(*args, **kwargs)

        def original(*args, **kwargs):
            seen.append("original")
            return 9

        executor = WrapperExecutor.new_executor(original, [scope.outer, second])
        self.assertEqual(executor.execute(None, None, None, self.schedule), 9)
        self.assertEqual(seen, ["second", "original"])

    def test_scope_passes_mode(self):
        scope = RunScope(.05, 1, 1, span="whole", mode="first_block", min_steps=4)

        def sample(*a, **k):
            self.assertEqual(scope.current().cache.mode, "first_block")
            self.assertEqual(scope.current().cache.span, "whole")
            return 1
        self.assertEqual(scope.outer(sample, None, None, None, [1., .5, 0.]), 1)

    def test_whole_disables_prefetch_without_mutating_input(self):
        original = {"prefetch_dynamic_vbars": True, "other": 7}
        whole = forward_options(original, "whole")
        self.assertFalse(whole["prefetch_dynamic_vbars"])
        self.assertTrue(original["prefetch_dynamic_vbars"])
        self.assertTrue(forward_options(original, "middle")["prefetch_dynamic_vbars"])

    def test_cache_step_wrapper_publishes_inputs_and_keeps_kwargs(self):
        state = CacheState(dict(MANUAL_DEFAULTS, span="whole"), max_mib=1)
        video, audio = object(), object()
        seen = {}
        original = {"prefetch_dynamic_vbars": True}
        make_cache_step_wrapper(state)(lambda *a, **k: seen.update(k), None, None, None,
                                       original, None, denoise_mask=video,
                                       audio_denoise_mask=audio)
        self.assertIs(seen["denoise_mask"], video)
        self.assertIs(seen["audio_denoise_mask"], audio)
        self.assertFalse(seen["transformer_options"]["prefetch_dynamic_vbars"])
        published = seen["transformer_options"][NS]["cache_inputs"]
        self.assertIs(published["denoise_mask"], video)
        self.assertIs(published["audio_denoise_mask"], audio)
        self.assertEqual(original, {"prefetch_dynamic_vbars": True})

    def test_cache_step_wrapper_rejects_multigpu(self):
        state = CacheState(dict(MANUAL_DEFAULTS), max_mib=1)
        with self.assertRaisesRegex(ValueError, "multigpu"):
            make_cache_step_wrapper(state)(lambda *a, **k: 0, None, None, None,
                                           {"multigpu_thread_device": 1}, None)
