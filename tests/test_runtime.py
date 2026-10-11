import unittest
from uuid import UUID

from comfy.patcher_extension import WrapperExecutor
from curvattn_test_package.cache import (
    RunScope, cache_input_flags, cache_metadata, forward_options, make_cache_step_wrapper,
    CacheState, MANUAL_DEFAULTS,
)
from curvattn_test_package.common import NS, exact_step, schedule_tuple


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

    def test_masks_bypass(self):
        for key in ("denoise_mask", "audio_denoise_mask"):
            self.assertFalse(self.meta(kwargs={key: object()})["eligible"])

    def test_references_bypass(self):
        self.assertFalse(self.meta(payload={"refs": [object()]})["eligible"])
        self.assertFalse(self.meta(payload={"keyframes": [object()]})["eligible"])
        self.assertTrue(self.meta(payload={"refs": []})["eligible"])

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
        self.assertTrue(seen["transformer_options"][NS]["cache_inputs"]["denoise_mask"])
        self.assertEqual(original, {"prefetch_dynamic_vbars": True})

    def test_cache_step_wrapper_rejects_multigpu(self):
        state = CacheState(dict(MANUAL_DEFAULTS), max_mib=1)
        with self.assertRaisesRegex(ValueError, "multigpu"):
            make_cache_step_wrapper(state)(lambda *a, **k: 0, None, None, None,
                                           {"multigpu_thread_device": 1}, None)
