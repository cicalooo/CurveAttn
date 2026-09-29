import unittest
from uuid import UUID
from curve_test_package.runtime import RunScope, cache_metadata, exact_step, schedule_tuple


class RuntimeTests(unittest.TestCase):
    schedule = (1., .875, .75, .625, .5, .375, .25, .125, 0.)

    def meta(self, options=None, payload=None, kwargs=None, **changes):
        opts = dict(sigmas=.625, cond_or_uncond=[0],
                    uuids=[UUID("00000000-0000-0000-0000-000000000001")])
        opts.update(options or {})
        settings = dict(schedule=self.schedule, enabled=True,
                        protect_audio=False, policy=(.2,))
        settings.update(changes)
        return cache_metadata(opts, payload, kwargs or {}, **settings)

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

    def test_audio_protection_bypasses(self):
        self.assertFalse(self.meta(protect_audio=True)["eligible"])

    def test_multigpu_bypasses(self):
        self.assertFalse(self.meta(options={"multigpu_thread_device": object()})["eligible"])

    def test_branch_batch_bypasses(self):
        meta = self.meta(options={"cond_or_uncond": [0, 1]})
        self.assertFalse(meta["eligible"])
        self.assertIn("cond_or_uncond length 2", meta["reason"])

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

        def normal(*args, **kwargs):
            runs.append(scope.current())
            return 7

        self.assertEqual(scope.outer(normal, None, None, None, self.schedule), 7)
        self.assertIsNot(runs[0], runs[1])
        self.assertIsNone(scope.current())

    def test_executor_chain(self):
        from comfy.patcher_extension import WrapperExecutor
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
