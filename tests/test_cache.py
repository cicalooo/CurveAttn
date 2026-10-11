import types
import unittest
import torch
from curvattn_test_package.residual_cache import MiddleCache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = MiddleCache(threshold=.01, max_hits=1, max_bytes=4096)
        self.calls = []
        self.layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 6, "video")))

    @torch.no_grad()
    def run_chain(self, step=2, key=(0,), dense=False, policy=(.2,),
                  x=None, eligible=True, fail=None, total=8):
        x = torch.ones(6, 4) if x is None else x.clone()
        meta = dict(eligible=eligible, key=key, step=step, total=total,
                    dense=dense, policy=policy)
        for index in range(4):
            def original(args, extra, index=index):
                self.calls.append(index)
                if fail == index:
                    raise RuntimeError("block failed")
                args["img"].add_(index + 1)
                return {"img": args["img"]}
            args = dict(img=x, layout=self.layout, transformer_options={})
            x = self.cache.call(index, 4, args, {}, meta, original)["img"]
        return x

    def test_off(self):
        self.run_chain(eligible=False)
        self.run_chain(step=3, eligible=False)
        self.assertEqual(self.calls, [0, 1, 2, 3] * 2)
        self.assertEqual(self.cache.bytes, 0)

    def test_hit(self):
        a = self.run_chain()
        b = self.run_chain(step=3)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(self.calls, [0, 1, 2, 3, 0, 3])
        self.assertEqual(self.cache.hits, 1)

    def test_max_hits(self):
        for step in (2, 3, 4):
            self.run_chain(step=step)
        self.assertEqual(self.calls, [0, 1, 2, 3, 0, 3, 0, 1, 2, 3])
        self.assertEqual(self.cache.hits, 1)

    def test_inplace_native_block(self):
        self.run_chain()
        e = next(iter(self.cache.entries.values()))
        torch.testing.assert_close(e.buffer, torch.full((6, 4), 5.))
        self.assertTrue(e.valid)

    def test_branch_isolation(self):
        self.run_chain(key=(0,))
        self.run_chain(step=3, key=(1,))
        self.assertEqual(self.cache.hits, 0)
        self.run_chain(step=3, key=(0,))
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(len(self.cache.entries), 2)

    def test_budget(self):
        self.cache.max_bytes = 96
        self.run_chain(key=(0,))
        self.run_chain(step=3, key=(1,))
        self.assertEqual(self.cache.bytes, 96)
        self.assertEqual(len(self.cache.entries), 1)
        self.cache.clear()
        self.cache.max_bytes = 0
        self.run_chain()
        self.assertEqual(self.cache.bytes, 0)

    def test_audio_drift(self):
        self.run_chain()
        x = torch.ones(6, 4)
        x[2:4] += 1
        self.run_chain(step=3, x=x)
        self.assertEqual(self.cache.hits, 0)

    def test_sign_flip(self):
        self.run_chain(x=torch.full((6, 4), 2.))
        self.run_chain(step=3, x=torch.full((6, 4), -4.))
        self.assertEqual(self.cache.hits, 0)

    def test_dense_refresh(self):
        self.run_chain()
        self.run_chain(step=3, dense=True)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.calls, [0, 1, 2, 3] * 2)

    def test_repeat_and_jump(self):
        for step in (2, 2, 4):
            self.run_chain(step=step)
        self.assertEqual(self.cache.hits, 0)

    def test_policy_change(self):
        self.run_chain()
        self.run_chain(step=3, policy=(.3,))
        self.assertEqual(self.cache.hits, 0)

    def test_clear(self):
        self.run_chain()
        self.cache.clear()
        self.assertEqual(self.cache.entries, {})
        self.assertIsNone(self.cache.active)
        self.assertEqual(self.cache.bytes, 0)

    def test_failed_middle_clears(self):
        with self.assertRaisesRegex(RuntimeError, "block failed"):
            self.run_chain(fail=2)
        self.assertEqual(self.cache.entries, {})
        self.assertIsNone(self.cache.active)

    def test_nonfinite(self):
        self.run_chain()
        self.run_chain(step=3, x=torch.full((6, 4), float("nan")))
        self.assertEqual(self.cache.hits, 0)

    def test_last_step(self):
        self.run_chain(step=6)
        self.run_chain(step=7)
        self.assertEqual(self.cache.hits, 0)

    def test_short_schedule(self):
        self.run_chain(step=1, total=4)
        self.run_chain(step=2, total=4)
        self.assertEqual(self.cache.hits, 0)

    def test_signature_change(self):
        self.run_chain()
        self.run_chain(step=3, x=torch.ones(6, 4, dtype=torch.float64))
        self.assertEqual(self.cache.hits, 0)
        e = next(iter(self.cache.entries.values()))
        self.assertEqual(e.buffer.dtype, torch.float64)

    def test_third_branch(self):
        self.cache.max_bytes = 10 ** 9
        self.run_chain(key=(0,))
        self.run_chain(key=(1,))
        self.run_chain(key=(2,))
        self.assertEqual(len(self.cache.entries), 2)
        self.assertEqual(self.calls[-4:], [0, 1, 2, 3])

    def test_grad_enabled(self):
        x = torch.ones(6, 4, requires_grad=True)
        meta = dict(eligible=True, key=(0,), step=2, total=8, dense=False, policy=(.2,))
        def original(args, extra):
            self.calls.append(0)
            return {"img": args["img"]}
        out = self.cache.call(0, 4, dict(img=x, layout=self.layout), {}, meta, original)
        self.assertEqual(self.cache.bytes, 0)
        self.assertIs(out["img"], x)
