import types
import unittest
from unittest.mock import Mock

import torch

from curvattn_test_package.blocks import CurveBlockDispatch
from curvattn_test_package.cache import CacheState, MANUAL_DEFAULTS
from helpers import SCHEDULE, sparse_state, step_options

LAYOUT = types.SimpleNamespace(segments=((0, 2, "text"), (2, 4, "audio"), (4, 6, "video")))
META = dict(eligible=True, key=(0,), dense=False, policy=(.2,), total=8)


class FixedMetaCache(CacheState):
    """CacheState whose metadata is supplied by the test."""

    def __init__(self, **meta):
        super().__init__(dict(MANUAL_DEFAULTS), max_mib=1)
        self.meta = meta

    def metadata(self, options, run):
        return dict(META, **self.meta, step=options["step"])


class DispatchTests(unittest.TestCase):
    def test_native_fallback(self):
        sentinel = object()
        result = CurveBlockDispatch(0, 4, None)(
            {"transformer_options": {}}, {"original_block": lambda a: {"img": sentinel, "extra": 7}})
        self.assertIs(result["img"], sentinel)
        self.assertEqual(result["extra"], 7)

    def chain(self, cache, step, calls, fail=None):
        x = torch.ones(6, 4)
        for i in range(4):
            def original(a, i=i):
                calls.append(i)
                if fail == i:
                    raise RuntimeError("mid")
                return {"img": a["img"].add_(i + 1)}
            x = CurveBlockDispatch(i, 4, None, cache)(
                dict(img=x, layout=LAYOUT, transformer_options={"step": step}),
                {"original_block": original})["img"]
        return x

    def test_middle_skip_keeps_edge_blocks(self):
        cache = FixedMetaCache()
        calls, outputs = [], []

        def sample(*args, **kwargs):
            for step in (2, 3):
                outputs.append(self.chain(cache, step, calls))
            self.assertEqual(cache.run_scope.current().cache.hits, 1)

        with torch.no_grad():
            cache.run_scope.outer(sample, None, None, None, SCHEDULE)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertEqual(calls, [0, 1, 2, 3, 0, 3])

    def test_metadata_is_lazy_and_computed_once_per_forward(self):
        cache = FixedMetaCache()
        seen = []
        original = cache.metadata
        cache.metadata = lambda options, run: seen.append(1) or original(options, run)
        with torch.no_grad():
            cache.run_scope.outer(lambda *a, **k: self.chain(cache, 2, []), None, None, None, SCHEDULE)
        self.assertEqual(seen, [1])

    def test_exception_drops_snapshot(self):
        cache = FixedMetaCache()
        with self.assertRaisesRegex(RuntimeError, "mid"):
            cache.run_scope.outer(lambda *a, **k: self.chain(cache, 2, [], fail=2),
                                  None, None, None, SCHEDULE)
        self.assertIsNone(cache.run_scope.current())

    def test_chunked_injected_only_for_the_owner(self):
        seen = []

        def original(args):
            seen.append(args.get("attention"))
            return {"img": args["img"]}

        args = lambda to: {"img": torch.ones(1, 1), "transformer_options": to}
        CurveBlockDispatch(0, 3, None)(args({}), {"original_block": original})
        a, b = sparse_state(dense_steps=frozenset()), sparse_state(dense_steps=frozenset())
        fa, fb = Mock(name="a"), Mock(name="b")
        dispatch = CurveBlockDispatch(0, 3, None, providers=((a, fa), (b, fb)))
        dispatch(args({}), {"original_block": original})                    # no wrapper: newest
        dispatch(args(step_options(a)), {"original_block": original})      # a owns
        dispatch(args(step_options(a, dense=True)), {"original_block": original})  # nobody
        self.assertEqual(seen, [None, fb, fa, None])
