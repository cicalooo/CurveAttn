"""Cache presets, windows, guards, whole span, first_block, telemetry, validation."""
import math
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from curvattn_test_package import residual_cache as rc
from curvattn_test_package.cache import (
    CACHE_PRESETS, MANUAL_DEFAULTS, apply_cache_preset, validate_cache,
)
from curvattn_test_package.residual_cache import MiddleCache, format_cache_report

VALID = dict(preset="manual", max_mib=1024, **MANUAL_DEFAULTS)


class CacheFeatureTests(unittest.TestCase):
    def setUp(self):
        self.layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 8, "video")))

    @torch.no_grad()
    def run_chain(self, cache, step=2, key=(0,), x=None, eligible=True, total=8, rows=8, **extra):
        calls = []
        x = torch.ones(rows, 4) if x is None else x.clone()
        meta = dict(eligible=eligible, key=key, step=step, total=total, dense=False,
                    policy=(.2,), **extra)
        for index in range(4):
            def original(args, extra, index=index):
                calls.append(index)
                args["img"].add_(index + 1)
                return {"img": args["img"]}
            x = cache.call(index, 4, dict(img=x, layout=self.layout, transformer_options={}),
                           {}, meta, original)["img"]
        return x, calls

    # -- presets / validation -------------------------------------------------

    def test_preset_maps(self):
        base = dict(MANUAL_DEFAULTS)
        self.assertEqual(list(CACHE_PRESETS), ["manual", "turbo_safe", "native", "fast"])
        self.assertEqual(apply_cache_preset("manual", base), base)
        expected = {"turbo_safe": (0.05, 0.15, 0.85, 4, False),
                    "native": (0.08, 0.15, 0.85, 6, True),
                    "fast": (0.10, 0.10, 0.90, 6, True)}
        for name, (threshold, start, end, min_steps, temporal) in expected.items():
            got = apply_cache_preset(name, base)
            self.assertEqual((got["threshold"], got["start_percent"], got["end_percent"],
                              got["min_steps"], got["temporal_guard"]),
                             (threshold, start, end, min_steps, temporal), name)
            self.assertTrue(got["audio_guard"])
            self.assertEqual(got["span"], "middle")
            self.assertNotIn("max_mib", CACHE_PRESETS[name])
            validate_cache(preset=name, max_mib=1024, **got)
        with self.assertRaises(ValueError):
            apply_cache_preset("Native/3090", base)

    def test_validation(self):
        validate_cache(**VALID)
        for change in [dict(threshold=float("nan")), dict(threshold=.3), dict(max_hits=0),
                       dict(max_mib=-1), dict(span="x"), dict(mode="tea"),
                       dict(start_percent=.9, end_percent=.1), dict(min_steps=1),
                       dict(preset="off"), dict(temporal_guard=1)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cache(**(VALID | change))

    # -- windows and guards ---------------------------------------------------

    def test_start_end_percent_gates(self):
        window = dict(start_percent=0.4, end_percent=0.6, min_steps=4)
        cache = MiddleCache(threshold=.05, max_hits=2, max_bytes=10 ** 9, **window)
        self.run_chain(cache, step=2, **window)
        _, calls = self.run_chain(cache, step=3, **window)
        self.assertEqual((cache.hits, calls), (1, [0, 3]))
        cache2 = MiddleCache(threshold=.05, max_hits=2, max_bytes=10 ** 9, **window)
        self.run_chain(cache2, step=3, **window)
        self.run_chain(cache2, step=5, **window)
        self.assertEqual(cache2.hits, 0)

    def test_min_steps_4_can_hit_and_6_cannot(self):
        window = dict(min_steps=4, start_percent=0.15, end_percent=0.85)
        cache = MiddleCache(threshold=.05, max_bytes=10 ** 9, **window)
        self.run_chain(cache, step=1, total=4, **window)
        self.run_chain(cache, step=2, total=4, **window)
        self.assertEqual(cache.hits, 1)
        strict = MiddleCache(threshold=.05, max_bytes=10 ** 9, min_steps=6)
        self.run_chain(strict, step=1, total=4, min_steps=6)
        self.run_chain(strict, step=2, total=4, min_steps=6)
        self.assertEqual(strict.hits, 0)

    def test_temporal_guard_blocks_local_video_drift(self):
        self.layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 20, "video")))
        window = dict(min_steps=4, start_percent=0.0, end_percent=1.0)
        guarded = MiddleCache(threshold=.05, max_bytes=10 ** 9, temporal_guard=True, **window)
        self.run_chain(guarded, rows=20, **window)
        x = torch.ones(20, 4)
        x[4:6] += 2.0
        self.run_chain(guarded, step=3, x=x, rows=20, **window)
        self.assertEqual(guarded.hits, 0)
        quiet = MiddleCache(threshold=.05, max_bytes=10 ** 9, **window)
        self.run_chain(quiet, rows=20, **window)
        self.run_chain(quiet, step=3, rows=20, **window)
        self.assertEqual(quiet.hits, 1)

    def test_audio_guard_soft_refuses(self):
        window = dict(min_steps=4, start_percent=0.0, end_percent=1.0)
        cache = MiddleCache(threshold=.05, max_bytes=10 ** 9, audio_guard=True, **window)
        self.run_chain(cache, **window)
        x = torch.ones(8, 4)
        x[2:4] += 5
        self.run_chain(cache, step=3, x=x, **window)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(len(cache.entries), 1)

    def test_first_block_mode_refresh_and_hit(self):
        window = dict(min_steps=4, start_percent=0.0, end_percent=1.0)
        cache = MiddleCache(threshold=.05, max_bytes=10 ** 9, mode="first_block", **window)
        self.run_chain(cache, **window)
        self.assertGreaterEqual(cache.refreshes, 1)
        self.run_chain(cache, step=3, **window)
        self.assertEqual(cache.hits, 1)

    def test_lazy_meta_callable(self):
        cache = MiddleCache(max_bytes=4096)
        calls = []
        meta = dict(eligible=True, key=(0,), step=2, total=8, dense=False, policy=())
        x = torch.ones(8, 4)
        for index in range(4):
            x = cache.call(index, 4, dict(img=x, layout=self.layout), {},
                           lambda: calls.append(1) or meta,
                           lambda a, e: {"img": a["img"]})["img"]
        self.assertEqual(calls, [1])

    # -- whole span / speed plan ------------------------------------------------

    def test_whole_span_skips_all_block_compute(self):
        cache = MiddleCache(max_bytes=4096, span="whole")
        fresh, fresh_calls = self.run_chain(cache)
        reused, hit_calls = self.run_chain(cache, step=3)
        torch.testing.assert_close(fresh, reused, rtol=0, atol=0)
        self.assertEqual((fresh_calls, hit_calls, cache.hits), ([0, 1, 2, 3], [], 1))

    def test_whole_span_contract(self):
        cache = MiddleCache(max_bytes=4096, span="whole")
        self.run_chain(cache)
        _, other = self.run_chain(cache, step=3, key=(1,))
        self.assertEqual(other, [0, 1, 2, 3])
        _, hit = self.run_chain(cache, step=3)
        self.assertEqual(hit, [])
        _, capped = self.run_chain(cache, step=4)
        self.assertEqual(capped, [0, 1, 2, 3])
        cache.clear()
        self.assertEqual(cache.bytes, 0)
        with self.assertRaisesRegex(ValueError, "invalid cache span"):
            MiddleCache(span="unknown")

    def test_whole_rejects_audio_drift_and_nonfinite(self):
        for bad in ("audio", "nan", "sign"):
            with self.subTest(bad=bad):
                cache = MiddleCache(max_bytes=4096, span="whole")
                self.run_chain(cache)
                x = torch.ones(8, 4)
                if bad == "audio":
                    x[2:4] += 1
                elif bad == "nan":
                    x[:] = float("nan")
                else:
                    x.neg_()
                _, calls = self.run_chain(cache, step=3, x=x)
                self.assertEqual((calls, cache.hits), ([0, 1, 2, 3], 0))

    def test_whole_rejects_ineligible_repeat_and_jump(self):
        cache = MiddleCache(max_bytes=4096, span="whole")
        for step in (2, 2, 4):
            self.assertEqual(self.run_chain(cache, step=step)[1], [0, 1, 2, 3])
        self.assertEqual(self.run_chain(cache, step=5, eligible=False)[1], [0, 1, 2, 3])
        self.assertEqual(cache.entries, {})

    def test_whole_boundary_steps_are_fresh(self):
        cache = MiddleCache(max_bytes=4096, span="whole")
        for step in (0, 1, 6, 7):
            self.assertEqual(self.run_chain(cache, step=step)[1], [0, 1, 2, 3])

    def test_drift_one_host_read_and_no_python_tensor_bool(self):
        samples = tuple(torch.ones(2, 2) for _ in range(3))
        real_item = torch.Tensor.item
        calls = []

        def item(tensor, *args):
            calls.append(1)
            return real_item(tensor, *args)

        def forbidden_bool(tensor):
            raise AssertionError("tensor truth-value forces another host decision")
        cache = MiddleCache(max_bytes=4096)
        self.run_chain(cache)
        with patch.object(torch.Tensor, "item", item), \
             patch.object(torch.Tensor, "__bool__", forbidden_bool):
            self.run_chain(cache, step=3)
        self.assertEqual((len(calls), cache.hits), (1, 1))
        self.assertEqual(rc.relative_drift(samples, samples), 0.0)
        bad = (torch.full((2, 2), float("nan")),) + samples[1:]
        self.assertTrue(math.isinf(rc.relative_drift(bad, samples)))

    def test_cache_begin_paused_but_block_compute_unpaused(self):
        events = []

        @contextmanager
        def pause():
            events.append("enter")
            yield
            events.append("exit")
        cache = MiddleCache(max_bytes=4096)
        original_begin = cache._begin

        def begin(*args):
            self.assertEqual(events[-1], "enter")
            return original_begin(*args)
        with patch.object(rc, "pause_malloc_graph", pause), patch.object(cache, "_begin", begin):
            self.run_chain(cache)
        self.assertEqual(events, ["enter", "exit"])

    def test_budget_failure_is_explained(self):
        cache = MiddleCache(max_bytes=1)
        self.run_chain(cache)
        self.assertEqual((cache.budget_bypasses, cache.required_entry_bytes, cache.bytes),
                         (1, 128, 0))

    def test_format_cache_report(self):
        report = format_cache_report(MiddleCache(), span="middle", mode="fingerprint",
                                     start_percent=0.2, end_percent=0.8)
        for token in ("CurveAttn cache: hits=", "refreshes=", "bypasses=",
                      "budget_bypasses=", "bytes=", "span=middle", "mode=fingerprint"):
            self.assertIn(token, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
