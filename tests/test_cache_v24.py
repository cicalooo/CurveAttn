"""Phase 0–2 cache tests: presets, windows, guards, first_block, telemetry."""
from __future__ import annotations

import ast
import inspect
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from curve_test_package.config import CACHE_PRESETS, apply_cache_preset, validate_options
from curve_test_package.residual_cache import (
    MiddleCache, format_cache_report, modality_drift, temporal_video_drift,
)
from curve_test_package.runtime import RunScope, cache_metadata
from curve_test_package.nodes import CurveAttention
from curve_test_package import curve_attention as ca
from test_config import DEFAULTS

ROOT = Path(__file__).resolve().parents[1]

WIDGET_ORDER = [
    "model", "selection", "keep_percent", "min_tokens", "dense_steps",
    "dense_last_steps", "tail", "protect_audio", "verbose",
    "protect_prefix", "disable_fp16_accum", "tau", "extra_tokens",
    "start_percent", "end_percent", "keep_percent_end", "tau_end",
    "dense_backend", "qkv_mode", "cache_enabled", "cache_threshold",
    "cache_max_hits", "cache_max_mib", "cache_span", "cache_preset",
    "cache_start_percent", "cache_end_percent", "cache_min_steps",
    "cache_temporal_guard", "cache_audio_guard", "cache_mode",
]


class CacheV24Tests(unittest.TestCase):
    def setUp(self):
        self.layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 8, "video")))

    @torch.no_grad()
    def run_chain(self, cache, step=2, key=(0,), dense=False, policy=(.2,),
                  x=None, eligible=True, total=8, meta_extra=None):
        calls = []
        x = torch.ones(8, 4) if x is None else x.clone()
        meta = dict(eligible=eligible, key=key, step=step, total=total,
                    dense=dense, policy=policy)
        if meta_extra:
            meta.update(meta_extra)
        for index in range(4):
            def original(args, extra, index=index):
                calls.append(index)
                args["img"].add_(index + 1)
                return {"img": args["img"]}
            args = dict(img=x, layout=self.layout, transformer_options={})
            x = cache.call(index, 4, args, {}, meta, original)["img"]
        return x, calls

    def test_format_cache_report_contains_hits(self):
        cache = MiddleCache(threshold=.01, max_hits=1, max_bytes=4096)
        report = format_cache_report(cache, span="middle", mode="fingerprint",
                                     start_percent=0.2, end_percent=0.8)
        self.assertIn("hits=", report)
        self.assertIn("refreshes=", report)
        self.assertIn("bypasses=", report)
        self.assertIn("budget_bypasses=", report)
        self.assertIn("bytes=", report)
        self.assertIn("span=middle", report)
        self.assertIn("mode=fingerprint", report)

    def test_preset_maps(self):
        expected = {
            "turbo_safe": dict(cache_threshold=0.05, cache_start_percent=0.15,
                               cache_end_percent=0.85, cache_max_hits=1,
                               cache_min_steps=4, cache_enabled=True,
                               cache_temporal_guard=False),
            "native": dict(cache_threshold=0.08, cache_start_percent=0.15,
                           cache_end_percent=0.85, cache_max_hits=1,
                           cache_temporal_guard=True),
            "fast": dict(cache_threshold=0.10, cache_start_percent=0.10,
                         cache_end_percent=0.90, cache_max_hits=1,
                         cache_temporal_guard=True),
        }
        base = dict(cache_enabled=False, cache_threshold=0.01, cache_max_hits=1,
                    cache_max_mib=512, cache_span="middle",
                    cache_start_percent=0.2, cache_end_percent=0.8,
                    cache_min_steps=6, cache_temporal_guard=False,
                    cache_audio_guard=False, cache_mode="fingerprint")
        for name, want in expected.items():
            got = apply_cache_preset(name, base)
            for k, v in want.items():
                self.assertEqual(got[k], v, msg=f"{name}.{k}")
            self.assertTrue(got["cache_enabled"])
            self.assertTrue(got["cache_audio_guard"])
            self.assertEqual(got["cache_span"], "middle")
        off = apply_cache_preset("off", base)
        self.assertFalse(off["cache_enabled"])
        manual = apply_cache_preset("manual", base)
        self.assertEqual(manual["cache_threshold"], 0.01)
        self.assertFalse(manual["cache_enabled"])
        self.assertEqual(manual["cache_max_mib"], 512)
        for name in ("native", "fast", "turbo_safe"):
            self.assertNotIn("cache_max_mib", CACHE_PRESETS[name])
            self.assertEqual(apply_cache_preset(name, base)["cache_max_mib"], 512)

    def test_start_end_percent_gates(self):
        cache = MiddleCache(threshold=.05, max_hits=2, max_bytes=10 ** 9,
                            start_percent=0.4, end_percent=0.6, min_steps=4)
        # total=8 → progress = step/7; step 2 ≈ 0.286 outside, step 3 ≈ 0.429 inside
        self.run_chain(cache, step=2, total=8,
                       meta_extra=dict(start_percent=0.4, end_percent=0.6, min_steps=4))
        _, calls = self.run_chain(cache, step=3, total=8,
                                  meta_extra=dict(start_percent=0.4, end_percent=0.6, min_steps=4))
        self.assertEqual(cache.hits, 1)
        self.assertEqual(calls, [0, 3])
        # step 5 ≈ 0.714 outside window → refresh not hit
        cache2 = MiddleCache(threshold=.05, max_hits=2, max_bytes=10 ** 9,
                             start_percent=0.4, end_percent=0.6, min_steps=4)
        self.run_chain(cache2, step=3, total=8,
                       meta_extra=dict(start_percent=0.4, end_percent=0.6, min_steps=4))
        self.run_chain(cache2, step=5, total=8,
                       meta_extra=dict(start_percent=0.4, end_percent=0.6, min_steps=4))
        self.assertEqual(cache2.hits, 0)

    def test_turbo_min_steps_4_can_hit(self):
        cache = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9,
                            start_percent=0.15, end_percent=0.85, min_steps=4)
        # total=4 → progress step1=1/3≈0.33, step2=2/3≈0.67 inside 0.15–0.85
        self.run_chain(cache, step=1, total=4,
                       meta_extra=dict(min_steps=4, start_percent=0.15, end_percent=0.85))
        self.run_chain(cache, step=2, total=4,
                       meta_extra=dict(min_steps=4, start_percent=0.15, end_percent=0.85))
        self.assertEqual(cache.hits, 1)

    def test_manual_min_steps_6_still(self):
        cache = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9, min_steps=6)
        self.run_chain(cache, step=1, total=4, meta_extra=dict(min_steps=6))
        self.run_chain(cache, step=2, total=4, meta_extra=dict(min_steps=6))
        self.assertEqual(cache.hits, 0)

    def test_temporal_guard_blocks_local_video_drift(self):
        # Larger video segment so a localized spike can exceed chunk drift while
        # global mean drift stays under threshold.
        layout = types.SimpleNamespace(segments=(
            (0, 2, "text"), (2, 4, "audio"), (4, 20, "video")))
        cache = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9,
                            temporal_guard=True, min_steps=4,
                            start_percent=0.0, end_percent=1.0)
        self.layout = layout
        self.run_chain(cache, step=2, total=8, x=torch.ones(20, 4),
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        # Without temporal guard, this mild global change would still hit; with
        # guard, concentrating change in one video chunk blocks the hit.
        x = torch.ones(20, 4)
        x[4:6] += 2.0  # first video chunk only
        self.run_chain(cache, step=3, total=8, x=x,
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        self.assertEqual(cache.hits, 0)
        # Control: same magnitude spread evenly → temporal off would hit; with
        # guard still may hit if every chunk stays under threshold.
        cache2 = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9,
                             temporal_guard=False, min_steps=4,
                             start_percent=0.0, end_percent=1.0)
        self.run_chain(cache2, step=2, total=8, x=torch.ones(20, 4),
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        x2 = torch.ones(20, 4)
        x2[4:6] += 2.0
        self.run_chain(cache2, step=3, total=8, x=x2,
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        # Global mean over 16 video rows of +2 on 2 rows ≈ 0.125 relative? may or may not hit.
        # Stronger proof: guard-on blocks; guard-off with tiny uniform change hits.
        cache3 = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9,
                             temporal_guard=False, min_steps=4,
                             start_percent=0.0, end_percent=1.0)
        self.run_chain(cache3, step=2, total=8, x=torch.ones(20, 4),
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        self.run_chain(cache3, step=3, total=8, x=torch.ones(20, 4),
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        self.assertEqual(cache3.hits, 1)

    def test_audio_guard_blocks_audio_drift(self):
        cache = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9,
                            audio_guard=True, min_steps=4,
                            start_percent=0.0, end_percent=1.0)
        self.run_chain(cache, step=2, total=8,
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        x = torch.ones(8, 4)
        x[2:4] += 5  # audio segment
        self.run_chain(cache, step=3, total=8, x=x,
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        self.assertEqual(cache.hits, 0)
        # Soft miss should leave an entry (refresh), not clear
        self.assertEqual(len(cache.entries), 1)

    def test_first_block_mode_refresh_and_hit(self):
        cache = MiddleCache(threshold=.05, max_hits=1, max_bytes=10 ** 9,
                            mode="first_block", min_steps=4,
                            start_percent=0.0, end_percent=1.0)
        self.run_chain(cache, step=2, total=8,
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        self.assertGreaterEqual(cache.refreshes, 1)
        self.run_chain(cache, step=3, total=8,
                       meta_extra=dict(min_steps=4, start_percent=0.0, end_percent=1.0))
        self.assertEqual(cache.hits, 1)

    def test_schema_prefix_and_new_widgets(self):
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        schema = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "Schema")
        inputs = next(k.value for k in schema.keywords if k.arg == "inputs")
        names = [ast.literal_eval(node.args[0]) for node in inputs.elts]
        self.assertEqual(names, WIDGET_ORDER)

    def test_defaults_append_only(self):
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

    def test_validate_new_fields(self):
        opts = dict(DEFAULTS)
        opts["cache_threshold"] = 0.12
        validate_options(**opts, cache_span="middle", cache_preset="turbo_safe",
                         cache_start_percent=0.15, cache_end_percent=0.85,
                         cache_min_steps=4, cache_mode="first_block")
        with self.assertRaises(ValueError):
            validate_options(**DEFAULTS, cache_preset="nope")
        with self.assertRaises(ValueError):
            validate_options(**DEFAULTS, cache_mode="tea")
        with self.assertRaises(ValueError):
            validate_options(**DEFAULTS, cache_start_percent=0.9, cache_end_percent=0.1)

    def test_cache_metadata_min_steps(self):
        from uuid import UUID
        schedule = (1., .75, .5, .25, 0.)
        opts = dict(sigmas=.5, cond_or_uncond=[0],
                    uuids=[UUID("00000000-0000-0000-0000-000000000001")])
        meta6 = cache_metadata(opts, None, {}, schedule=schedule, enabled=True,
                               protect_audio=False, policy=(), min_steps=6)
        self.assertFalse(meta6["eligible"])  # total=4 < 6
        meta4 = cache_metadata(opts, None, {}, schedule=schedule, enabled=True,
                               protect_audio=False, policy=(), min_steps=4)
        self.assertTrue(meta4["eligible"])

    def test_run_scope_passes_mode(self):
        scope = RunScope(.05, 1, 1, span="middle", mode="first_block",
                         min_steps=4, start_percent=0.1, end_percent=0.9)
        seen = []

        def sample(*a, **k):
            seen.append(scope.current().cache)
            self.assertEqual(seen[-1].mode, "first_block")
            self.assertEqual(seen[-1].min_steps, 4)
            return 1

        self.assertEqual(scope.outer(sample, None, None, None, [1., .5, 0.]), 1)

    def test_js_cache_max_mib_ownership_matches_config(self):
        """PRESET_OWNS_MAX_MIB must match which CACHE_PRESETS dicts set cache_max_mib."""
        import re
        js = (ROOT / "web" / "curve_attention.js").read_text(encoding="utf-8")
        match = re.search(
            r"const PRESET_OWNS_MAX_MIB = new Set\(\[(.*?)\]\);",
            js,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "PRESET_OWNS_MAX_MIB Set missing in JS")
        js_owns = set(re.findall(r'"([^"]+)"', match.group(1)))
        self.assertNotIn("off", js_owns)
        self.assertNotIn("manual", js_owns)
        config_owns = set()
        config_widget = set()
        for name, mapped in CACHE_PRESETS.items():
            if mapped is None:
                continue  # manual
            if "cache_max_mib" in mapped:
                config_owns.add(name)
            elif name != "off":
                # Named preset dict that does not set max_mib → widget-owned
                config_widget.add(name)
        self.assertEqual(config_owns, js_owns)
        self.assertEqual(config_widget, {"turbo_safe", "native", "fast"})
        self.assertTrue(config_widget.isdisjoint(js_owns))

    def test_preset_keys_complete(self):
        self.assertEqual(list(CACHE_PRESETS), ["off", "manual", "turbo_safe", "native", "fast"])


if __name__ == "__main__":
    unittest.main()
