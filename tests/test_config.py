import unittest
from curve_test_package.config import validate_options

DEFAULTS = dict(keep_percent=20., min_tokens=12288, dense_last_steps=0,
                selection="sla", tau=1.3, extra_tokens=0, start_percent=0.,
                end_percent=1., keep_percent_end=-1., tau_end=-1.,
                dense_backend="comfy_kitchen_int8", qkv_mode="native",
                cache_threshold=.01, cache_max_hits=1, cache_max_mib=1024)


class ConfigTests(unittest.TestCase):
    def test_valid(self):
        validate_options(**DEFAULTS)

    def test_invalid(self):
        invalid = [dict(tau=float("nan")), dict(keep_percent=float("inf")),
                   dict(extra_tokens=1), dict(extra_tokens=65),
                   dict(start_percent=.8, end_percent=.2),
                   dict(keep_percent_end=-.5), dict(tau_end=-.5),
                   dict(dense_backend="unknown"), dict(qkv_mode="unknown"),
                   dict(cache_threshold=float("nan")), dict(cache_max_hits=0),
                   dict(cache_max_mib=-1), dict(min_tokens=True)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_options(**(DEFAULTS | change))
