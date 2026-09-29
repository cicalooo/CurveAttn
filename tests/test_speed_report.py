import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SpeedReportTests(unittest.TestCase):
    def test_gate_rejects_quality_failure_and_slow_candidate(self):
        path = ROOT / "tools" / "check_speed_report.py"
        self.assertTrue(path.is_file(), "report checker missing")
        spec = importlib.util.spec_from_file_location("curve_speed_report", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def rows(case, seconds, quality="yes"):
            return [dict(case=case, trial=str(i), steps="8",
                         sampling_seconds=str(seconds), generation_seconds=str(seconds+10),
                         quality_pass=quality, cache_hits="1") for i in range(3)]
        reference = rows("A", 95) + rows("B", 100)
        self.assertTrue(module.evaluate(reference + rows("E", 85))["timing_quality_gate"])
        self.assertFalse(module.evaluate(reference + rows("E", 96))["timing_quality_gate"])
        self.assertFalse(module.evaluate(reference + rows("E", 85, "no"))["timing_quality_gate"])
