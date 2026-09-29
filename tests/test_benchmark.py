import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("curve_bench_cpu", ROOT / "benchmark_harness.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def test_no_implicit_gpu(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "benchmark_harness.py")],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout.strip(),
                         "GPU permission required; no GPU work performed.")
        self.assertNotIn("torch", result.stderr.lower())

    def test_aggregate(self):
        result = bench.aggregate([1, 2, 3, 4, 100], [10, 50, 20, 30, 40])
        self.assertEqual(result, dict(median_ms=3, p95_ms=100, peak_allocated_mib=50))
        with self.assertRaises(ValueError):
            bench.aggregate([], [])
