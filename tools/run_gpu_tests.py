from __future__ import annotations
import argparse
import sys
import unittest
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--allow-gpu", action="store_true")
parser.add_argument("selectors", nargs="*")
options = parser.parse_args()
if not options.allow_gpu:
    print("GPU permission required; no GPU work performed.")
    raise SystemExit(2)

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "tests"), str(root / "gpu_tests"), str(root.parents[1])]
sys.argv = [sys.argv[0]]
from comfy.cli_args import args
args.cpu = False
suite = unittest.TestLoader().loadTestsFromNames(
    options.selectors or ["test_attention.NativeOperatorTests"])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and not result.skipped and result.testsRun else 1)
