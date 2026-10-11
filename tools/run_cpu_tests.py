from __future__ import annotations
import os
import sys
import types
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(COMFY)]

import torch

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda._lazy_init = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("GPU forbidden in CPU tests")
)

if sys.argv[1:] == ["--self-test"]:
    assert torch.cuda.is_available() is False
    try:
        torch.cuda._lazy_init()
    except AssertionError as exc:
        assert str(exc) == "GPU forbidden in CPU tests"
        print("CPU_GUARD_OK")
        raise SystemExit(0)
    raise AssertionError("GPU allocation did not hit guard")

selectors = sys.argv[1:]
sys.argv = [sys.argv[0]]
from comfy.cli_args import args
args.cpu = True
package = types.ModuleType("curvattn_test_package")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
loader = unittest.TestLoader()
suite = (loader.loadTestsFromNames(selectors) if selectors else
         loader.discover(str(ROOT / "tests"), pattern="test_*.py"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
