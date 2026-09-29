import ast
from pathlib import Path
import types
import unittest
import torch


class CoreContractTests(unittest.TestCase):
    def test_native_boundary(self):
        path = Path(__file__).resolve().parents[3] / "comfy" / "ldm" / "minimax" / "model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "MiniMaxH3Model")
        forward = next(n for n in model.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_forward")
        names = [a.arg for a in forward.args.args]
        self.assertIn("denoise_mask", names)
        self.assertIn("audio_denoise_mask", names)
        constants = {n.value for n in ast.walk(forward)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertIn("minimax_h3_layout", constants)
        self.assertIn("double_block", constants)
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)]
        self.assertTrue(any(k.arg == "attention" and isinstance(k.value, ast.Call)
                            and isinstance(k.value.func, ast.Attribute)
                            and k.value.func.attr == "get" and k.value.args
                            and isinstance(k.value.args[0], ast.Constant)
                            and k.value.args[0].value == "attention"
                            for call in calls for k in call.keywords))

    @torch.no_grad()
    def test_native_block_passthrough(self):
        from comfy.ldm.minimax.model import DiTBlock
        from curve_test_package.block_dispatch import make_block_dispatch
        from curve_test_package.runtime import RunScope
        torch.manual_seed(7321)
        block = DiTBlock(hidden=256, heads=2, head_dim=128, ffn=128, t_dim=4,
                         eps=1e-6, qk_eps=1e-6, dtype=torch.float32,
                         device=torch.device("cpu"), operations=torch.nn).eval()
        state = types.SimpleNamespace(qkv_mode="native", cache_enabled=False,
                                      run_scope=RunScope(.01, 1, 1))
        x = torch.randn(6, 256)
        t = torch.randn(1, 4)
        segments = [(0, 2, 1), (2, 4, 2), (4, 6, 0)]
        expected = block(x.clone(), t, segments, None, transformer_options={})
        args = dict(img=x.clone(), t_emb=t, mod_segments=segments,
                    rope_freqs=None, transformer_options={})

        def original(a):
            return {"img": block(a["img"], a["t_emb"], a["mod_segments"],
                                 a["rope_freqs"], transformer_options=a["transformer_options"],
                                 attention=a.get("attention"))}

        actual = make_block_dispatch(0, 3, block.attn, state)(
            args, {"original_block": original})["img"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
