"""VSA tile plan and the sparse override's gather/scatter. No GPU."""
import types
import unittest
from unittest.mock import patch

import torch

from comfy.ldm.minimax.model import PackedLayout
from comfy_extras.nodes_sparse_attention import SparseAttnPatch
from curvattn_test_package import sparse as sp
from curvattn_test_package.vsa import build_vsa_plan, gather_qkv, scatter_rows
from helpers import sparse_state as state


def _core_plan(layout, device):
    patch = SparseAttnPatch(tau=0.0, topk_ratio=0.1, vsa=True, sigma_start=1.0, sigma_end=0.0,
                            min_tokens=0, dense_blocks=set(), sink_conditioning="off",
                            extra_tokens=0, verbose=False)
    return patch.vsa_plan(layout, device)


class VsaPlanTests(unittest.TestCase):
    def test_plan_matches_core_including_keyframes(self):
        keyframes = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 4)}]
        device = torch.device("cpu")
        for layout in (PackedLayout(2, 2, 4, 4, 2),
                       PackedLayout(3, 4, 8, 8, 2, keyframes=keyframes)):
            with self.subTest(seq=layout.seq_len):
                got = build_vsa_plan(layout, layout.seq_len, device)
                core = _core_plan(layout, device)
                self.assertEqual((got["n"], got["n_prefix"]), (core["n"], core["n_prefix"]))
                torch.testing.assert_close(got["src"], core["src"])
                torch.testing.assert_close(got["inv"], core["inv"])
                torch.testing.assert_close(got["block_len"], core["block_len"])
                self.assertEqual(int(got["block_len"].sum()), layout.seq_len)

    def test_missing_or_mismatched_layout_stays_unplanned(self):
        device = torch.device("cpu")
        layout = PackedLayout(2, 2, 4, 4, 2)
        self.assertIsNone(build_vsa_plan(None, layout.seq_len, device))
        self.assertIsNone(build_vsa_plan(layout, layout.seq_len + 1, device))
        bad = types.SimpleNamespace(
            seq_len=10, signature=(1, 2, 4, 4, 1),
            segments=((0, 2, "text"), (2, 4, "audio"), (4, 10, "video")))
        self.assertIsNone(build_vsa_plan(bad, 10, device))

    def test_gather_zeros_pad_and_scatter_restores_rows(self):
        layout = PackedLayout(2, 2, 4, 4, 2)
        plan = build_vsa_plan(layout, layout.seq_len, torch.device("cpu"))
        q = torch.arange(layout.seq_len, dtype=torch.float32).view(1, -1, 1, 1).expand(
            1, layout.seq_len, 2, 4).contiguous()
        gathered = gather_qkv(q, q, q, plan["src"])[0]
        pad = plan["src"] < 0
        self.assertTrue(torch.all(gathered[0, pad] == 0))
        live = ~pad
        torch.testing.assert_close(gathered[0, live, 0, 0], plan["src"][live].float())
        restored = scatter_rows(gathered, plan["inv"])
        torch.testing.assert_close(restored, q)


class VsaOverrideTests(unittest.TestCase):
    def test_override_tiles_calls_kernel_and_scatters(self):
        layout = PackedLayout(2, 2, 4, 4, 2)
        plan = build_vsa_plan(layout, layout.seq_len, torch.device("cpu"))
        s = state(selection="vsa", keep_ratio=0.1, tail=True, token_aug=128,
                  protect_audio=True, protect_prefix=True)
        qkv = [torch.randn(1, 2, layout.seq_len, 128) for _ in range(3)]
        seen = {}

        def kernel(q, k, v, **kwargs):
            seen["shape"] = tuple(q.shape)
            seen["kwargs"] = kwargs
            return torch.arange(q.shape[1], dtype=q.dtype).view(1, -1, 1, 1).expand_as(q).contiguous()

        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "sol_attn_zerocopy", kernel):
            out = sp.make_attention_override(s)(
                None, *qkv, 2, skip_reshape=True,
                transformer_options={"minimax_h3_layout": layout})
        self.assertEqual(seen["shape"], (1, plan["n"], 2, 128))
        self.assertEqual(seen["kwargs"]["sink_blocks"], [0, plan["n_prefix"]])
        self.assertEqual(seen["kwargs"]["sink_q"], [0, plan["n_prefix"]])
        self.assertFalse(seen["kwargs"]["tail"])
        self.assertEqual(seen["kwargs"]["token_aug"], 0)
        self.assertEqual(seen["kwargs"]["tau"], 0.0)
        self.assertAlmostEqual(seen["kwargs"]["topk_ratio"], 0.1)
        torch.testing.assert_close(seen["kwargs"]["block_len"], plan["block_len"])
        self.assertEqual(tuple(out.shape), (1, layout.seq_len, 256))
        # kernel wrote the padded index; scatter must put packed rows back
        torch.testing.assert_close(out[0, :, 0], plan["inv"].to(out.dtype))

    def test_missing_layout_stays_dense(self):
        layout = PackedLayout(2, 2, 4, 4, 2)
        s = state(selection="vsa", keep_ratio=0.1)
        qkv = [torch.randn(1, 2, layout.seq_len, 128) for _ in range(3)]
        with patch.object(sp, "_ineligible", return_value=None):
            out = sp.make_attention_override(s)(
                None, *qkv, 2, skip_reshape=True, transformer_options={})
        self.assertEqual(out, "dense")
        self.assertEqual(s.sparse_calls, 0)
        self.assertEqual(s.dense_calls, 1)
        self.assertIn("unavailable for VSA", s.last_reason)
