"""VSA coarse gate, chunked VSA and the plan cache. No GPU."""
import types
import unittest
from unittest.mock import patch

import torch

from comfy.ldm.minimax.model import PackedLayout
from curvattn_test_package import chunked_attention as ch
from curvattn_test_package import sparse as sp
from curvattn_test_package import vsa
from helpers import sparse_state as state, step_options

CPU = torch.device("cpu")


class PlanCacheTests(unittest.TestCase):
    def test_fast_path_lru_and_cleanup(self):
        s = state(selection="vsa", keep_ratio=0.1)
        layouts = [PackedLayout(2, 2, 4, 4 + 2 * i, 2) for i in range(6)]
        first = s.vsa_plan({"minimax_h3_layout": layouts[0]}, layouts[0].seq_len, CPU)
        self.assertIs(s.vsa_plan({"minimax_h3_layout": layouts[0]}, layouts[0].seq_len, CPU), first)
        with patch.object(sp, "build_vsa_plan", wraps=sp.build_vsa_plan) as build:
            for lay in layouts[1:4]:
                s.vsa_plan({"minimax_h3_layout": lay}, lay.seq_len, CPU)
            # touch layouts[0] (LRU hit, no rebuild) then add two more
            again = PackedLayout(2, 2, 4, 4, 2)
            self.assertIs(s.vsa_plan({"minimax_h3_layout": again}, again.seq_len, CPU), first)
            for lay in layouts[4:6]:
                s.vsa_plan({"minimax_h3_layout": lay}, lay.seq_len, CPU)
            self.assertEqual(build.call_count, 5)
        self.assertEqual(len(s._vsa_plans), sp.VSA_PLAN_CACHE)
        keys = [k[0] for k in s._vsa_plans]
        self.assertIn(tuple(layouts[0].signature), keys)       # recently used survives
        self.assertNotIn(tuple(layouts[1].signature), keys)    # oldest evicted
        s.reset()
        self.assertEqual(len(s._vsa_plans), sp.VSA_PLAN_CACHE)  # kept across runs
        s.cleanup()
        self.assertEqual(len(s._vsa_plans), 0)
        self.assertIsNone(s._vsa_last)

    def test_token_mismatch_is_none(self):
        s = state(selection="vsa", keep_ratio=0.1)
        lay = PackedLayout(2, 2, 4, 4, 2)
        self.assertIsNotNone(s.vsa_plan({"minimax_h3_layout": lay}, lay.seq_len, CPU))
        self.assertIsNone(s.vsa_plan({"minimax_h3_layout": lay}, lay.seq_len + 1, CPU))


def fake_attention(heads=2, head_dim=128, hidden=16, gate=True):
    attn = types.SimpleNamespace(heads=heads, head_dim=head_dim)
    attn.to_gate_compress = (torch.nn.Linear(hidden, heads * head_dim, bias=False)
                             if gate else None)
    return attn


class GateProviderTests(unittest.TestCase):
    def test_gate_reaches_override_kernel(self):
        lay = PackedLayout(2, 2, 4, 4, 2)
        s = state(selection="vsa", keep_ratio=0.1, dense_steps=frozenset())
        attn = fake_attention()
        x = torch.randn(lay.seq_len, 16)
        seen = {}
        override = sp.make_attention_override(s)

        def kernel(q, k, v, **kwargs):
            seen.update(kwargs)
            return torch.zeros_like(q)

        def module(h, rope_freqs=None, transformer_options=None):
            qkv = [torch.randn(1, 2, lay.seq_len, 128) for _ in range(3)]
            return override(None, *qkv, 2, skip_reshape=True,
                            transformer_options=transformer_options)

        opts = step_options(s, minimax_h3_layout=lay)
        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "sol_attn_zerocopy", kernel):
            plan = s.vsa_plan(opts, lay.seq_len, CPU)
            gate = vsa.coarse_gate(attn.to_gate_compress, x, plan, 2, 128)
            module(x, transformer_options=vsa.with_gate(opts, s, plan, gate))
        self.assertIn("coarse_gate", seen)
        self.assertEqual(tuple(seen["coarse_gate"].shape), (1, plan["n"], 2, 128))
        expect = attn.to_gate_compress(vsa.permute_rows(x, plan["src"])).view(1, plan["n"], 2, 128)
        torch.testing.assert_close(seen["coarse_gate"].float(), expect.to(torch.bfloat16).float())
        self.assertFalse(seen["tail"])

    def test_provider_calls_module_with_gate_or_without(self):
        lay = PackedLayout(2, 2, 4, 4, 2)
        s = state(selection="vsa", keep_ratio=0.1, dense_steps=frozenset())
        got = []

        class Module:
            heads, head_dim = 2, 128

            def __init__(self, gate):
                self.to_gate_compress = torch.nn.Linear(16, 256, bias=False) if gate else None

            def __call__(self, x, rope_freqs=None, transformer_options=None):
                got.append((transformer_options.get(sp.NS) or {}).get(vsa.GATE_KEY))
                return "ran"

        x = torch.randn(lay.seq_len, 16)
        opts = step_options(s, minimax_h3_layout=lay)
        self.assertEqual(vsa.make_vsa_gate_attention(Module(True), s)(x, None, opts), "ran")
        self.assertEqual(got[-1][0], s.uid)
        self.assertEqual(tuple(got[-1][2].shape), (1, s.vsa_plan(opts, lay.seq_len, CPU)["n"], 2, 128))
        self.assertNotIn(vsa.GATE_KEY, opts[sp.NS])   # caller dict untouched
        vsa.make_vsa_gate_attention(Module(False), s)(x, None, opts)
        self.assertIsNone(got[-1])
        dense_opts = step_options(s, dense=True, minimax_h3_layout=lay)
        vsa.make_vsa_gate_attention(Module(True), s)(x, None, dense_opts)
        self.assertIsNone(got[-1])

    def test_take_gate_rejects_foreign(self):
        s1 = state(selection="vsa", keep_ratio=0.1)
        s2 = state(selection="vsa", keep_ratio=0.1)
        plan = {"n": 64}
        opts = vsa.with_gate({}, s1, plan, "g")
        self.assertIsNone(vsa.take_gate(opts, s2, plan))
        self.assertIsNone(vsa.take_gate(opts, s1, {"n": 64}))
        self.assertEqual(vsa.take_gate(opts, s1, plan), "g")
        self.assertIsNone(vsa.take_gate(opts, s1, plan))


class InstallTests(unittest.TestCase):
    def test_gate_provider_installed_only_with_gate_layers(self):
        from helpers import h3_patcher, kitchen, dit
        with kitchen():
            plain = sp.apply_sparse(h3_patcher(), selection="vsa", keep_percent=10.)
            self.assertEqual(dit(plain), {})
            model = h3_patcher()
            for block in model.get_model_object("diffusion_model").blocks:
                block.attn = torch.nn.Linear(4, 4)
                block.attn.to_gate_compress = torch.nn.Linear(4, 4)
            gated = sp.apply_sparse(model, selection="vsa", keep_percent=10.)
            self.assertEqual(len(dit(gated)), 4)
            self.assertTrue(all(d.providers for d in dit(gated).values()))
            chunked = sp.apply_sparse(model, selection="vsa", keep_percent=10., qkv_mode="chunked")
            self.assertEqual(len(dit(chunked)), 4)


class ChunkedVsaTests(unittest.TestCase):
    def test_permutes_rows_rope_gate_and_scatters(self):
        lay = PackedLayout(2, 2, 4, 4, 2)
        s = state(selection="vsa", keep_ratio=0.1)
        plan = s.vsa_plan({"minimax_h3_layout": lay}, lay.seq_len, CPU)
        heads, hd, hidden = 2, 128, 16
        attn = types.SimpleNamespace(
            heads=heads, head_dim=hd,
            to_gate_compress=torch.nn.Linear(hidden, heads * hd, bias=False),
            qkv_proj=lambda xc: xc[:, :1].repeat(1, 3 * heads * hd),
            q_norm=types.SimpleNamespace(weight=torch.ones(hd), eps=1e-6),
            k_norm=types.SimpleNamespace(weight=torch.ones(hd), eps=1e-6),
            out_proj=lambda o: o)
        x = torch.arange(lay.seq_len, dtype=torch.float32).view(-1, 1).repeat(1, hidden) + 1
        rope = torch.randn(1, lay.seq_len, 1, 4, 2, 2)
        seen = {}

        def kernel(factory, t, h, rope_freqs, qk_norm_weights, **kw):
            rows = torch.cat([c[:, 0].float() for c in factory()])
            seen.update(kw, t=t, rope=rope_freqs)
            out = rows.view(-1, 1, 1).expand(t, h, hd).contiguous()
            return out, None, None

        out = ch.run_chunked_vsa(attn, x, s.vsa_rope_freqs(rope, plan),
                                 s.sparse_kwargs(None, (0, 0), (0, 0)), plan,
                                 kernel=kernel, cast=lambda w: w, chunk_tokens=64)
        self.assertEqual(seen["t"], plan["n"])
        self.assertEqual(seen["sink_blocks"], [0, plan["n_prefix"]])
        self.assertFalse(seen["tail"])
        torch.testing.assert_close(seen["block_len"], plan["block_len"])
        self.assertEqual(tuple(seen["coarse_gate"].shape), (1, plan["n"], heads, hd))
        live = plan["src"] >= 0
        torch.testing.assert_close(seen["rope"][0, live], rope[0, plan["src"][live]])
        self.assertTrue(torch.all(seen["rope"][0, ~live] == 0))
        # packed order restored: row i carries x[i] (= i + 1)
        torch.testing.assert_close(out[:, 0], torch.arange(lay.seq_len, dtype=torch.float32) + 1)
        self.assertIs(s.vsa_rope_freqs(rope, plan), seen["rope"])


if __name__ == "__main__":
    unittest.main()
