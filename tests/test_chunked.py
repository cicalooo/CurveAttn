import types
import unittest
from unittest.mock import Mock, patch
import torch
from curvattn_test_package import chunked_attention as ch
from helpers import sparse_state, step_options


class ChunkedTests(unittest.TestCase):
    def attention(self):
        return types.SimpleNamespace(
            heads=2, head_dim=128,
            qkv_proj=lambda x: torch.zeros(x.shape[0], 768, dtype=torch.bfloat16),
            out_proj=lambda x: x,
            q_norm=types.SimpleNamespace(weight=torch.ones(128), eps=1e-6),
            k_norm=types.SimpleNamespace(weight=torch.ones(128), eps=1e-6))

    def test_cpu_falls_back(self):
        state = types.SimpleNamespace(min_tokens=0, failure_reason=None)
        with patch.object(ch.ck, "sol_attn_is_available") as availability:
            reason = ch.chunked_reason(self.attention(), torch.ones(130, 256),
                                       None, state, {})
        self.assertEqual(reason, "not CUDA")
        availability.assert_not_called()

    def test_factory_replays(self):
        x = torch.ones(130, 256, dtype=torch.bfloat16)
        factory = ch.projection_chunks(self.attention(), x, 64)
        expected = [(64, 768), (64, 768), (2, 768)]
        self.assertEqual([tuple(t.shape) for t in factory()], expected)
        self.assertEqual([tuple(t.shape) for t in factory()], expected)

    def test_contract(self):
        attn = self.attention()
        x = torch.ones(130, 256, dtype=torch.bfloat16)
        rope = torch.zeros(1, 130, 1, 32, 2, 2)
        kw = dict(topk_ratio=.2, tau=0., scale=None, tail=False,
                  sink_blocks=[0, 1], sink_q=[1, 2], token_aug=64)

        def kernel(factory, **got):
            self.assertIsNone(got["kmean"])
            self.assertIsNone(got["vscale"])
            self.assertIs(got["rope_freqs"], rope)
            self.assertIs(got["qk_norm_weights"][0], attn.q_norm.weight)
            self.assertEqual(got["rope_eps"], 1e-6)
            for key, value in kw.items():
                self.assertEqual(got[key], value)
            for _ in range(2):
                chunks = list(factory())
                self.assertEqual(sum(c.shape[0] for c in chunks), 130)
                self.assertTrue(all(c.shape[1] == 768 for c in chunks))
            return torch.zeros(1, 130, 2, 128, dtype=torch.bfloat16), None, None

        out = ch.run_chunked(attn, x, rope, kw, kernel=kernel, cast=lambda w: w)
        self.assertEqual(tuple(out.shape), (130, 256))

    def test_previous_stats_are_forwarded_and_kept(self):
        attn = self.attention()
        x = torch.ones(130, 256, dtype=torch.bfloat16)
        rope = torch.zeros(1, 130, 1, 32, 2, 2)
        seen = {}
        nxt = (torch.ones(2, 128), torch.full((2, 128), 2.))

        def kernel(factory, **got):
            seen["kmean"] = got["kmean"]
            seen["vscale"] = got["vscale"]
            return torch.zeros(1, 130, 2, 128, dtype=torch.bfloat16), nxt[0], nxt[1]

        kept = {}
        ch.run_chunked(attn, x, rope, {}, kernel=kernel, cast=lambda w: w,
                       kmean=nxt[0], vscale=nxt[1],
                       on_stats=lambda k, v: kept.update(kmean=k, vscale=v))
        self.assertIs(seen["kmean"], nxt[0])
        self.assertIs(seen["vscale"], nxt[1])
        self.assertIs(kept["kmean"], nxt[0])

    def test_non_owner_step_uses_native_attention(self):
        native = Mock(return_value="native")
        state = sparse_state(dense_steps=frozenset())
        call = ch.make_chunked_attention(native, state)
        x = torch.ones(130, 256, dtype=torch.bfloat16)
        with patch.object(ch, "run_chunked") as run:
            self.assertEqual(call(x, transformer_options=step_options(state, dense=True)), "native")
            run.assert_not_called()
        native.assert_called_once()

    def test_failure_after_launch_propagates(self):
        kernel = Mock(side_effect=RuntimeError("launch failed"))
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            ch.run_chunked(self.attention(), torch.ones(130, 256), None, {},
                           kernel=kernel, cast=lambda w: w)
        kernel.assert_called_once()
