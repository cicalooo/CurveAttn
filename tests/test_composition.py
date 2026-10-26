"""How CurveAttn Sparse and CurveAttn Residual Cache stack on one MODEL."""
import types
import unittest
from unittest.mock import patch
from uuid import UUID

import torch

from comfy.patcher_extension import CallbacksMP, WrapperExecutor, WrappersMP
from curvattn_test_package import blocks, common
from curvattn_test_package import cache as cc
from curvattn_test_package import sparse as sp
from curvattn_test_package.nodes import CurveAttnCache, CurveAttnSparse
from helpers import SCHEDULE, dit, h3_patcher, kitchen, options

OVERRIDE = common.OVERRIDE_KEY
UID = UUID("00000000-0000-0000-0000-000000000001")
LAYOUT = types.SimpleNamespace(segments=((0, 2, "text"), (2, 4, "audio"), (4, 6, "video")))
REF_LAYOUTS = {name: types.SimpleNamespace(segments=segments) for name, segments in {
    "ref_img": ((0, 2, "text"), (2, 4, "ref_img"), (4, 6, "audio"), (6, 10, "video")),
    "ref_audio": ((0, 2, "text"), (2, 4, "ref_audio"), (4, 6, "audio"), (6, 10, "video")),
    "keyframes": ((0, 2, "text"), (2, 4, "cond"), (4, 5, "cond_audio"), (5, 6, "audio"),
                  (6, 10, "video")),
    "all": ((0, 1, "text"), (1, 2, "cond"), (2, 3, "cond_audio"), (3, 4, "ref_audio"),
            (4, 5, "ref_img"), (5, 6, "ref_img"), (6, 7, "audio"), (7, 10, "video")),
}.items()}


def chain(model):
    return list(common.override_chain(options(model).get(OVERRIDE)))


def prepare(model, rounds=2):
    """Run ON_PREPARE_STATE like the sampler does, on a copy of model_options."""
    model_options = {"transformer_options": dict(options(model))}
    for _ in range(rounds):
        for callback in model.get_all_callbacks(CallbacksMP.ON_PREPARE_STATE):
            callback(model, None, model_options)
    return model_options


class StackingTests(unittest.TestCase):
    def test_sparse_then_cache(self):
        with kitchen():
            model = cc.apply_cache(sp.apply_sparse(h3_patcher()))
        self.assertEqual(len(chain(model)), 1)
        dispatch = dit(model)
        self.assertEqual(set(dispatch), {("double_block", i) for i in range(4)})
        self.assertTrue(all(isinstance(d, blocks.CurveBlockDispatch) and d.cache is not None
                            and d.providers == () for d in dispatch.values()))

    def test_cache_and_chunked_sparse_compose_in_either_order(self):
        with kitchen():
            a = sp.apply_sparse(cc.apply_cache(h3_patcher()), qkv_mode="chunked")
            b = cc.apply_cache(sp.apply_sparse(h3_patcher(), qkv_mode="chunked"))
        for model in (a, b):
            with self.subTest(model=model):
                values = list(dit(model).values())
                self.assertEqual(len(values), 4)
                self.assertTrue(all(v.cache is not None and len(v.providers) == 1 for v in values))
                self.assertEqual(len({id(v.cache) for v in values}), 1)
                prepare(model)  # verify() accepts the shared dispatcher

    def test_cache_only_needs_no_sol_attn_and_no_override(self):
        with patch.object(sp.ck, "sol_attn_is_available", return_value=False) as probe:
            model = cc.apply_cache(h3_patcher())
        probe.assert_not_called()
        self.assertNotIn(OVERRIDE, options(model))
        self.assertEqual(len(dit(model)), 4)
        prepare(model)

    def test_cache_works_behind_model_attention_backend(self):
        from comfy.ldm.modules.attention import get_attention_function
        base = h3_patcher()
        base.set_model_optimized_attention(get_attention_function("pytorch"))
        model = cc.apply_cache(base)
        self.assertIs(options(model)[OVERRIDE], options(base)[OVERRIDE])

    def test_sparse_twice_stacks_and_reinstall_is_idempotent(self):
        with kitchen():
            first = sp.apply_sparse(h3_patcher())
            second = sp.apply_sparse(first, selection="sol-attn")
        stack = chain(second)
        self.assertEqual(len(stack), 2)
        top, below = stack
        self.assertEqual(top._curve_state.selection, "sol-attn")
        self.assertIs(below, options(first)[OVERRIDE])
        model_options = prepare(second, rounds=3)
        self.assertIs(model_options["transformer_options"][OVERRIDE], top)
        self.assertEqual(len(chain(first)), 1)

    def test_reinstall_goes_back_on_top_of_a_later_dense_backend(self):
        from comfy.ldm.modules.attention import get_attention_function
        with kitchen():
            model = sp.apply_sparse(h3_patcher())
        model.set_model_optimized_attention(get_attention_function("pytorch"))
        installed = prepare(model)["transformer_options"][OVERRIDE]
        self.assertTrue(sp.is_curve_override(installed))
        self.assertTrue(sp.is_dense_backend_override(installed._curve_previous))

    def run_attention(self, top, to):
        qkv = [torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16) for _ in range(3)]
        with patch.object(sp, "_ineligible", return_value=None), \
             patch.object(sp, "sol_attn_zerocopy", side_effect=lambda q, k, v, **kw: v):
            return top(None, *qkv, 2, skip_reshape=True, transformer_options=to)

    def test_later_sparse_owns_overlapping_steps_earlier_fills_its_dense_steps(self):
        with kitchen() as backend:
            first = sp.apply_sparse(h3_patcher(), dense_steps="", min_tokens=0)
            second = sp.apply_sparse(first, selection="sol-attn", dense_steps="1", min_tokens=0)
        top = options(second)[OVERRIDE]
        a, b = top._curve_previous._curve_state, top._curve_state
        for step, owner in ((0, b), (1, a)):
            to = {"sample_sigmas": SCHEDULE, "sigmas": SCHEDULE[step]}
            for state in (a, b):  # wrapper order must not matter
                to = sp.publish(to, state, state.decide(to))
            self.assertEqual(to[common.NS]["owner"], owner.uid)
            self.run_attention(top, to)
        self.assertEqual((a.sparse_calls, b.sparse_calls), (1, 1))
        self.assertEqual((a.dense_calls, b.dense_calls), (0, 0))
        backend.assert_not_called()

    def test_all_dense_step_reaches_the_bottom_dense_backend_once(self):
        with kitchen() as backend:
            first = sp.apply_sparse(h3_patcher(), dense_steps="0-0")
            second = sp.apply_sparse(first, dense_steps="0-0")
        top = options(second)[OVERRIDE]
        a, b = top._curve_previous._curve_state, top._curve_state
        to = {"sample_sigmas": SCHEDULE, "sigmas": SCHEDULE[0]}
        for state in (b, a):
            to = sp.publish(to, state, state.decide(to))
        self.assertEqual(self.run_attention(top, to), "dense")
        self.assertEqual(backend.call_count, 1)
        self.assertEqual((a.dense_calls, b.dense_calls), (1, 1))

    def test_no_duplicate_wrappers_or_limiters(self):
        with kitchen():
            model = cc.apply_cache(sp.apply_sparse(sp.apply_sparse(h3_patcher())))
        self.assertEqual(len(model.get_wrappers(WrappersMP.CALC_COND_BATCH, common.COND_BATCH_KEY)), 1)
        self.assertEqual(len(model.get_all_wrappers(WrappersMP.CALC_COND_BATCH)), 1)
        for kind in (WrappersMP.DIFFUSION_MODEL, WrappersMP.OUTER_SAMPLE):
            per_key = model.wrappers[kind]
            self.assertEqual(len(per_key), 3)
            self.assertTrue(all(len(v) == 1 for v in per_key.values()))
        self.assertEqual(len(model.get_all_callbacks(CallbacksMP.ON_PREPARE_STATE)), 3)

    def test_clone_isolation(self):
        base = h3_patcher()
        with kitchen():
            chunked = sp.apply_sparse(base, qkv_mode="chunked")
            cached = cc.apply_cache(chunked)
        self.assertNotIn(OVERRIDE, options(base))
        self.assertEqual(dit(base), {})
        self.assertEqual(base.wrappers, {})
        self.assertTrue(all(v.cache is None for v in dit(chunked).values()))
        self.assertTrue(all(v.cache is not None for v in dit(cached).values()))
        self.assertNotIn(cc.KEY, chunked.wrappers.get(WrappersMP.DIFFUSION_MODEL, {}))


class RejectionTests(unittest.TestCase):
    def test_foreign_override_rejected_before_device_probe(self):
        model = h3_patcher()
        options(model)[OVERRIDE] = lambda func, *a, **k: None
        with patch.object(sp.ck, "sol_attn_is_available") as probe:
            with self.assertRaisesRegex(ValueError, "non-CurveAttn attention patch"):
                sp.apply_sparse(model)
            probe.assert_not_called()

    def test_foreign_override_installed_later_is_caught_at_prepare(self):
        with kitchen():
            model = sp.apply_sparse(h3_patcher())
        options(model)[OVERRIDE] = lambda func, *a, **k: None
        with self.assertRaisesRegex(ValueError, "non-CurveAttn attention patch"):
            prepare(model)

    def test_foreign_dit_patch_rejected_by_both_nodes(self):
        for apply in (sp.apply_sparse, cc.apply_cache):
            model = h3_patcher()
            model.set_model_patch_replace(lambda *a, **k: None, "dit", "double_block", 0)
            with self.subTest(apply=apply.__name__), \
                 patch.object(sp.ck, "sol_attn_is_available") as probe, \
                 self.assertRaisesRegex(ValueError, "dit block replacements"):
                apply(model)
            probe.assert_not_called()

    def test_foreign_dit_patch_installed_later_is_caught_at_prepare(self):
        model = cc.apply_cache(h3_patcher())
        model.set_model_patch_replace(lambda *a, **k: None, "dit", "double_block", 1)
        with self.assertRaisesRegex(ValueError, "replaced CurveAttn dit patches"):
            prepare(model)

    def test_conflicting_object_patches(self):
        model = h3_patcher()
        model.add_object_patch("diffusion_model.blocks.0.attn.qkv_proj", object())
        with kitchen():
            self.assertIsNot(sp.apply_sparse(model), model)
        model.add_object_patch("diffusion_model._forward", object())
        for apply in (sp.apply_sparse, cc.apply_cache):
            with self.subTest(apply=apply.__name__), kitchen(), \
                 self.assertRaisesRegex(ValueError, "conflicting H3 object patch"):
                apply(model)

    def test_second_cache_rejected(self):
        with self.assertRaisesRegex(ValueError, "already has a CurveAttn residual cache"):
            cc.apply_cache(cc.apply_cache(h3_patcher()))

    def test_whole_cache_and_chunked_rejected_in_either_order(self):
        with kitchen():
            chunked = sp.apply_sparse(h3_patcher(), qkv_mode="chunked")
        with self.assertRaisesRegex(ValueError, "whole span requires native"):
            cc.apply_cache(chunked, span="whole")
        whole = cc.apply_cache(h3_patcher(), span="whole")
        with patch.object(sp.ck, "sol_attn_is_available") as probe:
            with self.assertRaisesRegex(ValueError, "requires native qkv_mode"):
                sp.apply_sparse(whole, qkv_mode="chunked")
            probe.assert_not_called()

    def test_cache_needs_three_blocks(self):
        with self.assertRaisesRegex(ValueError, "three"):
            cc.apply_cache(h3_patcher(n_blocks=2))


class RunTests(unittest.TestCase):
    """Drive the real DIFFUSION_MODEL wrapper chain over the shared dispatchers."""

    @torch.no_grad()
    def sample(self, model, steps, sigmas=None, layouts=None, payload=None, rows=6):
        """steps index SCHEDULE; sigmas (if given) are raw per-call sigmas instead.
        layouts optionally gives one layout per call; blocks add a row-dependent
        residual so a hit that misses any row would change the output."""
        dispatch = [d for _, d in sorted(dit(model).items(), key=lambda kv: kv[0][1])]
        calls = []
        row = torch.arange(float(rows)).unsqueeze(1)
        sigmas = [SCHEDULE[step] for step in steps] if sigmas is None else list(sigmas)
        layouts = [LAYOUT] * len(sigmas) if layouts is None else list(layouts)

        def forward(x, timestep, context, transformer_options=None, minimax_payload=None, **kw):
            layout = transformer_options["layout_for_test"]
            transformer_options["minimax_h3_layout"] = layout
            img = torch.ones(rows, 4)
            for index, d in enumerate(dispatch):
                def block(args, index=index):
                    calls.append(index)
                    return {"img": args["img"].add_(row * (index + 1) + 1)}
                img = d(dict(img=img, layout=layout, transformer_options=transformer_options),
                        {"original_block": block})["img"]
            return img

        executor = WrapperExecutor.new_executor(
            forward, model.get_all_wrappers(WrappersMP.DIFFUSION_MODEL))
        outer = WrapperExecutor.new_executor(
            lambda *a, **k: [executor.execute(
                None, torch.tensor([sigma * 1000.]), None, transformer_options=dict(
                    sample_sigmas=SCHEDULE, sigmas=sigma, cond_or_uncond=[0], uuids=[UID],
                    layout_for_test=layout), minimax_payload=payload or {})
                for sigma, layout in zip(sigmas, layouts)],
            model.get_all_wrappers(WrappersMP.OUTER_SAMPLE))
        cache = blocks.current_cache(model.model_options)
        seen = {}
        original = cache.run_scope.on_end
        with patch.object(cache.run_scope, "on_end",
                          lambda run: seen.update(hits=run.cache.hits, substeps=run.cache.substeps,
                                                  bypasses=run.cache.bypasses) or original(run)):
            outputs = outer.execute(None, None, None, SCHEDULE)
        self.seen = seen
        return outputs, calls, seen["hits"]

    def test_cache_hits_with_sparse_in_either_order(self):
        with kitchen():
            orders = [cc.apply_cache(sp.apply_sparse(h3_patcher(), dense_steps="")),
                      sp.apply_sparse(cc.apply_cache(h3_patcher()), dense_steps="")]
        for model in orders:
            with self.subTest(model=model):
                outputs, calls, hits = self.sample(model, (2, 3))
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                self.assertEqual(calls, [0, 1, 2, 3, 0, 3])
                self.assertEqual(hits, 1)

    def test_dense_sparse_step_refreshes_instead_of_hitting(self):
        with kitchen():
            model = cc.apply_cache(sp.apply_sparse(h3_patcher(), dense_steps="3"))
        _, calls, hits = self.sample(model, (2, 3))
        self.assertEqual(hits, 0)
        self.assertEqual(calls, [0, 1, 2, 3] * 2)

    def test_cache_only_on_dense_model_hits(self):
        _, calls, hits = self.sample(cc.apply_cache(h3_patcher()), (2, 3))
        self.assertEqual(hits, 1)

    def test_reference_and_keyframe_layouts_hit_end_to_end(self):
        for name, layout in REF_LAYOUTS.items():
            for sparse_first in (False, True):
                with self.subTest(layout=name, sparse_first=sparse_first), kitchen():
                    model = (cc.apply_cache(sp.apply_sparse(h3_patcher(), dense_steps="",
                                                            min_tokens=0))
                             if sparse_first else cc.apply_cache(h3_patcher()))
                    outputs, calls, hits = self.sample(model, (2, 3), layouts=[layout] * 2, rows=10)
                    self.assertEqual((hits, calls), (1, [0, 1, 2, 3, 0, 3]))
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_changed_reference_set_mid_run_refreshes(self):
        model = cc.apply_cache(h3_patcher())
        layouts = [REF_LAYOUTS["ref_img"], REF_LAYOUTS["keyframes"], REF_LAYOUTS["keyframes"]]
        _, calls, hits = self.sample(model, (2, 3, 4), layouts=layouts, rows=10)
        self.assertEqual(hits, 1)
        self.assertEqual(calls, [0, 1, 2, 3] * 2 + [0, 3])

    def test_conditioning_pin_crossing_forces_a_refresh(self):
        # aug .3: at step 2 (sigma .75, t_v .25) ref rows are pinned at .3; at
        # step 3 (t_v .375) they track t_v, so step 3 must not reuse step 2.
        payload = {"visual_cond_noise_aug": .3}
        _, _, hits = self.sample(cc.apply_cache(h3_patcher()), (2, 3),
                                 layouts=[REF_LAYOUTS["ref_img"]] * 2, payload=payload, rows=10)
        self.assertEqual(hits, 0)
        _, _, hits = self.sample(cc.apply_cache(h3_patcher()), (3, 4),
                                 layouts=[REF_LAYOUTS["ref_img"]] * 2, payload=payload, rows=10)
        self.assertEqual(hits, 1)
        # no visual conditioning rows: the aug value is irrelevant
        _, _, hits = self.sample(cc.apply_cache(h3_patcher()), (2, 3), payload=payload)
        self.assertEqual(hits, 1)

    def test_multistage_substeps_compute_fully_and_main_steps_still_hit(self):
        # res_2s / dpmpp_2s style: step, midpoint, step, midpoint
        sigmas = [SCHEDULE[2], .7, SCHEDULE[3], .55, SCHEDULE[4], .45]
        with kitchen():
            model = cc.apply_cache(sp.apply_sparse(h3_patcher(), dense_steps="", min_tokens=0),
                                   max_hits=2)
        outputs, calls, hits = self.sample(model, (), sigmas=sigmas,
                                           layouts=[REF_LAYOUTS["all"]] * 6, rows=10)
        self.assertEqual(hits, 2)
        self.assertEqual(self.seen["substeps"], 3)
        self.assertEqual(self.seen["bypasses"], 0)
        full = [0, 1, 2, 3]
        self.assertEqual(calls, full + full + [0, 3] + full + [0, 3] + full)
        for out in outputs[1:]:
            torch.testing.assert_close(out, outputs[0], rtol=0, atol=0)

    def test_off_schedule_sigmas_still_bypass(self):
        model = cc.apply_cache(h3_patcher())
        _, calls, hits = self.sample(model, (), sigmas=[SCHEDULE[2], 1.5, SCHEDULE[3]])
        self.assertEqual((hits, self.seen["bypasses"]), (0, 1))


class NodeExecuteTests(unittest.TestCase):
    def test_sparse_dynamic_combo_values(self):
        with patch("curvattn_test_package.nodes.apply_sparse", return_value="m") as fn:
            CurveAttnSparse.execute("model", {"selection": "sol-attn", "tau": 2.0, "tau_end": 1.0},
                                    qkv_mode="chunked")
        kw = fn.call_args.kwargs
        self.assertEqual((kw["selection"], kw["tau"], kw["tau_end"], kw["qkv_mode"]),
                         ("sol-attn", 2.0, 1.0, "chunked"))
        self.assertEqual(kw["keep_percent"], 20.0)
        with patch("curvattn_test_package.nodes.apply_sparse", return_value="m") as fn:
            CurveAttnSparse.execute("model", {"selection": "sla", "keep_percent": 30.0,
                                              "keep_percent_end": 10.0})
        self.assertEqual((fn.call_args.kwargs["keep_percent"],
                          fn.call_args.kwargs["keep_percent_end"]), (30.0, 10.0))

    def test_cache_dynamic_combo_values(self):
        with patch("curvattn_test_package.nodes.apply_cache", return_value="m") as fn:
            CurveAttnCache.execute("model", {"preset": "native"}, max_mib=2048)
        self.assertEqual(fn.call_args.kwargs, dict(preset="native", max_mib=2048))
        with patch("curvattn_test_package.nodes.apply_cache", return_value="m") as fn:
            CurveAttnCache.execute("model", {"preset": "manual", "threshold": 0.02,
                                             "span": "whole"})
        self.assertEqual(fn.call_args.kwargs["threshold"], 0.02)
        self.assertEqual(fn.call_args.kwargs["span"], "whole")

    def test_schema_defaults_match_apply_defaults(self):
        import inspect
        schema = CurveAttnSparse.define_schema()
        signature = inspect.signature(sp.apply_sparse).parameters
        for entry in schema.inputs:
            if entry.id in ("model", "selection"):
                continue
            self.assertEqual(entry.default, signature[entry.id].default, entry.id)
        nested = {}
        for option in schema.inputs[1].options:
            for widget in option.inputs:
                nested.setdefault(widget.id, set()).add(widget.default)
        for name, defaults in nested.items():
            self.assertIn(signature[name].default, defaults, name)
        self.assertIn(10.0, nested["keep_percent"])
        manual = next(o for o in CurveAttnCache.define_schema().inputs[1].options
                      if o.key == "manual")
        self.assertEqual({i.id: i.default for i in manual.inputs}, cc.MANUAL_DEFAULTS)
        cache_sig = inspect.signature(cc.apply_cache).parameters
        for name, value in cc.MANUAL_DEFAULTS.items():
            self.assertEqual(cache_sig[name].default, value, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
