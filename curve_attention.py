"""MiniMax H3 sparse attention through stock Comfy Kitchen.

Native model math is retained. Native QKV views avoid local owned-BF16 copies;
optional residual caching is branch-scoped and approximate.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager, nullcontext
from functools import partial

import comfy_kitchen as ck
import torch

from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.modules.attention import AttentionTensorContainer, get_attention_function
from comfy.model_prefetch import pause_malloc_graph
from comfy.patcher_extension import CallbacksMP, WrappersMP

from . import __version__
from .block_dispatch import make_block_dispatch
from .config import apply_cache_preset, validate_options
from .residual_cache import format_cache_report
from .runtime import RunScope, cache_metadata, exact_step, limit_h3_cond_batch, schedule_tuple
from .sol_attn import check_prepared_available, consume_sla, prepare_sla, sol_attn_zerocopy

_HEAD_DIM = 128
_BLOCK = 64
_LOG = logging.getLogger("CurveAttention")
_DENSE_STEPS_TOKEN = re.compile(r"\d+\s*-\s*\d+|\d+")
_DENSE_STEPS_SEPARATORS = re.compile(r"[\s,;]+")


def _parse_steps(text: str) -> frozenset[int]:
    """Parse dense step specs like '1', '0-1', '0,2'. Reject leftover junk."""
    text = text or ""
    remainder = _DENSE_STEPS_TOKEN.sub("", text)
    remainder = _DENSE_STEPS_SEPARATORS.sub("", remainder)
    if remainder:
        raise ValueError(
            f"Invalid dense_steps {text!r}: unexpected characters {remainder!r} "
            "(only digits, ranges like 0-1, whitespace, commas, and semicolons are allowed)"
        )
    steps = set()
    for part in _DENSE_STEPS_TOKEN.findall(text):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            steps.update(range(min(a, b), max(a, b) + 1))
        else:
            steps.add(int(part))
    return frozenset(steps)


def _is_fatal_device_error(exc) -> bool:
    """True for CUDA/device-side failures that must not fall through to dense."""
    cuda_error = getattr(torch.cuda, "CudaError", None)
    if cuda_error is not None and isinstance(exc, cuda_error):
        return True
    type_name = type(exc).__name__
    if "Cuda" in type_name or "CUDA" in type_name:
        return True
    message = str(exc).lower()
    if "cuda error" in message or "illegal memory access" in message:
        return True
    return False


def _scalar(value) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return _scalar(value[0]) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lerp_schedule(start: float, end: float, t: float) -> float:
    """Linear interpolate start→end by progress t in [0, 1]."""
    return start + (end - start) * t


def schedule_progress(step0: int | None, n_steps: int) -> float:
    """Step progress for keep/tau schedules. 0 when step unknown or n_steps<=1."""
    if step0 is None or n_steps <= 1:
        return 0.0
    return float(step0) / float(n_steps - 1)


def outside_sigma_window(sigma: float | None, sigma_start: float | None,
                         sigma_end: float | None) -> bool:
    """Dense outside [sigma_end, sigma_start] (decreasing-sigma sense).

    Same rule as core Model Sparse Attention: dense when
    sigma > sigma_start or sigma < sigma_end. Returns False when any bound
    or sigma is unavailable so defaults do not force extra dense steps.
    """
    if sigma is None or sigma_start is None or sigma_end is None:
        return False
    return sigma > sigma_start or sigma < sigma_end


def resolve_sigma_bounds(model, start_percent: float, end_percent: float):
    """Map sampling percents to sigma via model_sampling when available."""
    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        return None, None
    if model_sampling is None or not hasattr(model_sampling, "percent_to_sigma"):
        return None, None
    try:
        sigma_start = float(model_sampling.percent_to_sigma(float(start_percent)))
        sigma_end = float(model_sampling.percent_to_sigma(float(end_percent)))
        return sigma_start, sigma_end
    except Exception as exc:
        _LOG.warning("CurveAttention: percent_to_sigma failed (%s); sigma window disabled", exc)
        return None, None


def resolve_step(transformer_options) -> tuple[int | None, int]:
    """Step index from sample_sigmas. Same exact tolerance as the cache clock."""
    schedule = schedule_tuple(transformer_options.get("sample_sigmas"))
    if not schedule:
        return None, 0
    return exact_step(schedule, transformer_options.get("sigmas"))


def forward_options(options, cache_enabled, cache_span):
    result = dict(options or {})
    if cache_enabled and cache_span == "whole":
        result["prefetch_dynamic_vbars"] = False
    return result


def cache_policy_key(state, dense, video_shift, audio_shift):
    """Identity of a residual. Excludes per-step lerped keep/tau.

    Those scalars change every step when a schedule is armed. Putting them in
    the signature deletes the anchor each step, so the cache can never hit.
    """
    return (
        state.selection, state.tail, state.token_aug, state.protect_prefix,
        state.protect_audio, state.dense_backend, state.qkv_mode,
        state.disable_fp16_accum, dense, video_shift, audio_shift,
        state.cache_span, state.cache_mode, state.cache_start_percent,
        state.cache_end_percent, state.cache_min_steps,
        state.cache_temporal_guard, state.cache_audio_guard,
    )


def _dit_replacements(model_options) -> bool:
    to = model_options.get("transformer_options") or {}
    dit = (to.get("patches_replace") or {}).get("dit")
    return bool(dit)


def _conflicting_h3_object_patch(key: str) -> bool:
    """True only for a replaced H3 forward or attention module, not Linear weights."""
    return key == "diffusion_model._forward" or key.endswith(".attn") or key.endswith(".attn.forward")


class CurveAttnState:
    def __init__(self, keep_ratio, min_tokens, dense_steps, dense_last_steps,
                 tail, protect_audio, verbose, dense_attention,
                 protect_prefix=False, disable_fp16_accum=True, prepared_available=False,
                 selection="sla", tau=1.3, token_aug=0,
                 sigma_start=None, sigma_end=None,
                 keep_ratio_end=None, tau_end=None,
                 dense_backend="comfy_kitchen_int8", qkv_mode="native",
                 cache_enabled=False, run_scope=None,
                 cache_span="middle", cache_mode="fingerprint",
                 cache_start_percent=0.2, cache_end_percent=0.8,
                 cache_min_steps=6, cache_temporal_guard=False,
                 cache_audio_guard=False, cache_preset="manual"):
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.dense_steps = dense_steps
        self.dense_last_steps = dense_last_steps
        self.tail = tail
        self.protect_audio = protect_audio
        self.protect_prefix = protect_prefix
        self.disable_fp16_accum = disable_fp16_accum
        self.verbose = verbose
        self.dense_attention = dense_attention
        self.prepared_available = prepared_available
        self.selection = selection
        self.tau = float(tau)
        self.token_aug = int(token_aug)
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.keep_ratio_end = keep_ratio_end
        self.tau_end = tau_end
        self.dense_backend = dense_backend
        self.qkv_mode = qkv_mode
        self.cache_enabled = bool(cache_enabled)
        self.run_scope = run_scope
        self.cache_span = str(cache_span or "middle")
        self.cache_mode = str(cache_mode or "fingerprint")
        self.cache_start_percent = float(cache_start_percent)
        self.cache_end_percent = float(cache_end_percent)
        self.cache_min_steps = int(cache_min_steps)
        self.cache_temporal_guard = bool(cache_temporal_guard)
        self.cache_audio_guard = bool(cache_audio_guard)
        self.cache_preset = str(cache_preset or "manual")
        self.last_cache_block = None
        self.attention_override = None
        self.previous_override = None
        self.reset()

    def reset(self):
        self._logged = set()
        self.sparse_calls = 0
        self.dense_calls = 0
        self.failure_reason = None
        self.prepared_enabled = self.prepared_available
        self.last_reason = None
        self.current_topk = self.keep_ratio
        self.current_tau = self.tau
        self.chunked_calls = 0
        self.cache_hits = 0
        self.cache_refreshes = 0
        self.cache_bypasses = 0
        self.cache_bytes = 0
        self.last_cache_block = None

    def log_once(self, key, message):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            _LOG.info("CurveAttention: %s", message)

    def warn_once(self, key, message):
        if key not in self._logged:
            self._logged.add(key)
            _LOG.warning("CurveAttention: %s", message)


def h3_sinks(transformer_options, tokens, protect_audio: bool, protect_prefix=False):
    """Optional native protections; both off matches the supplied SLA run."""
    if not protect_audio and not protect_prefix:
        return (0, 0), (0, 0)
    layout = transformer_options.get("minimax_h3_layout")
    if layout is None or getattr(layout, "seq_len", None) != tokens:
        return None
    video = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
    kv = (0, (video[0] + _BLOCK - 1) // _BLOCK) if protect_prefix and video else (0, 0)
    if not protect_audio:
        return kv, (0, 0)
    audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
    rows = (audio[0] // _BLOCK, (audio[1] + _BLOCK - 1) // _BLOCK) if audio else (0, 0)
    return kv, rows


def _ineligible(q, k, v, dim_head):
    if q.device.type != "cuda":
        return "not on CUDA"
    if not ck.sol_attn_is_available(q.device):
        return "no compiled sol_attn kernel for this GPU"
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {q.dtype}"
    if dim_head != _HEAD_DIM:
        return f"head_dim {dim_head} != {_HEAD_DIM}"
    if q.shape != k.shape or q.shape != v.shape:
        return "cross-attention or GQA"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "mixed dtypes"
    return None


def _prepare_owned(q, k, v, heads, skip_reshape, **options):
    """Consume the H3 containers in a separate frame from attention output.

    Restore ownership on preparation failure so the native sparse fallback
    can still use the original inputs. Success retains owned BF16 Q/K/V; an optional upstream adapter owns its own format.
    """
    tensors = [container.take() for container in (q, k, v)]
    try:
        if skip_reshape:
            views = [tensor.transpose(1, 2) for tensor in tensors]
        else:
            views = [tensor.view(tensor.shape[0], -1, heads, tensor.shape[-1] // heads)
                     for tensor in tensors]
        views = [tensor.to(torch.bfloat16) for tensor in views]
        return prepare_sla(*views, **options)
    except Exception:
        for container, tensor in zip((q, k, v), tensors):
            container.tensor = tensor
        raise


def _run_dense(state, q, k, v, heads, mask, attn_precision, skip_reshape,
               skip_output_reshape, kwargs, reason=None):
    """Kitchen dense path with container handoff preserved."""
    state.dense_calls += 1
    if reason:
        state.last_reason = reason
        state.log_once(("dense", reason), f"dense: {reason}")
    args = (q, k, v, heads)
    kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
              skip_output_reshape=skip_output_reshape, **kwargs)
    # Bypass override recursion, but retain the backend's container
    # handoff so scheduled dense steps can release float QKV early.
    kw["_inside_attn_wrapper"] = True
    previous = state.previous_override
    if previous is not None:
        return previous(state.dense_attention, *args, **kw)
    return state.dense_attention(*args, **kw)


def _consume_prepared(state, prepared, input_dtype, skip_output_reshape,
                      b, heads, dim_head, tokens, sink, sink_q):
    """Finish prepared-Sol consumption after QKV ownership has been released."""
    # QKV is now gone. A consumption/device failure cannot fall back
    # to dense using inputs that have already been released.
    out = consume_sla(prepared)
    selector = prepared.selector
    del prepared
    if input_dtype != torch.bfloat16:
        out = out.to(input_dtype)
    state.sparse_calls += 1
    state.log_once(
        ("prepared", tokens, sink, sink_q),
        f"sparse ({b}, {tokens}, {heads}, {dim_head}) sinks {sink}/{sink_q} "
        f"selector={selector} prepared_qkv=True block=64",
    )
    return out.transpose(1, 2) if skip_output_reshape else out.reshape(b, -1, heads * dim_head)


def _prepare_or_disable(state, q, k, v, heads, skip_reshape, b, dim_head, options):
    """Try prepared ownership handoff; restore views on recoverable failure.

    Caller must already have dropped borrowed input views. Returns
    (prepared, qs, ks, vs) where prepared is set on success and qs/ks/vs are
    set only after a recoverable fallback that re-peeks the containers.
    """
    try:
        return _prepare_owned(q, k, v, heads, skip_reshape, **options), None, None, None
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:
        if _is_fatal_device_error(exc):
            raise
        state.prepared_enabled = False
        _LOG.warning(
            "CurveAttention: prepared backend unavailable (%s); native SLA for this run",
            exc,
        )
        inputs = [tensor.peek() for tensor in (q, k, v)]
        if skip_reshape:
            qs, ks, vs = (tensor.transpose(1, 2) for tensor in inputs)
        else:
            qs, ks, vs = (tensor.view(b, -1, heads, dim_head) for tensor in inputs)
        return None, qs, ks, vs


def _sparse_options(state, scale, sink, sink_q):
    """Build Kitchen kwargs for the active selection and scheduled values."""
    if state.selection == "sol-attn":
        topk_ratio = 0.0
        tau = float(state.current_tau)
    else:
        topk_ratio = float(state.current_topk)
        tau = 0.0
    return dict(
        topk_ratio=topk_ratio,
        tau=tau,
        scale=scale,
        sink_blocks=list(sink),
        sink_q=list(sink_q),
        tail=state.tail,
        token_aug=int(state.token_aug),
    )


def _run_native_sparse(state, q, k, v, qs, ks, vs, heads, skip_reshape, skip_output_reshape,
                       b, dim_head, sink, sink_q, options, input_dtype, owned, dense):
    """Native sol_attn sparse path with recoverable failure → dense."""
    if input_dtype != torch.bfloat16:
        qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
    try:
        out = sol_attn_zerocopy(
            qs, ks, vs,
            **options,
        )
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        # A device-side CUDA error cannot safely fall through to another
        # kernel. Ordinary unsupported-operation failures can stay dense.
        if _is_fatal_device_error(exc):
            raise
        state.failure_reason = f"sol_attn failed: {exc}"
        _LOG.warning("CurveAttention: %s; using Kitchen dense for this run", state.failure_reason)
        return dense(state.failure_reason)
    if owned:
        for container in (q, k, v):
            container.take()
    if input_dtype != torch.bfloat16:
        out = out.to(input_dtype)
    state.sparse_calls += 1
    state.log_once(
        ("sparse", tuple(qs.shape), sink, sink_q),
        f"sparse {tuple(qs.shape)} sinks {sink}/{sink_q} selector=native block=64",
    )
    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(b, -1, heads * dim_head)


def make_attention_override(state: CurveAttnState):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}

        def dense(reason=None):
            return _run_dense(
                state, q, k, v, heads, mask, attn_precision, skip_reshape,
                skip_output_reshape, kwargs, reason=reason,
            )

        if mask is not None:
            return dense("attention mask present")
        if transformer_options.get("_curve_dense"):
            return dense(transformer_options.get("_curve_dense_reason") or "scheduled dense step")
        if state.failure_reason is not None:
            return dense(state.failure_reason)
        owned = isinstance(q, AttentionTensorContainer)
        inputs = [tensor.peek() for tensor in (q, k, v)] if owned else [q, k, v]
        tokens = inputs[0].shape[2] if skip_reshape else inputs[0].shape[1]
        if tokens < state.min_tokens:
            del inputs
            return dense(f"{tokens} tokens < min_tokens {state.min_tokens}")
        if skip_reshape:
            b, _, _, dim_head = inputs[0].shape
            qs, ks, vs = (t.transpose(1, 2) for t in inputs)
        else:
            b, _, dim_head = inputs[0].shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in inputs)
        reason = _ineligible(qs, ks, vs, dim_head)
        if reason is not None:
            return dense(reason)
        sinks = h3_sinks(
            transformer_options, tokens, state.protect_audio, state.protect_prefix
        )
        if sinks is None:
            return dense("H3 layout unavailable for requested protection")
        sink, sink_q = sinks
        options = _sparse_options(state, kwargs.get("scale"), sink, sink_q)
        # The accepted SLA Kitchen path uses BF16 even for FP16 inputs.
        input_dtype = qs.dtype
        prepared = None
        if owned and state.prepared_enabled:
            # The outer frame must not keep borrowed views alive while the
            # separate owner frame consumes QKV and returns prepared carriers.
            del inputs, qs, ks, vs
            prepared, qs, ks, vs = _prepare_or_disable(
                state, q, k, v, heads, skip_reshape, b, dim_head, options,
            )
        if prepared is not None:
            return _consume_prepared(
                state, prepared, input_dtype, skip_output_reshape,
                b, heads, dim_head, tokens, sink, sink_q,
            )
        return _run_native_sparse(
            state, q, k, v, qs, ks, vs, heads, skip_reshape, skip_output_reshape,
            b, dim_head, sink, sink_q, options, input_dtype, owned, dense,
        )

    override.container_function = partial(override, None)
    override._curve_attention = True
    override.container_function._curve_attention = True
    return override


def _is_curve_override(fn) -> bool:
    return bool(getattr(fn, "_curve_attention", False))


_DENSE_BACKEND_NAMES = ("pytorch", "comfy_kitchen_int8")


def _is_dense_backend_override(fn) -> bool:
    """True when the override is Model Attention Backend's one-line forwarder.

    Identifies it by the captured attention function, not by __qualname__.
    Only the dense backends this node uses as fallback are allowed to stay.
    """
    closure = getattr(fn, "__closure__", None)
    if not closure or len(closure) != 1:
        return False
    try:
        captured = closure[0].cell_contents
    except ValueError:
        return False
    from comfy.ldm.modules.attention import get_attention_function
    for name in _DENSE_BACKEND_NAMES:
        backend = get_attention_function(name, None)
        if backend is not None and captured is backend:
            return True
    return False


def install_override(state: CurveAttnState, transformer_options):
    current = transformer_options.get("optimized_attention_override")
    if current is state.attention_override and current is not None:
        return
    if _is_curve_override(current):
        raise ValueError(
            "CurveAttention: this MODEL already has CurveAttention. "
            "Wire MODEL from the loader or Model Attention Backend, not from another CurveAttention."
        )
    if current is not None and not _is_dense_backend_override(current):
        raise ValueError(
            "CurveAttention: MODEL already has another attention patch. "
            "Unwire H3 SLA, Model Sparse Attention, Kitchen Sparse, or Sage from this MODEL. "
            "Model Attention Backend can stay."
        )
    if current is not None:
        state.previous_override = current
    if state.attention_override is None:
        state.attention_override = make_attention_override(state)
    transformer_options["optimized_attention_override"] = state.attention_override


@contextmanager
def reduction_policy(disable):
    """Match SLA's reduction flags and restore them even after cancellation."""
    if not disable:
        yield
        return
    backend = torch.backends.cuda.matmul
    names = ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
    saved = {}
    try:
        for name in names:
            value = getattr(backend, name)
            split_k = getattr(backend, name + "_split_k", None)
            saved[name] = value if split_k is None else (value, split_k)
            setattr(backend, name, False)
        yield
    finally:
        for name, value in saved.items():
            setattr(backend, name, value)


def _update_schedules(state: CurveAttnState, step0: int | None, n_steps: int):
    t = schedule_progress(step0, n_steps)
    if state.keep_ratio_end is not None and state.selection == "sla":
        state.current_topk = lerp_schedule(state.keep_ratio, state.keep_ratio_end, t)
    else:
        state.current_topk = state.keep_ratio
    if state.tau_end is not None and state.selection == "sol-attn":
        state.current_tau = lerp_schedule(state.tau, state.tau_end, t)
    else:
        state.current_tau = state.tau


def make_step_wrapper(state: CurveAttnState):
    def wrapper(executor, x, timestep, context, transformer_options=None,
                minimax_payload=None, **kwargs):
        to = forward_options(transformer_options, state.cache_enabled, state.cache_span)
        scope = state.run_scope
        run = scope.current() if scope is not None else None
        if run is not None and run.schedule:
            step0, n_steps = exact_step(run.schedule, to.get("sigmas"))
        else:
            step0, n_steps = resolve_step(to)
        _update_schedules(state, step0, n_steps)
        dense, reason = False, None
        if step0 is None and (state.dense_steps or state.dense_last_steps):
            dense, reason = True, "sigma schedule unavailable"
        elif step0 is not None:
            if step0 in state.dense_steps:
                dense, reason = True, f"dense_steps contains {step0}"
            elif state.dense_last_steps > 0 and step0 >= n_steps - state.dense_last_steps:
                dense, reason = True, f"dense_last_steps ({step0}/{n_steps})"
        if not dense and outside_sigma_window(
            _scalar(to.get("sigmas")), state.sigma_start, state.sigma_end
        ):
            dense, reason = True, "outside start/end_percent"
        to["_curve_dense"] = dense
        to["_curve_dense_reason"] = reason
        policy = cache_policy_key(
            state, dense,
            to.get("minimax_h3_sigma_shift_video"),
            to.get("minimax_h3_sigma_shift_audio"))
        meta = cache_metadata(
            to, minimax_payload, kwargs,
            schedule=run.schedule if run is not None else (),
            enabled=state.cache_enabled and run is not None,
            protect_audio=state.protect_audio, policy=policy,
            min_steps=state.cache_min_steps,
            start_percent=state.cache_start_percent,
            end_percent=state.cache_end_percent)
        to["_curve_meta"] = meta
        if state.cache_enabled and meta.get("reason"):
            state.last_cache_block = meta["reason"]
            state.warn_once(("cache", meta["reason"]), "cache ineligible: " + meta["reason"])
        if "multigpu_thread_device" in to and (state.cache_enabled or state.qkv_mode == "chunked"):
            raise ValueError("CurveAttention: cache/chunked mode is not validated for multigpu")
        pause = pause_malloc_graph() if state.qkv_mode == "chunked" else nullcontext()
        try:
            with reduction_policy(state.disable_fp16_accum), pause:
                return executor(x, timestep, context, transformer_options=to,
                                minimax_payload=minimax_payload, **kwargs)
        except BaseException:
            if run is not None:
                run.cache.clear()
            raise

    return wrapper


def apply_curve_attention(
    model,
    keep_percent=20.0,
    min_tokens=12288,
    dense_steps="1",
    dense_last_steps=0,
    tail=False,
    protect_audio=False,
    verbose=False,
    protect_prefix=False,
    disable_fp16_accum=True,
    selection="sla",
    tau=1.3,
    extra_tokens=0,
    start_percent=0.0,
    end_percent=1.0,
    keep_percent_end=-1.0,
    tau_end=-1.0,
    dense_backend="comfy_kitchen_int8",
    qkv_mode="native",
    cache_enabled=False,
    cache_threshold=0.01,
    cache_max_hits=1,
    cache_max_mib=1024,
    cache_span="middle",
    cache_preset="manual",
    cache_start_percent=0.2,
    cache_end_percent=0.8,
    cache_min_steps=6,
    cache_temporal_guard=False,
    cache_audio_guard=False,
    cache_mode="fingerprint",
):
    selection = str(selection or "sla")
    cache_preset = str(cache_preset or "manual")
    resolved = apply_cache_preset(cache_preset, dict(
        cache_enabled=bool(cache_enabled),
        cache_threshold=float(cache_threshold),
        cache_max_hits=int(cache_max_hits),
        cache_max_mib=int(cache_max_mib),
        cache_span=str(cache_span or "middle"),
        cache_start_percent=float(cache_start_percent),
        cache_end_percent=float(cache_end_percent),
        cache_min_steps=int(cache_min_steps),
        cache_temporal_guard=bool(cache_temporal_guard),
        cache_audio_guard=bool(cache_audio_guard),
        cache_mode=str(cache_mode or "fingerprint"),
    ))
    cache_enabled = bool(resolved["cache_enabled"])
    cache_threshold = float(resolved["cache_threshold"])
    cache_max_hits = int(resolved["cache_max_hits"])
    cache_max_mib = int(resolved["cache_max_mib"])
    cache_span = str(resolved["cache_span"])
    cache_start_percent = float(resolved["cache_start_percent"])
    cache_end_percent = float(resolved["cache_end_percent"])
    cache_min_steps = int(resolved["cache_min_steps"])
    cache_temporal_guard = bool(resolved["cache_temporal_guard"])
    cache_audio_guard = bool(resolved["cache_audio_guard"])
    cache_mode = str(resolved["cache_mode"])
    validate_options(
        keep_percent=keep_percent, min_tokens=min_tokens,
        dense_last_steps=dense_last_steps, selection=selection, tau=tau,
        extra_tokens=extra_tokens, start_percent=start_percent,
        end_percent=end_percent, keep_percent_end=keep_percent_end,
        tau_end=tau_end, dense_backend=dense_backend, qkv_mode=qkv_mode,
        cache_threshold=cache_threshold, cache_max_hits=cache_max_hits,
        cache_max_mib=cache_max_mib,
        cache_span=cache_span,
        cache_preset=cache_preset,
        cache_start_percent=cache_start_percent,
        cache_end_percent=cache_end_percent,
        cache_min_steps=cache_min_steps,
        cache_temporal_guard=cache_temporal_guard,
        cache_audio_guard=cache_audio_guard,
        cache_mode=cache_mode,
    )
    if cache_enabled and cache_span == "whole" and qkv_mode != "native":
        raise ValueError("CurveAttention: whole cache requires native qkv_mode")
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError("CurveAttention only supports MiniMax H3 models.")
    if _dit_replacements(model.model_options):
        raise ValueError(
            "CurveAttention: model already has dit block replacements "
            "(for example Model Sparse Attention). Use one sparse-attention node."
        )
    patches = getattr(model, "object_patches", {})
    if any(_conflicting_h3_object_patch(key) for key in patches):
        raise ValueError("CurveAttention: conflicting H3 object patch")
    if not ck.sol_attn_is_available():
        raise RuntimeError("CurveAttention: Comfy Kitchen sol_attn is not available on this GPU.")
    dense_attention = get_attention_function(dense_backend, None)
    if dense_attention is None:
        raise RuntimeError(f"CurveAttention: dense backend {dense_backend} is unavailable.")
    prepared_available = False
    if qkv_mode == "owned_bf16":
        prepared_available = True
        try:
            check_prepared_available()
        except (RuntimeError, OSError) as exc:
            prepared_available = False
            _LOG.warning(
                "CurveAttention: %s; using native ck.sol_attn without owned BF16 copies",
                exc,
            )

    keep_ratio_end = None
    if float(keep_percent_end) >= 0.5 and selection == "sla":
        keep_ratio_end = float(keep_percent_end) / 100.0
    scheduled_tau_end = None
    if float(tau_end) >= 0.0 and selection == "sol-attn":
        scheduled_tau_end = float(tau_end)

    sigma_start, sigma_end = resolve_sigma_bounds(model, start_percent, end_percent)
    run_scope = RunScope(
        cache_threshold, cache_max_hits, cache_max_mib, span=cache_span,
        start_percent=cache_start_percent, end_percent=cache_end_percent,
        min_steps=cache_min_steps, temporal_guard=cache_temporal_guard,
        audio_guard=cache_audio_guard, mode=cache_mode,
    )

    state = CurveAttnState(
        keep_ratio=float(keep_percent) / 100.0,
        min_tokens=int(min_tokens),
        dense_steps=_parse_steps(dense_steps),
        dense_last_steps=int(dense_last_steps),
        tail=bool(tail),
        protect_audio=bool(protect_audio),
        verbose=bool(verbose),
        dense_attention=dense_attention,
        protect_prefix=bool(protect_prefix),
        disable_fp16_accum=bool(disable_fp16_accum),
        prepared_available=prepared_available,
        selection=selection,
        tau=float(tau),
        token_aug=int(extra_tokens),
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        keep_ratio_end=keep_ratio_end,
        tau_end=scheduled_tau_end,
        dense_backend=dense_backend,
        qkv_mode=qkv_mode,
        cache_enabled=cache_enabled,
        run_scope=run_scope,
        cache_span=cache_span,
        cache_mode=cache_mode,
        cache_start_percent=cache_start_percent,
        cache_end_percent=cache_end_percent,
        cache_min_steps=cache_min_steps,
        cache_temporal_guard=cache_temporal_guard,
        cache_audio_guard=cache_audio_guard,
        cache_preset=cache_preset,
    )
    m = model.clone()
    install_override(state, m.model_options["transformer_options"])
    owned_replacements = dict(
        (m.model_options.get("transformer_options") or {}).get("patches_replace", {}).get("dit", {})
    )

    def _prepare(model_patcher, timestep, model_options):
        install_override(state, model_options["transformer_options"])
        current = (model_options.get("transformer_options") or {}).get("patches_replace", {}).get("dit", {})
        if owned_replacements and current != owned_replacements:
            raise ValueError("CurveAttention: another node replaced owned dit patches")

    def _report(run):
        state.cache_hits = run.cache.hits
        state.cache_refreshes = run.cache.refreshes
        state.cache_bypasses = run.cache.bypasses
        state.cache_bytes = run.cache.bytes
        if state.cache_enabled:
            _LOG.info(format_cache_report(
                run.cache,
                span=state.cache_span,
                mode=state.cache_mode,
                start_percent=state.cache_start_percent,
                end_percent=state.cache_end_percent,
            ))
            _LOG.info(
                "CurveAttention cache admission: entry_bytes=%d budget_bytes=%d "
                "storage_bypasses=%d entries=%d prefetch=%s",
                run.cache.required_entry_bytes, run.cache.max_bytes,
                run.cache.budget_bypasses, len(run.cache.entries),
                "on_demand" if state.cache_span == "whole" else "inherit")
            if run.cache.hits == 0:
                blocked = f" last_block={state.last_cache_block}" if state.last_cache_block else ""
                _LOG.warning(
                    "CurveAttention cache had zero hits; inspect eligibility, schedule, "
                    "threshold and storage budget before expecting a cache speedup.%s",
                    blocked)
        if state.verbose or state.sparse_calls or state.dense_calls or run.cache.hits:
            _LOG.info(
                "CurveAttention: sparse_calls=%d dense_calls=%d chunked_calls=%d "
                "cache_hits=%d cache_refreshes=%d cache_bypasses=%d cache_bytes=%d "
                "dense_backend=%s last_dense=%s",
                state.sparse_calls, state.dense_calls, state.chunked_calls,
                run.cache.hits, run.cache.refreshes, run.cache.bypasses, run.cache.bytes,
                state.dense_backend, state.last_reason,
            )
        if state.sparse_calls == 0 and not state.failure_reason:
            _LOG.warning("CurveAttention: sparse was requested but sparse_calls=0")

    run_scope.on_begin = state.reset
    run_scope.on_end = _report
    m.add_callback_with_key(CallbacksMP.ON_PREPARE_STATE, "curve_attention", _prepare)
    m.add_callback_with_key(CallbacksMP.ON_CLEANUP, "curve_attention", lambda model_patcher: _cleanup(state))
    m.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, "curve_attention_batch", limit_h3_cond_batch)
    m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "curve_attention_run", run_scope.outer)
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "curve_attention", make_step_wrapper(state))
    if cache_enabled or qkv_mode == "chunked":
        blocks = diffusion_model.blocks
        if cache_enabled and len(blocks) < 3:
            raise ValueError("CurveAttention: middle cache needs at least three blocks")
        for index, block in enumerate(blocks):
            m.set_model_patch_replace(
                make_block_dispatch(index, len(blocks), block.attn, state),
                "dit", "double_block", index)
        owned_replacements = dict(
            (m.model_options.get("transformer_options") or {}).get("patches_replace", {}).get("dit", {})
        )
    _LOG.info(
        "CurveAttention %s: selection=%s keep=%.1f%% tau=%.3f extra_tokens=%d "
        "start_percent=%.3f end_percent=%.3f keep_percent_end=%.1f tau_end=%.3f "
        "sigma_start=%s sigma_end=%s block=64 qkv_mode=%s prepared_qkv=%s dense_backend=%s "
        "cache_enabled=%s cache_preset=%s cache_mode=%s cache_span=%s "
        "cache_window=%.2f-%.2f min_steps=%d temporal_guard=%s audio_guard=%s "
        "dense_steps=%s dense_last_steps=%d protect_prefix=%s protect_audio=%s tail=%s "
        "disable_fp16_accum=%s cfg_branches=serial",
        __version__, selection, keep_percent, float(tau), int(extra_tokens),
        float(start_percent), float(end_percent), float(keep_percent_end), float(tau_end),
        None if sigma_start is None else f"{sigma_start:.6g}",
        None if sigma_end is None else f"{sigma_end:.6g}",
        qkv_mode, state.prepared_enabled, dense_backend, cache_enabled, cache_preset,
        cache_mode, cache_span, cache_start_percent, cache_end_percent, cache_min_steps,
        cache_temporal_guard, cache_audio_guard, sorted(state.dense_steps),
        state.dense_last_steps, state.protect_prefix, state.protect_audio, state.tail,
        state.disable_fp16_accum,
    )
    return m


def _cleanup(state: CurveAttnState):
    if state.verbose or state.sparse_calls or state.dense_calls or state.chunked_calls or state.cache_enabled:
        _LOG.info(
            "CurveAttention: sparse_calls=%d dense_calls=%d chunked_calls=%d "
            "cache_hits=%d cache_refreshes=%d cache_bypasses=%d cache_bytes=%d "
            "dense_backend=%s last_dense=%s",
            state.sparse_calls, state.dense_calls, state.chunked_calls,
            state.cache_hits, state.cache_refreshes, state.cache_bypasses, state.cache_bytes,
            state.dense_backend, state.last_reason,
        )
    if state.run_scope is not None:
        state.run_scope.clear_current()
    state.reset()
