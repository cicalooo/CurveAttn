"""Reblock selection: content-clustered 64-token video blocks for the sparse call.

Grouping target-video tokens into similar blocks, routing them by top-k and
covering the skipped blocks with anchor-reweighted summaries keeps more
attention mass per kernel block. Original design based on the underlying
math; no external code is used and no patched kernel is needed.

Per call (per layer, per sampled step):

* Order. Keys are mapped through the Cholesky factor of the queries' second
  moment, so cosine distance follows how differently the queries see them.
  A capacity-balanced spherical 2-means bisection then cuts the video rows
  into full 64-row blocks per batch row and head. Queries reuse the key
  order (q from k), so one permutation serves Q, K and V.
* Scope. Rows before the target video (text, references, audio) and the
  ``video % 64`` highest-norm video rows sit in zero-padded prefix tiles that
  are exact keys and dense queries. Only target video -> target video is sparse.
* Tail. ``reweighted`` scores every skipped video block for a query as
  ``q.k_bar*scale + log_mass`` with value ``v_bar``, where ``k_bar``/``v_bar``
  are the block softmax average at a float32 per-head anchor (the mean video
  query) and ``log_mass`` is that softmax's entropy (the missing effective
  token count). It runs as one flex_attention call over the selected blocks
  plus the summary rows. ``pooled`` and ``off`` use Kitchen ``sol_attn``'s
  own pooled tail or none on the reblocked order.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BLOCK = 64
SUMMARIES = ("reweighted", "pooled", "off")
HEAD_CHUNK = 16
SKETCH = 32
_FLEX = None
_SKETCH_BASIS = {}
_PROBE = {}


# ---------------------------------------------------------------- order

def video_span(layout, tokens):
    """(start, end) of the target-video rows, or None when reblock must stay dense."""
    if layout is None or getattr(layout, "seq_len", None) != tokens:
        return None
    segments = tuple(getattr(layout, "segments", ()) or ())
    if not segments or segments[-1][2] != "video":
        return None
    start, end = int(segments[-1][0]), int(segments[-1][1])
    if end != tokens or end - start < 2 * BLOCK:
        return None
    return start, end


def key_features(q, k, ridge=1e-3):
    """Keys seen through the queries: ``k @ L`` with ``L L^T = E[q q^T] + ridge``.

    ``q``/``k`` are ``(G, V, D)`` float32. The second moment comes from one
    query per 64 rows. A failed factorisation falls back to the raw keys.
    """
    sample = q[:, BLOCK // 2::BLOCK]
    if sample.shape[1] == 0:
        sample = q
    d = q.shape[-1]
    m2 = sample.transpose(1, 2) @ sample / sample.shape[1]
    lam = ridge * m2.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-6)
    m2 = m2 + lam.view(-1, 1, 1) * torch.eye(d, device=q.device, dtype=q.dtype)
    factor, info = torch.linalg.cholesky_ex(m2)
    eye = torch.eye(d, device=q.device, dtype=q.dtype).expand_as(factor)
    factor = torch.where((info == 0).view(-1, 1, 1), factor, eye)
    return k @ factor


_SEGMENTS = {}


def _sketch_basis(d, device, dtype):
    """Fixed random rotation to SKETCH dims, cached per shape/device.

    QR on the host once; planning then pays only the matmul.
    """
    key = (d, str(device), str(dtype))
    hit = _SKETCH_BASIS.get(key)
    if hit is not None:
        return hit
    gen = torch.Generator().manual_seed(d)
    basis = torch.linalg.qr(torch.randn(d, SKETCH, generator=gen))[0] * (d / SKETCH) ** 0.5
    basis = basis.to(device=device, dtype=dtype)
    _SKETCH_BASIS[key] = basis
    return basis


def _probe_vector(d, device, dtype):
    key = (d, str(device))
    hit = _PROBE.get(key)
    if hit is None:
        hit = torch.linspace(1.0, -1.0, d, device=device, dtype=dtype)
        _PROBE[key] = hit
    return hit


def _segments(n_blocks, device):
    """Per bisection level: (segment id of every row position, segment count).

    The node layout only depends on ``n_blocks``, so it is built once on the
    host and cached on the device; planning then never copies or syncs."""
    key = (n_blocks, str(device))
    if key not in _SEGMENTS:
        levels, nodes = [], [(0, n_blocks)]
        while any(c > 1 for _, c in nodes):
            ids = torch.repeat_interleave(torch.arange(len(nodes)),
                                          torch.tensor([c * BLOCK for _, c in nodes]))
            levels.append((ids.to(device), len(nodes)))
            nodes = [child for s, c in nodes
                     for child in (((s, (c + 1) // 2), (s + (c + 1) // 2, c // 2))
                                   if c > 1 else ((s, c),))]
        _SEGMENTS[key] = levels
    return _SEGMENTS[key]


def _segment_sum(x, seg, count):
    g, n, d = x.shape
    total = x.new_zeros(g, count, d, dtype=torch.float32)
    total.index_add_(1, seg, x.float())
    return total


def balanced_order(x, n_blocks, updates=1):
    """Permutation of ``x``'s rows (``(G, n_blocks*64, D)``) where each run of
    64 rows is one cluster.

    Each level splits every node at once by one segmented sort along the node's
    principal direction (``1 + updates`` sign-power iterations from the
    centred mean-offset direction). Nodes are fixed position ranges, so the
    larger half (``ceil(c/2)`` blocks) simply is the front of the sorted node.
    All on device, no host syncs."""
    g, n, d = x.shape
    unit = F.normalize(x.float(), dim=-1)
    if d > SKETCH:  # cached random rotation to SKETCH dims: less traffic per level
        unit = unit @ _sketch_basis(d, unit.device, unit.dtype)
        d = SKETCH
    order = torch.arange(n, device=x.device).expand(g, n).contiguous()
    probe = _probe_vector(d, x.device, torch.float32)
    for seg, count in _segments(n_blocks, x.device):
        feat = unit.gather(1, order.unsqueeze(-1).expand(-1, -1, d))
        sizes = torch.bincount(seg, minlength=count).view(1, -1, 1)
        centred = feat.float() - (_segment_sum(feat, seg, count) / sizes)[:, seg]
        direction = probe.expand(g, count, d)
        for _ in range(1 + int(updates)):
            sign = torch.sign((centred * direction[:, seg]).sum(-1, keepdim=True))
            direction = F.normalize(_segment_sum(centred * sign, seg, count), dim=-1)
        proj = (centred * direction[:, seg]).sum(-1)
        key = seg - 0.49 * torch.tanh(proj)
        order = order.gather(1, key.argsort(dim=-1, stable=True))
        del feat, centred, direction, proj, key
    return order


def _split_spare(feat, spare):
    """(kept rows in packed order, ``spare`` highest-norm rows ascending)."""
    g, n, _ = feat.shape
    rows = torch.arange(n, device=feat.device).expand(g, n)
    if spare == 0:
        return rows, rows[:, :0]
    out = feat.norm(dim=-1).topk(spare, dim=-1).indices.sort(-1).values
    keep = torch.ones(g, n, dtype=torch.bool, device=feat.device)
    keep.scatter_(1, out, False)
    kept = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)[:, :n - spare]
    return kept, out


def build_reblock_plan(q, k, span, *, updates=1, head_chunk=HEAD_CHUNK):
    """Plan for ``(B, T, H, D)`` q/k, or None when the video is too short.

    ``src`` ``(B, H, n)`` maps a plan row to a packed row (-1 is pad), ``inv``
    ``(B, H, T)`` maps back. ``block_len`` (shared) is the live count of each
    64-row tile; the first ``n_prefix`` tiles are exact prefix tiles.
    """
    b, t, h, d = k.shape
    start, end = span
    n_video = end - start
    n_full = n_video // BLOCK
    if n_full < 2:
        return None
    spare = n_video - n_full * BLOCK
    device = k.device
    order = torch.empty(b, h, n_full * BLOCK, dtype=torch.int64, device=device)
    spares = torch.empty(b, h, spare, dtype=torch.int64, device=device)
    # Flatten (batch, head) so CFG rows and heads share one batched bisection per chunk.
    qv = q[:, start:end].permute(0, 2, 1, 3).reshape(b * h, n_video, d)
    kv = k[:, start:end].permute(0, 2, 1, 3).reshape(b * h, n_video, d)
    order, spares = order.view(b * h, -1), spares.view(b * h, -1)
    for g0 in range(0, b * h, head_chunk):
        g1 = min(b * h, g0 + head_chunk)
        feat = key_features(qv[g0:g1].float(), kv[g0:g1].float())
        keep, out = _split_spare(feat, spare)
        local = balanced_order(feat.gather(1, keep.unsqueeze(-1).expand(-1, -1, d)),
                               n_full, updates)
        order[g0:g1] = keep.gather(1, local) + start
        spares[g0:g1] = out + start
        del feat, keep, out, local
    del qv, kv
    order, spares = order.view(b, h, -1), spares.view(b, h, -1)
    head_rows = start + spare
    n_prefix = (head_rows + BLOCK - 1) // BLOCK
    n = (n_prefix + n_full) * BLOCK
    src = torch.full((b, h, n), -1, dtype=torch.int64, device=device)
    src[..., :start] = torch.arange(start, device=device)
    src[..., start:head_rows] = spares
    src[..., n_prefix * BLOCK:] = order
    live = torch.ones(n, dtype=torch.bool, device=device)
    live[head_rows:n_prefix * BLOCK] = False
    inv = torch.empty(b, h, t, dtype=torch.int64, device=device)
    live_idx = torch.cat([torch.arange(head_rows, device=device),
                          torch.arange(n_prefix * BLOCK, n, device=device)])
    inv.scatter_(2, src.index_select(2, live_idx), live_idx.expand(b, h, t).contiguous())
    block_len = torch.full((n // BLOCK,), BLOCK, dtype=torch.int32, device=device)
    if n_prefix:
        block_len[n_prefix - 1] = head_rows - (n_prefix - 1) * BLOCK
    return {"n": n, "n_prefix": n_prefix, "n_video_blocks": n_full, "src": src, "inv": inv,
            "live": live, "block_len": block_len, "dead": (head_rows, n_prefix * BLOCK)}


def gather_heads(q, k, v, src, dead=None):
    """``(B, T, H, D)`` rows into plan order per batch row and head. Pad rows are zero.

    ``dead`` ``(lo, hi)`` is the plan's pad slice; with it, one gather pass per
    tensor plus a tiny slice fill replaces the full-size live multiply.

    Per-head ``index_select`` avoids materialising the ``(B, n, H, D)`` int64
    index (``n*H*D`` entries) that ``gather`` needs when the index is expanded
    across the head-dim.
    """
    b, h, n = src.shape
    d = q.shape[-1]
    if dead is not None and b * h <= 64:
        src_c = src.clamp_min(0)
        out_q = q.new_empty((b, n, h, d))
        out_k = k.new_empty((b, n, h, d))
        out_v = v.new_empty((b, n, h, d))
        for bi in range(b):
            qb, kb, vb = q[bi], k[bi], v[bi]
            oq, ok, ov = out_q[bi], out_k[bi], out_v[bi]
            for hi in range(h):
                idx = src_c[bi, hi]
                oq[:, hi, :] = qb[:, hi, :].index_select(0, idx)
                ok[:, hi, :] = kb[:, hi, :].index_select(0, idx)
                ov[:, hi, :] = vb[:, hi, :].index_select(0, idx)
        lo, hi_ = dead
        if hi_ > lo:
            out_q[:, lo:hi_] = 0
            out_k[:, lo:hi_] = 0
            out_v[:, lo:hi_] = 0
        return out_q, out_k, out_v
    index = src.clamp_min(0).transpose(1, 2).unsqueeze(-1).expand(b, n, h, d)
    if dead is not None:
        out = tuple(tensor.gather(1, index) for tensor in (q, k, v))
        for tensor in out:
            tensor[:, dead[0]:dead[1]] = 0
        return out
    live = (src >= 0).transpose(1, 2).unsqueeze(-1).to(q.dtype)
    return tuple(tensor.gather(1, index) * live for tensor in (q, k, v))


def scatter_heads(out, inv):
    """Plan-order ``(B, n, H, D)`` output back to packed rows.

    Per-head ``index_select`` avoids the ``(B, T, H, D)`` int64 index.
    """
    b, h, t = inv.shape
    if b * h <= 64:
        packed = out.new_empty((b, t, h, out.shape[-1]))
        for bi in range(b):
            ob, pb = out[bi], packed[bi]
            for hi in range(h):
                pb[:, hi, :] = ob[:, hi, :].index_select(0, inv[bi, hi])
        return packed
    return out.gather(1, inv.transpose(1, 2).unsqueeze(-1).expand(b, t, h, out.shape[-1]))


# ---------------------------------------------------------------- reweighted tail

def block_summaries(q, k, v, n_prefix, scale, head_chunk=HEAD_CHUNK):
    """Anchor-reweighted summaries of every video key block.

    Returns ``k_bar``, ``v_bar`` ``(B, H, NV, D)`` and ``log_mass`` ``(B, H, NV)``,
    all float32. The anchor is the float32 mean video query of each head.
    """
    b, n, h, d = k.shape
    v0 = n_prefix * BLOCK
    nv = (n - v0) // BLOCK
    k_bar = k.new_empty((b, h, nv, d), dtype=torch.float32)
    v_bar = torch.empty_like(k_bar)
    log_mass = k.new_empty((b, h, nv), dtype=torch.float32)
    for h0 in range(0, h, head_chunk):
        h1 = min(h, h0 + head_chunk)
        anchor = q[:, v0:, h0:h1].float().mean(1)
        kb = k[:, v0:, h0:h1].float().view(b, nv, BLOCK, h1 - h0, d)
        z = torch.einsum("bnjhd,bhd->bhnj", kb, anchor) * scale
        p = torch.softmax(z, -1)
        kbar = torch.einsum("bhnj,bnjhd->bhnd", p, kb)
        del kb
        vb = v[:, v0:, h0:h1].float().view(b, nv, BLOCK, h1 - h0, d)
        v_bar[:, h0:h1] = torch.einsum("bhnj,bnjhd->bhnd", p, vb)
        del vb, p
        k_bar[:, h0:h1] = kbar
        log_mass[:, h0:h1] = (torch.logsumexp(z, -1)
                              - torch.einsum("bhnd,bhd->bhn", kbar, anchor) * scale)
    return k_bar, v_bar, log_mass


def select_blocks(q, k_bar, log_mass, n_prefix, ratio, scale):
    """``(B, H, NQ, NV)`` exact video key blocks per query tile: the top
    ``ratio`` by mass-corrected summary score, plus the +-1 band. Prefix query
    tiles keep every block (dense rows)."""
    b, n, h, d = q.shape
    nq, nv = n // BLOCK, k_bar.shape[2]
    q_mean = q.view(b, nq, BLOCK, h, d).mean(2, dtype=torch.float32)
    score = torch.einsum("bqhd,bhnd->bhqn", q_mean, k_bar) * scale + log_mass.unsqueeze(2)
    keep = min(nv, max(1, round(ratio * nv)))
    sel = torch.zeros(score.shape, dtype=torch.bool, device=q.device)
    sel.scatter_(-1, score.topk(keep, dim=-1).indices, True)
    qi = torch.arange(nq, device=q.device).view(-1, 1)
    kj = torch.arange(nv, device=q.device).view(1, -1) + n_prefix
    sel |= ((qi - kj).abs() <= 1)
    sel[:, :, :n_prefix] = True
    return sel


def reweighted_reference(q, k, v, live, n_prefix, sel, k_bar, v_bar, log_mass, scale):
    """Materialised exact + summary softmax. O(n^2): CPU tests and small shapes."""
    v0 = n_prefix * BLOCK
    qh, kh, vh = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    exact = torch.cat([sel.new_ones(sel.shape[:3] + (v0,)),
                       sel.repeat_interleave(BLOCK, -1)], -1) & live
    exact = exact.repeat_interleave(BLOCK, 2)
    s_tok = (qh @ kh.transpose(-1, -2) * scale).masked_fill(~exact, float("-inf"))
    s_sum = qh @ k_bar.transpose(-1, -2) * scale + log_mass.unsqueeze(2)
    s_sum = s_sum.masked_fill(sel.repeat_interleave(BLOCK, 2), float("-inf"))
    p = torch.softmax(torch.cat([s_tok, s_sum], -1), -1)
    out = p[..., :kh.shape[2]] @ vh + p[..., kh.shape[2]:] @ v_bar
    return out.permute(0, 2, 1, 3).to(q.dtype)


def _flex_kernel(device):
    global _FLEX
    from torch.nn.attention.flex_attention import flex_attention
    if device.type != "cuda":
        return flex_attention
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def reweighted_flex(q, k, v, live, block_len, n_prefix, sel, k_bar, v_bar, log_mass, scale):
    """Same result as ``reweighted_reference`` as one block-sparse flex_attention
    call: selected 64-key tiles are full blocks, prefix tiles mask their pad
    rows, and the summaries ride as extra key rows with a log-mass score bias."""
    from torch.nn.attention.flex_attention import BlockMask
    b, n, h, d = q.shape
    nb, nv = n // BLOCK, k_bar.shape[2]
    ns = (nv + BLOCK - 1) // BLOCK
    pad = ns * BLOCK - nv
    dtype = q.dtype
    k_sum = F.pad(k_bar, (0, 0, 0, pad)).to(dtype)
    v_sum = F.pad(v_bar, (0, 0, 0, pad)).to(dtype)
    qh = q.transpose(1, 2)
    kh = torch.cat([k.transpose(1, 2), k_sum], 2)
    vh = torch.cat([v.transpose(1, 2), v_sum], 2)
    del k_sum, v_sum
    full = torch.zeros(b, h, nb, nb + ns, dtype=torch.bool, device=q.device)
    part = torch.zeros_like(full)
    whole = block_len[:n_prefix] == BLOCK
    full[..., :n_prefix] = whole
    part[..., :n_prefix] = ~whole
    full[..., n_prefix:nb] = sel
    part[..., nb:] = (torch.arange(nb, device=q.device) >= n_prefix).view(-1, 1)

    def table(mask):
        return (mask.sum(-1, dtype=torch.int32),
                torch.argsort((~mask).to(torch.int8), dim=-1, stable=True).to(torch.int32))

    log_mass = log_mass.contiguous()

    def mask_mod(bi, hi, q_idx, kv_idx):
        # The block tables already skip unselected tiles; the full predicate is
        # still spelled out so partial blocks and unfused fallbacks agree.
        qb = q_idx // BLOCK
        j = (kv_idx - n).clamp(0, nv - 1)
        summary = (kv_idx - n < nv) & ~sel[bi, hi, qb, j]
        key = kv_idx.clamp(max=n - 1)
        tile = (key // BLOCK - n_prefix).clamp(0, nv - 1)
        exact = live[key] & ((key < n_prefix * BLOCK) | sel[bi, hi, qb, tile])
        return torch.where(kv_idx >= n, summary, exact)

    def score_mod(score, bi, hi, q_idx, kv_idx):
        j = (kv_idx - n).clamp(0, nv - 1)
        return torch.where(kv_idx >= n, score + log_mass[bi, hi, j], score)

    kv_num, kv_idx = table(part)
    full_num, full_idx = table(full)
    mask = BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=BLOCK,
                                    mask_mod=mask_mod, seq_lengths=(n, n + ns * BLOCK))
    options = {"BLOCK_M": BLOCK, "BLOCK_N": BLOCK} if q.device.type == "cuda" else None
    out = _flex_kernel(q.device)(qh, kh, vh, score_mod=score_mod, block_mask=mask,
                                 scale=scale, kernel_options=options)
    return out.transpose(1, 2)


def reweighted_attention(q, k, v, plan, ratio, scale=None):
    """Reblocked top-k with reweighted tail on plan-order ``(B, n, H, D)`` q/k/v."""
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    n_prefix = plan["n_prefix"]
    k_bar, v_bar, log_mass = block_summaries(q, k, v, n_prefix, scale)
    sel = select_blocks(q, k_bar, log_mass, n_prefix, ratio, scale)
    return reweighted_flex(q, k, v, plan["live"], plan["block_len"], n_prefix, sel,
                           k_bar, v_bar, log_mass, scale)


__all__ = ["SUMMARIES", "video_span", "build_reblock_plan", "gather_heads", "scatter_heads",
           "block_summaries", "select_blocks", "reweighted_reference", "reweighted_flex",
           "reweighted_attention"]
