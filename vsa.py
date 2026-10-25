"""VSA tiling and coarse gate for the sol_attn calls.

Matches ComfyUI's Model Sparse Attention cube plan: non-video segments are
zero-padded 64-row prefix tiles, target video is 4x4x4 cubes. In native QKV
mode RoPE is already on each token, so the override only permutes Q, K and V,
then scatters the kernel output back to packed order. The coarse branch needs
the block input x, so a block-level provider (``make_vsa_gate_attention``)
computes ``to_gate_compress`` on the permuted rows and hands it to the
override through ``transformer_options[NS]["vsa_gate"]``. Checkpoints without
``to_gate_compress`` run the fine stage only.
"""
from __future__ import annotations

import torch

from .common import NS

BLOCK = 64
GATE_CHUNK = 4096
GATE_KEY = "vsa_gate"
CUBE = (4, 4, 4)


def build_vsa_plan(layout, tokens, device):
    """Tile plan, or None when this call cannot use the cube order.

    ``src`` maps a padded row to a packed row (-1 is pad). ``inv`` maps a
    packed row back. ``block_len`` is the live count of each 64-row tile.
    """
    if layout is None or getattr(layout, "seq_len", None) != tokens:
        return None
    signature = getattr(layout, "signature", None)
    segments = tuple(getattr(layout, "segments", ()) or ())
    if not signature or len(signature) != 5 or not segments or segments[-1][2] != "video":
        return None
    try:
        _, latent_t, latent_h, latent_w, _ = signature
        grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    except (TypeError, ValueError):
        return None
    if any(axis < 1 for axis in grid):
        return None
    tiles = []
    n_prefix = 0
    for start, end, kind in segments:
        n = end - start
        if n < 1:
            return None
        if kind != "video":
            tiles.append(_prefix_tiles(start, n, device))
            n_prefix += tiles[-1].shape[0]
            continue
        if grid[0] * grid[1] * grid[2] != n:
            return None
        tiles.append(_video_cubes(start, grid, device))
    tiled = torch.cat(tiles)
    src = tiled.reshape(-1)
    live = src >= 0
    inv = torch.empty(tokens, dtype=torch.int64, device=device)
    inv[src[live]] = torch.nonzero(live).flatten()
    return {
        "n": int(src.numel()),
        "n_prefix": n_prefix,
        "src": src,
        "inv": inv,
        "block_len": (tiled >= 0).sum(1).to(torch.int32),
    }


def _prefix_tiles(start, n, device):
    m = (n + BLOCK - 1) // BLOCK
    seg = torch.full((m * BLOCK,), -1, dtype=torch.int64, device=device)
    seg[:n] = torch.arange(start, start + n, device=device)
    return seg.view(m, BLOCK)


def _video_cubes(start, grid, device):
    ct, ch, cw = CUBE
    pt, ph, pw = ((axis + cube - 1) // cube * cube for axis, cube in zip(grid, CUBE))
    padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
    padded[:grid[0], :grid[1], :grid[2]] = torch.arange(
        start, start + grid[0] * grid[1] * grid[2], device=device).view(*grid)
    cubes = (padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw)
             .permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK))
    order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
    return torch.gather(cubes, 1, order)


def gather_qkv(q, k, v, src):
    """Permute packed rows into the tile order. Pad rows are zero."""
    index = src.clamp_min(0)
    live = (src >= 0).to(dtype=q.dtype).view(1, -1, 1, 1)
    return tuple(tensor.index_select(1, index) * live for tensor in (q, k, v))


def scatter_rows(out, inv):
    """Map kernel rows back to packed order."""
    return out.index_select(1, inv)


def permute_rows(x, src):
    """Packed (S, C) rows into tile order for ``src``. Pad rows are zero."""
    return x[src.clamp_min(0)] * (src >= 0).unsqueeze(1).to(x.dtype)


def has_gate(diffusion_model) -> bool:
    return any(getattr(block.attn, "to_gate_compress", None) is not None
               for block in diffusion_model.blocks)


def coarse_gate(gate_layer, x, plan, heads, head_dim, chunk=GATE_CHUNK):
    """``to_gate_compress`` on the permuted rows as ``(1, n, H, D)``, built in
    chunks so the permuted x is never materialised whole."""
    n, src = plan["n"], plan["src"]
    gate = x.new_empty(n, heads * head_dim)
    for i in range(0, n, chunk):
        gate[i:i + chunk] = gate_layer(permute_rows(x, src[i:i + chunk]))
    return gate.view(1, n, heads, head_dim)


def with_gate(options, state, plan, gate):
    """Copy of options carrying this node's gate for the attention override."""
    out = dict(options or {})
    ns = dict(out.get(NS) or {})
    ns[GATE_KEY] = (state.uid, plan, gate)
    out[NS] = ns
    return out


def take_gate(options, state, plan):
    """The gate published for this node and plan, or None. Removes it so the
    tensor dies with the attention call."""
    ns = (options or {}).get(NS)
    entry = ns.get(GATE_KEY) if isinstance(ns, dict) else None
    if entry is None or entry[0] != state.uid or entry[1] is not plan:
        return None
    ns.pop(GATE_KEY, None)
    return entry[2]


def make_vsa_gate_attention(attention, state, index=0):
    """Block attention that adds the VSA coarse gate, then runs the normal
    attention module (and so the CurveAttn override chain)."""
    gate_layer = getattr(attention, "to_gate_compress", None)

    def call(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        decision, _, _ = state.resolve(options)
        plan = None
        if (gate_layer is not None and decision is not None and state.failure_reason is None
                and x.ndim == 2 and x.shape[0] >= state.min_tokens):
            plan = state.vsa_plan(options, x.shape[0], x.device)
        if plan is None:
            return attention(x, rope_freqs=rope_freqs, transformer_options=options)
        gated = with_gate(options, state, plan,
                          coarse_gate(gate_layer, x, plan, attention.heads, attention.head_dim))
        return attention(x, rope_freqs=rope_freqs, transformer_options=gated)

    return call
