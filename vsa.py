"""Fine-stage VSA tiling for the existing sol_attn call.

Matches ComfyUI's Model Sparse Attention cube plan: non-video segments are
zero-padded 64-row prefix tiles, target video is 4x4x4 cubes. There is no
coarse gate. RoPE is already on each token, so the override only permutes
Q, K and V, then scatters the kernel output back to packed order.
"""
from __future__ import annotations

import torch

BLOCK = 64
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
