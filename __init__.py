"""CurveAttn: MiniMax H3 sparse attention and residual cache for ComfyUI (V3 nodes)."""
from ._version import VERSION as __version__
from .nodes import CurveAttnCache, CurveAttnSparse, comfy_entrypoint

__all__ = ["CurveAttnSparse", "CurveAttnCache", "comfy_entrypoint", "__version__"]
