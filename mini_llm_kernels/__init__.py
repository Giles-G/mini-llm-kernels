"""
mini_llm_kernels Python 入口

M1：基础框架，_HAS_CUDA_OPS=False
M3：fused_add_rms_norm（带 PyTorch fallback）
M4：decode_paged_attention（带 PyTorch fallback）
"""

from __future__ import annotations

try:
    from ._C import add_tensors  # noqa: F401
    _HAS_CUDA_OPS = True
except ImportError:
    _HAS_CUDA_OPS = False

from mini_llm_kernels.kernels.fused_norm import fused_add_rms_norm        # noqa: F401
from mini_llm_kernels.kernels.decode_attention import decode_paged_attention  # noqa: F401

__all__ = ["_HAS_CUDA_OPS", "fused_add_rms_norm", "decode_paged_attention"]
if _HAS_CUDA_OPS:
    __all__ += ["add_tensors"]
