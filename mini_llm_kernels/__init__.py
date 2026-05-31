"""
mini_llm_kernels Python 入口

此包提供自定义 CUDA kernel 的 Python 绑定。
在没有 CUDA 编译环境时（如 M1 Mac），仅暴露 fallback 标识，
不 import 任何 C 扩展，保证主仓库 fallback 路径正常工作。
"""

from __future__ import annotations

# 尝试加载编译好的 C 扩展
try:
    from ._C import add_tensors  # noqa: F401
    _HAS_CUDA_OPS = True
except ImportError:
    _HAS_CUDA_OPS = False

__all__ = ["_HAS_CUDA_OPS"]
if _HAS_CUDA_OPS:
    __all__ += ["add_tensors"]
