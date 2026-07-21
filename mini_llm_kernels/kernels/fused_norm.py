"""
fused_norm.py — Python 绑定与 fallback 实现

提供 fused_add_rms_norm() 函数：
- 有 CUDA kernel：调用 mini_llm_kernels._C.fused_add_rms_norm
- 无 CUDA（M1/CPU）：纯 PyTorch fallback，语义完全相同

接口：
    x_normed, residual_out = fused_add_rms_norm(x, residual, gamma, eps)

    x:        [N, H]  attention/mlp 输出（delta）
    residual: [N, H]  上一子层残差流
    gamma:    [H]     RMSNorm 可学习缩放
    eps:      float

    x_normed:     [N, H]  归一化结果，供下一算子使用
    residual_out: [N, H]  x + residual，作为下一子层残差输入
"""

from __future__ import annotations
from typing import Tuple

import torch


def _fused_add_rms_norm_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """纯 PyTorch fallback（M1/CPU 使用，语义与 CUDA kernel 完全相同）"""
    orig_dtype = x.dtype
    residual_out = x + residual                              # Step 1: residual add
    x_fp32 = residual_out.float()                            # Step 2: RMSNorm（fp32 计算）
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_normed = (x_fp32 * rms).to(orig_dtype) * gamma        # Step 3: scale
    return x_normed, residual_out


# 运行时选择：有 CUDA 扩展时使用 kernel，否则 fallback
try:
    from mini_llm_kernels._C import fused_add_rms_norm as _cuda_fused_add_rms_norm
    _HAS_FUSED_NORM = True
except ImportError:
    _HAS_FUSED_NORM = False


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """融合 RMSNorm + Residual Add

    Returns:
        (x_normed, residual_out)
    """
    if _HAS_FUSED_NORM and x.is_cuda and residual.is_cuda and gamma.is_cuda:
        return _cuda_fused_add_rms_norm(x, residual, gamma, eps)
    return _fused_add_rms_norm_pytorch(x, residual, gamma, eps)
