"""
int4_matmul.py — INT4 fused dequantize+matmul (Python binding + fallback).

Provides:
    int4_dequant_matmul(x, w_packed, group_scales)
        Fused INT4 dequantize + matmul (calls CUDA kernel if available).

    _int4_dequant_matmul_pytorch(x, w_packed, group_scales)
        PyTorch fallback (dequantize + F.linear).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _int4_dequant_matmul_pytorch(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
) -> torch.Tensor:
    """PyTorch fallback: unpack INT4 → dequantize → matmul."""
    # w_packed: [N//2, K] uint8 — two int4 values per byte
    N2, K = w_packed.shape
    N = N2 * 2

    # Unpack: low nibble = even rows, high nibble = odd rows
    w_low  = (w_packed.to(torch.int16) & 0x0F).to(torch.int8) - 8  # [N//2, K]
    w_high = (w_packed.to(torch.int16) >> 4).to(torch.int8) - 8     # [N//2, K]

    # Interleave: row 0 = w_low[0], row 1 = w_high[0], row 2 = w_low[1], ...
    w_q = torch.empty(N, K, dtype=torch.int8, device=w_packed.device)
    w_q[0::2] = w_low
    w_q[1::2] = w_high

    # Determine group_size from shapes
    group_size = K // group_scales.shape[1]
    if N != group_scales.shape[0]:
        raise ValueError(
            f"group_scales N={group_scales.shape[0]} != unpacked N={N}"
        )

    # Dequantize: w_fp16 = w_q * group_scale
    w_float = w_q.float()
    scales_expanded = group_scales.float().unsqueeze(-1).expand(
        N, group_scales.shape[1], group_size
    ).reshape(N, K)
    w_fp16 = (w_float * scales_expanded).to(x.dtype)

    return F.linear(x, w_fp16)


# Runtime selection
try:
    from mini_llm_kernels._C import int4_dequant_matmul as _cuda_int4_matmul
    _HAS_INT4_CUDA = True
except ImportError:
    _HAS_INT4_CUDA = False


def int4_dequant_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
) -> torch.Tensor:
    """Fused INT4 dequantize + matmul.

    Args:
        x: [..., K] fp16/bf16 input activations.
        w_packed: [N//2, K] uint8 packed INT4 weights.
        group_scales: [N, K//group_size] fp16/bf16 group-wise scales.

    Returns:
        [..., N] output.
    """
    if _HAS_INT4_CUDA and x.is_cuda:
        return _cuda_int4_matmul(x, w_packed, group_scales)
    return _int4_dequant_matmul_pytorch(x, w_packed, group_scales)
