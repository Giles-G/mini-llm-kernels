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
import os


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

    group_size = K // group_scales.shape[1]
    if K % group_scales.shape[1] != 0:
        raise ValueError("K must be divisible by the number of scale groups")
    if group_scales.shape[0] != N:
        raise ValueError(
            f"group_scales N={group_scales.shape[0]} != unpacked N={N}"
        )

    # Dequantize: w_rounded = round_to_activation(w_q * group_scale).
    # The fallback contract is "rounded weight, FP32 matmul, cast output"; a
    # BF16 tensor-core F.linear is a different contract and exceeded the FP32
    # budget on per_layer_model_projection (107/170240 elements).
    w_float = w_q.float()
    scales_expanded = group_scales.float().unsqueeze(-1).expand(
        N, group_scales.shape[1], group_size
    ).reshape(N, K)
    w_rounded = (w_float * scales_expanded).to(x.dtype)

    return F.linear(x.float(), w_rounded.float()).to(x.dtype)


# Runtime selection
try:
    from mini_llm_kernels._C import int4_dequant_matmul as _cuda_int4_matmul
    from mini_llm_kernels._C import int4_dequant_matmul_legacy as _cuda_int4_matmul_legacy
    from mini_llm_kernels._C import int4_dequant_matmul_pair as _cuda_int4_matmul_pair
    _HAS_INT4_CUDA = True
except ImportError:
    _cuda_int4_matmul = None
    _cuda_int4_matmul_legacy = None
    _cuda_int4_matmul_pair = None
    _HAS_INT4_CUDA = False

# pair_vec and the attribute query were added later than pair; import them
# independently so an older extension still runs the scalar pair kernel
# instead of silently disabling all CUDA INT4 dispatch.
try:
    from mini_llm_kernels._C import int4_dequant_matmul_pair_vec as _cuda_int4_matmul_pair_vec
except ImportError:
    _cuda_int4_matmul_pair_vec = None
try:
    from mini_llm_kernels._C import int4_pair_kernel_attributes as _int4_pair_kernel_attributes
except ImportError:
    _int4_pair_kernel_attributes = None
try:
    from mini_llm_kernels._C import int4_dequant_dense as _cuda_int4_dequant_dense
except ImportError:
    _cuda_int4_dequant_dense = None
try:
    from mini_llm_kernels._C import int4_dequant_matmul_pair_vec8 as _cuda_int4_matmul_pair_vec8
except ImportError:
    _cuda_int4_matmul_pair_vec8 = None
try:
    from mini_llm_kernels._C import int4_dequant_matmul_pair_batch as _cuda_int4_matmul_pair_batch
except ImportError:
    _cuda_int4_matmul_pair_batch = None
try:
    from mini_llm_kernels._C import int4_dequant_matmul_pair_batch_parallel as _cuda_int4_matmul_pair_batch_parallel
except ImportError:
    _cuda_int4_matmul_pair_batch_parallel = None
try:
    from mini_llm_kernels._C import int4_dequant_matmul_tc as _cuda_int4_matmul_tc
except ImportError:
    _cuda_int4_matmul_tc = None


def int4_dequant_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
) -> torch.Tensor:
    """Fused INT4 dequantize + matmul.

    CUDA defaults to the vectorized row-pair kernel: one warp computes
    adjacent output rows, reuses each packed byte for both nibbles, and loads
    four packed columns per lane with one aligned 32-bit word. On the RTX
    3060 it cut lm_head from 1.57 ms (warp-per-row) to about 1.0 ms and
    gate/up from 0.034 to 0.024 ms. All four implementations (legacy, warp,
    pair, pair_vec) are selectable via ``MINI_LLM_INT4_IMPL``.

    Args:
        x: [..., K] fp16/bf16 input activations.
        w_packed: [N//2, K] uint8 packed INT4 weights.
        group_scales: [N, K//group_size] fp16/bf16 group-wise scales.

    Returns:
        [..., N] output.
    """
    K = x.shape[-1]
    group_size = K // group_scales.shape[1]
    impl = os.environ.get("MINI_LLM_INT4_IMPL", "pair_vec8").strip().lower()
    if impl not in {"legacy", "warp", "pair", "pair_vec", "pair_vec8"}:
        raise ValueError(
            "MINI_LLM_INT4_IMPL must be one of: legacy, warp, pair, "
            "pair_vec, pair_vec8"
        )
    if _HAS_INT4_CUDA and x.is_cuda and group_size == 64:
        if impl == "pair_vec" and (
            _cuda_int4_matmul_pair_vec is None or K % 4 != 0
        ):
            impl = "pair"
        if impl == "pair_vec8":
            batch_elems = 1
            for dim in x.shape[:-1]:
                batch_elems *= int(dim)
            if (
                _cuda_int4_matmul_pair_vec8 is None
                or K % 8 != 0
                or batch_elems != 1
            ):
                impl = "pair_vec" if _cuda_int4_matmul_pair_vec is not None else "pair"
        kernel = {
            "legacy": _cuda_int4_matmul_legacy,
            "warp": _cuda_int4_matmul,
            "pair": _cuda_int4_matmul_pair,
            "pair_vec": _cuda_int4_matmul_pair_vec,
            "pair_vec8": _cuda_int4_matmul_pair_vec8,
        }[impl]
        return kernel(x, w_packed, group_scales, group_size)
    return _int4_dequant_matmul_pytorch(x, w_packed, group_scales)


def int4_dequant_matmul_pair_batch(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
) -> torch.Tensor:
    """Weight-stationary INT4 GEMV for 2..4 activation rows."""
    if (
        _cuda_int4_matmul_pair_batch is None
        or not x.is_cuda
        or x.dim() != 2
        or x.shape[0] < 2
        or x.shape[0] > 4
        or x.shape[1] % 8
    ):
        raise ValueError("pair_batch path unavailable for this input")
    group_size = x.shape[1] // group_scales.shape[1]
    if group_size != 64:
        raise ValueError("pair_batch requires group_size=64")
    return _cuda_int4_matmul_pair_batch(x, w_packed, group_scales, group_size)


def int4_dequant_matmul_pair_batch_parallel(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
) -> torch.Tensor:
    """Parallel weight-stationary INT4 GEMV for 1..4 activation rows."""
    if (
        _cuda_int4_matmul_pair_batch_parallel is None
        or not x.is_cuda
        or x.dim() != 2
        or x.shape[0] < 1
        or x.shape[0] > 4
        or x.shape[1] % 16
        or (w_packed.shape[0] * 2) % 8
    ):
        raise ValueError("parallel pair_batch path unavailable for this input")
    group_size = x.shape[1] // group_scales.shape[1]
    if group_size != 64:
        raise ValueError("parallel pair_batch requires group_size=64")
    return _cuda_int4_matmul_pair_batch_parallel(
        x, w_packed, group_scales, group_size
    )


def int4_dequant_matmul_tc(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
) -> torch.Tensor:
    """Tensor-core tile-dequant INT4 matmul for 2..16 activation rows."""
    if (
        _cuda_int4_matmul_tc is None
        or not x.is_cuda
        or x.dim() != 2
        or x.dtype != torch.float16
        or x.shape[0] < 2
        or x.shape[0] > 16
        or x.shape[1] % 16
        or (w_packed.shape[0] * 2) % 16
    ):
        raise ValueError("tensor-core batch path unavailable for this input")
    group_size = x.shape[1] // group_scales.shape[1]
    if group_size != 64:
        raise ValueError("tensor-core batch path requires group_size=64")
    return _cuda_int4_matmul_tc(x, w_packed, group_scales, group_size)


def int4_dequant_dense(
    w_packed: torch.Tensor,
    group_scales: torch.Tensor,
    compute_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Fast INT4 unpack to a dense activation-dtype weight matrix."""
    if _cuda_int4_dequant_dense is not None and w_packed.is_cuda:
        group_size = w_packed.shape[1] // group_scales.shape[1]
        return _cuda_int4_dequant_dense(
            w_packed, group_scales, group_size,
            compute_dtype == torch.bfloat16,
        )
    # CPU/reference fallback.
    N2, K = w_packed.shape
    N = N2 * 2
    low = (w_packed.to(torch.int16) & 0x0F).to(torch.int8) - 8
    high = (w_packed.to(torch.int16) >> 4).to(torch.int8) - 8
    w_q = torch.empty((N, K), dtype=torch.int8, device=w_packed.device)
    w_q[0::2] = low
    w_q[1::2] = high
    group_size = K // group_scales.shape[1]
    w = w_q.view(N, group_scales.shape[1], group_size).to(compute_dtype)
    w = w * group_scales.to(compute_dtype).unsqueeze(-1)
    return w.view(N, K)


def int4_pair_kernel_attributes(
    pairs_per_block: int = 4,
    dtype: torch.dtype = torch.float16,
    vectorized: bool = True,
) -> dict:
    """Register/shared-memory/occupancy attributes for the row-pair kernels.

    This is the non-Nsight counter path: CUDA's own function attributes and
    occupancy calculator report registers per thread and active blocks/SM.
    """
    if not _HAS_INT4_CUDA or _int4_pair_kernel_attributes is None:
        return {"available": False}
    values = _int4_pair_kernel_attributes(
        pairs_per_block, dtype == torch.float16, vectorized
    )
    keys = (
        "num_regs", "shared_bytes", "local_bytes", "max_threads_per_block",
        "blocks_per_sm", "active_threads_per_sm", "binary_version", "ptx_version",
    )
    return {"available": True, **dict(zip(keys, values))}
