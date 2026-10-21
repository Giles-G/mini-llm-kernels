"""Parity tests for the Gemma4 safe paged decode-attention kernel."""

from __future__ import annotations

import pytest
import torch

from mini_llm_kernels.kernels.gemma4_attention import (
    _HAS_GEMMA4_ATTN,
    _cuda_gemma4_attention,
    _gemma4_decode_attention_pytorch,
    cuda_attention_status,
    cuda_attention_usable,
)


def _case(dtype, D, H_q, H_kv, block_size, ctx, window, N=2):
    max_blocks = (ctx + block_size - 1) // block_size + 1
    num_blocks = N * max_blocks + 2
    q = torch.randn(N, H_q, D, device="cuda", dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, H_kv, D, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, H_kv, D, device="cuda", dtype=dtype)
    table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1).repeat(N, 1)
    for r in range(N):
        table[r] = torch.roll(table[r], r * max_blocks)
    table = table[:, :max_blocks].contiguous()
    lens = torch.full((N,), ctx, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, table, lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not _HAS_GEMMA4_ATTN, reason="Gemma4 attention extension missing")
@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("H_q,H_kv", [(8, 1), (8, 2)])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("ctx", [1, 17, 129])
@pytest.mark.parametrize("window", [0, 8, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_safe_kernel_matches_reference(D, H_q, H_kv, block_size, ctx, window, dtype):
    torch.manual_seed(5)
    q, k_cache, v_cache, table, lens = _case(
        dtype, D, H_q, H_kv, block_size, ctx, window
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = _cuda_gemma4_attention(q, k_cache, v_cache, table, lens, window)
    stream.synchronize()
    ref = _gemma4_decode_attention_pytorch(q, k_cache, v_cache, table, lens, window)
    tol = 0.05 if dtype == torch.float16 else 0.5
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not _HAS_GEMMA4_ATTN, reason="Gemma4 attention extension missing")
@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("ctx", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_safe_kernel_long_context(D, ctx, window, dtype):
    torch.manual_seed(11)
    q, k_cache, v_cache, table, lens = _case(
        dtype, D, 8, 1, 128, ctx, window
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = _cuda_gemma4_attention(q, k_cache, v_cache, table, lens, window)
    stream.synchronize()
    ref = _gemma4_decode_attention_pytorch(q, k_cache, v_cache, table, lens, window)
    tol = 0.05 if dtype == torch.float16 else 0.5
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not _HAS_GEMMA4_ATTN, reason="Gemma4 attention extension missing")
def test_safe_kernel_status_is_enabled():
    assert cuda_attention_usable()
    assert cuda_attention_status().startswith("enabled")


def _warpsplit():
    from mini_llm_kernels import _C
    return getattr(_C, "gemma4_decode_attention_warpsplit", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not _HAS_GEMMA4_ATTN, reason="Gemma4 attention extension missing")
@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("ctx", [1, 17, 129, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_warpsplit_kernel_matches_reference(D, ctx, window, dtype):
    fn = _warpsplit()
    if fn is None:
        pytest.skip("warpsplit binding not built")
    torch.manual_seed(19)
    q, k_cache, v_cache, table, lens = _case(
        dtype, D, 8, 1, 128, ctx, window
    )
    out = fn(q, k_cache, v_cache, table, lens, window)
    ref = _gemma4_decode_attention_pytorch(q, k_cache, v_cache, table, lens, window)
    tol = 0.05 if dtype == torch.float16 else 0.5
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=2e-2)


def _splitk_out(q, k_cache, v_cache, table, lens, window, chunk_size=512):
    from mini_llm_kernels import _C
    fn = getattr(_C, "gemma4_decode_attention_splitk", None)
    if fn is None:
        return None
    N, Hq, D = q.shape
    capacity = table.shape[1] * k_cache.shape[1]
    chunks = max(1, (capacity + chunk_size - 1) // chunk_size)
    po = torch.empty((N, Hq, chunks, D), device="cuda", dtype=torch.float32)
    pm = torch.empty((N, Hq, chunks), device="cuda", dtype=torch.float32)
    pl = torch.empty((N, Hq, chunks), device="cuda", dtype=torch.float32)
    return fn(q, k_cache, v_cache, table, lens, window, po, pm, pl, chunk_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not _HAS_GEMMA4_ATTN, reason="Gemma4 attention extension missing")
@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("ctx", [17, 513, 1025, 2048, 4096])
@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_splitk_kernel_matches_reference(D, ctx, window, dtype):
    torch.manual_seed(23)
    q, k_cache, v_cache, table, lens = _case(dtype, D, 8, 1, 128, ctx, window)
    out = _splitk_out(q, k_cache, v_cache, table, lens, window)
    if out is None:
        pytest.skip("splitk binding not built")
    ref = _gemma4_decode_attention_pytorch(q, k_cache, v_cache, table, lens, window)
    tol = 0.05 if dtype == torch.float16 else 0.5
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=2e-2)
