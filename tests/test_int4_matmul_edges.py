"""INT4 GEMV edge-case coverage.

These tests target the cases the shape-based synthetic benchmark does not
enumerate: zero activations, cancellation, very small/large values, mixed
scales, N tails that do not fill a warp/block, multi-token batches, empty
batches, non-default streams, and dispatch fallback for unsupported groups.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mini_llm_kernels.kernels import int4_matmul as mod


def _pack(q: torch.Tensor) -> torch.Tensor:
    unsigned = (q.to(torch.int16) + 8).to(torch.uint8)
    return unsigned[0::2] | (unsigned[1::2] << 4)


def _case(pattern: str, n: int, k: int, tokens: int, dtype: torch.dtype, device: str):
    torch.manual_seed(1000 + n + k + tokens)
    q = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    if pattern == "zero":
        x = torch.zeros((tokens, k), dtype=dtype)
        scales = torch.full((n, k // 64), 0.05, dtype=torch.float16)
    elif pattern == "cancel":
        x = torch.empty((tokens, k), dtype=dtype)
        x[:, 0::2] = 1.0
        x[:, 1::2] = -1.0
        # 7*1 + 7*(-1) = 0 for every adjacent pair.
        q = torch.full((n, k), 7, dtype=torch.int8)
        scales = torch.full((n, k // 64), 0.125, dtype=torch.float16)
    elif pattern == "large":
        x = (torch.rand((tokens, k), dtype=torch.float32) * 100.0 - 50.0).to(dtype)
        scales = torch.full((n, k // 64), 0.05, dtype=torch.float16)
    elif pattern == "small":
        x = (torch.rand((tokens, k), dtype=torch.float32) * 2e-4 - 1e-4).to(dtype)
        scales = torch.full((n, k // 64), 1e-3, dtype=torch.float16)
    elif pattern == "mixed_scales":
        x = torch.randn((tokens, k), dtype=dtype)
        scales = torch.empty((n, k // 64), dtype=torch.float16)
        scales[:, 0::2] = 1e-4
        scales[:, 1::2] = 0.1
    else:
        raise AssertionError(pattern)

    packed = _pack(q).to(device)
    x = x.to(device)
    scales = scales.to(device)
    q = q.to(device)
    return x, packed, scales, q


def _oracle(x, q, scales):
    # CPU FP32 reference: no TF32 and no CUDA matmul path involved.
    group = x.shape[-1] // scales.shape[1]
    w = q.cpu().float() * scales.cpu().float().repeat_interleave(group, dim=1)
    return F.linear(x.detach().cpu().float(), w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("pattern", ["zero", "cancel", "large", "small", "mixed_scales"])
@pytest.mark.parametrize("n", [2, 6, 130, 258])
@pytest.mark.parametrize("k", [64, 128, 1536])
@pytest.mark.parametrize("tokens", [1, 19])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_edge_patterns_legacy_warp_reference(pattern, n, k, tokens, dtype):
    x, packed, scales, q = _case(pattern, n, k, tokens, dtype, "cuda")
    legacy = mod._cuda_int4_matmul_legacy(x, packed, scales, 64)
    warp = mod._cuda_int4_matmul(x, packed, scales, 64)
    pair = mod._cuda_int4_matmul_pair(x, packed, scales, 64)
    vec = mod._cuda_int4_matmul_pair_vec(x, packed, scales, 64)
    exact = _oracle(x, q, scales)

    assert legacy.shape == warp.shape == pair.shape == vec.shape == (tokens, n)
    assert torch.isfinite(legacy).all() and torch.isfinite(warp).all()
    assert torch.isfinite(pair).all() and torch.isfinite(vec).all()

    # legacy/warp/pair/pair_vec must agree tightly; all use FP32 accumulation.
    torch.testing.assert_close(warp.float(), legacy.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pair.float(), legacy.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(vec.float(), legacy.float(), atol=2e-2, rtol=2e-2)

    # The reference is the mathematically exact dequantized weight in FP32;
    # the kernel output is rounded to the activation dtype at the end.
    tolerance = 2e-2 if dtype == torch.float16 else 1.0
    torch.testing.assert_close(
        legacy.cpu().float(), exact, atol=tolerance, rtol=2e-2
    )
    torch.testing.assert_close(
        warp.cpu().float(), exact, atol=tolerance, rtol=2e-2
    )
    torch.testing.assert_close(
        pair.cpu().float(), exact, atol=tolerance, rtol=2e-2
    )
    torch.testing.assert_close(
        vec.cpu().float(), exact, atol=tolerance, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_warp_empty_batch_and_non_default_stream():
    x, packed, scales, _ = _case("mixed_scales", 6, 64, 1, torch.float16, "cuda")
    empty = torch.empty((0, 64), device="cuda", dtype=torch.float16)
    out_empty = mod._cuda_int4_matmul(empty, packed, scales, 64)
    out_empty_pair = mod._cuda_int4_matmul_pair(empty, packed, scales, 64)
    assert out_empty.shape == (0, 6)
    assert out_empty_pair.shape == (0, 6)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = mod._cuda_int4_matmul(x, packed, scales, 64)
        out_pair = mod._cuda_int4_matmul_pair(x, packed, scales, 64)
    stream.synchronize()
    torch.testing.assert_close(
        out.float(), mod._cuda_int4_matmul_legacy(x, packed, scales, 64).float(),
        atol=2e-2, rtol=2e-2,
    )
    torch.testing.assert_close(
        out_pair.float(), out.float(), atol=2e-2, rtol=2e-2,
    )


def test_unsupported_group_size_falls_back_on_cpu():
    torch.manual_seed(17)
    x = torch.randn(2, 32, dtype=torch.float16)
    q = torch.randint(-8, 8, (12, 32), dtype=torch.int8)
    scales = torch.rand(12, 1, dtype=torch.float16)
    packed = _pack(q)
    expected = mod._int4_dequant_matmul_pytorch(x, packed, scales)
    actual = mod.int4_dequant_matmul(x, packed, scales)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unsupported_group_size_falls_back_on_cuda():
    torch.manual_seed(18)
    x = torch.randn(1, 32, device="cuda", dtype=torch.float16)
    q = torch.randint(-8, 8, (12, 32), dtype=torch.int8)
    scales = torch.rand(12, 1, device="cuda", dtype=torch.float16)
    packed = _pack(q).to("cuda")
    expected = mod._int4_dequant_matmul_pytorch(x, packed, scales)
    actual = mod.int4_dequant_matmul(x, packed, scales)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pair_vec_kernel_attributes_and_alignment_guard():
    attrs = mod.int4_pair_kernel_attributes(
        pairs_per_block=4, dtype=torch.float16, vectorized=True
    )
    assert attrs["available"] is True
    assert attrs["num_regs"] > 0
    assert attrs["blocks_per_sm"] > 0
    assert attrs["active_threads_per_sm"] >= 128

    # group_size=64 implies K % 4 == 0, so every valid Gemma4 shape can use
    # the vectorized path. The direct binding still rejects a malformed K
    # before it can issue an unaligned 32-bit load.
    x, packed, scales, _ = _case("zero", 6, 66, 1, torch.float16, "cuda")
    with pytest.raises(RuntimeError, match="divisible by 4"):
        mod._cuda_int4_matmul_pair_vec(x, packed, scales, 64)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fast_dense_dequant_matches_unpack(dtype):
    _x, packed, scales, q = _case("large", 6, 64, 1, dtype, "cuda")
    if getattr(mod, "_cuda_int4_dequant_dense", None) is None:
        pytest.skip("fast dequant binding not built")
    out = mod.int4_dequant_dense(packed, scales, dtype)
    low = (packed.to(torch.int16) & 0x0F).to(torch.float32) - 8
    high = (packed.to(torch.int16) >> 4).to(torch.float32) - 8
    n = packed.shape[0] * 2
    ref = torch.empty((n, 64), device="cuda", dtype=dtype)
    expanded = scales.float().repeat_interleave(64, dim=1)
    ref[0::2] = (low * expanded[0::2]).to(dtype)
    ref[1::2] = (high * expanded[1::2]).to(dtype)
    assert out.shape == (n, 64) and out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("pattern", ["zero", "large", "mixed_scales"])
@pytest.mark.parametrize("n", [6, 130, 258])
@pytest.mark.parametrize("k", [64, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pair_vec8_matches_pair_vec(pattern, n, k, dtype):
    x, packed, scales, _ = _case(pattern, n, k, 1, dtype, "cuda")
    out8 = mod._cuda_int4_matmul_pair_vec8(x, packed, scales, 64)
    ref = mod._cuda_int4_matmul_pair_vec(x, packed, scales, 64)
    torch.testing.assert_close(out8.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("tokens", [2, 3, 4])
@pytest.mark.parametrize("n", [6, 130, 258])
@pytest.mark.parametrize("k", [64, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pair_batch_matches_pair_vec(tokens, n, k, dtype):
    x, packed, scales, _ = _case("large", n, k, tokens, dtype, "cuda")
    out = mod.int4_dequant_matmul_pair_batch(x, packed, scales)
    ref = mod._cuda_int4_matmul_pair_vec(x, packed, scales, 64)
    assert out.shape == (tokens, n)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
