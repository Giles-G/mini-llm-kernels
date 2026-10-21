"""Parity tests for the tensor-core and parallel INT4 batch kernels."""
import pytest
import torch

from mini_llm_kernels.kernels import int4_matmul as mod


def _inputs(n, k, batch, dtype=torch.float16):
    torch.manual_seed(11)
    x = torch.randn(batch, k, device="cuda", dtype=dtype) * 0.1
    q = torch.randint(0, 16, (n // 2, k), device="cuda", dtype=torch.uint8)
    scales = (
        torch.rand(n, k // 64, device="cuda", dtype=torch.float16) * 0.02
        + 0.001
    )
    return x, q, scales


def _fp32_reference(x, q, scales):
    n, k = q.shape[0] * 2, q.shape[1]
    low = (q.to(torch.int16) & 0x0F).to(torch.int16) - 8
    high = (q.to(torch.int16) >> 4).to(torch.int16) - 8
    w = torch.stack((low, high), dim=1).reshape(n, k).float()
    w = w * scales.float().repeat_interleave(64, dim=1)
    return x.float() @ w.t()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("n,k", [(64, 64), (1536, 1536), (1536, 6144)])
@pytest.mark.parametrize("batch", [2, 4])
def test_tc_matches_pair_batch(n, k, batch):
    x, q, scales = _inputs(n, k, batch)
    ref = mod.int4_dequant_matmul_pair_batch(x, q, scales)
    out = mod.int4_dequant_matmul_tc(x, q, scales)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("n,k", [(64, 64), (1536, 1536), (1536, 6144)])
@pytest.mark.parametrize("batch", [8, 16])
def test_tc_matches_fp32_reference(n, k, batch):
    x, q, scales = _inputs(n, k, batch)
    ref = _fp32_reference(x, q, scales)
    out = mod.int4_dequant_matmul_tc(x, q, scales)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("batch", range(2, 17))
def test_tc_supports_every_batch_size(batch):
    x, q, scales = _inputs(1536, 1536, batch)
    ref = _fp32_reference(x, q, scales)
    out = mod.int4_dequant_matmul_tc(x, q, scales)
    assert tuple(out.shape) == (batch, 1536)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("n,k", [(64, 64), (1536, 1536), (1536, 6144)])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_parallel_pair_batch_exact(n, k, batch):
    x, q, scales = _inputs(n, k, batch)
    out = mod.int4_dequant_matmul_pair_batch_parallel(x, q, scales)
    if batch == 1:
        ref = mod._cuda_int4_matmul_pair_vec8(x, q, scales, 64)
    else:
        ref = mod.int4_dequant_matmul_pair_batch(x, q, scales)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tc_rejects_unsupported_batch():
    x, q, scales = _inputs(1536, 1536, 17)
    with pytest.raises(ValueError):
        mod.int4_dequant_matmul_tc(x, q, scales)
