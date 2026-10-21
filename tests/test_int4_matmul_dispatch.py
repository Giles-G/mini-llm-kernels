from unittest.mock import Mock

import pytest
import torch

from mini_llm_kernels.kernels import int4_matmul as mod


def _inputs(device="cpu", dtype=torch.float16, batch_shape=(2,), n=6, k=64):
    torch.manual_seed(7)
    x = torch.randn(*batch_shape, k, device=device, dtype=dtype)
    q = torch.randint(0, 16, (n // 2, k), device=device, dtype=torch.uint8)
    scales = torch.rand(n, k // 64, device=device, dtype=torch.float16)
    return x, q, scales


def test_dispatch_defaults_to_pair_vec8(monkeypatch):
    x, q, scales = _inputs()
    calls = []
    fake_x = Mock(is_cuda=True, shape=(1, 64))
    monkeypatch.setattr(mod, "_HAS_INT4_CUDA", True)
    monkeypatch.setattr(mod, "_cuda_int4_matmul_legacy",
                        lambda *args: calls.append("legacy") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul",
                        lambda *args: calls.append("warp") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair",
                        lambda *args: calls.append("pair") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec",
                        lambda *args: calls.append("pair_vec") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec8",
                        lambda *args: calls.append("pair_vec8") or torch.empty(2, 6))
    monkeypatch.delenv("MINI_LLM_INT4_IMPL", raising=False)
    mod.int4_dequant_matmul(fake_x, q, scales)
    assert calls == ["pair_vec8"]


def test_dispatch_falls_back_to_pair_without_pair_vec(monkeypatch):
    x, q, scales = _inputs()
    calls = []
    fake_x = Mock(is_cuda=True, shape=x.shape)
    monkeypatch.setattr(mod, "_HAS_INT4_CUDA", True)
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair",
                        lambda *args: calls.append("pair") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec", None)
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec8", None)
    monkeypatch.delenv("MINI_LLM_INT4_IMPL", raising=False)
    mod.int4_dequant_matmul(fake_x, q, scales)
    assert calls == ["pair"]


def test_pair_vec8_falls_back_to_pair_vec_for_batch(monkeypatch):
    x, q, scales = _inputs()
    calls = []
    fake_x = Mock(is_cuda=True, shape=(2, 64))
    monkeypatch.setenv("MINI_LLM_INT4_IMPL", "pair_vec8")
    monkeypatch.setattr(mod, "_HAS_INT4_CUDA", True)
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec",
                        lambda *args: calls.append("pair_vec") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec8",
                        lambda *args: calls.append("pair_vec8") or torch.empty(2, 6))
    mod.int4_dequant_matmul(fake_x, q, scales)
    assert calls == ["pair_vec"]


@pytest.mark.parametrize("impl", ["legacy", "warp", "pair", "pair_vec", "pair_vec8"])
def test_dispatch_selects_requested_cuda_kernel(monkeypatch, impl):
    x, q, scales = _inputs()
    calls = []
    fake_x = Mock(is_cuda=True, shape=(1, 64))
    monkeypatch.setenv("MINI_LLM_INT4_IMPL", impl)
    monkeypatch.setattr(mod, "_HAS_INT4_CUDA", True)
    monkeypatch.setattr(mod, "_cuda_int4_matmul_legacy",
                        lambda *args: calls.append("legacy") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul",
                        lambda *args: calls.append("warp") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair",
                        lambda *args: calls.append("pair") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec",
                        lambda *args: calls.append("pair_vec") or torch.empty(2, 6))
    monkeypatch.setattr(mod, "_cuda_int4_matmul_pair_vec8",
                        lambda *args: calls.append("pair_vec8") or torch.empty(2, 6))
    mod.int4_dequant_matmul(fake_x, q, scales)
    assert calls == [impl]


def test_dispatch_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("MINI_LLM_INT4_IMPL", "bogus")
    x, q, scales = _inputs()
    with pytest.raises(ValueError, match="legacy, warp, pair, pair_vec, pair_vec8"):
        mod.int4_dequant_matmul(x, q, scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("n", [2, 6, 130])
@pytest.mark.parametrize("k", [64, 1536, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_cuda_legacy_warp_match(n, k, dtype, batch_shape):
    x, q, scales = _inputs("cuda", dtype, batch_shape, n, k)
    old = mod._cuda_int4_matmul_legacy(x, q, scales, 64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        new = mod._cuda_int4_matmul(x, q, scales, 64)
        new_pair = mod._cuda_int4_matmul_pair(x, q, scales, 64)
    torch.cuda.synchronize()
    diff = (new.float() - old.float()).abs()
    scale = old.float().abs().clamp_min(1e-3)
    assert torch.isfinite(new).all() and torch.isfinite(old).all()
    assert torch.all(diff <= 5e-2 + 5e-3 * scale)
    torch.testing.assert_close(
        new_pair.float(), old.float(), atol=2e-2, rtol=2e-2
    )

    # The oracle is explicitly FP32 and TF32 is disabled for this check.
    q_signed = torch.stack(
        ((q & 0x0F).to(torch.float32) - 8,
         (q >> 4).to(torch.float32) - 8), dim=1,
    ).reshape(n, k)
    scales_fp32 = scales.float().repeat_interleave(64, dim=1)
    oracle_w = q_signed * scales_fp32
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        oracle = torch.nn.functional.linear(x.float(), oracle_w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert torch.allclose(old.float(), oracle, rtol=2e-2, atol=5e-2)
    assert torch.allclose(new.float(), oracle, rtol=2e-2, atol=5e-2)
    assert torch.allclose(new_pair.float(), oracle, rtol=2e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_empty_batch_and_nondefault_stream():
    x, q, scales = _inputs("cuda", torch.float16, (0,), 6, 64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = mod._cuda_int4_matmul_legacy(x, q, scales, 64)
        out_pair = mod._cuda_int4_matmul_pair(x, q, scales, 64)
    stream.synchronize()
    assert out.shape == (0, 6)
    assert out_pair.shape == (0, 6)
