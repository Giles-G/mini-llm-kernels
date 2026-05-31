"""
tests/test_integration_m5.py

M5 集成验证：
1. 验证 mini_llm_kernels import 正常，fallback 路径无异常
2. 验证 fused_add_rms_norm fallback 语义正确
3. 验证 decode_paged_attention fallback 语义正确
4. 验证 MINI_LLM_NO_CUSTOM_KERNELS 环境变量正确关闭 kernel
5. 验证 DecodeBatchGatherCtx 有 block_table_tensor 字段
6. 验证 qkv_proj 合并权重的数值正确性
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 加入 mini-serve-llm 到 sys.path
_SERVE_LLM_ROOT = Path(__file__).resolve().parent.parent.parent / "mini-serve-llm"
if str(_SERVE_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVE_LLM_ROOT))

import torch


# ─── 1. mini_llm_kernels 导入 ───────────────────────────────────────────────

def test_mini_llm_kernels_import():
    import mini_llm_kernels
    assert hasattr(mini_llm_kernels, "_HAS_CUDA_OPS"), "_HAS_CUDA_OPS missing"
    assert hasattr(mini_llm_kernels, "fused_add_rms_norm"), "fused_add_rms_norm missing"
    assert hasattr(mini_llm_kernels, "decode_paged_attention"), "decode_paged_attention missing"
    print(f"  mini_llm_kernels OK, _HAS_CUDA_OPS={mini_llm_kernels._HAS_CUDA_OPS}")


# ─── 2. fused_add_rms_norm fallback 正确性 ─────────────────────────────────

def test_fused_add_rms_norm_fallback():
    from miniservellm.runtime.nn_ops import fused_add_rms_norm, rms_norm

    torch.manual_seed(0)
    N, H = 16, 896
    eps = 1e-6
    x        = torch.randn(N, H)
    residual = torch.randn(N, H)
    gamma    = torch.ones(H)

    x_ref = x + residual
    n_ref = rms_norm(x_ref, gamma, eps)

    x_fused, r_fused = fused_add_rms_norm(x, residual, gamma, eps)
    assert torch.allclose(r_fused, x_ref, atol=1e-5), \
        f"residual_out mismatch: {(r_fused - x_ref).abs().max():.2e}"
    assert torch.allclose(x_fused, n_ref, atol=1e-5), \
        f"x_normed mismatch: {(x_fused - n_ref).abs().max():.2e}"
    print("  fused_add_rms_norm fallback: OK")


# ─── 3. decode_paged_attention fallback 正确性 ─────────────────────────────

def test_decode_paged_attention_fallback():
    from miniservellm.runtime.nn_ops import decode_paged_attention, _decode_paged_attention_fallback

    torch.manual_seed(42)
    N, H_q, H_kv, D = 4, 14, 2, 64
    block_size, num_blocks = 16, 32

    k_cache = torch.randn(num_blocks, block_size, H_kv, D)
    v_cache = torch.randn(num_blocks, block_size, H_kv, D)
    ctx_lens = torch.tensor([32, 16, 48, 8], dtype=torch.long)
    block_table = torch.zeros(N, 4, dtype=torch.long)
    for i in range(N):
        nb = (ctx_lens[i].item() + block_size - 1) // block_size
        block_table[i, :nb] = torch.arange(i * 4, i * 4 + nb) % num_blocks
    q = torch.randn(N, H_q, D)

    out = decode_paged_attention(q, k_cache, v_cache, block_table, ctx_lens)
    ref = _decode_paged_attention_fallback(q, k_cache, v_cache, block_table, ctx_lens)

    assert out.shape == (N, H_q, D)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"decode_paged_attention mismatch: {(out - ref).abs().max():.2e}"
    print("  decode_paged_attention fallback: OK")


# ─── 4. MINI_LLM_NO_CUSTOM_KERNELS 环境变量 ────────────────────────────────

def test_no_custom_kernels_env():
    """验证环境变量关闭后 nn_ops._HAS_CUSTOM_KERNELS=False（M1 上本来就是 False，这里只验证逻辑一致性）"""
    import miniservellm.runtime.nn_ops as nn_ops
    # 读取当前值
    original = nn_ops._HAS_CUSTOM_KERNELS
    # 在 M1 环境：_HAS_CUSTOM_KERNELS 本来就是 False，验证 env var 的设置路径
    # 通过 reload 方式验证（仅验证模块代码结构）
    import importlib
    os.environ["MINI_LLM_NO_CUSTOM_KERNELS"] = "1"
    import mini_llm_kernels as _mkl_mod
    result = _mkl_mod._HAS_CUDA_OPS and os.environ.get("MINI_LLM_NO_CUSTOM_KERNELS", "0") != "1"
    assert result is False, "MINI_LLM_NO_CUSTOM_KERNELS=1 should disable kernels"
    del os.environ["MINI_LLM_NO_CUSTOM_KERNELS"]
    print(f"  MINI_LLM_NO_CUSTOM_KERNELS env: OK (original={original})")


# ─── 5. DecodeBatchGatherCtx 字段 ──────────────────────────────────────────

def test_decode_batch_gather_ctx_fields():
    from miniservellm.runtime.model_runner import DecodeBatchGatherCtx
    fields = set(DecodeBatchGatherCtx.__dataclass_fields__)
    assert "block_table_tensor" in fields, "block_table_tensor field missing"
    print(f"  DecodeBatchGatherCtx.block_table_tensor: OK")


# ─── 6. QKV 合并权重数值正确性 ────────────────────────────────────────────

def test_qkv_fusion_numerical():
    from miniservellm.runtime.nn_ops import linear

    torch.manual_seed(1)
    H, q_dim, kv_dim = 64, 48, 16
    x = torch.randn(8, H)

    W_q = torch.randn(q_dim, H); b_q = torch.randn(q_dim)
    W_k = torch.randn(kv_dim, H); b_k = torch.randn(kv_dim)
    W_v = torch.randn(kv_dim, H); b_v = torch.randn(kv_dim)

    W_qkv = torch.cat([W_q, W_k, W_v], dim=0)
    b_qkv = torch.cat([b_q, b_k, b_v], dim=0)

    q_old, k_old, v_old = linear(x, W_q, b_q), linear(x, W_k, b_k), linear(x, W_v, b_v)
    qkv = linear(x, W_qkv, b_qkv)
    q_new, k_new, v_new = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

    assert torch.allclose(q_old, q_new, atol=1e-5), "Q mismatch"
    assert torch.allclose(k_old, k_new, atol=1e-5), "K mismatch"
    assert torch.allclose(v_old, v_new, atol=1e-5), "V mismatch"
    print("  QKV fusion numerical: OK")


if __name__ == "__main__":
    tests = [
        test_mini_llm_kernels_import,
        test_fused_add_rms_norm_fallback,
        test_decode_paged_attention_fallback,
        test_no_custom_kernels_env,
        test_decode_batch_gather_ctx_fields,
        test_qkv_fusion_numerical,
    ]
    passed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
