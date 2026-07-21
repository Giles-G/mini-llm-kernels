"""
decode_attention.py — Python 绑定与 fallback 实现

提供 decode_paged_attention() 函数：
- 有 CUDA kernel：调用 mini_llm_kernels._C.decode_paged_attention
- 无 CUDA（M1/CPU）：PyTorch fallback，语义等价于当前 gather + batched_matmul

接口：
    out = decode_paged_attention(q, k_cache, v_cache, block_table, context_lens)

    q:            [N, H_q, D]                      fp16/bf16
    k_cache:      [num_blocks, block_size, H_kv, D] fp16/bf16
    v_cache:      [num_blocks, block_size, H_kv, D] fp16/bf16
    block_table:  [N, max_blocks]                   int32 / int64
    context_lens: [N]                               int32 / int64

    out:          [N, H_q, D]
"""

from __future__ import annotations
import torch


def _decode_paged_attention_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """PyTorch fallback：gather + batched GQA attention（与原 nn_ops.py 等价）"""
    N, H_q, D = q.shape
    block_size = k_cache.size(1)
    H_kv = k_cache.size(2)
    max_ctx = int(context_lens.max().item())
    max_blocks = block_table.size(1)

    # 构造 gather 索引（与 KVCacheManager.gather_kv_decode_batch 等价）
    pos = torch.arange(max_ctx, device=q.device, dtype=torch.long)
    blk_idx = pos // block_size      # [max_ctx]
    blk_off = pos % block_size       # [max_ctx]

    # block_table: [N, max_blocks] → 按 pos 索引 → [N, max_ctx]
    bt = block_table[:, :max_blocks].long()  # [N, max_blocks]
    bt_expanded = bt[:, blk_idx]             # [N, max_ctx]

    # gather: k_cache[bt_expanded, blk_off, :, :] → [N, max_ctx, H_kv, D]
    k_padded = k_cache[bt_expanded, blk_off]  # [N, max_ctx, H_kv, D]
    v_padded = v_cache[bt_expanded, blk_off]  # [N, max_ctx, H_kv, D]

    # GQA batched attention（与 nn_ops.gathered_paged_kv_decode_attention 等价）
    group = H_q // H_kv
    q_grouped = q.view(N, H_kv, group, D)
    k_p = k_padded.permute(0, 2, 1, 3)   # [N, H_kv, max_ctx, D]
    v_p = v_padded.permute(0, 2, 1, 3)

    scale = D ** -0.5
    scores = torch.matmul(q_grouped, k_p.transpose(-1, -2)) * scale  # [N, H_kv, group, max_ctx]

    # mask
    pos_idx = torch.arange(max_ctx, device=q.device).view(1, 1, 1, max_ctx)
    valid = pos_idx < context_lens.view(N, 1, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(weights, v_p)   # [N, H_kv, group, D]
    return out.reshape(N, H_q, D)


# 运行时选择
try:
    from mini_llm_kernels._C import decode_paged_attention as _cuda_decode_paged_attn
    from mini_llm_kernels._C import decode_paged_attention_partitioned as _cuda_decode_partitioned
    _HAS_DECODE_ATTN = True
except ImportError:
    _HAS_DECODE_ATTN = False
    _cuda_decode_partitioned = None


def decode_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """Block-aware decode attention (paged KV, online softmax).

    Automatically uses KV-sequence-partitioned kernel when batch=1 for
    better SM utilization on GPUs with many SMs (e.g., RTX 3060).
    """
    if _HAS_DECODE_ATTN and q.is_cuda:
        ctx_int32 = context_lens.to(torch.int32)
        bt_int32  = block_table.to(torch.int32)
        try:
            # P4: For small batches, use KV-partitioned kernel to fill more SMs
            if q.size(0) == 1 and _cuda_decode_partitioned is not None:
                return _cuda_decode_partitioned(q, k_cache, v_cache, bt_int32, ctx_int32, 2)
            return _cuda_decode_paged_attn(q, k_cache, v_cache, bt_int32, ctx_int32)
        except Exception:
            pass  # fall through to PyTorch fallback
    return _decode_paged_attention_pytorch(q, k_cache, v_cache, block_table, context_lens)
