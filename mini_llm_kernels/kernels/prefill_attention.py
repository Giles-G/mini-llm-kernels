"""B1 paged prefill attention binding and PyTorch fallback."""

from __future__ import annotations

import torch


def _paged_prefill_attention_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    chunk_lens: torch.Tensor,
    history_lens: torch.Tensor,
) -> torch.Tensor:
    """Reference fallback; keeps the public CUDA binding contract."""
    n, _, h_q, d = q.shape
    block_size = k_cache.size(1)
    h_kv = k_cache.size(2)
    group = h_q // h_kv
    out = torch.zeros_like(q)
    for i in range(n):
        t = int(chunk_lens[i].item())
        history = int(history_lens[i].item())
        total = history + t
        if t == 0:
            continue
        positions = torch.arange(total, device=q.device)
        blocks = block_table[i, positions // block_size].long()
        offsets = positions % block_size
        k = k_cache[blocks, offsets].permute(1, 0, 2)
        v = v_cache[blocks, offsets].permute(1, 0, 2)
        q_i = q[i, :t].view(t, h_kv, group, d).permute(1, 0, 2, 3)
        scores = torch.einsum("htgd,hkd->htgk", q_i.float(), k.float()) / (d ** 0.5)
        key_pos = positions.view(1, 1, 1, total)
        query_pos = (history + torch.arange(t, device=q.device)).view(1, t, 1, 1)
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        result = torch.einsum("htgk,hkd->htgd", weights, v.float())
        out[i, :t] = result.permute(1, 0, 2, 3).to(q.dtype).reshape(t, h_q, d)
    return out


try:
    from mini_llm_kernels._C import paged_prefill_attention as _cuda_paged_prefill_attention
    _HAS_PREFILL_ATTN = True
except ImportError:
    _cuda_paged_prefill_attention = None
    _HAS_PREFILL_ATTN = False


def paged_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    chunk_lens: torch.Tensor,
    history_lens: torch.Tensor,
) -> torch.Tensor:
    if _HAS_PREFILL_ATTN and q.is_cuda:
        try:
            return _cuda_paged_prefill_attention(
                q, k_cache, v_cache, block_table.to(torch.int32),
                chunk_lens.to(torch.int32), history_lens.to(torch.int32),
            )
        except Exception:
            pass
    return _paged_prefill_attention_pytorch(
        q, k_cache, v_cache, block_table, chunk_lens, history_lens,
    )
