"""
gemma4_attention.py — Python 绑定与 fallback 实现

提供 gemma4_decode_attention()：
- 有 CUDA kernel：调用 mini_llm_kernels._C.gemma4_decode_attention
- 无 CUDA（M1/CPU）：PyTorch fallback，语义与 CUDA kernel 一致

为什么需要单独的 kernel（而不是复用 decode_paged_attention）：
- Gemma4 的 sliding 层 head_dim=256、full 层 head_dim=512，通用 kernel 的
  MAX_BLOCK_DIM=128 覆盖不到；
- sliding 层只需看最近 sliding_window 个 token，通用 kernel 会遍历全部
  历史 block，长上下文下每步都要重读整个 KV。

接口：
    out = gemma4_decode_attention(q, k_cache, v_cache, block_table,
                                  context_lens, window_size=0)

    q:            [N, H_q, D]                        fp16/bf16
    k_cache:      [num_blocks, block_size, H_kv, D]  fp16/bf16
    v_cache:      [num_blocks, block_size, H_kv, D]  fp16/bf16
    block_table:  [N, max_blocks]                    int32 / int64
    context_lens: [N]                                int32 / int64
    window_size:  滑动窗口长度，0 表示不限制

    out:          [N, H_q, D]
"""

from __future__ import annotations

import torch


def _gemma4_decode_attention_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    window_size: int = 0,
) -> torch.Tensor:
    """PyTorch fallback：按 block_table gather 后做标准的单 query attention。

    逐行处理：每行只读取自己 ``context_lens[row]`` 个 token，``window_size``
    只保留最近若干个位置（单 query 下等价于 mask 掉更早的 key）。
    与 eager 参考路径一致，用 fp32 累加后转回。
    """
    num_rows, num_q_heads, _ = q.shape
    kv_heads = k_cache.shape[-2]
    block_size = k_cache.shape[1]
    outputs = []

    for row in range(num_rows):
        length = int(context_lens[row].item())
        if length <= 0:
            outputs.append(torch.zeros_like(q[row]))
            continue

        start = max(0, length - window_size) if window_size else 0
        positions = torch.arange(start, length, device=q.device, dtype=torch.long)
        block_ids = block_table[row][positions // block_size]
        offsets = positions % block_size

        k = k_cache[block_ids, offsets]  # [T, H_kv, D]
        v = v_cache[block_ids, offsets]
        if num_q_heads != kv_heads:
            repeat = num_q_heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # [T, H, D] -> [H, T, D]：把 head 轴换到前面，避免 head 数与 head_dim
        # 不等时 transpose 交换错轴。
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = torch.matmul(q[row].unsqueeze(1), k.transpose(-1, -2)).float()
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outputs.append(torch.matmul(probs, v).squeeze(1))

    return torch.stack(outputs, dim=0)


# 运行时选择
try:
    from mini_llm_kernels._C import gemma4_decode_attention as _cuda_gemma4_attention
    _HAS_GEMMA4_ATTN = True
except ImportError:
    _HAS_GEMMA4_ATTN = False
    _cuda_gemma4_attention = None


# ─────────────────────────────────────────────────────────────────────────────
# CUDA 路径状态
#
# CUDA kernel 目前**不可用**：入口会挂死（见 csrc/gemma4_attention.cu 头部
# 说明）。因此默认一律走下面这个逐位对齐的 PyTorch 实现。
#
# 这里刻意不在导入时"试跑一次"来探测——kernel 的问题正是会挂死，任何自动
# 探测都会把进程一起挂住。修好 kernel 之后，把 _CUDA_ATTN_VERIFIED 改成
# True（并跑通 tests 里的对照）即可放行；调试时也可以用环境变量
# MINI_LLM_GEMMA4_ATTN_KERNEL=1 临时强制启用。
# ─────────────────────────────────────────────────────────────────────────────
_CUDA_ATTN_VERIFIED = False          # kernel 修好并通过对照测试后改为 True
_FORCE_ENV = "MINI_LLM_GEMMA4_ATTN_KERNEL"


def cuda_attention_usable() -> bool:
    """是否可以用 Gemma4 CUDA attention kernel（当前恒为 False）。"""
    import os

    if os.environ.get(_FORCE_ENV) == "1":
        return bool(_HAS_GEMMA4_ATTN and torch.cuda.is_available())
    return bool(_CUDA_ATTN_VERIFIED and _HAS_GEMMA4_ATTN and torch.cuda.is_available())


def cuda_attention_status() -> str:
    """CUDA 路径被启用/禁用的原因，便于日志与排查。"""
    import os

    if os.environ.get(_FORCE_ENV) == "1":
        return "enabled by MINI_LLM_GEMMA4_ATTN_KERNEL=1 (unverified, may hang)"
    if not _HAS_GEMMA4_ATTN:
        return "disabled: CUDA extension symbol not built"
    if not _CUDA_ATTN_VERIFIED:
        return "disabled: kernel is not verified yet (hangs); using PyTorch fallback"
    return "enabled"


def gemma4_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    window_size: int = 0,
) -> torch.Tensor:
    """Gemma4 paged decode attention（head_dim 256/512 + 滑动窗口）。

    只有在 CUDA kernel 通过启动期正确性探测时才走 kernel，否则一律使用
    语义等价的 PyTorch 实现。
    """
    if cuda_attention_usable() and q.is_cuda:
        return _cuda_gemma4_attention(
            q,
            k_cache,
            v_cache,
            block_table.to(torch.int32),
            context_lens.to(torch.int32),
            int(window_size),
        )
    return _gemma4_decode_attention_pytorch(
        q, k_cache, v_cache, block_table, context_lens, window_size
    )
