# M6 Kernel Tuning Notes

Stage 8 kernel 调优方向与 Nsight Compute 分析指南。

## 快速 Profiling 命令

```bash
# 整体 timeline（系统级）
nsys profile -w true -t cuda,nvtx,osrt \
    python scripts/bench_stage8.py --model Qwen/Qwen2.5-0.5B-Instruct --greedy

# Kernel 级指标（M4 decode_attention_kernel）
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum \
    --target-processes all \
    python -c "import mini_llm_kernels; ..."
```

---

## M4 `decode_attention_kernel` 调优方向

### 1. block_size 选择

| block_size | 优势 | 适用场景 |
|---|---|---|
| 16 | 小 shared mem，高 occupancy | 短 context（≤512）|
| 32 | 均衡 | 通用 |
| 64 | 减少 block 迭代次数，cache 友好 | 长 context（≥1024）|
| 128 | 最少迭代，但 shared mem 压力大 | head_dim=128, A100 |

**当前默认**：`BLOCK_SIZE=32`（`kv_cache.py`），建议在 A100 长 context 下试验 64。

### 2. 向量化访存（float4 128-bit load）

当前 K/V 按标量加载，改为 `float4` 可将 global memory load 指令数减少 4×：

```cuda
// 当前（标量）
float val = k_cache[offset];

// 优化（向量化，需 head_dim % 4 == 0）
float4 vec = reinterpret_cast<float4*>(k_cache)[offset / 4];
```

适用条件：`head_dim` 为 64 或 128（Qwen2.5 均满足）。

### 3. Shared Memory Bank Conflict 消除

K_tile / V_tile 的 layout 为 `[BLOCK_SIZE, head_dim]`，当 `head_dim=64` 时：

```
每行 64 × 2(fp16) = 128 bytes = 4 个 bank × 32 bytes → 无 conflict
每行 128 × 2(fp16) = 256 bytes → 同样无 conflict
```

若出现 conflict（ncu 报 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld > 0`），
在声明时加 +1 列 padding：

```cuda
__shared__ __half K_tile[BLOCK_SIZE][HEAD_DIM + 1];
```

### 4. Warp 占用率（Occupancy）

目标：`sm__warps_active ≥ 50%`。

调整 `blockDim.x`（当前 = `min(head_dim, 128)`）：
- `head_dim=64`：blockDim.x=64（1 warp/block），考虑改为 128（2 warp/block）
- `head_dim=128`：blockDim.x=128（4 warp/block），通常已足够

Shared memory per block 上限（A100）：`164 KB`，计算公式：
```
shm = (BLOCK_SIZE * head_dim * 2) * 2  (K_tile + V_tile, fp16)
    = 32 * 128 * 2 * 2 = 16 KB  ← 远低于上限，可加大 blockDim
```

---

## M3 `fused_add_rms_norm_kernel` 调优方向

### 1. blockDim.x 选择

当前：`blockDim.x = min(hidden_size, 512)`。

hidden_size 常见值：
- Qwen2.5-0.5B：896 → blockDim.x=512（2 warp reduce passes）
- Qwen2.5-1.5B：1536 → blockDim.x=512
- Qwen2.5-7B：3584 → blockDim.x=512（7 elements/thread）

建议：hidden_size ≤ 1024 时用 256，> 1024 时用 512。

### 2. FP32 Accumulation

当前已在 warp reduce 中用 `float` 累加，确认 `__half2float` 正确调用即可。

---

## M2 QKV GEMM 调优方向

纯 Python 层，调优空间有限，但可考虑：

- **cuBLAS batch GEMM**：batch_size ≥ 32 时，用 `torch.baddbmm` 替代 `F.linear`
- **Triton / cutlass**：超大 hidden_size（≥4096）时手写 Triton kernel 优于 cuBLAS

---

## 预期性能（A100, Qwen2.5-0.5B, greedy）

| 场景 | Stage 7 基线 | M2 | M2+M3 | M2+M3+M4 | M6 调优后 |
|---|---|---|---|---|---|
| batch=1 | ~43 tok/s | ~55 | ~75 | ~150 | ~180 |
| batch=8 | ~304 tok/s | ~330 | ~400 | ~650 | ~750 |
| batch=16 | ~200 tok/s | ~220 | ~270 | ~500 | ~600 |

> 以上数据基于 CUDA GPU 估算。MPS/CPU 设备全程走 fallback，性能与 Stage 7 持平。
