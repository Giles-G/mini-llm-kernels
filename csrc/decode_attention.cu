/**
 * decode_attention.cu
 *
 * Block-aware Decode Attention Kernel（Stage 8 M4 优化）
 *
 * 替换主仓库当前的两步实现：
 *   1. advanced indexing gather（物化 KV 到连续内存）
 *   2. gathered_paged_kv_decode_attention（标准 batched matmul）
 *
 * 本 kernel 直接按 block_table 跳着读 KV，配合 online softmax，
 * 一次 kernel 完成 q_len=1 的 Flash Attention：
 *   - 不物化 gather tensor，消除额外 HBM 拷贝
 *   - online softmax：m/l/o 存寄存器，无需写回完整 score 矩阵
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 线程组织：
 *   grid:  [batch_size, num_q_heads]
 *          每个 (batch_idx, q_head_idx) 对应一个 block
 *   block: [BLOCK_DIM]，BLOCK_DIM = min(head_dim, 128)，对齐到 32
 *          多个线程协作处理 head_dim 方向
 *
 * shared memory 布局（per block）：
 *   K_tile: [block_size, head_dim]  fp16/bf16
 *   V_tile: [block_size, head_dim]  fp16/bf16
 *
 * 寄存器：
 *   m, l, o[head_dim]  → online softmax 状态
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 算法（简化 Flash Attention，q_len=1）：
 *   m = -inf, l = 0, o = 0
 *   for each kv_block in range(num_kv_blocks):
 *       block_id = block_table[batch, kv_block]
 *       load K_tile, V_tile from K_cache[block_id], V_cache[block_id]
 *       scores[i] = Q · K_tile[i] / sqrt(D)  for i in [0, block_size)
 *       mask scores beyond context_len to -inf
 *       m_new = max(m, max(scores))
 *       l = exp(m - m_new) * l + sum(exp(scores - m_new))
 *       o = exp(m - m_new) * o + exp(scores - m_new)_i · V_tile[i]
 *       m = m_new
 *   output = o / l
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 输入张量布局（与主仓库 KVCacheManager 一致）：
 *   Q:           [N, H_q, D]
 *   K_cache:     [num_kv_blocks, block_size, H_kv, D]
 *   V_cache:     [num_kv_blocks, block_size, H_kv, D]
 *   block_table: [N, max_blocks]   int32
 *   context_lens:[N]               int32
 *
 * 输出：
 *   out:         [N, H_q, D]
 *
 * GQA 支持：
 *   H_kv < H_q，kv_head_idx = q_head_idx / (H_q / H_kv)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <torch/extension.h>

// ─────────────────────────────────────────────────────────────────────────────
// 编译时常量
// ─────────────────────────────────────────────────────────────────────────────
#define MAX_BLOCK_DIM 128   // <= head_dim，对齐到 32

// ─────────────────────────────────────────────────────────────────────────────
// 辅助：类型转换
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__device__ __forceinline__ float elem_to_float(T v);
template<> __device__ __forceinline__ float elem_to_float(__half v)         { return __half2float(v); }
template<> __device__ __forceinline__ float elem_to_float(__nv_bfloat16 v)  { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T float_to_elem(float v);
template<> __device__ __forceinline__ __half        float_to_elem<__half>(float v)        { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 float_to_elem<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ─────────────────────────────────────────────────────────────────────────────
// warp reduce max / sum
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// 主 kernel
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void decode_paged_attention_kernel(
    const T*     __restrict__ Q,           // [N, H_q, D]
    const T*     __restrict__ K_cache,     // [num_kv_blocks, block_size, H_kv, D]
    const T*     __restrict__ V_cache,     // [num_kv_blocks, block_size, H_kv, D]
    const int*   __restrict__ block_table, // [N, max_blocks]
    const int*   __restrict__ context_lens,// [N]
    T*           __restrict__ out,         // [N, H_q, D]
    int N, int H_q, int H_kv, int D,
    int block_size, int max_blocks,
    float scale)
{
    // 每个 CUDA block 处理 (batch_idx, q_head_idx) 对
    int batch_idx  = blockIdx.x;
    int q_head_idx = blockIdx.y;
    // GQA：多个 Q head 共享一个 KV head
    int gqa_ratio  = H_q / H_kv;
    int kv_head_idx = q_head_idx / gqa_ratio;

    int ctx_len = context_lens[batch_idx];
    if (ctx_len == 0) {
        // 空上下文：输出零
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            out[batch_idx * H_q * D + q_head_idx * D + i] = float_to_elem<T>(0.0f);
        return;
    }

    int num_kv_blocks = (ctx_len + block_size - 1) / block_size;

    // ── shared memory: K_tile + V_tile ──
    // 大小 = 2 * block_size * D * sizeof(T)，由 kernel launch 指定
    extern __shared__ char smem[];
    T* K_tile = reinterpret_cast<T*>(smem);
    T* V_tile = K_tile + block_size * D;

    // ── 寄存器：online softmax 状态 ──
    // o_acc: 累积输出，分布在所有线程（每线程负责 [threadIdx.x, threadIdx.x + stride, ...]）
    // 用 fixed-size 数组，D <= MAX_BLOCK_DIM * 4（典型 D=64/128）
    float o_acc[MAX_BLOCK_DIM] = {};  // 初始化为 0
    float m_acc = -FLT_MAX;           // 当前最大值（用于 online softmax 数值稳定）
    float l_acc = 0.0f;               // 归一化因子

    // Q 行指针：[batch_idx, q_head_idx, :]
    const T* q_ptr = Q + batch_idx * H_q * D + q_head_idx * D;

    // ── 遍历每个 KV block ──
    for (int blk = 0; blk < num_kv_blocks; blk++) {
        int block_id = block_table[batch_idx * max_blocks + blk];

        // 本 block 的有效 token 数（最后一个 block 可能不满）
        int blk_start = blk * block_size;
        int blk_len   = min(block_size, ctx_len - blk_start);

        // ── 协作加载 K_tile, V_tile 到 shared memory ──
        // K_cache 布局: [num_kv_blocks, block_size, H_kv, D]
        // K_cache[block_id, slot, kv_head_idx, :] 的起始偏移：
        const T* k_block_base = K_cache + ((long long)block_id * block_size * H_kv + kv_head_idx) * D;
        const T* v_block_base = V_cache + ((long long)block_id * block_size * H_kv + kv_head_idx) * D;
        // k_block_base[slot * H_kv * D ... slot * H_kv * D + D)

        // 每线程分批加载（stride = blockDim.x）
        for (int slot = 0; slot < blk_len; slot++) {
            const T* k_row = k_block_base + (long long)slot * H_kv * D;
            const T* v_row = v_block_base + (long long)slot * H_kv * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                K_tile[slot * D + i] = k_row[i];
                V_tile[slot * D + i] = v_row[i];
            }
        }
        __syncthreads();

        // ── 计算 scores = Q · K_tile^T / scale，online softmax ──
        // 每个 token 一次内积（每线程贡献部分，warp reduce 汇总）
        for (int slot = 0; slot < blk_len; slot++) {
            // 计算点积
            float dot = 0.0f;
            const T* k_row = K_tile + slot * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x)
                dot += elem_to_float(q_ptr[i]) * elem_to_float(k_row[i]);
            dot = warp_reduce_sum(dot);
            dot *= scale;

            // 只有 lane 0 有完整的 dot（其余 lane 结果无效）
            // 用 __shfl_sync 广播到所有 lane
            dot = __shfl_sync(0xffffffff, dot, 0);

            // online softmax 更新
            float m_new = fmaxf(m_acc, dot);
            float exp_old = expf(m_acc - m_new);
            float exp_cur = expf(dot - m_new);

            // 更新 o_acc[i] = exp_old * o_acc[i] + exp_cur * V_tile[slot, i]
            const T* v_row = V_tile + slot * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                int li = i / blockDim.x;  // 本线程负责的 "逻辑索引"
                // 由于每线程处理多个 i，用线性下标
                o_acc[i] = exp_old * o_acc[i] + exp_cur * elem_to_float(v_row[i]);
            }

            l_acc = exp_old * l_acc + exp_cur;
            m_acc = m_new;
        }
        __syncthreads();
    }

    // ── 写输出：out[batch_idx, q_head_idx, :] = o_acc / l_acc ──
    T* out_ptr = out + batch_idx * H_q * D + q_head_idx * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = (l_acc > 0.0f) ? (o_acc[i] / l_acc) : 0.0f;
        out_ptr[i] = float_to_elem<T>(val);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python 绑定
// ─────────────────────────────────────────────────────────────────────────────

/**
 * decode_paged_attention
 *
 * Args:
 *   q:           [N, H_q, D]     fp16/bf16
 *   k_cache:     [num_blocks, block_size, H_kv, D]  fp16/bf16
 *   v_cache:     [num_blocks, block_size, H_kv, D]  fp16/bf16
 *   block_table: [N, max_blocks]  int32
 *   context_lens:[N]              int32
 *
 * Returns:
 *   out: [N, H_q, D]  fp16/bf16
 */
torch::Tensor decode_paged_attention(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor context_lens)
{
    TORCH_CHECK(q.is_cuda(),           "q must be CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda(),     "k_cache must be CUDA tensor");
    TORCH_CHECK(v_cache.is_cuda(),     "v_cache must be CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA tensor");
    TORCH_CHECK(context_lens.is_cuda(),"context_lens must be CUDA tensor");

    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "only fp16 / bf16 supported");
    TORCH_CHECK(q.dim() == 3, "q must be [N, H_q, D]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [num_blocks, block_size, H_kv, D]");

    int N          = q.size(0);
    int H_q        = q.size(1);
    int D          = q.size(2);
    int block_size = k_cache.size(1);
    int H_kv       = k_cache.size(2);
    int max_blocks = block_table.size(1);

    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv (GQA)");
    TORCH_CHECK(D <= MAX_BLOCK_DIM * 4, "head_dim too large (max 512)");

    float scale = 1.0f / sqrtf((float)D);

    auto out = torch::empty({N, H_q, D}, q.options());

    // block dim: 向上对齐 32，不超过 D 也不超过 128
    int threads = ((std::min(D, MAX_BLOCK_DIM) + 31) / 32) * 32;
    dim3 grid(N, H_q);

    // shared memory: 2 * block_size * D * sizeof(T)
    size_t smem_bytes = 2 * block_size * D * (q.dtype() == torch::kFloat16 ? 2 : 2);

    if (q.dtype() == torch::kFloat16) {
        decode_paged_attention_kernel<__half><<<grid, threads, smem_bytes>>>(
            reinterpret_cast<const __half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(k_cache.data_ptr()),
            reinterpret_cast<const __half*>(v_cache.data_ptr()),
            block_table.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr()),
            N, H_q, H_kv, D, block_size, max_blocks, scale
        );
    } else {
        decode_paged_attention_kernel<__nv_bfloat16><<<grid, threads, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            block_table.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            N, H_q, H_kv, D, block_size, max_blocks, scale
        );
    }

    return out;
}
