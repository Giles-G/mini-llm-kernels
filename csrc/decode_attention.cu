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

    // ── warp / lane / block 常量 ──
    int warp_id   = threadIdx.x / 32;
    int lane_id   = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    // ── 跨 warp 归约 shared memory（复用 smem 尾部） ──
    // 总 shared memory = 2 * block_size * D * sizeof(T) + num_warps * sizeof(float)
    float* warp_partial  = reinterpret_cast<float*>(V_tile + block_size * D);

    // ── 寄存器：online softmax 状态 ──
    // o_acc: 累积输出，分布在所有线程（每线程负责 [threadIdx.x, threadIdx.x + stride, ...]）
    // D 必须 <= MAX_BLOCK_DIM (128)，由 wrapper TORCH_CHECK 保证
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

        // 每线程分批加载（stride = blockDim.x），float4 向量化（128-bit）加载
        int vec_count = D / 8;  // 8 half = 1 float4
        for (int slot = 0; slot < blk_len; slot++) {
            const T* k_row = k_block_base + (long long)slot * H_kv * D;
            const T* v_row = v_block_base + (long long)slot * H_kv * D;

            float4*       k_tile_vec = reinterpret_cast<float4*>(K_tile + slot * D);
            float4*       v_tile_vec = reinterpret_cast<float4*>(V_tile + slot * D);
            const float4* k_row_vec  = reinterpret_cast<const float4*>(k_row);
            const float4* v_row_vec  = reinterpret_cast<const float4*>(v_row);

            for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
                k_tile_vec[i] = k_row_vec[i];
                v_tile_vec[i] = v_row_vec[i];
            }
        }
        __syncthreads();

        // ── 计算 scores = Q · K_tile^T / scale，online softmax ──
        // 每个 token 一次内积：
        //   1. 各线程计算 partial dot（stride = blockDim.x）
        //   2. warp 内部 reduce 得到 warp-level dot
        //   3. 跨 warp reduce（shared memory）得到完整 dot
        //   4. broadcast 到所有线程，保证 m_acc / l_acc 一致
        for (int slot = 0; slot < blk_len; slot++) {
            // Step 1 & 2: per-thread partial dot → warp reduce
            float dot = 0.0f;
            const T* k_row = K_tile + slot * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x)
                dot += elem_to_float(q_ptr[i]) * elem_to_float(k_row[i]);
            dot = warp_reduce_sum(dot);

            // Step 3: 跨 warp reduce — lane 0 写 shared memory
            if (lane_id == 0) warp_partial[warp_id] = dot;
            __syncthreads();

            // warp 0 的 lane 0..num_warps-1 各读一个 warp 的 partial sum 然后归约
            if (warp_id == 0) {
                dot = (lane_id < num_warps) ? warp_partial[lane_id] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_sync(0xffffffff, dot, offset);
                if (lane_id == 0) warp_partial[0] = dot;
            }
            __syncthreads();

            // Step 4: 所有线程从 shared memory 读取最终 dot
            dot = warp_partial[0];
            dot *= scale;

            // online softmax 更新
            float m_new = fmaxf(m_acc, dot);
            float exp_old = expf(m_acc - m_new);
            float exp_cur = expf(dot - m_new);

            // 更新 o_acc[i] = exp_old * o_acc[i] + exp_cur * V_tile[slot, i]
            const T* v_row = V_tile + slot * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x)
                o_acc[i] = exp_old * o_acc[i] + exp_cur * elem_to_float(v_row[i]);

            l_acc = exp_old * l_acc + exp_cur;
            m_acc = m_new;
            __syncthreads();  // 保证 o_acc 更新完成后再处理下一个 slot
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
    TORCH_CHECK(D <= MAX_BLOCK_DIM, "head_dim must be <= MAX_BLOCK_DIM=128 (o_acc stack size)");
    TORCH_CHECK(D % 8 == 0, "head_dim must be divisible by 8 (float4 alignment)");

    float scale = 1.0f / sqrtf((float)D);

    auto out = torch::empty({N, H_q, D}, q.options());

    // block dim: 向上对齐 32，不超过 D 也不超过 128
    int threads = ((std::min(D, MAX_BLOCK_DIM) + 31) / 32) * 32;
    dim3 grid(N, H_q);

    // shared memory: K_tile + V_tile + cross-warp reduction buffer
    int num_warps = threads / 32;
    size_t smem_bytes = 2 * block_size * D * sizeof(__half)
                      + num_warps * sizeof(float);

    // 检查 shared memory 是否超限，必要时申请扩展 shared memory
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (smem_bytes > prop.sharedMemPerBlock) {
        TORCH_CHECK(smem_bytes <= prop.sharedMemPerBlockOptin,
                    "shared memory demand (", smem_bytes,
                    " B) exceeds device max (", prop.sharedMemPerBlockOptin, " B)");
    }

    if (q.dtype() == torch::kFloat16) {
        if (smem_bytes > prop.sharedMemPerBlock)
            cudaFuncSetAttribute(decode_paged_attention_kernel<__half>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
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
        if (smem_bytes > prop.sharedMemPerBlock)
            cudaFuncSetAttribute(decode_paged_attention_kernel<__nv_bfloat16>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
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

// ─────────────────────────────────────────────────────────────────────────────
// KV-sequence-partitioned attention kernel（P4 优化）
//
// 每个 (batch, head) 启动 NUM_KV_PARTITIONS 个 block，每个 block 处理一段
// KV sequence。各 partition 独立计算 online softmax，写 partial (o, m, l)
// 到全局内存，再由 merge kernel 合并。batch=1 时 grid 从 (1, 14) 变为
// (1, 14, 2) = 28 blocks，SM 利用率从 50% 提升到 100%。
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void decode_paged_attention_partial_kernel(
    const T*     __restrict__ Q,
    const T*     __restrict__ K_cache,
    const T*     __restrict__ V_cache,
    const int*   __restrict__ block_table,
    const int*   __restrict__ context_lens,
    float*       __restrict__ o_partial,  // [num_partitions, N, H_q, D]
    float*       __restrict__ m_partial,  // [num_partitions, N, H_q]
    float*       __restrict__ l_partial,  // [num_partitions, N, H_q]
    int N, int H_q, int H_kv, int D,
    int block_size, int max_blocks,
    float scale, int num_partitions)
{
    int batch_idx    = blockIdx.x;
    int q_head_idx   = blockIdx.y;
    int partition_id = blockIdx.z;
    int gqa_ratio    = H_q / H_kv;
    int kv_head_idx  = q_head_idx / gqa_ratio;

    int ctx_len = context_lens[batch_idx];
    if (ctx_len == 0) {
        int global_head_idx = partition_id * N * H_q + batch_idx * H_q + q_head_idx;
        m_partial[global_head_idx] = -FLT_MAX;
        l_partial[global_head_idx] = 0.0f;
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            o_partial[global_head_idx * D + i] = 0.0f;
        return;
    }

    int num_kv_blocks = (ctx_len + block_size - 1) / block_size;
    int kv_per_part   = (num_kv_blocks + num_partitions - 1) / num_partitions;
    int kv_block_start = partition_id * kv_per_part;
    int kv_block_end   = min(kv_block_start + kv_per_part, num_kv_blocks);

    extern __shared__ char smem[];
    T* K_tile = reinterpret_cast<T*>(smem);
    T* V_tile = K_tile + block_size * D;
    float* warp_partial = reinterpret_cast<float*>(V_tile + block_size * D);

    int warp_id   = threadIdx.x / 32;
    int lane_id   = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    float o_acc[MAX_BLOCK_DIM] = {};
    float m_acc = -FLT_MAX;
    float l_acc = 0.0f;

    const T* q_ptr = Q + batch_idx * H_q * D + q_head_idx * D;

    for (int blk = kv_block_start; blk < kv_block_end; blk++) {
        int block_id = block_table[batch_idx * max_blocks + blk];
        int blk_start = blk * block_size;
        int blk_len   = min(block_size, ctx_len - blk_start);

        const T* k_block_base = K_cache + ((long long)block_id * block_size * H_kv + kv_head_idx) * D;
        const T* v_block_base = V_cache + ((long long)block_id * block_size * H_kv + kv_head_idx) * D;

        int vec_count = D / 8;
        for (int slot = 0; slot < blk_len; slot++) {
            const T* k_row = k_block_base + (long long)slot * H_kv * D;
            const T* v_row = v_block_base + (long long)slot * H_kv * D;
            float4*       k_tile_vec = reinterpret_cast<float4*>(K_tile + slot * D);
            float4*       v_tile_vec = reinterpret_cast<float4*>(V_tile + slot * D);
            const float4* k_row_vec  = reinterpret_cast<const float4*>(k_row);
            const float4* v_row_vec  = reinterpret_cast<const float4*>(v_row);
            for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
                k_tile_vec[i] = k_row_vec[i];
                v_tile_vec[i] = v_row_vec[i];
            }
        }
        __syncthreads();

        for (int slot = 0; slot < blk_len; slot++) {
            float dot = 0.0f;
            const T* k_row = K_tile + slot * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x)
                dot += elem_to_float(q_ptr[i]) * elem_to_float(k_row[i]);
            dot = warp_reduce_sum(dot);

            if (lane_id == 0) warp_partial[warp_id] = dot;
            __syncthreads();
            if (warp_id == 0) {
                dot = (lane_id < num_warps) ? warp_partial[lane_id] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_sync(0xffffffff, dot, offset);
                if (lane_id == 0) warp_partial[0] = dot;
            }
            __syncthreads();
            dot = warp_partial[0] * scale;

            float m_new = fmaxf(m_acc, dot);
            float exp_old = expf(m_acc - m_new);
            float exp_cur = expf(dot - m_new);

            const T* v_row = V_tile + slot * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x)
                o_acc[i] = exp_old * o_acc[i] + exp_cur * elem_to_float(v_row[i]);

            l_acc = exp_old * l_acc + exp_cur;
            m_acc = m_new;
            __syncthreads();
        }
        __syncthreads();
    }

    // Write partial results
    int global_head_idx = partition_id * N * H_q + batch_idx * H_q + q_head_idx;
    m_partial[global_head_idx] = m_acc;
    l_partial[global_head_idx] = l_acc;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        o_partial[global_head_idx * D + i] = o_acc[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax merge kernel: 合并各 partition 的 partial (o, m, l)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void decode_attention_merge_kernel(
    const float* __restrict__ o_partial,   // [num_part, N, H_q, D]
    const float* __restrict__ m_partial,   // [num_part, N, H_q]
    const float* __restrict__ l_partial,   // [num_part, N, H_q]
    float*       __restrict__ out,          // [N, H_q, D]
    int N, int H_q, int D, int num_partitions)
{
    int batch_idx  = blockIdx.x;
    int q_head_idx = blockIdx.y;

    // Find global max across partitions
    float m_global = -FLT_MAX;
    for (int p = 0; p < num_partitions; p++) {
        int idx = p * N * H_q + batch_idx * H_q + q_head_idx;
        m_global = fmaxf(m_global, m_partial[idx]);
    }

    // Compute global L = sum(exp(m_p - m_global) * l_p)
    float l_global = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        int idx = p * N * H_q + batch_idx * H_q + q_head_idx;
        l_global += expf(m_partial[idx] - m_global) * l_partial[idx];
    }

    // Weighted sum of partial outputs
    float* out_base = out + batch_idx * H_q * D + q_head_idx * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float sum = 0.0f;
        for (int p = 0; p < num_partitions; p++) {
            int idx = p * N * H_q + batch_idx * H_q + q_head_idx;
            sum += expf(m_partial[idx] - m_global) * o_partial[idx * D + i];
        }
        out_base[i] = (l_global > 0.0f) ? (sum / l_global) : 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KV-partitioned attention wrapper
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor decode_paged_attention_partitioned(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor context_lens,
    int num_partitions = 2)
{
    TORCH_CHECK(q.is_cuda(),           "q must be CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda(),     "k_cache must be CUDA tensor");
    TORCH_CHECK(v_cache.is_cuda(),     "v_cache must be CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA tensor");
    TORCH_CHECK(context_lens.is_cuda(),"context_lens must be CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "only fp16 / bf16 supported");
    TORCH_CHECK(k_cache.dtype() == q.dtype() && v_cache.dtype() == q.dtype(),
                "K/V cache dtype must match Q dtype");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(context_lens.dtype() == torch::kInt32, "context_lens must be int32");
    TORCH_CHECK(q.dim() == 3, "q must be [N, H_q, D]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [num_blocks, block_size, H_kv, D]");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be [num_blocks, block_size, H_kv, D]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [N, max_blocks]");
    TORCH_CHECK(context_lens.dim() == 1, "context_lens must be [N]");
    TORCH_CHECK(num_partitions > 0, "num_partitions must be > 0");

    int N          = q.size(0);
    int H_q        = q.size(1);
    int D          = q.size(2);
    int block_size = k_cache.size(1);
    int H_kv       = k_cache.size(2);
    int max_blocks = block_table.size(1);

    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv (GQA)");
    TORCH_CHECK(D <= MAX_BLOCK_DIM, "head_dim must be <= MAX_BLOCK_DIM=128 (o_acc stack size)");
    TORCH_CHECK(D % 8 == 0, "head_dim must be divisible by 8 (float4 alignment)");

    float scale = 1.0f / sqrtf((float)D);

    int threads = ((std::min(D, MAX_BLOCK_DIM) + 31) / 32) * 32;
    int num_warps = threads / 32;
    size_t smem_bytes = 2 * block_size * D * (q.dtype() == torch::kFloat16 ? 2 : 2)
                      + num_warps * sizeof(float);

    // Shared memory check
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    TORCH_CHECK(smem_bytes <= prop.sharedMemPerBlockOptin,
                "shared memory demand exceeds device maximum");

    // Temporary buffers (float32 for numerical stability in merge)
    auto o_partial = torch::empty({num_partitions, N, H_q, D},
                                  torch::dtype(torch::kFloat32).device(q.device()));
    auto m_partial = torch::empty({num_partitions, N, H_q},
                                  torch::dtype(torch::kFloat32).device(q.device()));
    auto l_partial = torch::empty({num_partitions, N, H_q},
                                  torch::dtype(torch::kFloat32).device(q.device()));
    // Output in fp32, cast to input dtype at the end
    auto out_fp32 = torch::empty({N, H_q, D},
                                 torch::dtype(torch::kFloat32).device(q.device()));

    dim3 partial_grid(N, H_q, num_partitions);

    // Launch partition kernel
    if (smem_bytes > prop.sharedMemPerBlock) {
        if (q.dtype() == torch::kFloat16) {
            cudaFuncSetAttribute(decode_paged_attention_partial_kernel<__half>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            decode_paged_attention_partial_kernel<__half><<<partial_grid, threads, smem_bytes>>>(
                reinterpret_cast<const __half*>(q.data_ptr()),
                reinterpret_cast<const __half*>(k_cache.data_ptr()),
                reinterpret_cast<const __half*>(v_cache.data_ptr()),
                block_table.data_ptr<int>(), context_lens.data_ptr<int>(),
                o_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                N, H_q, H_kv, D, block_size, max_blocks, scale, num_partitions);
        } else {
            cudaFuncSetAttribute(decode_paged_attention_partial_kernel<__nv_bfloat16>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            decode_paged_attention_partial_kernel<__nv_bfloat16><<<partial_grid, threads, smem_bytes>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
                block_table.data_ptr<int>(), context_lens.data_ptr<int>(),
                o_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                N, H_q, H_kv, D, block_size, max_blocks, scale, num_partitions);
        }
    } else {
        if (q.dtype() == torch::kFloat16) {
            decode_paged_attention_partial_kernel<__half><<<partial_grid, threads, smem_bytes>>>(
                reinterpret_cast<const __half*>(q.data_ptr()),
                reinterpret_cast<const __half*>(k_cache.data_ptr()),
                reinterpret_cast<const __half*>(v_cache.data_ptr()),
                block_table.data_ptr<int>(), context_lens.data_ptr<int>(),
                o_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                N, H_q, H_kv, D, block_size, max_blocks, scale, num_partitions);
        } else {
            decode_paged_attention_partial_kernel<__nv_bfloat16><<<partial_grid, threads, smem_bytes>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
                block_table.data_ptr<int>(), context_lens.data_ptr<int>(),
                o_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                N, H_q, H_kv, D, block_size, max_blocks, scale, num_partitions);
        }
    }

    // Launch merge kernel
    dim3 merge_grid(N, H_q);
    int merge_threads = ((std::min(D, MAX_BLOCK_DIM) + 31) / 32) * 32;
    decode_attention_merge_kernel<<<merge_grid, merge_threads>>>(
        o_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(), out_fp32.data_ptr<float>(),
        N, H_q, D, num_partitions);

    // Cast fp32 output back to input dtype
    return out_fp32.to(q.dtype());
}
