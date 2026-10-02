/**
 * gemma4_attention.cu
 *
 * Paged decode attention for Gemma4's heterogeneous head dimensions.
 *
 * Why a dedicated kernel instead of ``decode_paged_attention``:
 *   - Gemma4 uses head_dim=256 on sliding layers and head_dim=512 on full
 *     layers. The generic kernel caps head_dim at MAX_BLOCK_DIM=128 and
 *     keeps ``o_acc[MAX_BLOCK_DIM]`` in registers per thread.
 *   - Sliding layers only attend to the last ``sliding_window`` tokens, so
 *     the kernel must skip older KV instead of walking every block. Without
 *     that a 128K context would re-read the whole history every step.
 *
 * ⚠ STATUS: NOT USABLE YET — the Python wrapper disables this kernel.
 *   Two designs were tried and both fail at runtime: a per-tile loop version
 *   deadlocked, and the current split-K version also hangs (reproduced in a
 *   standalone .cu with the same nvcc flags: execution stops at the first
 *   warp shuffle that follows a divergently-defined ``p``). Root cause is not
 *   yet identified — a minimal repro of that exact shuffle pattern passes, so
 *   the trigger is elsewhere in this file.
 *
 *   ``mini_llm_kernels.cuda_attention_usable()`` therefore probes this kernel
 *   once at import and falls back to the bit-exact PyTorch implementation
 *   unless it returns the right answer. Do not route production traffic here
 *   until that probe passes. Set MINI_LLM_GEMMA4_ATTN_KERNEL=1 to force it on
 *   while debugging.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design: split-K over the context, two kernels.
 *
 *   partial kernel  grid = [N, H_q, num_chunks], BLOCK_DIM threads
 *     Each block owns one KV chunk of KV_CHUNK tokens and one warp per token,
 *     so the q·k reduction stays inside a warp (shuffle) and never needs a
 *     cross-warp scalar reduction. The block produces its own softmax
 *     statistics (m, l) plus the unnormalised weighted sum o, written to
 *     global scratch.
 *
 *   merge kernel    grid = [N, H_q], BLOCK_DIM threads
 *     Combines the per-chunk results with the standard online-softmax merge:
 *       m = max_c m_c
 *       l = Σ_c l_c · exp(m_c − m)
 *       o = Σ_c o_c · exp(m_c − m)
 *       out = o / l
 *
 *   Splitting the context this way also spreads long contexts over many more
 *   blocks than a single-block-per-head scan would.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Tensor layouts (matching Gemma4PagedKVCacheManager):
 *   Q:            [N, H_q, D]
 *   K_cache:      [num_blocks, block_size, H_kv, D]
 *   V_cache:      [num_blocks, block_size, H_kv, D]
 *   block_table:  [N, max_blocks]  int32
 *   context_lens: [N]              int32
 *   out:          [N, H_q, D]
 *
 * GQA: kv_head = q_head / (H_q / H_kv).
 * Sliding window: window_size > 0 keeps only the most recent ``window_size``
 * positions, which is equivalent to masking older keys for a single query.
 *
 * Output dtype is fp32 for the partial buffers (accuracy of the merge) and
 * the input dtype for the final result.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
// 编译时常量
// ─────────────────────────────────────────────────────────────────────────────
#define G4_MAX_BLOCK_DIM 256   // threads per block
#define G4_KV_TILE       128   // KV positions staged per online-softmax step

// ─────────────────────────────────────────────────────────────────────────────
// 辅助：类型转换
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__device__ __forceinline__ float g4_to_float(T v);
template<> __device__ __forceinline__ float g4_to_float(__half v)         { return __half2float(v); }
template<> __device__ __forceinline__ float g4_to_float(__nv_bfloat16 v)  { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T g4_from_float(float v);
template<> __device__ __forceinline__ __half        g4_from_float<__half>(float v)        { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 g4_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ─────────────────────────────────────────────────────────────────────────────
// warp 内归约
//
// 全掩码 shuffle，因此调用点必须在统一控制流上：如果部分 lane 已经跳过该
// 调用（例如放在 `if (tid < tile_count)` 里），__shfl_xor_sync 的行为未定义，
// 实测会直接挂死。
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float g4_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ float g4_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// 主 kernel
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Split-K 两段式实现
//
// 第一段（partial）：grid = [N, H_q, num_kv_blocks]，每个 block 只处理一段
//   KV（KV_CHUNK 个 token），一个 warp 负责一个 token。段内没有跨迭代的
//   barrier，也没有共享槽位复用，因此不存在前几版那种归约互相覆盖 /
//   条件屏障导致的死锁。每段输出自己的 softmax 统计量 (m, l) 和归一化前的
//   部分输出 o，写回 global memory。
//
// 第二段（merge）：grid = [N, H_q]，把各段按标准 online-softmax 合并：
//   m = max_b m_b ; l = Σ_b l_b * exp(m_b - m) ; o = Σ_b o_b * exp(m_b - m)
//   out = o / l
//
// 这个结构对长上下文也更友好：段之间并行，SM 利用率远高于单 block 扫全程。
// ─────────────────────────────────────────────────────────────────────────────
constexpr int KV_CHUNK = 128;   // 每个 partial block 处理的 token 数

template<typename T, int BLOCK_DIM>
__global__ void gemma4_attention_partial_kernel(
    const T*     __restrict__ Q,           // [N, H_q, D]
    const T*     __restrict__ K_cache,     // [num_blocks, block_size, H_kv, D]
    const T*     __restrict__ V_cache,     // [num_blocks, block_size, H_kv, D]
    const int*   __restrict__ block_table, // [N, max_blocks]
    const int*   __restrict__ context_lens,// [N]
    float*       __restrict__ partial_o,   // [N, H_q, num_chunks, D] fp32
    float*       __restrict__ partial_m,   // [N, H_q, num_chunks]
    float*       __restrict__ partial_l,   // [N, H_q, num_chunks]
    int H_q, int H_kv, int D,
    int block_size, int max_blocks,
    int window_size, float scale, int num_chunks)
{
    static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM must be a multiple of the warp size");
    constexpr int WARPS = BLOCK_DIM / 32;

    const int batch_idx  = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int chunk      = blockIdx.z;
    const int gqa_ratio  = H_q / H_kv;
    const int kv_head_idx = q_head_idx / gqa_ratio;
    const int tid        = threadIdx.x;
    const int lane       = tid & 31;
    const int warp       = tid >> 5;

    const int ctx_len = context_lens[batch_idx];
    const int start = (window_size > 0 && ctx_len > window_size) ? (ctx_len - window_size) : 0;
    const int num_kv = max(0, ctx_len - start);

    const int kv_begin = chunk * KV_CHUNK;
    const int kv_end   = min(num_kv, kv_begin + KV_CHUNK);

    const T* __restrict__ q_row = Q + ((long long)batch_idx * H_q + q_head_idx) * D;
    constexpr int MAX_NUM_D = (512 + BLOCK_DIM - 1) / BLOCK_DIM;

    float o_acc[MAX_NUM_D];
    float q_reg[MAX_NUM_D];
    #pragma unroll
    for (int j = 0; j < MAX_NUM_D; j++) {
        const int d = tid + j * BLOCK_DIM;
        o_acc[j] = 0.0f;
        q_reg[j] = (d < D) ? g4_to_float(q_row[d]) : 0.0f;
    }

    __shared__ float smem_max[WARPS];
    __shared__ float smem_sum[WARPS];

    const int t_local = tid % KV_CHUNK;                 // 本线程负责的段内 token
    const int pos = kv_begin + t_local;
    const bool active = (pos < kv_end);

    // ── 本 warp 负责的 token 的 logit（点积在 warp 内求和） ──
    float p = -FLT_MAX;
    if (active) {
        const int logical = start + pos;
        const int block_id = block_table[(long long)batch_idx * max_blocks + logical / block_size];
        const int slot     = logical % block_size;
        const T* k_row = K_cache + (((long long)block_id * block_size + slot) * H_kv + kv_head_idx) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int j = 0; j < MAX_NUM_D; j++) {
            const int d = lane + j * 32;
            if (d < D) partial += q_reg[j] * g4_to_float(k_row[d]);
        }
        p = g4_warp_sum(partial) * scale;
    }

    // 段内最大 logit（-FLT_MAX 为中性元，空 lane 不影响结果）
    const float wmax = g4_warp_max(p);
    if (lane == 0) smem_max[warp] = wmax;
    __syncthreads();
    if (tid == 0) {
        float m = -FLT_MAX;
        #pragma unroll
        for (int wi = 0; wi < WARPS; wi++) m = fmaxf(m, smem_max[wi]);
        smem_max[0] = m;
    }
    __syncthreads();
    const float m_chunk = smem_max[0];

    const float w = active ? expf(p - m_chunk) : 0.0f;

    // ── 段内加权和 o = Σ w * v ──
    if (active) {
        const int logical = start + pos;
        const int block_id = block_table[(long long)batch_idx * max_blocks + logical / block_size];
        const int slot     = logical % block_size;
        const T* v_row = V_cache + (((long long)block_id * block_size + slot) * H_kv + kv_head_idx) * D;
        #pragma unroll
        for (int j = 0; j < MAX_NUM_D; j++) {
            const int d = tid + j * BLOCK_DIM;
            if (d < D) o_acc[j] += w * g4_to_float(v_row[d]);
        }
    }

    // ── 段内权重和 l = Σ w ──
    const float wsum = g4_warp_sum(w);
    if (lane == 0) smem_sum[warp] = wsum;
    __syncthreads();
    if (tid == 0) {
        float sw = 0.0f;
        #pragma unroll
        for (int wi = 0; wi < WARPS; wi++) sw += smem_sum[wi];
        smem_sum[0] = sw;
    }
    __syncthreads();
    const float l_chunk = smem_sum[0];

    // ── 写回本段结果 ──
    const long long head_base = ((long long)batch_idx * H_q + q_head_idx) * num_chunks + chunk;
    if (tid == 0) {
        partial_m[head_base] = (l_chunk > 0.0f) ? m_chunk : 0.0f;
        partial_l[head_base] = l_chunk;
    }
    float* __restrict__ o_out = partial_o + ((long long)batch_idx * H_q + q_head_idx) * num_chunks * D
                                         + (long long)chunk * D;
    #pragma unroll
    for (int j = 0; j < MAX_NUM_D; j++) {
        const int d = tid + j * BLOCK_DIM;
        if (d < D) o_out[d] = o_acc[j];
    }
}

template<typename T, int BLOCK_DIM>
__global__ void gemma4_attention_merge_kernel(
    const float* __restrict__ partial_o,   // [N, H_q, num_chunks, D]
    const float* __restrict__ partial_m,   // [N, H_q, num_chunks]
    const float* __restrict__ partial_l,   // [N, H_q, num_chunks]
    T*           __restrict__ out,         // [N, H_q, D]
    int N, int H_q, int D, int num_chunks)
{
    const int batch_idx  = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int tid        = threadIdx.x;

    const long long head_base = ((long long)batch_idx * H_q + q_head_idx) * num_chunks;
    __shared__ float smem_m;
    __shared__ float smem_l;

    if (tid == 0) {
        float m = -FLT_MAX;
        for (int c = 0; c < num_chunks; c++) m = fmaxf(m, partial_m[head_base + c]);
        float l = 0.0f;
        for (int c = 0; c < num_chunks; c++) l += partial_l[head_base + c] * expf(partial_m[head_base + c] - m);
        smem_m = m;
        smem_l = l;
    }
    __syncthreads();
    const float inv_l = (smem_l > 0.0f) ? (1.0f / smem_l) : 0.0f;

    const float* __restrict__ po = partial_o + head_base * D;
    T* __restrict__ out_row = out + ((long long)batch_idx * H_q + q_head_idx) * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < num_chunks; c++) {
            acc += po[(long long)c * D + d] * expf(partial_m[head_base + c] - smem_m);
        }
        out_row[d] = g4_from_float<T>(acc * inv_l);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python 绑定
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemma4_decode_attention(
    torch::Tensor q,               // [N, H_q, D] fp16/bf16
    torch::Tensor k_cache,         // [num_blocks, block_size, H_kv, D]
    torch::Tensor v_cache,         // [num_blocks, block_size, H_kv, D]
    torch::Tensor block_table,     // [N, max_blocks] int32
    torch::Tensor context_lens,    // [N] int32
    int64_t window_size)
{
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "gemma4_decode_attention: q/k_cache/v_cache must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "gemma4_decode_attention: q must be fp16/bf16");
    TORCH_CHECK(k_cache.dtype() == q.dtype() && v_cache.dtype() == q.dtype(),
                "gemma4_decode_attention: k/v cache dtype must match q");
    TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4,
                "gemma4_decode_attention: expected q [N,H_q,D] and caches [blocks,bs,H_kv,D]");

    auto q_c = q.contiguous();
    auto k_c = k_cache.contiguous();
    auto v_c = v_cache.contiguous();
    auto bt  = block_table.to(torch::kInt32).contiguous();
    auto ctx = context_lens.to(torch::kInt32).contiguous();

    const int N    = (int)q_c.size(0);
    const int H_q  = (int)q_c.size(1);
    const int D    = (int)q_c.size(2);
    const int H_kv = (int)k_c.size(2);
    const int block_size = (int)k_c.size(1);
    const int max_blocks = (int)bt.size(1);

    TORCH_CHECK(H_kv > 0 && H_q % H_kv == 0, "gemma4_decode_attention: H_q must be a multiple of H_kv");
    TORCH_CHECK(k_c.size(0) == v_c.size(0) && k_c.size(3) == D && v_c.size(3) == D,
                "gemma4_decode_attention: cache shape mismatch");
    TORCH_CHECK(bt.size(0) == N && ctx.size(0) == N,
                "gemma4_decode_attention: block_table/context_lens must have N rows");
    TORCH_CHECK(D <= 512, "gemma4_decode_attention: head_dim above 512 is not supported");

    auto out = torch::empty_like(q_c);
    if (N == 0 || H_q == 0) return out;

    // 段数按最长上下文确定，保证每个 head 的段数一致（merge 阶段好索引）
    const int max_ctx = ctx.max().item<int>();
    const int win = (int)window_size;
    const int eff = (win > 0) ? std::min(max_ctx, win) : max_ctx;
    const int num_chunks = std::max(1, (eff + KV_CHUNK - 1) / KV_CHUNK);

    auto opts_f = q_c.options().dtype(torch::kFloat32);
    auto partial_o = torch::zeros({N, H_q, num_chunks, D}, opts_f);
    auto partial_m = torch::full({N, H_q, num_chunks}, -std::numeric_limits<float>::infinity(), opts_f);
    auto partial_l = torch::zeros({N, H_q, num_chunks}, opts_f);

    const int block_dim = (D <= 128) ? 128 : 256;
    dim3 grid_p((unsigned)N, (unsigned)H_q, (unsigned)num_chunks);
    dim3 grid_m((unsigned)N, (unsigned)H_q);
    const float scale = rsqrtf((float)D);

    const auto stream = at::cuda::getCurrentCUDAStream();

#define G4_LAUNCH_PARTIAL(TYPE, BDIM)                                                   \
    gemma4_attention_partial_kernel<TYPE, BDIM><<<grid_p, BDIM, 0, stream>>>(           \
        reinterpret_cast<const TYPE*>(q_c.data_ptr()),                                  \
        reinterpret_cast<const TYPE*>(k_c.data_ptr()),                                  \
        reinterpret_cast<const TYPE*>(v_c.data_ptr()),                                  \
        bt.data_ptr<int>(), ctx.data_ptr<int>(),                                        \
        partial_o.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), \
        H_q, H_kv, D, block_size, max_blocks, win, scale, num_chunks)

#define G4_LAUNCH_MERGE(TYPE, BDIM)                                                     \
    gemma4_attention_merge_kernel<TYPE, BDIM><<<grid_m, BDIM, 0, stream>>>(             \
        partial_o.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), \
        reinterpret_cast<TYPE*>(out.data_ptr()), N, H_q, D, num_chunks)

    if (q_c.dtype() == torch::kFloat16) {
        if (block_dim == 128) { G4_LAUNCH_PARTIAL(__half, 128); }
        else                  { G4_LAUNCH_PARTIAL(__half, 256); }
    } else {
        if (block_dim == 128) { G4_LAUNCH_PARTIAL(__nv_bfloat16, 128); }
        else                  { G4_LAUNCH_PARTIAL(__nv_bfloat16, 256); }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (q_c.dtype() == torch::kFloat16) {
        if (block_dim == 128) { G4_LAUNCH_MERGE(__half, 128); }
        else                  { G4_LAUNCH_MERGE(__half, 256); }
    } else {
        if (block_dim == 128) { G4_LAUNCH_MERGE(__nv_bfloat16, 128); }
        else                  { G4_LAUNCH_MERGE(__nv_bfloat16, 256); }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef G4_LAUNCH_PARTIAL
#undef G4_LAUNCH_MERGE
    return out;
}
