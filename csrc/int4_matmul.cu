/**
 * int4_matmul.cu
 *
 * Fused INT4 dequantize + matmul kernels.  The legacy kernel is retained
 * beside the newer warp-per-row kernel so callers can choose explicitly.
 *
 * Packed layout:
 *   w_packed is [N/2, K] uint8, row-major.  Each byte at [n >> 1, k]
 *   contains the two output rows' values at column k:
 *
 *       low nibble  = even output row n
 *       high nibble = odd output row n
 *
 *   The packed-row stride is K bytes.  In particular, the column index is k,
 *   not k >> 1: each byte stores two rows at the same K position.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>

template<typename T>
__device__ __forceinline__ float _to_float(T v);
template<> __device__ __forceinline__ float _to_float(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float _to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T _from_float(float v);
template<> __device__ __forceinline__ __half _from_float<__half>(float v) {
    return __float2half(v);
}

union TcH2U { __half2 h; uint32_t u; };

__device__ __forceinline__ uint32_t tc_pack_half2(float a, float b)
{
    TcH2U v;
    v.h = __floats2half2_rn(a, b);
    return v.u;
}

__device__ __forceinline__ void tc_mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t tc_b_from_raw(
    const uint8_t* __restrict__ raw, int n_local, int kk, int k_off, float scale)
{
    const int p = n_local >> 1;
    const uint16_t v = *reinterpret_cast<const uint16_t*>(
        raw + p * 64 + kk + k_off);
    const int shift = (n_local & 1) ? 4 : 0;
    const int v0 = (int)((v >> shift) & 0x0F) - 8;
    const int v1 = (int)((v >> (shift + 8)) & 0x0F) - 8;
    return tc_pack_half2((float)v0 * scale, (float)v1 * scale);
}

__device__ __forceinline__ void tc_dequant16_h2(
    const uint8_t* __restrict__ bytes,
    __half* __restrict__ even,
    __half* __restrict__ odd,
    float se,
    float so)
{
    union H2U { __half2 h; uint32_t u; };
    uint32_t ev[8];
    uint32_t od[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint16_t pair =
            (uint16_t)bytes[2 * j] | ((uint16_t)bytes[2 * j + 1] << 8);
        const int e0 = (pair & 0x0F) - 8;
        const int e1 = ((pair >> 8) & 0x0F) - 8;
        const int o0 = ((pair >> 4) & 0x0F) - 8;
        const int o1 = ((pair >> 12) & 0x0F) - 8;
        H2U e;
        H2U o;
        e.h = __floats2half2_rn((float)e0 * se, (float)e1 * se);
        o.h = __floats2half2_rn((float)o0 * so, (float)o1 * so);
        ev[j] = e.u;
        od[j] = o.u;
    }
    *reinterpret_cast<uint4*>(even) =
        make_uint4(ev[0], ev[1], ev[2], ev[3]);
    *reinterpret_cast<uint4*>(even + 8) =
        make_uint4(ev[4], ev[5], ev[6], ev[7]);
    *reinterpret_cast<uint4*>(odd) =
        make_uint4(od[0], od[1], od[2], od[3]);
    *reinterpret_cast<uint4*>(odd + 8) =
        make_uint4(od[4], od[5], od[6], od[7]);
}

// Faster equivalent of tc_dequant16_h2.  A nibble n maps to the exact half
// bit pattern 0x3C00 | n = 1 + n/1024; prmt builds two such halves at once,
// then one FMA subtracts 1032 to recover n-8 exactly.  Multiplying by the
// half scale rounds identically to the float path (n and scale are exact in
// fp16/fp32), and the bitwise result matches tc_dequant16_h2.
__device__ __forceinline__ uint32_t tc_nib_pair(
    uint32_t packed_nibbles, uint32_t selector, __half2 scale)
{
    uint32_t bits = __byte_perm(packed_nibbles, 0x3C3C3C3Cu, selector);
    __half2 value = *reinterpret_cast<__half2*>(&bits);
    value = __hfma2(value, __floats2half2_rn(1024.0f, 1024.0f),
                    __floats2half2_rn(-1032.0f, -1032.0f));
    value = __hmul2(value, scale);
    return *reinterpret_cast<uint32_t*>(&value);
}

__device__ __forceinline__ void tc_dequant16_h2_fast(
    const uint8_t* __restrict__ bytes,
    __half* __restrict__ even,
    __half* __restrict__ odd,
    float se,
    float so)
{
    const uint4 raw = *reinterpret_cast<const uint4*>(bytes);
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
    const __half2 s_even = __float2half2_rn(se);
    const __half2 s_odd = __float2half2_rn(so);
    uint32_t ev[8];
    uint32_t od[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t e = words[i] & 0x0F0F0F0Fu;
        const uint32_t o = (words[i] >> 4) & 0x0F0F0F0Fu;
        ev[2 * i] = tc_nib_pair(e, 0x4140u, s_even);
        ev[2 * i + 1] = tc_nib_pair(e, 0x4342u, s_even);
        od[2 * i] = tc_nib_pair(o, 0x4140u, s_odd);
        od[2 * i + 1] = tc_nib_pair(o, 0x4342u, s_odd);
    }
    *reinterpret_cast<uint4*>(even) = make_uint4(ev[0], ev[1], ev[2], ev[3]);
    *reinterpret_cast<uint4*>(even + 8) = make_uint4(ev[4], ev[5], ev[6], ev[7]);
    *reinterpret_cast<uint4*>(odd) = make_uint4(od[0], od[1], od[2], od[3]);
    *reinterpret_cast<uint4*>(odd + 8) = make_uint4(od[4], od[5], od[6], od[7]);
}

template<> __device__ __forceinline__ __nv_bfloat16 _from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

namespace {

constexpr int kKTile = 64;
constexpr int kLegacyThreads = 128;
constexpr int kLegacyRowsPerBlock = 128;
constexpr int kWarpThreads = 128;
constexpr int kWarpRowsPerBlock = 4;
constexpr int kWarpPairsPerBlock = 4;
constexpr int kPairRowsPerBlock = kWarpPairsPerBlock * 2;
constexpr int kPairThreads = kWarpPairsPerBlock * 32;

}  // namespace

// The HEAD implementation: one thread computes one output row.
template<typename T>
__global__ void int4_dequant_matmul_legacy_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size)
{
    const int batch_idx = blockIdx.y;
    const int n_start = blockIdx.x * kLegacyRowsPerBlock;
    const int tid = threadIdx.x;
    const int n_this = min(kLegacyRowsPerBlock, N - n_start);
    if (n_this <= 0) return;

    const int n_groups = K / group_size;
    const T* __restrict__ x_row = x + (long long)batch_idx * K;
    T* __restrict__ y_row = y + (long long)batch_idx * N + n_start;

    __shared__ float smem_nib[16];
    __shared__ T smem_x[kKTile];

    if (tid < 16) smem_nib[tid] = (float)(tid - 8);
    __syncthreads();

    float acc = 0.0f;
    for (int k_start = 0; k_start < K; k_start += kKTile) {
        const int k_this = min(kKTile, K - k_start);
        const int g = k_start / group_size;
        if (tid < k_this) smem_x[tid] = x_row[k_start + tid];
        __syncthreads();

        const int n_local = tid;
        if (n_local < n_this) {
            const int n_global = n_start + n_local;
            const uint8_t* w_pair =
                w_packed + (long long)(n_global >> 1) * K;
            const bool high_row = (n_global & 1) != 0;
            float dot = 0.0f;
            #pragma unroll 4
            for (int kb = 0; kb < k_this; ++kb) {
                const uint8_t byte = w_pair[k_start + kb];
                const int code = high_row ? (int)(byte >> 4) : (int)(byte & 0x0F);
                dot += _to_float(smem_x[kb]) * smem_nib[code];
            }
            acc += dot * _to_float(group_scales[(long long)n_global * n_groups + g]);
        }
        __syncthreads();
    }

    if (tid < n_this) y_row[tid] = _from_float<T>(acc);
}

// New kernel: one warp computes one output row.
template<typename T>
__global__ void int4_dequant_matmul_warp_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size)
{
    const int batch_idx = blockIdx.y;
    const int n_global = blockIdx.x * kWarpRowsPerBlock + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const bool valid_row = n_global < N;
    const int n_groups = K / group_size;

    const T* x_row = x + (long long)batch_idx * K;
    T* y_row = y + (long long)batch_idx * N;
    const uint8_t* w_pair = valid_row
        ? w_packed + (long long)(n_global >> 1) * K
        : nullptr;
    const bool high_row = (n_global & 1) != 0;

    float acc = 0.0f;
    for (int g = 0; valid_row && g < n_groups; ++g) {
        const int k_start = g * kKTile;
        const int k0 = k_start + lane;
        const int k1 = k0 + 32;
        const uint8_t b0 = w_pair[k0];
        const uint8_t b1 = w_pair[k1];
        const int q0 = high_row ? (int)(b0 >> 4) : (int)(b0 & 0x0F);
        const int q1 = high_row ? (int)(b1 >> 4) : (int)(b1 & 0x0F);
        const float scale = _to_float(group_scales[(long long)n_global * n_groups + g]);
        acc += (_to_float(x_row[k0]) * (float)(q0 - 8)
              + _to_float(x_row[k1]) * (float)(q1 - 8)) * scale;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (valid_row && lane == 0) y_row[n_global] = _from_float<T>(acc);
}

// Row-pair kernel: one warp computes the adjacent rows (2p, 2p+1) that share
// the same packed bytes. The byte is loaded once and both nibbles are used,
// halving weight-byte traffic relative to warp-per-row.
template<typename T>
__global__ void int4_dequant_matmul_pair_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size, int pairs_per_block)
{
    const int batch_idx = blockIdx.y;
    const int pair_idx = blockIdx.x * pairs_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int n_pairs = N >> 1;
    const bool valid = pair_idx < n_pairs;
    const int n_groups = K / group_size;

    const T* x_row = x + (long long)batch_idx * K;
    T* y_row = y + (long long)batch_idx * N;
    const uint8_t* w_pair = valid
        ? w_packed + (long long)pair_idx * K
        : nullptr;

    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    for (int g = 0; valid && g < n_groups; ++g) {
        const int k_start = g * kKTile;
        const int k0 = k_start + lane;
        const int k1 = k0 + 32;
        const uint8_t b0 = w_pair[k0];
        const uint8_t b1 = w_pair[k1];
        const float x0 = _to_float(x_row[k0]);
        const float x1 = _to_float(x_row[k1]);
        const long long scale_base =
            (long long)(pair_idx * 2) * n_groups + g;
        const float s_even = _to_float(group_scales[scale_base]);
        const float s_odd = _to_float(group_scales[scale_base + n_groups]);
        acc_even += (x0 * (float)((b0 & 0x0F) - 8)
                   + x1 * (float)((b1 & 0x0F) - 8)) * s_even;
        acc_odd += (x0 * (float)((b0 >> 4) - 8)
                  + x1 * (float)((b1 >> 4) - 8)) * s_odd;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_even += __shfl_down_sync(0xffffffff, acc_even, offset);
        acc_odd += __shfl_down_sync(0xffffffff, acc_odd, offset);
    }
    if (valid && lane == 0) {
        y_row[pair_idx * 2] = _from_float<T>(acc_even);
        y_row[pair_idx * 2 + 1] = _from_float<T>(acc_odd);
    }
}

// Vectorized row-pair kernel: one warp computes the adjacent rows (2p, 2p+1)
// and each lane loads one aligned 32-bit word (four packed columns).  A warp
// therefore covers two 64-column groups per iteration, cutting weight-load
// instructions by 4x versus the scalar row-pair kernel while keeping the same
// byte traffic.
template<typename T>
__global__ void int4_dequant_matmul_pair_vec_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size, int pairs_per_block)
{
    const int batch_idx = blockIdx.y;
    const int pair_idx = blockIdx.x * pairs_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int n_pairs = N >> 1;
    const bool valid = pair_idx < n_pairs;
    const int n_groups = K / group_size;

    const T* x_row = x + (long long)batch_idx * K;
    T* y_row = y + (long long)batch_idx * N;
    const uint8_t* w_pair = valid
        ? w_packed + (long long)pair_idx * K
        : nullptr;

    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    const int half = lane >> 4;          // two 64-column groups per iteration
    const int sublane = lane & 15;       // 16 lanes * 4 columns = 64 columns
    for (int g0 = 0; valid && g0 < n_groups; g0 += 2) {
        const int g = g0 + half;
        if (g >= n_groups) continue;
        const int col = g * kKTile + sublane * 4;
        const uint32_t packed =
            *reinterpret_cast<const uint32_t*>(w_pair + col);
        const T* xp = x_row + col;
        const float x0 = _to_float(xp[0]);
        const float x1 = _to_float(xp[1]);
        const float x2 = _to_float(xp[2]);
        const float x3 = _to_float(xp[3]);
        const int b0 = (int)(packed & 0xFFu);
        const int b1 = (int)((packed >> 8) & 0xFFu);
        const int b2 = (int)((packed >> 16) & 0xFFu);
        const int b3 = (int)((packed >> 24) & 0xFFu);
        const long long scale_base =
            (long long)(pair_idx * 2) * n_groups + g;
        const float s_even = _to_float(group_scales[scale_base]);
        const float s_odd = _to_float(group_scales[scale_base + n_groups]);
        acc_even += (x0 * (float)((b0 & 0x0F) - 8)
                   + x1 * (float)((b1 & 0x0F) - 8)
                   + x2 * (float)((b2 & 0x0F) - 8)
                   + x3 * (float)((b3 & 0x0F) - 8)) * s_even;
        acc_odd += (x0 * (float)((b0 >> 4) - 8)
                  + x1 * (float)((b1 >> 4) - 8)
                  + x2 * (float)((b2 >> 4) - 8)
                  + x3 * (float)((b3 >> 4) - 8)) * s_odd;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_even += __shfl_down_sync(0xffffffff, acc_even, offset);
        acc_odd += __shfl_down_sync(0xffffffff, acc_odd, offset);
    }
    if (valid && lane == 0) {
        y_row[pair_idx * 2] = _from_float<T>(acc_even);
        y_row[pair_idx * 2 + 1] = _from_float<T>(acc_odd);
    }
}


// Wider row-pair kernel for batch=1 decode: every lane loads one aligned
// 64-bit word (8 packed columns) and reuses it for both rows. A warp covers
// four 64-column groups per iteration, halving the loop count and load
// instructions relative to pair_vec.
template<typename T>
__global__ void int4_dequant_matmul_pair_vec8_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int N, int K, int group_size, int pairs_per_block)
{
    const int pair_idx = blockIdx.x * pairs_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int n_pairs = N >> 1;
    const bool valid = pair_idx < n_pairs;
    const int n_groups = K / group_size;
    const uint8_t* w_pair = valid
        ? w_packed + (long long)pair_idx * K
        : nullptr;

    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    const int group_off = lane >> 3;   // four groups per warp
    const int sublane = lane & 7;      // eight lanes cover 64 columns
    for (int g0 = 0; valid && g0 < n_groups; g0 += 4) {
        const int g = g0 + group_off;
        if (g >= n_groups) continue;
        const int col = g * kKTile + sublane * 8;
        const uint2 raw =
            *reinterpret_cast<const uint2*>(w_pair + col);
        const uint32_t words[2] = {raw.x, raw.y};
        const T* xp = x + col;
        const long long scale_base =
            (long long)(pair_idx * 2) * n_groups + g;
        const float s_even = _to_float(group_scales[scale_base]);
        const float s_odd = _to_float(group_scales[scale_base + n_groups]);
        float dot_even = 0.0f;
        float dot_odd = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int byte = (int)((words[j >> 2] >> (8 * (j & 3))) & 0xFFu);
            const float xv = _to_float(xp[j]);
            dot_even += xv * (float)((byte & 0x0F) - 8);
            dot_odd += xv * (float)((byte >> 4) - 8);
        }
        acc_even += dot_even * s_even;
        acc_odd += dot_odd * s_odd;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_even += __shfl_down_sync(0xffffffff, acc_even, offset);
        acc_odd += __shfl_down_sync(0xffffffff, acc_odd, offset);
    }
    if (valid && lane == 0) {
        y[pair_idx * 2] = _from_float<T>(acc_even);
        y[pair_idx * 2 + 1] = _from_float<T>(acc_odd);
    }
}

// Weight-stationary batch GEMV for 2..4 decode rows. One warp computes the
// adjacent output rows and reuses each packed word across all batch rows.
template<typename T, int MAX_B>
__global__ void int4_dequant_matmul_pair_batch_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size, int pairs_per_block)
{
    const int pair_idx = blockIdx.x * pairs_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int n_pairs = N >> 1;
    const bool valid = pair_idx < n_pairs;
    const int n_groups = K / group_size;
    const uint8_t* w_pair = valid
        ? w_packed + (long long)pair_idx * K
        : nullptr;

    float acc_even[MAX_B];
    float acc_odd[MAX_B];
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) {
        acc_even[b] = 0.0f;
        acc_odd[b] = 0.0f;
    }

    const int group_off = lane >> 3;
    const int sublane = lane & 7;
    for (int g0 = 0; valid && g0 < n_groups; g0 += 4) {
        const int g = g0 + group_off;
        if (g >= n_groups) continue;
        const int col = g * kKTile + sublane * 8;
        const uint2 raw = *reinterpret_cast<const uint2*>(w_pair + col);
        const uint32_t words[2] = {raw.x, raw.y};
        const long long scale_base =
            (long long)(pair_idx * 2) * n_groups + g;
        const float s_even = _to_float(group_scales[scale_base]);
        const float s_odd = _to_float(group_scales[scale_base + n_groups]);
        for (int b = 0; b < B; ++b) {
            const T* xp = x + (long long)b * K + col;
            float de = 0.0f;
            float do_ = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int byte = (int)((words[j >> 2] >> (8 * (j & 3))) & 0xFFu);
                const float xv = _to_float(xp[j]);
                de += xv * (float)((byte & 0x0F) - 8);
                do_ += xv * (float)((byte >> 4) - 8);
            }
            acc_even[b] += de * s_even;
            acc_odd[b] += do_ * s_odd;
        }
    }

    for (int b = 0; b < B; ++b) {
        float even = acc_even[b];
        float odd = acc_odd[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            even += __shfl_down_sync(0xffffffff, even, offset);
            odd += __shfl_down_sync(0xffffffff, odd, offset);
        }
        if (valid && lane == 0) {
            y[(long long)b * N + pair_idx * 2] = _from_float<T>(even);
            y[(long long)b * N + pair_idx * 2 + 1] = _from_float<T>(odd);
        }
    }
}

// Parallel weight-stationary batch GEMV for 2..4 rows.  Unlike pair_batch,
// the batch dimension is mapped to warps: each block stages four packed
// output-row pairs once in shared memory, then one warp per (row, pair)
// computes its dot product.  This keeps the weight traffic of pair_batch but
// removes its serial B loop.
constexpr int kPairBatchPairs = 4;

template<typename T>
__global__ void int4_dequant_matmul_pair_batch_parallel_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size)
{
    extern __shared__ __align__(32) uint8_t smem[];
    uint8_t* sW = smem;
    __half* sS = reinterpret_cast<__half*>(smem + kPairBatchPairs * K);

    const int pair_base = blockIdx.x * kPairBatchPairs;
    const int n_groups = K / group_size;
    const int total_bytes = kPairBatchPairs * K;
    const int scale_count = 2 * kPairBatchPairs * n_groups;

    for (int i = threadIdx.x; i < total_bytes / 16; i += blockDim.x) {
        reinterpret_cast<uint4*>(sW)[i] = reinterpret_cast<const uint4*>(
            w_packed + (long long)pair_base * K)[i];
    }
    for (int i = threadIdx.x; i < scale_count; i += blockDim.x) {
        sS[i] = group_scales[(long long)(pair_base * 2) * n_groups + i];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = warp >> 2;
    const int p = warp & 3;
    const int n0 = 2 * (pair_base + p);
    const uint8_t* w_pair = sW + p * K;
    const T* x_row = x + (long long)b * K;

    const int group_off = lane >> 3;
    const int sublane = lane & 7;
    float acc_even = 0.0f;
    float acc_odd = 0.0f;
    for (int g0 = 0; g0 < n_groups; g0 += 4) {
        const int g = g0 + group_off;
        if (g >= n_groups) continue;
        const int col = g * kKTile + sublane * 8;
        const uint2 raw = *reinterpret_cast<const uint2*>(w_pair + col);
        const uint32_t words[2] = {raw.x, raw.y};
        const T* xp = x_row + col;
        const float s_even = __half2float(sS[(2 * p) * n_groups + g]);
        const float s_odd = __half2float(sS[(2 * p + 1) * n_groups + g]);
        float dot_even = 0.0f;
        float dot_odd = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int byte = (int)((words[j >> 2] >> (8 * (j & 3))) & 0xFFu);
            const float xv = _to_float(xp[j]);
            dot_even += xv * (float)((byte & 0x0F) - 8);
            dot_odd += xv * (float)((byte >> 4) - 8);
        }
        acc_even += dot_even * s_even;
        acc_odd += dot_odd * s_odd;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_even += __shfl_down_sync(0xffffffff, acc_even, offset);
        acc_odd += __shfl_down_sync(0xffffffff, acc_odd, offset);
    }
    if (lane == 0) {
        y[(long long)b * N + n0] = _from_float<T>(acc_even);
        y[(long long)b * N + n0 + 1] = _from_float<T>(acc_odd);
    }
}

// B=16 split-K across four warps for narrow output tiles (down/qkv/o).  Each
// warp owns one K quarter of the same 16 output rows; partial FP32 states are
// reduced once in shared memory.  This raises the block count for linears
// whose output width would otherwise expose only 24-96 warps.
template<int WARPS, int KT>
__global__ __launch_bounds__(WARPS * 32, 3)
void int4_dequant_matmul_tc16_splitk_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int LD = KT + 8;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * NT;
    const int p0 = n0 >> 1;
    const int n_groups = K / group_size;
    const int k_per_split = K / WARPS;
    const int k_begin = warp * k_per_split;
    const int k_end = k_begin + k_per_split;

    __shared__ __align__(32) __half sB[WARPS][NT * LD];
    __shared__ __align__(32) float sC[WARPS][M * NT];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = k_begin; k0 < k_end; k0 += KT) {
        #pragma unroll
        for (int chunk = lane; chunk < 8 * (KT / 16); chunk += 32) {
            const int lp = chunk / (KT / 16);
            const int c = (chunk - lp * (KT / 16)) * 16;
            const int g = (k0 + c) / group_size;
            const uint4 raw = *reinterpret_cast<const uint4*>(
                w_packed + (long long)(p0 + lp) * K + k0 + c);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
            const float se = __half2float(
                group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
            const float so = __half2float(
                group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
            tc_dequant16_h2_fast(
                bytes, &sB[warp][(2 * lp) * LD + c],
                &sB[warp][(2 * lp + 1) * LD + c], se, so
            );
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(
                a_frag, x + (long long)k0 + kk, K
            );
            nvcuda::wmma::load_matrix_sync(
                b_frag, sB[warp] + kk, LD
            );
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(
        sC[warp], acc, NT, nvcuda::wmma::mem_row_major
    );
    __syncthreads();
    if (warp == 0) {
        for (int idx = lane; idx < M * NT; idx += 32) {
            float sum = 0.0f;
            #pragma unroll
            for (int s = 0; s < WARPS; ++s) sum += sC[s][idx];
            const int r = idx / NT;
            const int c = idx - r * NT;
            y[(long long)r * N + n0 + c] = __float2half(sum);
        }
    }
}


__device__ __forceinline__ void tc_cp_async_16(
    void* __restrict__ smem, const void* __restrict__ gmem)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
#endif
}

__device__ __forceinline__ void tc_cp_async_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int N>
__device__ __forceinline__ void tc_cp_async_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}


__device__ __forceinline__ uint4 tc_load_b16_tiled(
    const uint8_t* __restrict__ w_tiled, int n0, int lp, int c,
    int k, int K)
{
    if (k >= K) return make_uint4(0, 0, 0, 0);
    const int tile = n0 >> 4;
    const int kt = k >> 6;
    const long long base =
        (((long long)tile * (K >> 6) + kt) * 8 + lp) * 64 + c;
    return *reinterpret_cast<const uint4*>(w_tiled + base);
}

__device__ __forceinline__ uint4 tc_load_b16(
    const uint8_t* __restrict__ w_packed, int p0, int lp, int c,
    int k, int K);

// B=16 register-mma kernel: packed B is staged coalesced in 512 bytes of
// shared memory per warp, then dequantized straight into the two mma B
// registers. A is shared per block.
template<int WARPS>
__global__ __launch_bounds__(WARPS * 32, 8)
void int4_dequant_matmul_tc16_reg_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int N, int K, int group_size)
{
    constexpr int LD = 72;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / 16;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * 16 : 0;
    const int p0 = n0 >> 1;
    const int group = lane >> 2;
    const int tid = lane & 3;
    const int n_groups = K / group_size;

    __shared__ __align__(32) __half sA[16 * LD];
    __shared__ __align__(32) uint8_t sRaw[WARPS][512];

    float ca0 = 0, ca1 = 0, ca2 = 0, ca3 = 0;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;

    for (int k0 = 0; k0 < K; k0 += 64) {
        #pragma unroll
        for (int chunk = threadIdx.x; chunk < 16 * 8; chunk += WARPS * 32) {
            const int r = chunk / 8;
            const int c = (chunk - r * 8) * 8;
            const uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + c);
            *reinterpret_cast<uint4*>(&sA[r * LD + c]) = v;
        }
        if (active) {
            const int lp = lane >> 2;
            const int c = (lane & 3) * 16;
            const uint4 raw = *reinterpret_cast<const uint4*>(
                w_packed + (long long)(p0 + lp) * K + k0 + c);
            *reinterpret_cast<uint4*>(&sRaw[warp][lp * 64 + c]) = raw;
        }
        __syncthreads();
        if (active) {
            const int n_a = n0 + group;
            const int n_b = n0 + 8 + group;
            #pragma unroll
            for (int kk = 0; kk < 64; kk += 16) {
                const int g = (k0 + kk) / group_size;
                const float sa = __half2float(
                    group_scales[(long long)n_a * n_groups + g]);
                const float sb = __half2float(
                    group_scales[(long long)n_b * n_groups + g]);
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                    &sA[group * LD + kk + tid * 2]);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                    &sA[(group + 8) * LD + kk + tid * 2]);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                    &sA[group * LD + kk + tid * 2 + 8]);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                    &sA[(group + 8) * LD + kk + tid * 2 + 8]);
                const uint8_t* raw = sRaw[warp];
                const uint32_t ba0 = tc_b_from_raw(raw, group, kk, tid * 2, sa);
                const uint32_t ba1 = tc_b_from_raw(raw, group, kk, tid * 2 + 8, sa);
                const uint32_t bb0 = tc_b_from_raw(raw, 8 + group, kk, tid * 2, sb);
                const uint32_t bb1 = tc_b_from_raw(raw, 8 + group, kk, tid * 2 + 8, sb);
                tc_mma_m16n8k16(ca0, ca1, ca2, ca3, a0, a1, a2, a3, ba0, ba1);
                tc_mma_m16n8k16(cb0, cb1, cb2, cb3, a0, a1, a2, a3, bb0, bb1);
            }
        }
        __syncthreads();
    }
    if (!active) return;
    y[(long long)group * N + n0 + tid * 2] = __float2half(ca0);
    y[(long long)group * N + n0 + tid * 2 + 1] = __float2half(ca1);
    y[(long long)(group + 8) * N + n0 + tid * 2] = __float2half(ca2);
    y[(long long)(group + 8) * N + n0 + tid * 2 + 1] = __float2half(ca3);
    y[(long long)group * N + n0 + 8 + tid * 2] = __float2half(cb0);
    y[(long long)group * N + n0 + 8 + tid * 2 + 1] = __float2half(cb1);
    y[(long long)(group + 8) * N + n0 + 8 + tid * 2] = __float2half(cb2);
    y[(long long)(group + 8) * N + n0 + 8 + tid * 2 + 1] = __float2half(cb3);
}

// B=16, two K-partition blocks per output tile.  Each block stages only B
// (no shared A/C), giving ~20 resident warps/SM; the two FP32 partials are
// reduced by a tiny epilogue kernel.  This targets the memory-latency limit
// of the narrow down/qkv/o linears and the large gate/up/lm_head shapes.
template<int WARPS, int KT>
__global__ __launch_bounds__(WARPS * 32, 3)
void int4_dequant_matmul_tc16_split2_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    float* __restrict__ partial,
    int N, int K, int group_size, int parts)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int LD = KT + 8;
    constexpr int NRAW = KT / 64;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * NT;
    const int bs = blockIdx.y;
    const int p0 = n0 >> 1;
    const int n_groups = K / group_size;
    const int k_per = K / (parts * WARPS);
    const int k_begin = (bs * WARPS + warp) * k_per;
    const int k_end = k_begin + k_per;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sB[WARPS][NT * LD];
    __shared__ __align__(32) float sC[WARPS][M * NT];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    uint4 raw_cur[NRAW];
    uint4 raw_next[NRAW];
    #pragma unroll
    for (int r = 0; r < NRAW; ++r) {
        raw_cur[r] = tc_load_b16(w_packed, p0, lp, c + r * 64, k_begin, K);
        raw_next[r] = tc_load_b16(
            w_packed, p0, lp, c + r * 64, k_begin + KT, K);
    }
    for (int k0 = k_begin; k0 < k_end; k0 += KT) {
        uint4 raw_future[NRAW];
        #pragma unroll
        for (int r = 0; r < NRAW; ++r) {
            raw_future[r] = tc_load_b16(
                w_packed, p0, lp, c + r * 64, k0 + 2 * KT, K);
        }
        #pragma unroll
        for (int r = 0; r < NRAW; ++r) {
            const int cr = c + r * 64;
            const int g = (k0 + cr) / group_size;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw_cur[r]);
            const float se = __half2float(
                group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
            const float so = __half2float(
                group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
            tc_dequant16_h2_fast(
                bytes, &sB[warp][(2 * lp) * LD + cr],
                &sB[warp][(2 * lp + 1) * LD + cr], se, so
            );
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(
                a_frag, x + (long long)k0 + kk, K
            );
            nvcuda::wmma::load_matrix_sync(b_frag, sB[warp] + kk, LD);
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncwarp();
        #pragma unroll
        for (int r = 0; r < NRAW; ++r) {
            raw_cur[r] = raw_next[r];
            raw_next[r] = raw_future[r];
        }
    }

    nvcuda::wmma::store_matrix_sync(
        sC[warp], acc, NT, nvcuda::wmma::mem_row_major
    );
    __syncthreads();
    if (warp == 0) {
        for (int idx = lane; idx < M * NT; idx += 32) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) sum += sC[w][idx];
            const int r = idx / NT;
            const int c2 = idx - r * NT;
            partial[(long long)bs * 16 * N + (long long)r * N + n0 + c2] = sum;
        }
    }
}


// B=16 split-2 with two output tiles per warp.  A fragments are loaded once
// per kk and reused for both tiles, halving the duplicated activation traffic
// that dominates the wide-K down_proj shape (N=1536, K=6144).
template<int WARPS, int TILES>
__global__ __launch_bounds__(WARPS * 32, 3)
void int4_dequant_matmul_tc16_split2_mt2_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    float* __restrict__ partial,
    __half* __restrict__ y,
    int N, int K, int group_size, int parts)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * (TILES * NT);
    const int bs = blockIdx.y;
    const int p0 = n0 >> 1;
    const int n_groups = K / group_size;
    const int k_per = K / (parts * WARPS);
    const int k_begin = (bs * WARPS + warp) * k_per;
    const int k_end = k_begin + k_per;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sB[WARPS][TILES][NT * LD];
    __shared__ __align__(32) float sC[WARPS][M * NT];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc[TILES];
    #pragma unroll
    for (int t = 0; t < TILES; ++t) nvcuda::wmma::fill_fragment(acc[t], 0.0f);

    uint4 raw_cur[TILES];
    uint4 raw_next[TILES];
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
        raw_cur[t] = tc_load_b16(w_packed, p0 + 8 * t, lp, c, k_begin, K);
        raw_next[t] = tc_load_b16(w_packed, p0 + 8 * t, lp, c, k_begin + KT, K);
    }
    for (int k0 = k_begin; k0 < k_end; k0 += KT) {
        uint4 raw_future[TILES];
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            raw_future[t] = tc_load_b16(
                w_packed, p0 + 8 * t, lp, c, k0 + 2 * KT, K);
        }
        const int g = (k0 + c) / group_size;
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int nt0 = n0 + t * NT;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw_cur[t]);
            const float se = __half2float(
                group_scales[(long long)(nt0 + 2 * lp) * n_groups + g]);
            const float so = __half2float(
                group_scales[(long long)(nt0 + 2 * lp + 1) * n_groups + g]);
            tc_dequant16_h2_fast(
                bytes, &sB[warp][t][(2 * lp) * LD + c],
                &sB[warp][t][(2 * lp + 1) * LD + c], se, so);
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(
                a_frag, x + (long long)k0 + kk, K);
            #pragma unroll
            for (int t = 0; t < TILES; ++t) {
                nvcuda::wmma::load_matrix_sync(
                    b_frag, sB[warp][t] + kk, LD);
                nvcuda::wmma::mma_sync(acc[t], a_frag, b_frag, acc[t]);
            }
        }
        __syncwarp();
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            raw_cur[t] = raw_next[t];
            raw_next[t] = raw_future[t];
        }
    }

    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
        nvcuda::wmma::store_matrix_sync(
            sC[warp], acc[t], NT, nvcuda::wmma::mem_row_major);
        __syncwarp();
        __syncthreads();
        if (warp == 0) {
            for (int idx = lane; idx < M * NT; idx += 32) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) sum += sC[w][idx];
                const int r = idx / NT;
                const int c2 = idx - r * NT;
                if (parts == 1) {
                    y[(long long)r * N + n0 + t * NT + c2] =
                        __float2half(sum);
                } else {
                    partial[(long long)bs * 16 * N + (long long)r * N +
                            n0 + t * NT + c2] = sum;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void tc16_reduce_kernel(
    const float* __restrict__ partial, __half* __restrict__ y,
    int n, int chunk, int parts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int p = 0; p < parts; ++p) {
            sum += partial[(long long)p * chunk + i];
        }
        y[i] = __float2half(sum);
    }
}

// B=16 specialization: keep A and the accumulator in registers/global so the
// only shared allocation is the 16x64 dequantized B tile.  Removing the
// per-warp A/C tiles raises occupancy from ~8 to ~20 warps/SM, which is the
// main limiter on the small-N decode linears.
template<int WARPS>
__global__ void int4_dequant_matmul_tc16_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int p0 = n0 >> 1;

    __shared__ __align__(32) __half sB[WARPS][NT * LD];
    __shared__ __align__(32) float sC[WARPS][M * NT];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += KT) {
        if (active) {
            #pragma unroll
            for (int chunk = lane; chunk < 8 * (KT / 16); chunk += 32) {
                const int lp = chunk / (KT / 16);
                const int c = (chunk - lp * (KT / 16)) * 16;
                const int g = (k0 + c) / group_size;
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    w_packed + (long long)(p0 + lp) * K + k0 + c);
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
                const float se = __half2float(
                    group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
                const float so = __half2float(
                    group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
                tc_dequant16_h2_fast(
                    bytes, &sB[warp][(2 * lp) * LD + c],
                    &sB[warp][(2 * lp + 1) * LD + c], se, so
                );
            }
        }
        __syncwarp();
        if (active) {
            #pragma unroll
            for (int kk = 0; kk < KT; kk += 16) {
                nvcuda::wmma::load_matrix_sync(
                    a_frag, x + (long long)k0 + kk, K
                );
                nvcuda::wmma::load_matrix_sync(
                    b_frag, sB[warp] + kk, LD
                );
                nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }
        __syncwarp();
    }

    if (active) {
        nvcuda::wmma::store_matrix_sync(
            sC[warp], acc, NT, nvcuda::wmma::mem_row_major
        );
        __syncwarp();
        for (int idx = lane; idx < 16 * NT; idx += 32) {
            const int r = idx / NT;
            const int c = idx - r * NT;
            y[(long long)r * N + n0 + c] = __float2half(sC[warp][idx]);
        }
    }
}

__device__ __forceinline__ uint4 tc_load_b16(
    const uint8_t* __restrict__ w_packed, int p0, int lp, int c,
    int k, int K)
{
    if (k >= K) return make_uint4(0, 0, 0, 0);
    const uint8_t* ptr = w_packed + (long long)(p0 + lp) * K + k + c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint4 v;
    asm volatile(
        "ld.global.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
#else
    return *reinterpret_cast<const uint4*>(ptr);
#endif
}

__global__ void tc_f16_convert_kernel(
    const float* __restrict__ src, __half* __restrict__ dst, int n)
{
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        *reinterpret_cast<__half2*>(dst + i) =
            __floats2half2_rn(src[i], src[i + 1]);
    } else if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

// B=16, large-N pipelined variant: preload the next two K tiles of packed
// weights into registers so the global load overlaps the current dequant/MMA.
// A and C stay in shared memory; the ptx prefetch hint alone did not hide the
// HBM latency.
template<int WARPS>
__global__ __launch_bounds__(WARPS * 32, 4)
void int4_dequant_matmul_tc16_pipe_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int p0 = n0 >> 1;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sA[M * LD];
    __shared__ __align__(32) __half sB[WARPS][NT * LD];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    if (!active) {
        // Inactive warps still participate in the block-wide A load/sync.
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    uint4 raw_cur = tc_load_b16(w_packed, p0, lp, c, 0, K);
    uint4 raw_next = tc_load_b16(w_packed, p0, lp, c, KT, K);
    for (int k0 = 0; k0 < K; k0 += KT) {
        const uint4 raw_future =
            tc_load_b16(w_packed, p0, lp, c, k0 + 2 * KT, K);

        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int cc = (chunk - r * (KT / 8)) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
        }

        const int g = (k0 + c) / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
        const float so = __half2float(
            group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw_cur);
        tc_dequant16_h2_fast(
            bytes, &sB[warp][(2 * lp) * LD + c],
            &sB[warp][(2 * lp + 1) * LD + c], se, so
        );

        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, sA + kk, LD);
            nvcuda::wmma::load_matrix_sync(b_frag, sB[warp] + kk, LD);
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
        raw_cur = raw_next;
        raw_next = raw_future;
    }

    // sB is dead after the final MMA, so reuse this warp's 2.3 KiB tile as
    // the 1 KiB FP32 accumulator staging area instead of growing shared.
    float* sC = reinterpret_cast<float*>(sB[warp]);
    nvcuda::wmma::store_matrix_sync(
        sC, acc, NT, nvcuda::wmma::mem_row_major
    );
    __syncwarp();
    #pragma unroll
    for (int pair = lane; pair < M * NT / 2; pair += 32) {
        const int idx = pair * 2;
        const int r = idx / NT;
        const int c = idx - r * NT;
        *reinterpret_cast<__half2*>(y + (long long)r * N + n0 + c) =
            __floats2half2_rn(sC[idx], sC[idx + 1]);
    }
}


// B=16 pipelined variant with a deeper raw-weight register queue.  The
// two-stage pipe hides only one HBM round trip; four stages keep roughly
// 2 KiB per warp in flight, which is what the narrow-down / large-vocab
// shapes need to approach the memory roofline.
template<int WARPS, int DEPTH>
__global__ __launch_bounds__(WARPS * 32, 4)
void int4_dequant_matmul_tc16_pipe_d_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    float* __restrict__ tmp,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int p0 = n0 >> 1;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sA[M * LD];
    __shared__ __align__(32) __half sB[WARPS][NT * LD];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    if (!active) {
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    uint4 raw[DEPTH];
    #pragma unroll
    for (int s = 0; s < DEPTH; ++s) {
        raw[s] = tc_load_b16(w_packed, p0, lp, c, s * KT, K);
    }
    for (int k0 = 0; k0 < K; k0 += KT) {
        const uint4 raw_future =
            tc_load_b16(w_packed, p0, lp, c, k0 + DEPTH * KT, K);

        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int cc = (chunk - r * (KT / 8)) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
        }

        const int g = (k0 + c) / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
        const float so = __half2float(
            group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw[0]);
        tc_dequant16_h2_fast(
            bytes, &sB[warp][(2 * lp) * LD + c],
            &sB[warp][(2 * lp + 1) * LD + c], se, so
        );

        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, sA + kk, LD);
            nvcuda::wmma::load_matrix_sync(b_frag, sB[warp] + kk, LD);
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s + 1 < DEPTH; ++s) raw[s] = raw[s + 1];
        raw[DEPTH - 1] = raw_future;
    }

    nvcuda::wmma::store_matrix_sync(
        tmp + n0, acc, N, nvcuda::wmma::mem_row_major
    );
}



// B=16 pipe with cp.async staging of the packed weight tile.  The previous
// pipe kept two uint4 prefetches in registers; this version issues a 512-byte
// async copy per warp per K tile into a small shared ring, so the global load
// overlaps dequant + WMMA instead of occupying registers and issue slots.
template<int WARPS, int STAGES>
__global__ __launch_bounds__(WARPS * 32, 4)
void int4_dequant_matmul_tc16_pipe_ca_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int p0 = n0 >> 1;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sA[M * LD];
    __shared__ __align__(32) __half sB[WARPS][NT * LD];
    __shared__ __align__(16) uint8_t sRaw[WARPS][STAGES][8 * 64];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    if (!active) {
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    // Prime STAGES groups.  Each lane copies its own 16-byte row segment.
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        const int k0 = s * KT;
        if (k0 < K) {
            tc_cp_async_16(
                &sRaw[warp][s][lp * 64 + c],
                w_packed + (long long)(p0 + lp) * K + k0 + c);
        }
        tc_cp_async_commit();
    }

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += KT) {
        tc_cp_async_wait<STAGES - 1>();
        __syncwarp();

        const int g = (k0 + c) / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
        const float so = __half2float(
            group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
        tc_dequant16_h2_fast(
            &sRaw[warp][stage][lp * 64 + c],
            &sB[warp][(2 * lp) * LD + c],
            &sB[warp][(2 * lp + 1) * LD + c], se, so);
        __syncwarp();

        // sRaw[stage] has been consumed, so it can receive the next tile.
        const int future = k0 + STAGES * KT;
        if (future < K) {
            tc_cp_async_16(
                &sRaw[warp][stage][lp * 64 + c],
                w_packed + (long long)(p0 + lp) * K + future + c);
        }
        tc_cp_async_commit();

        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int cc = (chunk - r * (KT / 8)) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, sA + kk, LD);
            nvcuda::wmma::load_matrix_sync(b_frag, sB[warp] + kk, LD);
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
        stage = (stage + 1) % STAGES;
    }

    float* sC = reinterpret_cast<float*>(sB[warp]);
    nvcuda::wmma::store_matrix_sync(sC, acc, NT, nvcuda::wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int pair = lane; pair < M * NT / 2; pair += 32) {
        const int idx = pair * 2;
        const int r = idx / NT;
        const int cc = idx - r * NT;
        *reinterpret_cast<__half2*>(y + (long long)r * N + n0 + cc) =
            __floats2half2_rn(sC[idx], sC[idx + 1]);
    }
}


// Tile-major packed weights: a warp's 32 lanes cover one contiguous 512-byte
// block per K tile instead of eight 64-byte row segments separated by K.
template<int WARPS>
__global__ __launch_bounds__(WARPS * 32, 4)
void int4_dequant_matmul_tc16_pipe_tiled_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_tiled,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int p0 = n0 >> 1;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sA[M * LD];
    __shared__ __align__(32) __half sB[WARPS][NT * LD];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    if (!active) {
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    uint4 raw_cur = tc_load_b16_tiled(w_tiled, n0, lp, c, 0, K);
    uint4 raw_next = tc_load_b16_tiled(w_tiled, n0, lp, c, KT, K);
    for (int k0 = 0; k0 < K; k0 += KT) {
        const uint4 raw_future =
            tc_load_b16_tiled(w_tiled, n0, lp, c, k0 + 2 * KT, K);
        const int g = (k0 + c) / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
        const float so = __half2float(
            group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw_cur);
        tc_dequant16_h2_fast(
            bytes, &sB[warp][(2 * lp) * LD + c],
            &sB[warp][(2 * lp + 1) * LD + c], se, so);

        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int cc = (chunk - r * (KT / 8)) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, sA + kk, LD);
            nvcuda::wmma::load_matrix_sync(b_frag, sB[warp] + kk, LD);
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
        raw_cur = raw_next;
        raw_next = raw_future;
    }

    float* sC = reinterpret_cast<float*>(sB[warp]);
    nvcuda::wmma::store_matrix_sync(sC, acc, NT, nvcuda::wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int pair = lane; pair < M * NT / 2; pair += 32) {
        const int idx = pair * 2;
        const int r = idx / NT;
        const int cc = idx - r * NT;
        *reinterpret_cast<__half2*>(y + (long long)r * N + n0 + cc) =
            __floats2half2_rn(sC[idx], sC[idx + 1]);
    }
}


// ---------------------------------------------------------------------------
// Fragment-major INT4 prepack + direct-B MMA.
//
// One lane's 16-byte prepacked block contains exactly the 32 nibbles needed
// for its B operands across a 64-wide K tile: four kk sub-tiles times the two
// n8 groups.  The kernel dequantizes those nibbles straight into the mma B
// registers, so there is no dequantized B tile in shared memory at all.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t tc_frag_half2(
    uint32_t two_nibble_bytes, __half2 scale)
{
    uint32_t bits = __byte_perm(two_nibble_bytes, 0x3C3C3C3Cu, 0x4140u);
    __half2 v = *reinterpret_cast<__half2*>(&bits);
    v = __hfma2(v, __floats2half2_rn(1024.0f, 1024.0f),
                __floats2half2_rn(-1032.0f, -1032.0f));
    return *reinterpret_cast<uint32_t*>(&__hmul2(v, scale));
}

__device__ __forceinline__ void tc_frag_b(
    uint16_t packed, __half2 scale, uint32_t& b0, uint32_t& b1)
{
    const uint32_t k01 = (packed & 0x0Fu) | (((packed >> 4) & 0x0Fu) << 8);
    const uint32_t k89 = ((packed >> 8) & 0x0Fu) |
                         (((packed >> 12) & 0x0Fu) << 8);
    b0 = tc_frag_half2(k01, scale);
    b1 = tc_frag_half2(k89, scale);
}

__device__ __forceinline__ uint32_t tc_frag_half2_raw(
    uint32_t two_nibble_bytes)
{
    uint32_t bits = __byte_perm(two_nibble_bytes, 0x3C3C3C3Cu, 0x4140u);
    __half2 v = *reinterpret_cast<__half2*>(&bits);
    return *reinterpret_cast<uint32_t*>(
        &__hfma2(v, __floats2half2_rn(1024.0f, 1024.0f),
                 __floats2half2_rn(-1032.0f, -1032.0f)));
}

__device__ __forceinline__ void tc_frag_b_raw(
    uint16_t packed, uint32_t& b0, uint32_t& b1)
{
    const uint32_t k01 = (packed & 0x0Fu) | (((packed >> 4) & 0x0Fu) << 8);
    const uint32_t k89 = ((packed >> 8) & 0x0Fu) |
                         (((packed >> 12) & 0x0Fu) << 8);
    b0 = tc_frag_half2_raw(k01);
    b1 = tc_frag_half2_raw(k89);
}

__global__ void int4_prepack_frag_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int K)
{
    const int tile = blockIdx.x;
    const int kt = blockIdx.y;
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int tin = lane & 3;
    const int n_groups = K >> 6;
    __align__(16) uint16_t out[8];
    #pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
        const int kbase = kt * 64 + ki * 16 + tin * 2;
        #pragma unroll
        for (int gi = 0; gi < 2; ++gi) {
            const int n = tile * 16 + g + 8 * gi;
            const uint8_t* row = src + (long long)(n >> 1) * K;
            const int shift = (n & 1) * 4;
            const int k0 = kbase, k1 = kbase + 1;
            const int k8 = kbase + 8, k9 = kbase + 9;
            const uint32_t v =
                ((row[k0] >> shift) & 0xFu) |
                (((row[k1] >> shift) & 0xFu) << 4) |
                (((row[k8] >> shift) & 0xFu) << 8) |
                (((row[k9] >> shift) & 0xFu) << 12);
            out[2 * ki + gi] = (uint16_t)v;
        }
    }
    const long long off =
        (((long long)tile * n_groups + kt) * 32 + lane) * 16;
    *reinterpret_cast<uint4*>(dst + off) = *reinterpret_cast<const uint4*>(out);
}

template<int WARPS>
__global__ __launch_bounds__(WARPS * 32, 8)
void int4_dequant_matmul_tc16_frag_ga_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_frag,
    const __half* __restrict__ group_scales,
    float* __restrict__ partial,
    __half* __restrict__ y,
    int N, int K, int group_size, int parts)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    if (tile >= n_tiles) return;
    const int n0 = tile * NT;
    const int n_groups = K / group_size;
    const int g = lane >> 2;
    const int tin = lane & 3;
    const int part = blockIdx.y;
    const int kt_begin = (K / KT) * part / parts;
    const int kt_end = (K / KT) * (part + 1) / parts;

    float ca0 = 0, ca1 = 0, ca2 = 0, ca3 = 0;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;

    const int nt = K / KT;
    auto load_frag = [&](int kt) -> uint4 {
        const long long off =
            (((long long)tile * nt + kt) * 32 + lane) * 16;
        return *reinterpret_cast<const uint4*>(w_frag + off);
    };

    uint4 raw = load_frag(kt_begin);
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int k0 = kt * KT;
        const uint4 next = (kt + 1 < kt_end) ? load_frag(kt + 1)
                                             : make_uint4(0, 0, 0, 0);
        const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        const int gi = k0 / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + g) * n_groups + gi]);
        const float so = __half2float(
            group_scales[(long long)(n0 + g + 8) * n_groups + gi]);
        const __half2 s_lo = __floats2half2_rn(se, se);
        const __half2 s_hi = __floats2half2_rn(so, so);

        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int kk = k0 + ki * 16 + tin * 2;
            const uint16_t wa0 = (uint16_t)(w[ki] & 0xFFFFu);
            const uint16_t wa1 = (uint16_t)(w[ki] >> 16);
            uint32_t b00, b01, b10, b11;
            tc_frag_b(wa0, s_lo, b00, b01);
            tc_frag_b(wa1, s_hi, b10, b11);
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                x + (long long)g * K + kk);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                x + (long long)(g + 8) * K + kk);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                x + (long long)g * K + kk + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                x + (long long)(g + 8) * K + kk + 8);
            tc_mma_m16n8k16(ca0, ca1, ca2, ca3, a0, a1, a2, a3, b00, b01);
            tc_mma_m16n8k16(cb0, cb1, cb2, cb3, a0, a1, a2, a3, b10, b11);
        }
        raw = next;
    }

    if (parts == 1) {
        y[(long long)g * N + n0 + tin * 2] = __float2half(ca0);
        y[(long long)g * N + n0 + tin * 2 + 1] = __float2half(ca1);
        y[(long long)(g + 8) * N + n0 + tin * 2] = __float2half(ca2);
        y[(long long)(g + 8) * N + n0 + tin * 2 + 1] = __float2half(ca3);
        y[(long long)g * N + n0 + 8 + tin * 2] = __float2half(cb0);
        y[(long long)g * N + n0 + 8 + tin * 2 + 1] = __float2half(cb1);
        y[(long long)(g + 8) * N + n0 + 8 + tin * 2] = __float2half(cb2);
        y[(long long)(g + 8) * N + n0 + 8 + tin * 2 + 1] = __float2half(cb3);
    } else {
        float* dst = partial + (long long)part * 16 * N;
        dst[(long long)g * N + n0 + tin * 2] = ca0;
        dst[(long long)g * N + n0 + tin * 2 + 1] = ca1;
        dst[(long long)(g + 8) * N + n0 + tin * 2] = ca2;
        dst[(long long)(g + 8) * N + n0 + tin * 2 + 1] = ca3;
        dst[(long long)g * N + n0 + 8 + tin * 2] = cb0;
        dst[(long long)g * N + n0 + 8 + tin * 2 + 1] = cb1;
        dst[(long long)(g + 8) * N + n0 + 8 + tin * 2] = cb2;
        dst[(long long)(g + 8) * N + n0 + 8 + tin * 2 + 1] = cb3;
    }
}

template<int WARPS>
__global__ __launch_bounds__(WARPS * 32, 8)
void int4_dequant_matmul_tc16_frag_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_frag,
    const __half* __restrict__ group_scales,
    float* __restrict__ partial,
    __half* __restrict__ y,
    int N, int K, int group_size, int parts)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int g = lane >> 2;
    const int tin = lane & 3;
    const int part = blockIdx.y;
    const int kt_begin = (K / KT) * part / parts;
    const int kt_end = (K / KT) * (part + 1) / parts;

    __shared__ __align__(32) __half sA[M * LD];

    float ca0 = 0, ca1 = 0, ca2 = 0, ca3 = 0;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;

    if (!active) {
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    const int nt = K / KT;
    auto load_frag = [&](int kt) -> uint4 {
        const long long off =
            (((long long)tile * nt + kt) * 32 + lane) * 16;
        return *reinterpret_cast<const uint4*>(w_frag + off);
    };

    uint4 raw = load_frag(kt_begin);
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int k0 = kt * KT;
        const uint4 next = (kt + 1 < kt_end) ? load_frag(kt + 1)
                                             : make_uint4(0, 0, 0, 0);

        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int cc = (chunk - r * (KT / 8)) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
        }
        __syncthreads();

        const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        const int gi = k0 / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + g) * n_groups + gi]);
        const float so = __half2float(
            group_scales[(long long)(n0 + g + 8) * n_groups + gi]);
        const __half2 s_lo = __floats2half2_rn(se, se);
        const __half2 s_hi = __floats2half2_rn(so, so);

        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int kk = ki * 16;
            const uint16_t wa0 = (uint16_t)(w[ki] & 0xFFFFu);
            const uint16_t wa1 = (uint16_t)(w[ki] >> 16);
            uint32_t b00, b01, b10, b11;
            tc_frag_b(wa0, s_lo, b00, b01);
            tc_frag_b(wa1, s_hi, b10, b11);
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                &sA[g * LD + kk + tin * 2]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                &sA[(g + 8) * LD + kk + tin * 2]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                &sA[g * LD + kk + tin * 2 + 8]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                &sA[(g + 8) * LD + kk + tin * 2 + 8]);
            tc_mma_m16n8k16(ca0, ca1, ca2, ca3, a0, a1, a2, a3, b00, b01);
            tc_mma_m16n8k16(cb0, cb1, cb2, cb3, a0, a1, a2, a3, b10, b11);
        }
        __syncthreads();
        raw = next;
    }

    float* dst = (parts == 1) ? nullptr : partial + (long long)part * 16 * N;
    if (parts == 1) {
        y[(long long)g * N + n0 + tin * 2] = __float2half(ca0);
        y[(long long)g * N + n0 + tin * 2 + 1] = __float2half(ca1);
        y[(long long)(g + 8) * N + n0 + tin * 2] = __float2half(ca2);
        y[(long long)(g + 8) * N + n0 + tin * 2 + 1] = __float2half(ca3);
        y[(long long)g * N + n0 + 8 + tin * 2] = __float2half(cb0);
        y[(long long)g * N + n0 + 8 + tin * 2 + 1] = __float2half(cb1);
        y[(long long)(g + 8) * N + n0 + 8 + tin * 2] = __float2half(cb2);
        y[(long long)(g + 8) * N + n0 + 8 + tin * 2 + 1] = __float2half(cb3);
    } else {
        dst[(long long)g * N + n0 + tin * 2] = ca0;
        dst[(long long)g * N + n0 + tin * 2 + 1] = ca1;
        dst[(long long)(g + 8) * N + n0 + tin * 2] = ca2;
        dst[(long long)(g + 8) * N + n0 + tin * 2 + 1] = ca3;
        dst[(long long)g * N + n0 + 8 + tin * 2] = cb0;
        dst[(long long)g * N + n0 + 8 + tin * 2 + 1] = cb1;
        dst[(long long)(g + 8) * N + n0 + 8 + tin * 2] = cb2;
        dst[(long long)(g + 8) * N + n0 + 8 + tin * 2 + 1] = cb3;
    }
}

// B=16 multi-tile pipe: each warp owns NTILE adjacent 16-column tiles and
// reuses one block-wide shared A tile across all of them.  This cuts the
// duplicated A traffic for the very wide lm_head shape and keeps NTILE
// independent packed-weight loads in flight per warp.
template<int WARPS>
__global__ __launch_bounds__(WARPS * 32, 8)
void int4_dequant_matmul_tc16_frag_ldm_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_frag,
    const __half* __restrict__ group_scales,
    float* __restrict__ partial,
    __half* __restrict__ y,
    int N, int K, int group_size, int parts)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;
    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int g = lane >> 2;
    const int tin = lane & 3;
    const int part = blockIdx.y;
    const int kt_begin = (K / KT) * part / parts;
    const int kt_end = (K / KT) * (part + 1) / parts;

    __shared__ __align__(32) __half sA[16 * 64];

    float ca0 = 0, ca1 = 0, ca2 = 0, ca3 = 0;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;

    if (!active) {
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    const int nt = K / KT;
    auto load_frag = [&](int kt) -> uint4 {
        const long long off =
            (((long long)tile * nt + kt) * 32 + lane) * 16;
        return *reinterpret_cast<const uint4*>(w_frag + off);
    };

    uint4 raw = load_frag(kt_begin);
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int k0 = kt * KT;
        const uint4 next = (kt + 1 < kt_end) ? load_frag(kt + 1)
                                             : make_uint4(0, 0, 0, 0);

        // Store the 16x64 A tile as sixteen conflict-free 8x8 blocks.
        for (int t = threadIdx.x; t < 128; t += THREADS) {
            const int b = t >> 3;
            const int row = t & 7;
            const int ki = b >> 2;
            const int rem = b & 3;
            const int rr = (rem >> 1) * 8 + row;
            const int cc = ki * 16 + (rem & 1) * 8;
            const uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)rr * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[b * 64 + row * 8]) = v;
        }
        __syncthreads();

        const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        const int gi = k0 / group_size;
        const float se = __half2float(
            group_scales[(long long)(n0 + g) * n_groups + gi]);
        const float so = __half2float(
            group_scales[(long long)(n0 + g + 8) * n_groups + gi]);
        const __half2 s_lo = __floats2half2_rn(se, se);
        const __half2 s_hi = __floats2half2_rn(so, so);

        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int kk = ki * 16;
            const uint16_t wa0 = (uint16_t)(w[ki] & 0xFFFFu);
            const uint16_t wa1 = (uint16_t)(w[ki] >> 16);
            uint32_t b00, b01, b10, b11;
            tc_frag_b(wa0, s_lo, b00, b01);
            tc_frag_b(wa1, s_hi, b10, b11);
            const int grp = lane >> 3;
            const int blk = ki * 4 +
                (grp == 0 ? 0 : (grp == 1 ? 2 : (grp == 2 ? 1 : 3)));
            const __half* ap = sA + blk * 64 + (lane & 7) * 8;
            uint32_t a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0,%1,%2,%3}, [%4];"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"((unsigned)__cvta_generic_to_shared(ap)));
            tc_mma_m16n8k16(ca0, ca1, ca2, ca3, a0, a1, a2, a3, b00, b01);
            tc_mma_m16n8k16(cb0, cb1, cb2, cb3, a0, a1, a2, a3, b10, b11);
        }
        __syncthreads();
        raw = next;
    }

    float* dst = (parts == 1) ? nullptr : partial + (long long)part * 16 * N;
    if (parts == 1) {
        y[(long long)g * N + n0 + tin * 2] = __float2half(ca0);
        y[(long long)g * N + n0 + tin * 2 + 1] = __float2half(ca1);
        y[(long long)(g + 8) * N + n0 + tin * 2] = __float2half(ca2);
        y[(long long)(g + 8) * N + n0 + tin * 2 + 1] = __float2half(ca3);
        y[(long long)g * N + n0 + 8 + tin * 2] = __float2half(cb0);
        y[(long long)g * N + n0 + 8 + tin * 2 + 1] = __float2half(cb1);
        y[(long long)(g + 8) * N + n0 + 8 + tin * 2] = __float2half(cb2);
        y[(long long)(g + 8) * N + n0 + 8 + tin * 2 + 1] = __float2half(cb3);
    } else {
        dst[(long long)g * N + n0 + tin * 2] = ca0;
        dst[(long long)g * N + n0 + tin * 2 + 1] = ca1;
        dst[(long long)(g + 8) * N + n0 + tin * 2] = ca2;
        dst[(long long)(g + 8) * N + n0 + tin * 2 + 1] = ca3;
        dst[(long long)g * N + n0 + 8 + tin * 2] = cb0;
        dst[(long long)g * N + n0 + 8 + tin * 2 + 1] = cb1;
        dst[(long long)(g + 8) * N + n0 + 8 + tin * 2] = cb2;
        dst[(long long)(g + 8) * N + n0 + 8 + tin * 2 + 1] = cb3;
    }
}

// B=16 multi-tile pipe: each warp owns NTILE adjacent 16-column tiles and
// reuses one block-wide shared A tile across all of them.  This cuts the
// duplicated A traffic for the very wide lm_head shape and keeps NTILE
// independent packed-weight loads in flight per warp.
template<int WARPS, int NTILE>
__global__ __launch_bounds__(WARPS * 32, 4)
void int4_dequant_matmul_tc16_pipe_mt_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    float* __restrict__ tmp,
    int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int KT = 64;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int block_tile = blockIdx.x * (WARPS * NTILE);
    const int tile0 = block_tile + warp * NTILE;
    const int n_tiles = N / NT;
    const bool active = tile0 < n_tiles;
    const int n_groups = K / group_size;
    const int lp = lane >> 2;
    const int c = (lane & 3) * 16;

    __shared__ __align__(32) __half sA[M * LD];
    __shared__ __align__(32) __half sB[WARPS][NT * LD];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc[NTILE];
    #pragma unroll
    for (int t = 0; t < NTILE; ++t) {
        nvcuda::wmma::fill_fragment(acc[t], 0.0f);
    }

    if (!active) {
        for (int k0 = 0; k0 < K; k0 += KT) {
            #pragma unroll
            for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
                const int r = chunk / (KT / 8);
                const int cc = (chunk - r * (KT / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < 16) {
                    v = *reinterpret_cast<const uint4*>(
                        x + (long long)r * K + k0 + cc);
                }
                *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    uint4 raw[NTILE];
    uint4 next[NTILE];
    #pragma unroll
    for (int t = 0; t < NTILE; ++t) {
        const int n0 = (tile0 + t) * NT;
        raw[t] = (n0 < N) ? tc_load_b16(w_packed, n0 >> 1, lp, c, 0, K)
                          : make_uint4(0, 0, 0, 0);
    }
    for (int k0 = 0; k0 < K; k0 += KT) {
        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int cc = (chunk - r * (KT / 8)) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(
                x + (long long)r * K + k0 + cc);
            *reinterpret_cast<uint4*>(&sA[r * LD + cc]) = v;
        }
        #pragma unroll
        for (int t = 0; t < NTILE; ++t) {
            const int n0 = (tile0 + t) * NT;
            next[t] = (n0 < N) ? tc_load_b16(w_packed, n0 >> 1, lp, c, k0 + KT, K)
                               : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int t = 0; t < NTILE; ++t) {
            const int n0 = (tile0 + t) * NT;
            const int p0 = n0 >> 1;
            const int g = (k0 + c) / group_size;
            const float se = __half2float(
                group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
            const float so = __half2float(
                group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw[t]);
            tc_dequant16_h2_fast(
                bytes, &sB[warp][(2 * lp) * LD + c],
                &sB[warp][(2 * lp + 1) * LD + c], se, so);
            if (t == 0) __syncthreads();
            __syncwarp();
            #pragma unroll
            for (int kk = 0; kk < KT; kk += 16) {
                nvcuda::wmma::load_matrix_sync(a_frag, sA + kk, LD);
                nvcuda::wmma::load_matrix_sync(b_frag, sB[warp] + kk, LD);
                nvcuda::wmma::mma_sync(acc[t], a_frag, b_frag, acc[t]);
            }
            __syncwarp();
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < NTILE; ++t) raw[t] = next[t];
    }

    #pragma unroll
    for (int t = 0; t < NTILE; ++t) {
        const int n0 = (tile0 + t) * NT;
        if (n0 < N) {
            nvcuda::wmma::store_matrix_sync(
                tmp + n0, acc[t], N, nvcuda::wmma::mem_row_major);
        }
    }
}

// Tensor-core batch GEMV/GEMM for decode rows.  The batch rows are padded to
// the WMMA M tile (16).  A is staged once per block and shared by all four
// output-tile warps; each warp dequantizes its own 16xKT B tile.  A 128-wide
// K tile amortizes the shared-memory round trip over eight MMAs.
template<int WARPS, int KT>
__global__ void int4_dequant_matmul_tc_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    __half* __restrict__ y,
    int B, int N, int K, int group_size)
{
    constexpr int M = 16;
    constexpr int NT = 16;
    constexpr int LD = KT + 8;
    constexpr int THREADS = WARPS * 32;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * WARPS + warp;
    const int n_tiles = N / NT;
    const bool active = tile < n_tiles;

    const int n0 = active ? tile * NT : 0;
    const int n_groups = K / group_size;
    const int p0 = n0 >> 1;

    __shared__ __align__(32) __half sA[M * LD];
    __shared__ __align__(32) __half sB[WARPS][NT * LD];
    __shared__ __align__(32) float sC[WARPS][M * NT];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, NT, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, NT, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, NT, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += KT) {
        // A: one shared copy per block, 16 rows x KT columns.
        #pragma unroll
        for (int chunk = threadIdx.x; chunk < M * (KT / 8); chunk += THREADS) {
            const int r = chunk / (KT / 8);
            const int c = (chunk - r * (KT / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < B) {
                v = *reinterpret_cast<const uint4*>(
                    x + (long long)r * K + k0 + c);
            }
            *reinterpret_cast<uint4*>(&sA[r * LD + c]) = v;
        }

        // B: eight packed source rows per output tile.
        if (active) {
            #pragma unroll
            for (int chunk = lane; chunk < 8 * (KT / 16); chunk += 32) {
                const int lp = chunk / (KT / 16);
                const int c = (chunk - lp * (KT / 16)) * 16;
                const int g = (k0 + c) / group_size;
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    w_packed + (long long)(p0 + lp) * K + k0 + c);
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
                const float se = __half2float(
                    group_scales[(long long)(n0 + 2 * lp) * n_groups + g]);
                const float so = __half2float(
                    group_scales[(long long)(n0 + 2 * lp + 1) * n_groups + g]);
                tc_dequant16_h2_fast(
                    bytes, &sB[warp][(2 * lp) * LD + c],
                    &sB[warp][(2 * lp + 1) * LD + c], se, so
                );
            }
        }

        __syncthreads();
        if (active) {
            #pragma unroll
            for (int kk = 0; kk < KT; kk += 16) {
                nvcuda::wmma::load_matrix_sync(a_frag, sA + kk, LD);
                nvcuda::wmma::load_matrix_sync(
                    b_frag, sB[warp] + kk, LD
                );
                nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }
        __syncthreads();
    }

    if (!active) return;
    nvcuda::wmma::store_matrix_sync(sC[warp], acc, NT, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < B * NT; idx += 32) {
        const int r = idx / NT;
        const int c = idx - r * NT;
        y[(long long)r * N + n0 + c] = __float2half(sC[warp][idx]);
    }
}

struct Int4LaunchShape {
    torch::Tensor x_2d;
    torch::Tensor y;
    int64_t batch_elems;
    int N;
    int K;
    int group_size;
};

Int4LaunchShape check_int4_inputs(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& group_scales,
    int group_size)
{
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && group_scales.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() >= 1, "x must have rank >= 1");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must have rank 2");
    TORCH_CHECK(group_scales.dim() == 2, "group_scales must have rank 2");
    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
                "x must be fp16/bf16");
    TORCH_CHECK(w_packed.dtype() == torch::kUInt8, "w_packed must be uint8");
    TORCH_CHECK(group_scales.dtype() == torch::kFloat16,
                "Gemma4 INT4 group scales must be float16");
    TORCH_CHECK(group_size == kKTile, "INT4 CUDA kernel requires group_size=64");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() &&
                    group_scales.is_contiguous(),
                "INT4 CUDA inputs must be contiguous");

    const int64_t K64 = x.size(-1);
    TORCH_CHECK(w_packed.size(0) <= INT_MAX / 2,
                "packed N exceeds int32 range");
    const int64_t N64 = w_packed.size(0) * 2;
    int64_t batch_elems = 1;
    for (int64_t d = 0; d + 1 < x.dim(); ++d) {
        TORCH_CHECK(x.size(d) >= 0, "x dimensions must be non-negative");
        TORCH_CHECK(x.size(d) <= INT_MAX, "x dimension exceeds int32 range");
        batch_elems *= x.size(d);
        TORCH_CHECK(batch_elems <= INT_MAX, "flattened batch exceeds int32 range");
    }
    TORCH_CHECK(K64 > 0 && K64 <= INT_MAX, "x K dimension must be positive and fit int32");
    TORCH_CHECK(N64 > 0 && N64 <= INT_MAX, "unpacked N must be positive and fit int32");
    TORCH_CHECK(w_packed.size(1) == K64, "w_packed K dim must match x last dim");
    TORCH_CHECK(K64 % 2 == 0, "K must be even for INT4 packing");
    TORCH_CHECK(K64 % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_scales.size(0) == N64, "group_scales N must match unpacked N");
    TORCH_CHECK(group_scales.size(1) == K64 / group_size, "group_scales K dim mismatch");
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(w_packed.device() == x.device() &&
                    group_scales.device() == x.device(),
                "INT4 CUDA inputs must be on the same device");
    TORCH_CHECK(batch_elems <= 65535, "INT4 CUDA batch grid.y exceeds CUDA limit");

    auto x_2d = x.reshape({batch_elems, K64});
    auto y = torch::empty({batch_elems, N64}, x.options());
    return {x_2d, y, batch_elems, (int)N64, (int)K64, group_size};
}

template<int Mode>
torch::Tensor int4_dequant_matmul_impl(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size, int pairs_per_block = kWarpPairsPerBlock)
{
    static_assert(Mode >= 0 && Mode <= 3, "unknown INT4 GEMV mode");
    TORCH_CHECK(pairs_per_block > 0 && pairs_per_block <= 16,
                "pairs_per_block must be in [1, 16]");
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    auto orig_shape = x.sizes();
    if (shape.batch_elems != 0) {
        const int rows_per_block = Mode == 0
            ? kLegacyRowsPerBlock
            : (Mode == 1 ? kWarpRowsPerBlock : pairs_per_block * 2);
        const dim3 grid(
            (unsigned)(((int64_t)shape.N + rows_per_block - 1) / rows_per_block),
            (unsigned)shape.batch_elems);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (x.dtype() == torch::kFloat16) {
            if constexpr (Mode == 0) {
                int4_dequant_matmul_legacy_kernel<__half><<<grid, kLegacyThreads, 0, stream>>>(
                    (const __half*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__half*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            } else if constexpr (Mode == 1) {
                int4_dequant_matmul_warp_kernel<__half><<<grid, kWarpThreads, 0, stream>>>(
                    (const __half*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__half*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            } else {
                if constexpr (Mode == 3) {
                    int4_dequant_matmul_pair_vec_kernel<__half><<<grid, pairs_per_block * 32, 0, stream>>>(
                        (const __half*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                        (const __half*)group_scales.data_ptr(), (__half*)shape.y.data_ptr(),
                        (int)shape.batch_elems, shape.N, shape.K, shape.group_size, pairs_per_block);
                } else {
                    int4_dequant_matmul_pair_kernel<__half><<<grid, pairs_per_block * 32, 0, stream>>>(
                        (const __half*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                        (const __half*)group_scales.data_ptr(), (__half*)shape.y.data_ptr(),
                        (int)shape.batch_elems, shape.N, shape.K, shape.group_size, pairs_per_block);
                }
            }
        } else {
            if constexpr (Mode == 0) {
                int4_dequant_matmul_legacy_kernel<__nv_bfloat16><<<grid, kLegacyThreads, 0, stream>>>(
                    (const __nv_bfloat16*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__nv_bfloat16*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            } else if constexpr (Mode == 1) {
                int4_dequant_matmul_warp_kernel<__nv_bfloat16><<<grid, kWarpThreads, 0, stream>>>(
                    (const __nv_bfloat16*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__nv_bfloat16*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            } else {
                if constexpr (Mode == 3) {
                    int4_dequant_matmul_pair_vec_kernel<__nv_bfloat16><<<grid, pairs_per_block * 32, 0, stream>>>(
                        (const __nv_bfloat16*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                        (const __half*)group_scales.data_ptr(), (__nv_bfloat16*)shape.y.data_ptr(),
                        (int)shape.batch_elems, shape.N, shape.K, shape.group_size, pairs_per_block);
                } else {
                    int4_dequant_matmul_pair_kernel<__nv_bfloat16><<<grid, pairs_per_block * 32, 0, stream>>>(
                        (const __nv_bfloat16*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                        (const __half*)group_scales.data_ptr(), (__nv_bfloat16*)shape.y.data_ptr(),
                        (int)shape.batch_elems, shape.N, shape.K, shape.group_size, pairs_per_block);
                }
            }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (orig_shape.size() != 2) {
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_shape.push_back(shape.N);
        shape.y = shape.y.reshape(out_shape);
    }
    return shape.y;
}

torch::Tensor int4_dequant_matmul(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales, int group_size)
{
    return int4_dequant_matmul_impl<1>(x, w_packed, group_scales, group_size);
}

torch::Tensor int4_dequant_matmul_legacy(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales, int group_size)
{
    return int4_dequant_matmul_impl<0>(x, w_packed, group_scales, group_size);
}

torch::Tensor int4_dequant_matmul_pair(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales, int group_size)
{
    return int4_dequant_matmul_impl<2>(
        x, w_packed, group_scales, group_size, kWarpPairsPerBlock);
}

torch::Tensor int4_dequant_matmul_pair_tuned(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size, int pairs_per_block)
{
    return int4_dequant_matmul_impl<2>(
        x, w_packed, group_scales, group_size, pairs_per_block);
}

torch::Tensor int4_dequant_matmul_pair_vec(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size)
{
    // The vectorized kernel needs aligned 4-byte loads. K is always a
    // multiple of 4 for Gemma4 (K % 64 == 0), but keep the scalar pair
    // kernel as a correct fallback for arbitrary test shapes.
    TORCH_CHECK(x.size(-1) % 4 == 0,
                "pair_vec requires K divisible by 4");
    return int4_dequant_matmul_impl<3>(
        x, w_packed, group_scales, group_size, kWarpPairsPerBlock);
}

template<typename T>
__global__ void int4_dequant_dense_kernel(
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ out,
    long long total, int N, int K, int group_size)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int pair = (int)(idx / K);
    const int k = (int)(idx - (long long)pair * K);
    const int g = k / group_size;
    const int n_groups = K / group_size;
    const uint8_t byte = w_packed[idx];
    const float s0 = _to_float(group_scales[(long long)(pair * 2) * n_groups + g]);
    const float s1 = _to_float(group_scales[(long long)(pair * 2 + 1) * n_groups + g]);
    const float low = (float)((int)(byte & 0x0F) - 8);
    const float high = (float)((int)(byte >> 4) - 8);
    out[(long long)(pair * 2) * K + k] = _from_float<T>(low * s0);
    out[(long long)(pair * 2 + 1) * K + k] = _from_float<T>(high * s1);
}

torch::Tensor int4_dequant_dense(
    torch::Tensor w_packed, torch::Tensor group_scales,
    int64_t group_size, bool bf16)
{
    TORCH_CHECK(w_packed.is_cuda() && group_scales.is_cuda(),
                "dequant inputs must be CUDA");
    TORCH_CHECK(w_packed.dim() == 2 && group_scales.dim() == 2,
                "dequant inputs must be 2D");
    TORCH_CHECK(w_packed.dtype() == torch::kUInt8,
                "packed weights must be uint8");
    TORCH_CHECK(group_scales.dtype() == torch::kFloat16,
                "scales must be float16");
    TORCH_CHECK(w_packed.is_contiguous() && group_scales.is_contiguous(),
                "dequant inputs must be contiguous");
    const int K = (int)w_packed.size(1);
    const int N = (int)w_packed.size(0) * 2;
    TORCH_CHECK(group_size > 0 && K % group_size == 0,
                "K must be divisible by group_size");
    TORCH_CHECK(group_scales.size(0) == N &&
                    group_scales.size(1) == K / group_size,
                "scale shape mismatch");
    const long long total = (long long)w_packed.size(0) * K;
    auto options = torch::TensorOptions()
        .dtype(bf16 ? torch::kBFloat16 : torch::kFloat16)
        .device(w_packed.device());
    auto out = torch::empty({N, K}, options);
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (bf16) {
        int4_dequant_dense_kernel<__nv_bfloat16><<<(unsigned)blocks, threads, 0, stream>>>(
            w_packed.data_ptr<uint8_t>(),
            (const __half*)group_scales.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), total, N, K, (int)group_size);
    } else {
        int4_dequant_dense_kernel<__half><<<(unsigned)blocks, threads, 0, stream>>>(
            w_packed.data_ptr<uint8_t>(),
            (const __half*)group_scales.data_ptr(),
            (__half*)out.data_ptr(), total, N, K, (int)group_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int4_dequant_matmul_pair_vec8(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size)
{
    TORCH_CHECK(x.size(-1) % 8 == 0,
                "pair_vec8 requires K divisible by 8");
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    if (shape.batch_elems == 0) return shape.y;
    TORCH_CHECK(shape.batch_elems == 1,
                "pair_vec8 supports a single activation row");
    constexpr int pairs_per_block = kWarpPairsPerBlock;
    const int n_pairs = shape.N >> 1;
    TORCH_CHECK(shape.N % 2 == 0, "pair_vec8 requires even N");
    const dim3 grid(
        (unsigned)((n_pairs + pairs_per_block - 1) / pairs_per_block));
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (x.dtype() == torch::kFloat16) {
        int4_dequant_matmul_pair_vec8_kernel<__half>
            <<<grid, pairs_per_block * 32, 0, stream>>>(
                (const __half*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__half*)shape.y.data_ptr(), shape.N, shape.K,
                shape.group_size, pairs_per_block);
    } else {
        int4_dequant_matmul_pair_vec8_kernel<__nv_bfloat16>
            <<<grid, pairs_per_block * 32, 0, stream>>>(
                (const __nv_bfloat16*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__nv_bfloat16*)shape.y.data_ptr(), shape.N, shape.K,
                shape.group_size, pairs_per_block);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.y;
}

torch::Tensor int4_dequant_matmul_pair_batch(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size)
{
    TORCH_CHECK(x.size(-1) % 8 == 0,
                "pair_batch requires K divisible by 8");
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    if (shape.batch_elems == 0) return shape.y;
    TORCH_CHECK(shape.batch_elems >= 2 && shape.batch_elems <= 4,
                "pair_batch supports 2..4 activation rows");
    constexpr int pairs_per_block = kWarpPairsPerBlock;
    const int n_pairs = shape.N >> 1;
    TORCH_CHECK(shape.N % 2 == 0, "pair_batch requires even N");
    const dim3 grid(
        (unsigned)((n_pairs + pairs_per_block - 1) / pairs_per_block));
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (x.dtype() == torch::kFloat16) {
        int4_dequant_matmul_pair_batch_kernel<__half, 4>
            <<<grid, pairs_per_block * 32, 0, stream>>>(
                (const __half*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__half*)shape.y.data_ptr(), (int)shape.batch_elems,
                shape.N, shape.K, shape.group_size, pairs_per_block);
    } else {
        int4_dequant_matmul_pair_batch_kernel<__nv_bfloat16, 4>
            <<<grid, pairs_per_block * 32, 0, stream>>>(
                (const __nv_bfloat16*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__nv_bfloat16*)shape.y.data_ptr(), (int)shape.batch_elems,
                shape.N, shape.K, shape.group_size, pairs_per_block);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.y;
}


torch::Tensor int4_dequant_matmul_pair_batch16(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size)
{
    TORCH_CHECK(x.size(-1) % 8 == 0,
                "pair_batch16 requires K divisible by 8");
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    if (shape.batch_elems == 0) return shape.y;
    TORCH_CHECK(shape.batch_elems >= 5 && shape.batch_elems <= 16,
                "pair_batch16 supports 5..16 activation rows");
    TORCH_CHECK(x.dtype() == torch::kFloat16,
                "pair_batch16 currently supports fp16");
    constexpr int pairs_per_block = kWarpPairsPerBlock;
    const int n_pairs = shape.N >> 1;
    TORCH_CHECK(shape.N % 2 == 0, "pair_batch16 requires even N");
    const dim3 grid(
        (unsigned)((n_pairs + pairs_per_block - 1) / pairs_per_block));
    const auto stream = at::cuda::getCurrentCUDAStream();
    int4_dequant_matmul_pair_batch_kernel<__half, 16>
        <<<grid, pairs_per_block * 32, 0, stream>>>(
            (const __half*)shape.x_2d.data_ptr(),
            w_packed.data_ptr<uint8_t>(),
            (const __half*)group_scales.data_ptr(),
            (__half*)shape.y.data_ptr(), (int)shape.batch_elems,
            shape.N, shape.K, shape.group_size, pairs_per_block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.y;
}

torch::Tensor int4_dequant_matmul_pair_batch_parallel(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size)
{
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    if (shape.batch_elems == 0) return shape.y;
    TORCH_CHECK(shape.batch_elems >= 1 && shape.batch_elems <= 4,
                "parallel pair_batch supports 1..4 activation rows");
    TORCH_CHECK(shape.N % (2 * kPairBatchPairs) == 0,
                "parallel pair_batch requires N divisible by 8");
    TORCH_CHECK(shape.K % 16 == 0,
                "parallel pair_batch requires K divisible by 16");
    const int n_pairs = shape.N >> 1;
    const int n_groups = shape.K / shape.group_size;
    const size_t smem_bytes =
        (size_t)kPairBatchPairs * shape.K * sizeof(uint8_t)
        + (size_t)2 * kPairBatchPairs * n_groups * sizeof(__half);
    const dim3 grid((unsigned)(n_pairs / kPairBatchPairs));
    const int threads = (int)shape.batch_elems * kPairBatchPairs * 32;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (x.dtype() == torch::kFloat16) {
        int4_dequant_matmul_pair_batch_parallel_kernel<__half>
            <<<grid, threads, smem_bytes, stream>>>(
                (const __half*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__half*)shape.y.data_ptr(), (int)shape.batch_elems,
                shape.N, shape.K, shape.group_size);
    } else {
        int4_dequant_matmul_pair_batch_parallel_kernel<__nv_bfloat16>
            <<<grid, threads, smem_bytes, stream>>>(
                (const __nv_bfloat16*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__nv_bfloat16*)shape.y.data_ptr(), (int)shape.batch_elems,
                shape.N, shape.K, shape.group_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.y;
}


__global__ void int4_prepack_tiles_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int P, int K)
{
    const int tile = blockIdx.x;
    const int kt = blockIdx.y;
    const int lp = threadIdx.x >> 2;
    const int c = (threadIdx.x & 3) * 16;
    const long long s_off =
        (long long)(tile * 8 + lp) * K + (long long)kt * 64 + c;
    const long long d_off =
        (((long long)tile * (K >> 6) + kt) * 8 + lp) * 64 + c;
    *reinterpret_cast<uint4*>(dst + d_off) =
        *reinterpret_cast<const uint4*>(src + s_off);
}


torch::Tensor int4_prepack_frag(torch::Tensor w_packed)
{
    TORCH_CHECK(w_packed.is_cuda() && w_packed.scalar_type() == torch::kUInt8,
                "frag prepack expects a CUDA uint8 tensor");
    TORCH_CHECK(w_packed.dim() == 2 && w_packed.is_contiguous(),
                "frag prepack expects [N/2, K] contiguous");
    const int P = (int)w_packed.size(0);
    const int K = (int)w_packed.size(1);
    const int N = P * 2;
    TORCH_CHECK(N % 16 == 0 && K % 64 == 0,
                "frag prepack requires N%16==0 and K%64==0");
    auto out = torch::empty(
        {N / 16, K / 64, 32, 16},
        torch::TensorOptions().dtype(torch::kUInt8).device(w_packed.device()));
    const dim3 grid((unsigned)(N / 16), (unsigned)(K / 64));
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_prepack_frag_kernel<<<grid, 32, 0, stream>>>(
        w_packed.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int4_dequant_matmul_tc_frag(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts)
{
    TORCH_CHECK(x.is_cuda() && w_frag.is_cuda() && group_scales.is_cuda(),
                "frag TC inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat16 &&
                    x.size(0) == 16,
                "frag TC requires a 16-row fp16 activation");
    TORCH_CHECK(w_frag.dim() == 4 && w_frag.size(2) == 32 &&
                    w_frag.size(3) == 16,
                "frag weights must be [n_tiles, K/64, 32, 16]");
    const int K = (int)x.size(1);
    const int N = (int)(w_frag.size(0) * 16);
    TORCH_CHECK((int)(w_frag.size(1) * 64) == K, "frag K mismatch");
    TORCH_CHECK(parts >= 1 && parts <= 16, "frag parts must be 1..8");
    TORCH_CHECK((K / 64) % parts == 0, "frag K tiles must divide parts");
    auto y = torch::empty({16, N}, x.options());
    const int chunk = 16 * N;
    auto partial = torch::empty(
        {parts, chunk},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const int n_tiles = N / 16;
    const dim3 grid((unsigned)((n_tiles + 3) / 4), (unsigned)parts);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_dequant_matmul_tc16_frag_kernel<4>
        <<<grid, 128, 0, stream>>>(
            (const __half*)x.data_ptr(),
            w_frag.data_ptr<uint8_t>(),
            (const __half*)group_scales.data_ptr(),
            partial.data_ptr<float>(),
            (__half*)y.data_ptr(), N, K, 64, (int)parts);
    if (parts > 1) {
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), (__half*)y.data_ptr(),
            chunk, chunk, (int)parts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}



torch::Tensor int4_dequant_matmul_tc_frag_warps(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts, int64_t warps)
{
    TORCH_CHECK(x.is_cuda() && w_frag.is_cuda() && group_scales.is_cuda(),
                "frag-warps inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat16 &&
                    x.size(0) == 16,
                "frag-warps requires a 16-row fp16 activation");
    TORCH_CHECK(w_frag.dim() == 4 && w_frag.size(2) == 32 &&
                    w_frag.size(3) == 16,
                "frag-warps weights must be [n_tiles, K/64, 32, 16]");
    TORCH_CHECK(parts >= 1 && parts <= 16, "frag-warps parts must be 1..16");
    TORCH_CHECK(warps == 4 || warps == 8 || warps == 16, "frag-warps supports 4, 8 or 16 warps");
    const int K = (int)x.size(1);
    const int N = (int)(w_frag.size(0) * 16);
    TORCH_CHECK((int)(w_frag.size(1) * 64) == K, "frag-warps K mismatch");
    TORCH_CHECK((K / 64) % parts == 0, "frag-warps K tiles must divide parts");
    auto y = torch::empty({16, N}, x.options());
    const int chunk = 16 * N;
    auto partial = torch::empty(
        {parts, chunk},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const int n_tiles = N / 16;
    const int w = (int)warps;
    const dim3 grid((unsigned)((n_tiles + w - 1) / w), (unsigned)parts);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (w == 16) {
        int4_dequant_matmul_tc16_frag_kernel<16>
            <<<grid, 512, 0, stream>>>(
                (const __half*)x.data_ptr(),
                w_frag.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                partial.data_ptr<float>(), (__half*)y.data_ptr(),
                N, K, 64, (int)parts);
    } else if (w == 8) {
        int4_dequant_matmul_tc16_frag_kernel<8>
            <<<grid, 256, 0, stream>>>(
                (const __half*)x.data_ptr(),
                w_frag.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                partial.data_ptr<float>(), (__half*)y.data_ptr(),
                N, K, 64, (int)parts);
    } else {
        int4_dequant_matmul_tc16_frag_kernel<4>
            <<<grid, 128, 0, stream>>>(
                (const __half*)x.data_ptr(),
                w_frag.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                partial.data_ptr<float>(), (__half*)y.data_ptr(),
                N, K, 64, (int)parts);
    }
    if (parts > 1) {
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), (__half*)y.data_ptr(),
            chunk, chunk, (int)parts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


torch::Tensor int4_dequant_matmul_tc_frag_ga(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts)
{
    TORCH_CHECK(x.is_cuda() && w_frag.is_cuda() && group_scales.is_cuda(),
                "frag-ga inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat16 &&
                    x.size(0) == 16,
                "frag-ga requires a 16-row fp16 activation");
    TORCH_CHECK(w_frag.dim() == 4 && w_frag.size(2) == 32 &&
                    w_frag.size(3) == 16,
                "frag-ga weights must be [n_tiles, K/64, 32, 16]");
    TORCH_CHECK(parts >= 1 && parts <= 16, "frag-ga parts must be 1..16");
    const int K = (int)x.size(1);
    const int N = (int)(w_frag.size(0) * 16);
    TORCH_CHECK((int)(w_frag.size(1) * 64) == K, "frag-ga K mismatch");
    TORCH_CHECK((K / 64) % parts == 0, "frag-ga K tiles must divide parts");
    auto y = torch::empty({16, N}, x.options());
    const int chunk = 16 * N;
    auto partial = torch::empty(
        {parts, chunk},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const int n_tiles = N / 16;
    const dim3 grid((unsigned)((n_tiles + 3) / 4), (unsigned)parts);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_dequant_matmul_tc16_frag_ga_kernel<4>
        <<<grid, 128, 0, stream>>>(
            (const __half*)x.data_ptr(),
            w_frag.data_ptr<uint8_t>(),
            (const __half*)group_scales.data_ptr(),
            partial.data_ptr<float>(),
            (__half*)y.data_ptr(), N, K, 64, (int)parts);
    if (parts > 1) {
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), (__half*)y.data_ptr(),
            chunk, chunk, (int)parts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


torch::Tensor int4_dequant_matmul_tc_frag_ldm(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts, int64_t warps)
{
    TORCH_CHECK(x.is_cuda() && w_frag.is_cuda() && group_scales.is_cuda(),
                "frag-ldm inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat16 &&
                    x.size(0) == 16,
                "frag-ldm requires a 16-row fp16 activation");
    TORCH_CHECK(w_frag.dim() == 4 && w_frag.size(2) == 32 &&
                    w_frag.size(3) == 16,
                "frag-ldm weights must be [n_tiles, K/64, 32, 16]");
    TORCH_CHECK(parts >= 1 && parts <= 16, "frag-ldm parts must be 1..16");
    TORCH_CHECK(warps == 4 || warps == 8, "frag-ldm supports 4 or 8 warps");
    const int K = (int)x.size(1);
    const int N = (int)(w_frag.size(0) * 16);
    TORCH_CHECK((int)(w_frag.size(1) * 64) == K, "frag-ldm K mismatch");
    TORCH_CHECK((K / 64) % parts == 0, "frag-ldm K tiles must divide parts");
    auto y = torch::empty({16, N}, x.options());
    const int chunk = 16 * N;
    auto partial = torch::empty(
        {parts, chunk},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const int n_tiles = N / 16;
    const int w = (int)warps;
    const dim3 grid((unsigned)((n_tiles + w - 1) / w), (unsigned)parts);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (w == 8) {
        int4_dequant_matmul_tc16_frag_ldm_kernel<8>
            <<<grid, 256, 0, stream>>>(
                (const __half*)x.data_ptr(),
                w_frag.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                partial.data_ptr<float>(), (__half*)y.data_ptr(),
                N, K, 64, (int)parts);
    } else {
        int4_dequant_matmul_tc16_frag_ldm_kernel<4>
            <<<grid, 128, 0, stream>>>(
                (const __half*)x.data_ptr(),
                w_frag.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                partial.data_ptr<float>(), (__half*)y.data_ptr(),
                N, K, 64, (int)parts);
    }
    if (parts > 1) {
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), (__half*)y.data_ptr(),
            chunk, chunk, (int)parts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor int4_prepack_tiles(torch::Tensor w_packed)
{
    TORCH_CHECK(w_packed.is_cuda() && w_packed.scalar_type() == torch::kUInt8,
                "prepack expects a CUDA uint8 tensor");
    TORCH_CHECK(w_packed.dim() == 2 && w_packed.is_contiguous(),
                "prepack expects [N/2, K] contiguous");
    const int P = (int)w_packed.size(0);
    const int K = (int)w_packed.size(1);
    const int N = P * 2;
    TORCH_CHECK(N % 16 == 0, "prepack requires N divisible by 16");
    TORCH_CHECK(K % 64 == 0, "prepack requires K divisible by 64");
    const int n_tiles = N / 16;
    auto out = torch::empty(
        {n_tiles, K / 64, 8, 64},
        torch::TensorOptions().dtype(torch::kUInt8).device(w_packed.device()));
    const dim3 grid((unsigned)n_tiles, (unsigned)(K / 64));
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_prepack_tiles_kernel<<<grid, 32, 0, stream>>>(
        w_packed.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), P, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int4_dequant_matmul_tc_tiled(
    torch::Tensor x, torch::Tensor w_tiled, torch::Tensor group_scales)
{
    TORCH_CHECK(x.is_cuda() && w_tiled.is_cuda() && group_scales.is_cuda(),
                "tiled TC inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat16,
                "tiled TC requires a 2-D fp16 activation");
    TORCH_CHECK(w_tiled.dim() == 4 && w_tiled.size(2) == 8 &&
                    w_tiled.size(3) == 64,
                "tiled weights must be [n_tiles, K/64, 8, 64]");
    TORCH_CHECK(x.size(0) == 16, "tiled TC currently requires exactly 16 rows");
    const int K = (int)x.size(1);
    const int N = (int)(w_tiled.size(0) * 16);
    TORCH_CHECK((int)(w_tiled.size(1) * 64) == K, "tiled K mismatch");
    auto y = torch::empty({16, N}, x.options());
    const int n_tiles = N / 16;
    const dim3 grid((unsigned)((n_tiles + 3) / 4));
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_dequant_matmul_tc16_pipe_tiled_kernel<4>
        <<<grid, 128, 0, stream>>>(
            (const __half*)x.data_ptr(),
            w_tiled.data_ptr<uint8_t>(),
            (const __half*)group_scales.data_ptr(),
            (__half*)y.data_ptr(), N, K, 64);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor int4_dequant_matmul_tc(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size)
{
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    if (shape.batch_elems == 0) return shape.y;
    TORCH_CHECK(shape.batch_elems >= 2 && shape.batch_elems <= 16,
                "tensor-core batch path supports 2..16 activation rows");
    TORCH_CHECK(x.dtype() == torch::kFloat16,
                "tensor-core batch path currently requires fp16 activations");
    TORCH_CHECK(shape.N % 16 == 0, "tensor-core path requires N divisible by 16");
    TORCH_CHECK(shape.K % 16 == 0, "tensor-core path requires K divisible by 16");
    if (shape.batch_elems < 16) {
        const int B = (int)shape.batch_elems;
        auto x_pad = torch::zeros({16, shape.K}, x.options());
        x_pad.narrow(0, 0, B).copy_(shape.x_2d);
        auto y_pad = int4_dequant_matmul_tc(
            x_pad, w_packed, group_scales, group_size
        );
        shape.y.copy_(y_pad.narrow(0, 0, B));
        return shape.y;
    }
    constexpr int WARPS = 4;
    const int n_tiles = shape.N / 16;
    const dim3 grid((unsigned)((n_tiles + WARPS - 1) / WARPS));
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (shape.batch_elems == 16) {
        TORCH_CHECK(shape.K % 64 == 0,
                    "tensor-core path requires K divisible by 64");
        if (shape.N <= 1536 && shape.K >= 6144 && shape.K % 1024 == 0) {
            const int chunk = 16 * shape.N;
            constexpr int parts = 4;
            auto partial = torch::empty(
                {parts, chunk},
                torch::TensorOptions().dtype(torch::kFloat32).device(x.device())
            );
            if (shape.N % 32 == 0) {
                const dim3 sgrid((unsigned)(n_tiles / 2), parts);
                int4_dequant_matmul_tc16_split2_mt2_kernel<WARPS, 2>
                    <<<sgrid, WARPS * 32, 0, stream>>>(
                        (const __half*)shape.x_2d.data_ptr(),
                        w_packed.data_ptr<uint8_t>(),
                        (const __half*)group_scales.data_ptr(),
                        partial.data_ptr<float>(),
                        (__half*)shape.y.data_ptr(),
                        shape.N, shape.K, shape.group_size, parts);
            } else {
                const dim3 sgrid((unsigned)n_tiles, parts);
                int4_dequant_matmul_tc16_split2_kernel<WARPS, 64>
                    <<<sgrid, WARPS * 32, 0, stream>>>(
                        (const __half*)shape.x_2d.data_ptr(),
                        w_packed.data_ptr<uint8_t>(),
                        (const __half*)group_scales.data_ptr(),
                        partial.data_ptr<float>(),
                        shape.N, shape.K, shape.group_size, parts);
            }
            tc16_reduce_kernel<<<
                (chunk + 255) / 256, 256, 0, stream>>>(
                    partial.data_ptr<float>(),
                    (__half*)shape.y.data_ptr(), chunk, chunk, parts);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return shape.y;
        }
        if (shape.N <= 8192 && shape.K % 256 == 0) {
            const dim3 sgrid((unsigned)n_tiles);
            int4_dequant_matmul_tc16_splitk_kernel<WARPS, 64>
                <<<sgrid, WARPS * 32, 0, stream>>>(
                    (const __half*)shape.x_2d.data_ptr(),
                    w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(),
                    (__half*)shape.y.data_ptr(),
                    shape.N, shape.K, shape.group_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return shape.y;
        }
        int4_dequant_matmul_tc16_pipe_kernel<WARPS>
            <<<grid, WARPS * 32, 0, stream>>>(
                (const __half*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__half*)shape.y.data_ptr(),
                shape.N, shape.K, shape.group_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return shape.y;
    }
    if (shape.K % 128 == 0) {
        int4_dequant_matmul_tc_kernel<WARPS, 128>
            <<<grid, WARPS * 32, 0, stream>>>(
                (const __half*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__half*)shape.y.data_ptr(), (int)shape.batch_elems,
                shape.N, shape.K, shape.group_size);
    } else {
        TORCH_CHECK(shape.K % 64 == 0,
                    "tensor-core path requires K divisible by 64");
        int4_dequant_matmul_tc_kernel<WARPS, 64>
            <<<grid, WARPS * 32, 0, stream>>>(
                (const __half*)shape.x_2d.data_ptr(),
                w_packed.data_ptr<uint8_t>(),
                (const __half*)group_scales.data_ptr(),
                (__half*)shape.y.data_ptr(), (int)shape.batch_elems,
                shape.N, shape.K, shape.group_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.y;
}


// Experimental dispatch used by the shape tuner.  variant selects the
// split-K/split-2 warp count and partition count so the best configuration can
// be measured per linear shape before it becomes the default.
torch::Tensor int4_dequant_matmul_tc_variant(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size, int64_t variant)
{
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    TORCH_CHECK(shape.batch_elems == 16,
                "variant tuner requires exactly 16 activation rows");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "variant tuner requires fp16");
    TORCH_CHECK(shape.N % 16 == 0 && shape.K % 64 == 0,
                "variant tuner requires N%16==0 and K%64==0");
    const int n_tiles = shape.N / 16;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const __half* xp = (const __half*)shape.x_2d.data_ptr();
    const uint8_t* wp = w_packed.data_ptr<uint8_t>();
    const __half* sp = (const __half*)group_scales.data_ptr();
    __half* yp = (__half*)shape.y.data_ptr();

    auto launch_splitk128 = [&](int warps) {
        TORCH_CHECK(shape.K % (warps * 128) == 0,
                    "splitk128 K must divide warps*128");
        const dim3 sgrid((unsigned)n_tiles);
        int4_dequant_matmul_tc16_splitk_kernel<4, 128>
            <<<sgrid, 128, 0, stream>>>(
                xp, wp, sp, yp, shape.N, shape.K, shape.group_size);
    };
    auto launch_split2 = [&](int warps, int parts) {
        TORCH_CHECK(shape.K % (warps * parts * 64) == 0,
                    "split2 K must divide warps*parts*64");
        const int chunk = 16 * shape.N;
        auto partial = torch::empty(
            {parts, chunk},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        const dim3 sgrid((unsigned)n_tiles, parts);
        #define LAUNCH_SPLIT2(W, P)                                            \
            int4_dequant_matmul_tc16_split2_kernel<W, 64>                      \
                <<<sgrid, W * 32, 0, stream>>>(                                \
                    xp, wp, sp, partial.data_ptr<float>(),                     \
                    shape.N, shape.K, shape.group_size, P)
        if (warps == 4 && parts == 2) { LAUNCH_SPLIT2(4, 2); }
        else if (warps == 4 && parts == 3) { LAUNCH_SPLIT2(4, 3); }
        else if (warps == 4 && parts == 4) { LAUNCH_SPLIT2(4, 4); }
        else if (warps == 4 && parts == 6) { LAUNCH_SPLIT2(4, 6); }
        else if (warps == 4 && parts == 8) { LAUNCH_SPLIT2(4, 8); }
        else if (warps == 8 && parts == 2) { LAUNCH_SPLIT2(8, 2); }
        else if (warps == 8 && parts == 4) { LAUNCH_SPLIT2(8, 4); }
        else if (warps == 8 && parts == 8) { LAUNCH_SPLIT2(8, 8); }
        else { TORCH_CHECK(false, "unsupported split2 variant"); }
        #undef LAUNCH_SPLIT2
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), yp, chunk, chunk, parts);
    };
    auto launch_splitk = [&](int warps) {
        TORCH_CHECK(shape.K % (warps * 64) == 0,
                    "splitk K must divide warps*64");
        const dim3 sgrid((unsigned)n_tiles);
        #define LAUNCH_SPLITK(W)                                               \
            int4_dequant_matmul_tc16_splitk_kernel<W, 64>                      \
                <<<sgrid, W * 32, 0, stream>>>(                                \
                    xp, wp, sp, yp, shape.N, shape.K, shape.group_size)
        if (warps == 8) { LAUNCH_SPLITK(8); }
        else if (warps == 12) { LAUNCH_SPLITK(12); }
        else { TORCH_CHECK(false, "unsupported splitk variant"); }
        #undef LAUNCH_SPLITK
    };
    auto launch_split2_mt4 = [&](int parts) {
        TORCH_CHECK(shape.N % 64 == 0, "split2-mt4 requires N%64==0");
        TORCH_CHECK(shape.K % (4 * parts * 64) == 0,
                    "split2-mt4 K must divide 4*parts*64");
        const int chunk = 16 * shape.N;
        auto partial = torch::empty(
            {parts, chunk},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        const dim3 sgrid((unsigned)(n_tiles / 4), parts);
        int4_dequant_matmul_tc16_split2_mt2_kernel<4, 4>
            <<<sgrid, 128, 0, stream>>>(
                xp, wp, sp, partial.data_ptr<float>(), yp,
                shape.N, shape.K, shape.group_size, parts);
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), yp, chunk, chunk, parts);
    };
    auto launch_split2_mt2 = [&](int parts, int warps) {
        TORCH_CHECK(shape.N % 32 == 0, "split2-mt2 requires N%32==0");
        TORCH_CHECK(shape.K % (4 * parts * 64) == 0,
                    "split2-mt2 K must divide 4*parts*64");
        const int chunk = 16 * shape.N;
        auto partial = torch::empty(
            {(parts == 1 ? 0 : parts), chunk},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        const dim3 sgrid((unsigned)(n_tiles / 2), parts);
        if (warps == 4) {
            int4_dequant_matmul_tc16_split2_mt2_kernel<4, 2>
                <<<sgrid, 128, 0, stream>>>(
                    xp, wp, sp,
                    parts == 1 ? nullptr : partial.data_ptr<float>(), yp,
                    shape.N, shape.K, shape.group_size, parts);
        } else {
            int4_dequant_matmul_tc16_split2_mt2_kernel<8, 2>
                <<<sgrid, 256, 0, stream>>>(
                    xp, wp, sp,
                    parts == 1 ? nullptr : partial.data_ptr<float>(), yp,
                    shape.N, shape.K, shape.group_size, parts);
        }
        if (parts > 1) {
            tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
                partial.data_ptr<float>(), yp, chunk, chunk, parts);
        }
    };
    auto launch_split2_128 = [&](int parts) {
        TORCH_CHECK(shape.K % (4 * parts * 128) == 0,
                    "split2-128 K must divide 4*parts*128");
        const int chunk = 16 * shape.N;
        auto partial = torch::empty(
            {parts, chunk},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        const dim3 sgrid((unsigned)n_tiles, parts);
        int4_dequant_matmul_tc16_split2_kernel<4, 128>
            <<<sgrid, 128, 0, stream>>>(
                xp, wp, sp, partial.data_ptr<float>(),
                shape.N, shape.K, shape.group_size, parts);
        tc16_reduce_kernel<<<(chunk + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), yp, chunk, chunk, parts);
    };
    auto launch_pipe_mt = [&](int warps, int ntile) {
        auto tmp = torch::empty(
            {16, shape.N},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        const dim3 pgrid((unsigned)((n_tiles + warps * ntile - 1) / (warps * ntile)));
        #define LAUNCH_PIPEMT(W, T)                                            \
            int4_dequant_matmul_tc16_pipe_mt_kernel<W, T>                      \
                <<<pgrid, W * 32, 0, stream>>>(                                \
                    xp, wp, sp, tmp.data_ptr<float>(),                         \
                    shape.N, shape.K, shape.group_size)
        if (warps == 4 && ntile == 2) { LAUNCH_PIPEMT(4, 2); }
        else if (warps == 4 && ntile == 4) { LAUNCH_PIPEMT(4, 4); }
        else if (warps == 4 && ntile == 8) { LAUNCH_PIPEMT(4, 8); }
        else if (warps == 8 && ntile == 2) { LAUNCH_PIPEMT(8, 2); }
        else { TORCH_CHECK(false, "unsupported pipe mt variant"); }
        #undef LAUNCH_PIPEMT
        const int conv_n = 16 * shape.N;
        tc_f16_convert_kernel<<<(conv_n / 2 + 255) / 256, 256, 0, stream>>>(
            tmp.data_ptr<float>(), yp, conv_n);
    };
    auto launch_pipe_ca = [&](int stages) {
        const dim3 pgrid((unsigned)((n_tiles + 3) / 4));
        #define LAUNCH_PIPECA(S)                                               \
            int4_dequant_matmul_tc16_pipe_ca_kernel<4, S>                      \
                <<<pgrid, 128, 0, stream>>>(                                   \
                    xp, wp, sp, yp, shape.N, shape.K, shape.group_size)
        if (stages == 2) { LAUNCH_PIPECA(2); }
        else if (stages == 3) { LAUNCH_PIPECA(3); }
        else if (stages == 4) { LAUNCH_PIPECA(4); }
        else { TORCH_CHECK(false, "unsupported cp.async pipe stages"); }
        #undef LAUNCH_PIPECA
    };
    auto launch_pipe_d = [&](int warps, int depth) {
        auto tmp = torch::empty(
            {16, shape.N},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        const dim3 pgrid((unsigned)((n_tiles + warps - 1) / warps));
        #define LAUNCH_PIPED(W, D)                                             \
            int4_dequant_matmul_tc16_pipe_d_kernel<W, D>                       \
                <<<pgrid, W * 32, 0, stream>>>(                                \
                    xp, wp, sp, tmp.data_ptr<float>(),                         \
                    shape.N, shape.K, shape.group_size)
        if (warps == 4 && depth == 4) { LAUNCH_PIPED(4, 4); }
        else if (warps == 4 && depth == 6) { LAUNCH_PIPED(4, 6); }
        else if (warps == 8 && depth == 4) { LAUNCH_PIPED(8, 4); }
        else { TORCH_CHECK(false, "unsupported pipe depth"); }
        #undef LAUNCH_PIPED
        const int conv_n = 16 * shape.N;
        tc_f16_convert_kernel<<<(conv_n / 2 + 255) / 256, 256, 0, stream>>>(
            tmp.data_ptr<float>(), yp, conv_n);
    };
    auto launch_reg = [&](int warps) {
        const dim3 rgrid((unsigned)((n_tiles + warps - 1) / warps));
        #define LAUNCH_REG(W)                                                  \
            int4_dequant_matmul_tc16_reg_kernel<W>                             \
                <<<rgrid, W * 32, 0, stream>>>(                                \
                    xp, wp, sp, yp, shape.N, shape.K, shape.group_size)
        if (warps == 4) { LAUNCH_REG(4); }
        else if (warps == 8) { LAUNCH_REG(8); }
        else { TORCH_CHECK(false, "unsupported reg variant"); }
        #undef LAUNCH_REG
    };
    auto launch_pipe = [&](int warps) {
        const dim3 pgrid((unsigned)((n_tiles + warps - 1) / warps));
        #define LAUNCH_PIPE(W)                                                 \
            int4_dequant_matmul_tc16_pipe_kernel<W>                            \
                <<<pgrid, W * 32, 0, stream>>>(                                \
                    xp, wp, sp, yp,                                            \
                    shape.N, shape.K, shape.group_size)
        if (warps == 8) { LAUNCH_PIPE(8); }
        else if (warps == 2) { LAUNCH_PIPE(2); }
        else { TORCH_CHECK(false, "unsupported pipe variant"); }
        #undef LAUNCH_PIPE
    };

    switch (variant) {
        case 1: launch_splitk(8); break;
        case 3: launch_splitk(12); break;
        case 4: launch_split2(8, 2); break;
        case 5: launch_split2(8, 4); break;
        case 6: launch_split2(8, 8); break;
        case 7: launch_split2(4, 8); break;
        case 10: launch_pipe(8); break;
        case 12: launch_reg(4); break;
        case 13: launch_reg(8); break;
        case 14: launch_pipe_d(4, 4); break;
        case 15: launch_pipe_d(4, 6); break;
        case 16: launch_pipe_d(8, 4); break;
        case 17: launch_split2(4, 2); break;
        case 18: launch_split2(4, 3); break;
        case 19: launch_split2(4, 4); break;
        case 20: launch_split2(4, 6); break;
        case 21: launch_split2(4, 8); break;
        case 22: launch_pipe_mt(4, 2); break;
        case 23: launch_pipe_mt(4, 4); break;
        case 24: launch_pipe_mt(4, 8); break;
        case 25: launch_pipe_mt(8, 2); break;
        case 26: launch_splitk128(4); break;
        case 27: launch_split2_128(2); break;
        case 28: launch_split2_128(4); break;
        case 29: launch_split2_128(8); break;
        case 30: launch_split2_mt2(4, 4); break;
        case 31: launch_split2_mt2(2, 4); break;
        case 32: launch_split2_mt2(8, 4); break;
        case 33: launch_split2_mt2(2, 8); break;
        case 34: launch_split2_mt2(4, 8); break;
        case 35: launch_split2_mt4(4); break;
        case 36: launch_split2_mt4(2); break;
        case 37: launch_split2_mt4(1); break;
        case 40: launch_pipe_ca(2); break;
        case 41: launch_pipe_ca(3); break;
        case 42: launch_pipe_ca(4); break;
        case 45: launch_split2_mt2(1, 4); break;
        case 46: launch_split2_mt2(1, 8); break;
        case 47: launch_pipe(2); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.y;
}


std::vector<int64_t> tc_kernel_attributes(int64_t which)
{
    const void* func = nullptr;
    int threads = 128;
    if (which == 0) {
        func = (const void*)int4_dequant_matmul_tc16_pipe_kernel<4>;
    } else if (which == 1) {
        func = (const void*)int4_dequant_matmul_tc16_splitk_kernel<4, 64>;
    } else if (which == 2) {
        func = (const void*)int4_dequant_matmul_tc16_split2_mt2_kernel<4, 2>;
    } else if (which == 3) {
        func = (const void*)int4_dequant_matmul_tc16_frag_kernel<4>;
    } else if (which == 4) {
        func = (const void*)int4_dequant_matmul_tc16_split2_kernel<4, 64>;
    } else {
        TORCH_CHECK(false, "unknown tc kernel");
    }
    cudaFuncAttributes attr{};
    cudaError_t status = cudaFuncGetAttributes(&attr, func);
    TORCH_CHECK(status == cudaSuccess, "cudaFuncGetAttributes failed");
    int blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, func, threads, 0);
    TORCH_CHECK(status == cudaSuccess, "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed");
    return {(int64_t)attr.numRegs, (int64_t)attr.sharedSizeBytes,
            (int64_t)attr.localSizeBytes, (int64_t)blocks,
            (int64_t)blocks * threads};
}

std::vector<int64_t> int4_pair_kernel_attributes(
    int pairs_per_block, bool fp16, bool vectorized)
{
    TORCH_CHECK(pairs_per_block > 0 && pairs_per_block <= 16,
                "pairs_per_block must be in [1, 16]");
    cudaFuncAttributes attr{};
    const void* func = nullptr;
    if (fp16) {
        func = vectorized
            ? (const void*)int4_dequant_matmul_pair_vec_kernel<__half>
            : (const void*)int4_dequant_matmul_pair_kernel<__half>;
    } else {
        func = vectorized
            ? (const void*)int4_dequant_matmul_pair_vec_kernel<__nv_bfloat16>
            : (const void*)int4_dequant_matmul_pair_kernel<__nv_bfloat16>;
    }
    cudaError_t status = cudaFuncGetAttributes(&attr, func);
    TORCH_CHECK(status == cudaSuccess, "cudaFuncGetAttributes failed: ",
                cudaGetErrorString(status));
    int blocks_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, func, pairs_per_block * 32, 0);
    TORCH_CHECK(status == cudaSuccess,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: ",
                cudaGetErrorString(status));
    return {
        (int64_t)attr.numRegs,
        (int64_t)attr.sharedSizeBytes,
        (int64_t)attr.localSizeBytes,
        (int64_t)attr.maxThreadsPerBlock,
        (int64_t)blocks_per_sm,
        (int64_t)blocks_per_sm * pairs_per_block * 32,
        (int64_t)attr.binaryVersion,
        (int64_t)attr.ptxVersion,
    };
}
