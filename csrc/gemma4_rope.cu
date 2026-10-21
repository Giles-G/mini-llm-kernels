// Fused Gemma4 RoPE for FP16 decode/prefill.
//
// The eager path uses three kernels per q/k tensor (mul + two addcmul) and
// rounds each step to FP16.  This kernel performs the same two rounding steps
// in one launch and reads both halves of the rotary pair before writing.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// Accurate libdevice tanh; the fast-math flag would otherwise emit MUFU.TANH,
// which changes ~0.6% of FP16 GELU values by one ulp versus PyTorch.
extern "C" __device__ float __nv_tanhf(float);

__global__ void gemma4_rope_fused_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    __half* __restrict__ y,
    int pairs_per_row, int rows, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * heads * pairs_per_row;
    if (idx >= total) return;

    const int pair = idx % pairs_per_row;
    const int head = (idx / pairs_per_row) % heads;
    const int t = idx / (pairs_per_row * heads);
    const int base = (t * heads + head) * dim;

    const __half x0 = x[base + pair];
    const __half x1 = x[base + pair + pairs_per_row];
    const __half c0 = cos[t * dim + pair];
    const __half c1 = cos[t * dim + pair + pairs_per_row];
    const __half s0 = sin[t * dim + pair];
    const __half s1 = sin[t * dim + pair + pairs_per_row];

    // Match the eager path exactly: out = round(x*cos), then
    // out = round(out + (-/+other)*sin).
    const float low0 = __half2float(x0) * __half2float(c0);
    __half low = __float2half(low0);
    low = __float2half(
        __half2float(low) - __half2float(x1) * __half2float(s0));
    const float high0 = __half2float(x1) * __half2float(c1);
    __half high = __float2half(high0);
    high = __float2half(
        __half2float(high) + __half2float(x0) * __half2float(s1));

    y[base + pair] = low;
    y[base + pair + pairs_per_row] = high;
}

torch::Tensor gemma4_rope_fused(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin)
{
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(),
                "rope inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
                    cos.scalar_type() == torch::kFloat16 &&
                    sin.scalar_type() == torch::kFloat16,
                "rope fused path requires fp16 tensors");
    TORCH_CHECK(x.dim() == 3, "rope x must be [rows, heads, dim]");
    TORCH_CHECK(x.is_contiguous() && cos.is_contiguous() && sin.is_contiguous(),
                "rope inputs must be contiguous");
    const int rows = (int)x.size(0);
    const int heads = (int)x.size(1);
    const int dim = (int)x.size(2);
    TORCH_CHECK(dim % 2 == 0, "rope dim must be even");
    const int pairs = dim / 2;
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == rows && cos.size(1) == dim,
                "rope cos must be [rows, dim]");
    TORCH_CHECK(sin.sizes() == cos.sizes(), "rope sin must match cos");
    auto y = torch::empty_like(x);
    const int total = rows * heads * pairs;
    if (total == 0) return y;
    constexpr int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    gemma4_rope_fused_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)x.data_ptr(),
        (const __half*)cos.data_ptr(),
        (const __half*)sin.data_ptr(),
        (__half*)y.data_ptr(),
        pairs, rows, heads, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Fused GELU(tanh) * up for the Gemma4 MLP.  GELU is rounded to FP16 before
// the multiply, matching the eager gelu + mul sequence exactly.
__device__ __forceinline__ float gelu_tanh_float(float x)
{
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    return 0.5f * x * (1.0f + __nv_tanhf(kAlpha * (x + kBeta * x * x * x)));
}

__global__ void gemma4_gelu_mul_strided_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int rows, int row_pairs, long long gate_stride, long long up_stride)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * row_pairs;
    if (i >= total) return;
    const int row = i / row_pairs;
    const int j = (i - row * row_pairs) * 2;
    const __half* gp = gate + (long long)row * gate_stride + j;
    const __half* up_p = up + (long long)row * up_stride + j;
    __half* op = out + (long long)row * (long long)(row_pairs * 2) + j;
    const float g0 = gelu_tanh_float(__half2float(gp[0]));
    const float g1 = gelu_tanh_float(__half2float(gp[1]));
    const __half2 h = __floats2half2_rn(g0, g1);
    const __half2 u = *reinterpret_cast<const __half2*>(up_p);
    *reinterpret_cast<__half2*>(op) = __hmul2(h, u);
}

torch::Tensor gemma4_gelu_mul(torch::Tensor gate, torch::Tensor up)
{
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "gelu_mul inputs must be CUDA");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat16 &&
                    up.scalar_type() == torch::kFloat16,
                "gelu_mul requires fp16 tensors");
    TORCH_CHECK(gate.dim() == 2 && up.dim() == 2,
                "gelu_mul requires 2-D tensors");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gelu_mul shapes must match");
    TORCH_CHECK(gate.stride(1) == 1 && up.stride(1) == 1,
                "gelu_mul requires contiguous rows");
    const int rows = (int)gate.size(0);
    const int n = (int)gate.size(1);
    TORCH_CHECK(n % 2 == 0, "gelu_mul requires an even row width");
    auto out = torch::empty({rows, n}, gate.options());
    const int row_pairs = n / 2;
    const int total = rows * row_pairs;
    if (total == 0) return out;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    gemma4_gelu_mul_strided_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)gate.data_ptr(),
        (const __half*)up.data_ptr(),
        (__half*)out.data_ptr(), rows, row_pairs,
        gate.stride(0), up.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// (a + b) * scalar in one launch.  The add and multiply each round to FP16,
// matching the eager residual.add_() + mul_() sequence bit for bit.
__global__ void gemma4_add_mul_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const __half* __restrict__ scalar,
    __half* __restrict__ out,
    int pairs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const int j = i * 2;
    const __half2 av = *reinterpret_cast<const __half2*>(a + j);
    const __half2 bv = *reinterpret_cast<const __half2*>(b + j);
    const __half s = scalar[0];
    const __half2 sv = __halves2half2(s, s);
    *reinterpret_cast<__half2*>(out + j) = __hmul2(__hadd2(av, bv), sv);
}

torch::Tensor gemma4_add_mul(
    torch::Tensor a, torch::Tensor b, torch::Tensor scalar)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && scalar.is_cuda(),
                "add_mul inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 &&
                    b.scalar_type() == torch::kFloat16 &&
                    scalar.scalar_type() == torch::kFloat16,
                "add_mul requires fp16 tensors");
    TORCH_CHECK(a.sizes() == b.sizes() && a.is_contiguous() &&
                    b.is_contiguous() && scalar.numel() == 1,
                "add_mul expects matching contiguous tensors");
    TORCH_CHECK(a.numel() % 2 == 0, "add_mul requires an even element count");
    auto out = torch::empty_like(a);
    const int pairs = (int)(a.numel() / 2);
    if (pairs == 0) return out;
    constexpr int threads = 256;
    const int blocks = (pairs + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    gemma4_add_mul_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)a.data_ptr(), (const __half*)b.data_ptr(),
        (const __half*)scalar.data_ptr(), (__half*)out.data_ptr(), pairs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Fused q/k/v RMSNorm (+ RoPE for q/k).  One block per (token, head):
// the block reduces the FP32 sum of squares, rounds the normalized value to
// FP16, then applies the same two-step RoPE rounding as the eager path.
__global__ void gemma4_qkv_norm_rope_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ qw,
    const __half* __restrict__ kw,
    const __half* __restrict__ vw,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    __half* __restrict__ qo,
    __half* __restrict__ ko,
    __half* __restrict__ vo,
    int rows, int hq, int hk, int dim, float eps)
{
    const int b = blockIdx.x;
    const int thq = rows * hq;
    const int thk = rows * hk;
    const int tid = threadIdx.x;
    const bool is_q = b < thq;
    const bool is_k = b >= thq && b < thq + thk;
    int t, head;
    const __half* src;
    const __half* weight;
    __half* dst;
    if (is_q) {
        t = b / hq; head = b - t * hq;
        src = q + (long long)(t * hq + head) * dim;
        weight = qw;
        dst = qo + (long long)(t * hq + head) * dim;
    } else if (is_k) {
        const int c = b - thq;
        t = c / hk; head = c - t * hk;
        src = k + (long long)(t * hk + head) * dim;
        weight = kw;
        dst = ko + (long long)(t * hk + head) * dim;
    } else {
        const int c = b - thq - thk;
        t = c / hk; head = c - t * hk;
        src = v + (long long)(t * hk + head) * dim;
        weight = vw;
        dst = vo + (long long)(t * hk + head) * dim;
    }

    __shared__ __align__(16) __half sh[512];
    __shared__ float red[32];

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (dim + 31) >> 5;

    float ss = 0.0f;
    if (tid < dim) {
        const float xv = __half2float(src[tid]);
        ss = xv * xv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < nwarps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0) red[0] = s;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)dim + eps);
    if (tid < dim) {
        const float hv = __half2float(src[tid]) * inv *
                         __half2float(weight[tid]);
        sh[tid] = __float2half(hv);
    }
    __syncthreads();

    if (is_q || is_k) {
        const int half = dim >> 1;
        if (tid < half) {
            const __half lo = sh[tid];
            const __half hi = sh[tid + half];
            const __half c0 = cos[(long long)t * dim + tid];
            const __half c1 = cos[(long long)t * dim + tid + half];
            const __half s0 = sin[(long long)t * dim + tid];
            const __half s1 = sin[(long long)t * dim + tid + half];
            __half out_lo = __float2half(__half2float(lo) * __half2float(c0));
            out_lo = __float2half(__half2float(out_lo) -
                                  __half2float(hi) * __half2float(s0));
            __half out_hi = __float2half(__half2float(hi) * __half2float(c1));
            out_hi = __float2half(__half2float(out_hi) +
                                  __half2float(lo) * __half2float(s1));
            dst[tid] = out_lo;
            dst[tid + half] = out_hi;
        }
    } else if (tid < dim) {
        dst[tid] = sh[tid];
    }
}

std::vector<torch::Tensor> gemma4_qkv_norm_rope(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qw, torch::Tensor kw, torch::Tensor vw,
    torch::Tensor cos, torch::Tensor sin, double eps)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() &&
                    qw.is_cuda() && kw.is_cuda() && vw.is_cuda() &&
                    cos.is_cuda() && sin.is_cuda(),
                "qkv norm/rope inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                    k.scalar_type() == torch::kFloat16 &&
                    v.scalar_type() == torch::kFloat16,
                "qkv norm/rope requires fp16");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "qkv norm/rope expects [rows, heads, dim]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "qkv norm/rope expects contiguous q/k/v");
    const int rows = (int)q.size(0);
    const int hq = (int)q.size(1);
    const int hk = (int)k.size(1);
    const int dim = (int)q.size(2);
    TORCH_CHECK(dim > 0 && dim <= 512 && dim % 2 == 0,
                "qkv norm/rope supports even dim <= 512");
    TORCH_CHECK((int)k.size(0) == rows && (int)v.size(0) == rows &&
                    (int)k.size(2) == dim && (int)v.size(2) == dim &&
                    (int)v.size(1) == hk,
                "qkv norm/rope shape mismatch");
    TORCH_CHECK(qw.numel() == dim && kw.numel() == dim && vw.numel() == dim,
                "qkv norm weights must have dim elements");
    TORCH_CHECK(cos.numel() == (long long)rows * dim &&
                    sin.numel() == (long long)rows * dim,
                "qkv rope cos/sin shape mismatch");
    auto qo = torch::empty_like(q);
    auto ko = torch::empty_like(k);
    auto vo = torch::empty_like(v);
    const int blocks = rows * (hq + 2 * hk);
    auto stream = at::cuda::getCurrentCUDAStream();
    gemma4_qkv_norm_rope_kernel<<<blocks, dim, 0, stream>>>(
        (const __half*)q.data_ptr(), (const __half*)k.data_ptr(),
        (const __half*)v.data_ptr(), (const __half*)qw.data_ptr(),
        (const __half*)kw.data_ptr(), (const __half*)vw.data_ptr(),
        (const __half*)cos.data_ptr(), (const __half*)sin.data_ptr(),
        (__half*)qo.data_ptr(), (__half*)ko.data_ptr(), (__half*)vo.data_ptr(),
        rows, hq, hk, dim, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qo, ko, vo};
}

// Warp-per-head q/k/v RMSNorm + q/k RoPE.  Each lane owns nper elements,
// which are exactly the elements paired by RoPE (j and j + dim/2), so the
// rotary pair never leaves the lane and no shared memory or block sync is
// needed.
__global__ void gemma4_qkv_norm_rope_warp_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ qw,
    const __half* __restrict__ kw,
    const __half* __restrict__ vw,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    __half* __restrict__ qo,
    __half* __restrict__ ko,
    __half* __restrict__ vo,
    int rows, int hq, int hk, int dim, float eps)
{
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int thq = rows * hq;
    const int thk = rows * hk;
    if (gwarp >= thq + 2 * thk) return;
    const bool is_q = gwarp < thq;
    const bool is_k = gwarp >= thq && gwarp < thq + thk;
    int t, head;
    const __half* src;
    const __half* weight;
    __half* dst;
    if (is_q) {
        t = gwarp / hq; head = gwarp - t * hq;
        src = q + (long long)(t * hq + head) * dim;
        weight = qw;
        dst = qo + (long long)(t * hq + head) * dim;
    } else if (is_k) {
        const int c = gwarp - thq;
        t = c / hk; head = c - t * hk;
        src = k + (long long)(t * hk + head) * dim;
        weight = kw;
        dst = ko + (long long)(t * hk + head) * dim;
    } else {
        const int c = gwarp - thq - thk;
        t = c / hk; head = c - t * hk;
        src = v + (long long)(t * hk + head) * dim;
        weight = vw;
        dst = vo + (long long)(t * hk + head) * dim;
    }

    const int nper = dim >> 5;  // dim / 32, 8 or 16
    float ss = 0.0f;
    __half hv[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i < nper) {
            const int j = lane + i * 32;
            hv[i] = src[j];
            const float xv = __half2float(hv[i]);
            ss += xv * xv;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    ss = __shfl_sync(0xffffffffu, ss, 0);
    const float inv = rsqrtf(ss / (float)dim + eps);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i < nper) {
            const int j = lane + i * 32;
            hv[i] = __float2half(__half2float(src[j]) * inv *
                                 __half2float(weight[j]));
        }
    }

    if (is_q || is_k) {
        const int half_stride = nper >> 1;
        const int half = dim >> 1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < half_stride) {
                const int j = lane + i * 32;
                const __half lo = hv[i];
                const __half hi = hv[i + half_stride];
                const __half c0 = cos[(long long)t * dim + j];
                const __half c1 = cos[(long long)t * dim + j + half];
                const __half s0 = sin[(long long)t * dim + j];
                const __half s1 = sin[(long long)t * dim + j + half];
                __half out_lo = __float2half(__half2float(lo) * __half2float(c0));
                out_lo = __float2half(__half2float(out_lo) -
                                      __half2float(hi) * __half2float(s0));
                __half out_hi = __float2half(__half2float(hi) * __half2float(c1));
                out_hi = __float2half(__half2float(out_hi) +
                                      __half2float(lo) * __half2float(s1));
                dst[j] = out_lo;
                dst[j + half] = out_hi;
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < nper) dst[lane + i * 32] = hv[i];
        }
    }
}

std::vector<torch::Tensor> gemma4_qkv_norm_rope_warp(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qw, torch::Tensor kw, torch::Tensor vw,
    torch::Tensor cos, torch::Tensor sin, double eps)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "warp qkv norm/rope requires CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                    k.scalar_type() == torch::kFloat16 &&
                    v.scalar_type() == torch::kFloat16,
                "warp qkv norm/rope requires fp16");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "warp qkv norm/rope expects [rows, heads, dim]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "warp qkv norm/rope expects contiguous q/k/v");
    const int rows = (int)q.size(0);
    const int hq = (int)q.size(1);
    const int hk = (int)k.size(1);
    const int dim = (int)q.size(2);
    TORCH_CHECK(dim == 256 || dim == 512, "warp qkv norm/rope dim must be 256/512");
    auto qo = torch::empty_like(q);
    auto ko = torch::empty_like(k);
    auto vo = torch::empty_like(v);
    const int warps = rows * (hq + 2 * hk);
    constexpr int threads = 256;
    const int blocks = (warps + threads / 32 - 1) / (threads / 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    gemma4_qkv_norm_rope_warp_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)q.data_ptr(), (const __half*)k.data_ptr(),
        (const __half*)v.data_ptr(), (const __half*)qw.data_ptr(),
        (const __half*)kw.data_ptr(), (const __half*)vw.data_ptr(),
        (const __half*)cos.data_ptr(), (const __half*)sin.data_ptr(),
        (__half*)qo.data_ptr(), (__half*)ko.data_ptr(), (__half*)vo.data_ptr(),
        rows, hq, hk, dim, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qo, ko, vo};
}
