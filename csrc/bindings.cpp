/**
 * bindings.cpp — Python module registration for mini_llm_kernels._C
 *
 * All CUDA kernel functions are declared extern and registered here.
 * Without this file, the compiled .so has no PyInit__C symbol.
 */

#include <torch/extension.h>

// ── Declarations (defined in .cu files) ──
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b);
std::pair<torch::Tensor, torch::Tensor> fused_add_rms_norm(
    torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, float eps);
torch::Tensor decode_paged_attention(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens);
torch::Tensor decode_paged_attention_partitioned(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens, int num_partitions);
torch::Tensor paged_prefill_attention(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor chunk_lens, torch::Tensor history_lens);
torch::Tensor int4_dequant_matmul(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);

// ── Module registration ──
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "Element-wise tensor addition");
    m.def("fused_add_rms_norm", &fused_add_rms_norm,
          "Fused residual add + RMSNorm");
    m.def("decode_paged_attention", &decode_paged_attention,
          "Block-aware paged decode attention");
    m.def("decode_paged_attention_partitioned", &decode_paged_attention_partitioned,
          "KV-sequence-partitioned decode attention");
    m.def("paged_prefill_attention", &paged_prefill_attention,
          "Block-aware paged prefill attention");
    m.def("int4_dequant_matmul", &int4_dequant_matmul,
          "Fused INT4 dequantize + matmul");
}
