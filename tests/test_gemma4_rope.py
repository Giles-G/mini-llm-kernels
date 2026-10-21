import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

from mini_llm_kernels._C import gemma4_gelu_mul, gemma4_rope_fused


@pytest.mark.parametrize("rows,heads,dim", [(16, 8, 256), (16, 1, 256), (16, 8, 512)])
def test_rope_fused_matches_eager(rows, heads, dim):
    torch.manual_seed(0)
    x = torch.randn(rows, heads, dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
    half = dim // 2
    ref = x * cos[:, None, :]
    ref[..., :half].addcmul_(x[..., half:], sin[:, None, :half], value=-1.0)
    ref[..., half:].addcmul_(x[..., :half], sin[:, None, half:], value=1.0)
    out = gemma4_rope_fused(x, cos, sin)
    assert torch.equal(out, ref)


def test_gelu_mul_matches_eager_strided():
    torch.manual_seed(1)
    x = torch.randn(16, 2 * 6144, device="cuda", dtype=torch.float16)
    gate = x[:, :6144]
    up = x[:, 6144:]
    ref = torch.nn.functional.gelu(gate, approximate="tanh") * up
    assert torch.equal(gemma4_gelu_mul(gate, up), ref)

    big = torch.randn(16, 5, 1536, device="cuda", dtype=torch.float16)
    gate2 = torch.randn(16, 1536, device="cuda", dtype=torch.float16)
    up2 = big[:, 3, :]
    ref2 = torch.nn.functional.gelu(gate2, approximate="tanh") * up2
    assert torch.equal(gemma4_gelu_mul(gate2, up2), ref2)
