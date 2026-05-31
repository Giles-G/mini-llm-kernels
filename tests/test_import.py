"""
test_import.py

M1 验收测试：验证 mini_llm_kernels 可以正常 import，
fallback 模式下 _HAS_CUDA_OPS=False，不抛异常。
"""

import sys
import mini_llm_kernels


def test_import():
    print(f"mini_llm_kernels imported successfully")
    print(f"_HAS_CUDA_OPS = {mini_llm_kernels._HAS_CUDA_OPS}")

    if mini_llm_kernels._HAS_CUDA_OPS:
        import torch
        a = torch.ones(4, device="cuda")
        b = torch.ones(4, device="cuda") * 2
        out = mini_llm_kernels.add_tensors(a, b)
        assert out.tolist() == [3.0, 3.0, 3.0, 3.0], f"unexpected output: {out}"
        print("CUDA add_tensors: PASS")
    else:
        print("Fallback mode (no CUDA): PASS")


if __name__ == "__main__":
    test_import()
    print("ok")
