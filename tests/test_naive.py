"""Test Naive Self-Attention CUDA kernel against PyTorch reference."""
import torch
import sys
sys.path.insert(0, '..')
from attention import attention_naive


def test_naive(N: int, B: int = 2, H: int = 4, D: int = 64):
    """Compare Naive kernel output with PyTorch math backend at sequence length N."""
    torch.manual_seed(42)

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    scale = D ** -0.5

    # PyTorch reference: force math backend (disable flash/mem-efficient)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    # Our naive kernel
    out = attention_naive(q, k, v, scale=scale)

    # Compare
    diff = (out - ref).abs().max().item()
    return diff


def main():
    Ns = [64, 128, 256, 512, 1024]
    all_pass = True

    for N in Ns:
        diff = test_naive(N)
        status = "PASS" if diff < 1e-3 else "FAIL"
        print(f"N={N:5d} | max_diff={diff:.6e} | {status}")
        if diff >= 1e-3:
            all_pass = False

    if all_pass:
        print("\n✅ All Naive kernel tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
