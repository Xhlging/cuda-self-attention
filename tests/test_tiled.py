"""Test Tiled Self-Attention CUDA kernel against PyTorch reference and Naive kernel."""
import torch
import sys
sys.path.insert(0, '..')
from attention import attention_naive, attention_tiled


def test_tiled_vs_ref(N: int, B: int = 2, H: int = 4, D: int = 64):
    """Compare Tiled kernel output with PyTorch math backend."""
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')
    scale = D ** -0.5

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    out = attention_tiled(q, k, v, scale=scale)
    diff = (out - ref).abs().max().item()
    return diff


def test_tiled_vs_naive(N: int, B: int = 2, H: int = 4, D: int = 64):
    """Cross-validate: Tiled and Naive should produce identical results."""
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')
    scale = D ** -0.5

    out_naive = attention_naive(q, k, v, scale=scale)
    out_tiled = attention_tiled(q, k, v, scale=scale)
    diff = (out_naive - out_tiled).abs().max().item()
    return diff


def main():
    print("=== Tiled vs PyTorch Reference ===")
    Ns = [32, 64, 128, 256, 512, 1024]
    all_pass = True

    for N in Ns:
        diff = test_tiled_vs_ref(N)
        status = "PASS" if diff < 1e-3 else "FAIL"
        print(f"N={N:5d} | max_diff={diff:.6e} | {status}")
        if diff >= 1e-3:
            all_pass = False

    print("\n=== Tiled vs Naive (cross-validation) ===")
    for N in Ns:
        diff = test_tiled_vs_naive(N)
        status = "PASS" if diff < 1e-5 else "FAIL"
        print(f"N={N:5d} | naive_tiled_diff={diff:.6e} | {status}")
        if diff >= 1e-5:
            all_pass = False

    if all_pass:
        print("\n✅ All Tiled kernel tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
