"""Benchmark: Naive vs Tiled vs PyTorch math backend."""
import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention import attention_naive, attention_tiled


def benchmark_one(N, B=2, H=4, D=64, warmup=5, iters=50):
    """Benchmark a single N configuration. Returns times in microseconds."""
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')
    scale = D ** -0.5

    def time_fn(fn, *args):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e6  # us

    t_naive = time_fn(attention_naive, q, k, v, scale)

    t_tiled = time_fn(attention_tiled, q, k, v, scale)

    def pt_attention(q, k, v, scale):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    t_pytorch = time_fn(pt_attention, q, k, v, scale)

    return t_naive, t_tiled, t_pytorch


def main():
    Ns = [32, 64, 128, 256, 512, 1024]
    device_name = torch.cuda.get_device_name(0)

    print(f"{'N':>6} | {'Naive (us)':>12} | {'Tiled (us)':>12} | {'PyTorch (us)':>14} | {'Speedup':>8}")
    print("-" * 65)

    results = []
    for N in Ns:
        t_naive, t_tiled, t_pt = benchmark_one(N)
        speedup = t_naive / t_tiled if t_tiled > 0 else 0
        results.append((N, t_naive, t_tiled, t_pt, speedup))
        print(f"{N:6d} | {t_naive:12.1f} | {t_tiled:12.1f} | {t_pt:14.1f} | {speedup:7.2f}x")

    # Summary
    print("\n=== Environment ===")
    print(f"GPU: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"B=2, H=4, D=64, float32")
    print(f"Naive: no shared memory, brute-force dot products")
    print(f"Tiled: TILE_N=32, shared memory Q_tile+KV_tile, online softmax")
    print(f"PyTorch: math backend (flash/mem-efficient disabled for fair comparison)")


if __name__ == '__main__':
    main()
