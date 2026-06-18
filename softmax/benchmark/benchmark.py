import torch
import time
import sys

sys.path.insert(0, '.')
from softmax import softmax_naive, softmax_warp

WARMUP = 5
ITERS = 30

def benchmark(dims, rows=32768):
    rows = rows if dims[0] <= 512 else 16384
    results = []
    for D in dims:
        x = torch.randn(rows, D, device='cuda', dtype=torch.float32)
        scale = D ** -0.5

        # Warmup
        for _ in range(WARMUP):
            softmax_naive(x)
            softmax_warp(x)
            torch.softmax(x, dim=-1)
        torch.cuda.synchronize()

        # Our Naive
        start = time.perf_counter()
        for _ in range(ITERS):
            softmax_naive(x)
        torch.cuda.synchronize()
        t_naive = (time.perf_counter() - start) / ITERS * 1e6

        # Our Warp
        start = time.perf_counter()
        for _ in range(ITERS):
            softmax_warp(x)
        torch.cuda.synchronize()
        t_warp = (time.perf_counter() - start) / ITERS * 1e6

        # PyTorch (cuDNN/cuBLAS backend)
        start = time.perf_counter()
        for _ in range(ITERS):
            torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - start) / ITERS * 1e6

        # Verify correctness
        ref = torch.softmax(x, dim=-1)
        d_naive = (softmax_naive(x) - ref).abs().max().item()
        d_warp = (softmax_warp(x) - ref).abs().max().item()

        results.append({
            'D': D, 'rows': rows,
            'naive_us': t_naive, 'warp_us': t_warp, 'torch_us': t_torch,
            'warp_vs_naive': t_naive / t_warp,
            'warp_vs_torch': t_warp / t_torch,
            'naive_diff': d_naive, 'warp_diff': d_warp,
        })
    return results

def main():
    print("=" * 70)
    print("Softmax Performance Benchmark")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    dims = [64, 128, 256, 512, 1024]
    results = benchmark(dims)

    # Table 1: Latency
    print(f"{'D':>6} | {'Naive (us)':>10} | {'Warp (us)':>10} | {'torch (us)':>10} | {'Warp/Naive':>10} | {'Warp/torch':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['D']:>6} | {r['naive_us']:>10.0f} | {r['warp_us']:>10.0f} | {r['torch_us']:>10.0f} | {r['warp_vs_naive']:>9.1f}x | {r['warp_vs_torch']:>9.2f}x")

    # Table 2: Correctness
    print(f"\n{'D':>6} | {'Naive err':>12} | {'Warp err':>12} | {'Status':>8}")
    print("-" * 50)
    for r in results:
        s = '✅' if r['warp_diff'] < 1e-5 else '❌'
        print(f"{r['D']:>6} | {r['naive_diff']:>12.2e} | {r['warp_diff']:>12.2e} | {s:>8}")

    # Table 3: Throughput
    print(f"\n{'D':>6} | {'Naive (M rows/s)':>16} | {'Warp (M rows/s)':>16} | {'torch (M rows/s)':>16}")
    print("-" * 60)
    for r in results:
        tp_naive = r['rows'] / r['naive_us']  # rows per us = M rows per s
        tp_warp  = r['rows'] / r['warp_us']
        tp_torch = r['rows'] / r['torch_us']
        print(f"{r['D']:>6} | {tp_naive:>16.2f} | {tp_warp:>16.2f} | {tp_torch:>16.2f}")

    # Environment info
    print(f"\n=== Environment ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}, dtype: float32")

if __name__ == '__main__':
    main()
