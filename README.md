# CUDA Self-Attention

从零实现的 Scaled Dot-Product Self-Attention CUDA kernel，封装为 PyTorch 自定义算子。
并行计算课程项目。

## Overview

实现了两个版本的 Self-Attention 前向 kernel：

| Version | Description | Techniques |
|---------|-------------|------------|
| **Naive** | 无优化基准实现 | 每个 thread 处理一个 query 位置，局部数组存储 scores |
| **Tiled** | 共享内存优化版 | Q/K/V tile 加载、online softmax、共享内存复用 |

与 PyTorch `scaled_dot_product_attention` (math backend) 做了正确性交叉验证和性能对比。

## Environment

- **GPU:** NVIDIA GeForce RTX 4060 Laptop (SM 8.9, 24 SMs, 8GB)
- **CUDA Toolkit:** 12+ (tested with 13.3)
- **PyTorch:** 2.0+ (tested with 2.11.0)
- **Python:** 3.12+
- **Compiler:** GCC 12+ (conda g++ 15.2 recommended)

## Quick Start

```bash
# 1. Install CUDA toolkit + PyTorch
conda create -n cuda-attention python=3.12 -y
conda activate cuda-attention
conda install -c nvidia cuda-nvcc cuda-cudart -y
pip install torch ninja

# 2. Build
cd ~/projects/cuda-self-attention
export CUDA_HOME=$CONDA_PREFIX
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
python3 setup.py build_ext --inplace

# 3. Test
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(python3 -c "import torch; print('/'.join(torch.__file__.split('/')[:-1] + ['lib']))")
PYTHONPATH=. python3 tests/test_naive.py
PYTHONPATH=. python3 tests/test_tiled.py
```

> **Note:** If using the system GCC, you may encounter C++ template compatibility issues with PyTorch headers. The conda-provided GCC (via `cxx_linux-64` package) is recommended.

## Usage

```python
import torch
from attention import attention_naive, attention_tiled

B, H, N, D = 2, 4, 128, 64
q = torch.randn(B, H, N, D, device='cuda')
k = torch.randn(B, H, N, D, device='cuda')
v = torch.randn(B, H, N, D, device='cuda')
scale = D ** -0.5

out_naive = attention_naive(q, k, v, scale)  # baseline
out_tiled = attention_tiled(q, k, v, scale)  # optimized
```

## Correctness

Both kernels verified against PyTorch `scaled_dot_product_attention` with math backend (FlashAttention and MemEfficientAttention disabled for fair comparison).

```
N      | Naive diff   | Tiled diff    | Tiled vs Naive
───────┼──────────────┼───────────────┼───────────────
   64  | 4.77e-07 ✅  | 5.36e-07 ✅   | 5.36e-07 ✅
  128  | 4.77e-07 ✅  | 5.96e-07 ✅   | 6.56e-07 ✅
  256  | 4.17e-07 ✅  | 4.77e-07 ✅   | 4.47e-07 ✅
  512  | 4.32e-07 ✅  | 7.45e-07 ✅   | 5.66e-07 ✅
 1024  | 4.17e-07 ✅  | 7.15e-07 ✅   | 6.56e-07 ✅
```

All max absolute deviations < 1e-6, well within 1e-3 tolerance.

## Performance

Conditions: B=2, H=4, D=64, float32, RTX 4060 Laptop GPU.

| N | Naive (μs) | Tiled (μs) | PyTorch (μs) | Speedup |
|---|------------|------------|--------------|---------|
| 32 | 158.7 | 166.2 | 173.2 | 0.96× |
| 64 | 385.4 | 308.6 | 136.3 | 1.25× |
| 128 | 945.8 | 642.4 | 215.9 | 1.47× |
| 256 | 2,617.6 | 1,301.6 | 232.3 | 2.01× |
| 512 | 8,208.3 | 4,850.3 | 211.9 | 1.69× |
| 1024 | 33,145.3 | 15,114.2 | 1,142.1 | 2.19× |

Key observations:

- **Tiled vs Naive:** 1–2× speedup from shared memory tiling; gains increase with sequence length N
- **PyTorch vs Tiled:** 7–20× faster (cuBLAS GEMM + Tensor Cores vs. hand-written dot products)
- **N=32:** Tiled ≈ Naive (single tile, shared memory overhead dominates)

## Project Structure

```
cuda-self-attention/
├── setup.py                        # PyTorch CUDAExtension build
├── csrc/
│   ├── attention_kernel.cu         # CUDA kernels (naive + tiled)
│   └── attention_bindings.cpp      # PYBIND11 + Torch bindings
├── attention/
│   └── __init__.py                 # Python API
├── tests/
│   ├── test_naive.py               # Naive correctness tests
│   └── test_tiled.py               # Tiled correctness + cross-validation
├── benchmark/
│   ├── benchmark_speed.py          # Performance benchmark
│   └── results/summary.md          # Full project summary
└── .hermes/plans/                  # Implementation plan
```

## Key Implementation Details

### Naive Kernel

Each CUDA block handles one `(batch, head)` pair. Each thread handles one query position and stores all N scores in a local `float[1024]` array. No shared memory — intentionally the simplest possible baseline.

### Tiled Kernel

- **TILE_N = 32**: chosen to keep shared memory under 48 KB limit (2 × 32 × 64 × 4 = 16 KB)
- **Shared memory reuse**: K and V tiles share the same memory region
- **Online softmax**: running max + running denominator maintained across tiles to avoid storing the full N×N attention matrix
- **Grid**: 2D `(ceil(N/TILE_N), B×H)` mapping tile index and batch-head

## License

MIT
