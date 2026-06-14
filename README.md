# CUDA Self-Attention

从零实现的 Scaled Dot-Product Self-Attention CUDA kernel，封装为 PyTorch 自定义算子。
并行计算课程项目。

## Overview

实现了两个版本的 Self-Attention 前向 kernel：

| Version | Description | Techniques |
|---------|-------------|------------|
| **Naive** | 无优化基准实现 | 每个 thread 处理一个 query 位置，局部数组存储 scores |
| **Tiled** | 共享内存优化版 | Q/K/V tile 加载、online softmax、D_PAD bank conflict 消除、K+V 同时加载 |

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
   32  | 4.77e-07 ✅  | 7.15e-07 ✅   | 4.77e-07 ✅
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
| 32 | 316.5 | 139.0 | 243.3 | 2.28× |
| 64 | 544.1 | 306.5 | 222.1 | 1.78× |
| 128 | 1,030.5 | 374.6 | 159.7 | 2.75× |
| 256 | 2,891.9 | 465.4 | 148.2 | 6.21× |
| 512 | 8,327.5 | 1,351.0 | 206.3 | 6.16× |
| 1024 | 47,779.8 | 4,684.5 | 1,464.3 | 10.20× |

Key observations:

- **Tiled vs Naive:** 2–10× speedup after bank conflict elimination + K/V simultaneous load
- **PyTorch vs Tiled:** 3–4× faster (cuBLAS GEMM + Tensor Cores vs. hand-written dot products), narrowed from original 7–20× gap
- **Bank conflict fix (D_PAD = D + 1):** single largest contributor, eliminates 32-way shared memory bank conflicts
- **N=1024:** Tiled outperforms Naive by 10×, demonstrating the impact of shared memory tiling combined with conflict-free access

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

- **TILE_N = 32**: chosen to keep shared memory under 48 KB limit
- **D_PAD = D + 1**: row stride padded to 65 to eliminate 32-way shared memory bank conflicts (`bank = (ti * 65 + d) % 32 = (ti + d) % 32`, stride=1 across warp threads)
- **K and V loaded simultaneously**: 3 separate smem regions (Q_smem + K_smem + V_smem, ~25 KB total), reducing `__syncthreads()` from 4 to 2 per tile
- **Online softmax**: running max + running denominator maintained across tiles to avoid storing the full N×N attention matrix
- **Compile-time loop unrolling**: kernels templated on `D` for full inner-loop unrolling by nvcc
- **Grid**: 2D `(ceil(N/TILE_N), B×H)` mapping tile index and batch-head

### Optimization History

The tiled kernel originally suffered from 32-way bank conflicts due to the shared memory stride D=64 (`bank = (ti * 64 + d) % 32 = d % 32`). Fixing this with a D+1=65 padding column was the single most impactful optimization, delivering 1.4–3.3× speedup alone.

## License

MIT
