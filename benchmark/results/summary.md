# CUDA Self-Attention 项目 — 阶段性总结

> 并行计算课程项目 | 2026-06-11

---

## 项目概览

从零实现了两个版本的 Scaled Dot-Product Self-Attention CUDA kernel，封装为 PyTorch 自定义算子，并通过正确性验证和性能基准测试。

### 项目结构

```
~/projects/cuda-self-attention/
├── setup.py                       # PyTorch CUDAExtension 编译配置
├── env.sh                         # 环境变量辅助脚本
├── csrc/
│   ├── attention_kernel.cu        # CUDA kernel 实现（Naive + Tiled）
│   └── attention_bindings.cpp     # PYBIND11 + PyTorch 绑定
├── attention/
│   └── __init__.py                # Python API 封装
├── tests/
│   ├── test_naive.py              # Naive kernel 正确性测试
│   ├── test_tiled.py              # Tiled kernel 正确性测试
│   └── debug_tiled.py             # 调试用
├── benchmark/
│   └── benchmark_speed.py         # 性能基准测试
└── .hermes/plans/
    └── 2026-06-11_132700-cuda-self-attention.md  # 实现计划
```

---

## 环境配置（关键经验）

| 项目 | 实际使用 | 备注 |
|------|----------|------|
| GPU | RTX 4060 Laptop (SM 8.9, 24 SMs, 8GB) | |
| CUDA Toolkit | 13.3 (conda env `cuda-attention`) | `conda install -c nvidia cuda-nvcc cuda-cudart` |
| PyTorch | 2.11.0+cu130 (base conda env) | CUDA 13.0 编译，与 13.3 toolkit 兼容 |
| Host Compiler | g++ 15.2 (conda) | 系统 g++ 13.3 与 PyTorch 头文件不兼容 |
| Python | 3.13 | 编译 OK，PyTorch 2.11.0 支持 3.13 |

### 编译关键变量

```bash
# 必须设置的环境变量
export CUDA_HOME=~/miniconda3/envs/cuda-attention
export CXX=~/miniconda3/envs/cuda-attention/bin/x86_64-conda-linux-gnu-c++
export CC=~/miniconda3/envs/cuda-attention/bin/x86_64-conda-linux-gnu-cc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:~/miniconda3/lib/python3.13/site-packages/torch/lib
```

### 遇到的编译问题

1. **g++ 13.3 + PyTorch 2.11.0 头文件不兼容**：`List_inl.h:202` 的 C++ 模板解析错误。解决：拆分为 `.cu` + `.cpp`，`.cpp` 用 conda 的 g++ 15.2 编译。
2. **Windows nvcc.exe 找不到 cl.exe**：WSL 下系统 nvcc 是 Windows CUDA toolkit 的包装脚本，需要 MSVC。解决：通过 conda 安装 Linux 版 CUDA toolkit。

---

## Task 1: Naive Self-Attention Kernel

### 设计要点

- **每个 block 处理一个 (batch, head)**，grid = `(B*H,)`
- **每个 thread 处理一个 query 位置**，block = `(N,)`，N ≤ 1024
- **不使用共享内存**：每个线程用局部数组 `float scores[1024]` 存储所有 score
- **三阶段计算**：计算 score → safe softmax → 加权求和

### 正确性验证

```
N      | max_diff (vs PyTorch math backend)
───────┼────────────────────────────────────
   64  | 4.77e-07  ✅
  128  | 4.77e-07  ✅
  256  | 4.17e-07  ✅
  512  | 4.32e-07  ✅
 1024  | 4.17e-07  ✅
```

所有 N 值偏差 < 1e-6，远优于 1e-3 的容限。

---

## Task 2: Tiled Self-Attention Kernel

### 关键优化

| 优化点 | 实现方式 | 效果 |
|--------|----------|------|
| **共享内存 Tiling** | Q_tile + KV_tile 加载到 `__shared__`，K/V tile 复用 | 全局访存减少 O(N / TILE_N) |
| **Online Softmax** | 分块维护 running max/sum，避免存储 N×N 注意力矩阵 | 显存复杂度 O(N) 而非 O(N²) |
| **Tile 大小** | TILE_N = 32 | 共享内存 16 KB，安全余量 |
| **K/V tile 复用** | K_tile 和 V_tile 使用同一块共享内存 | 共享内存减半 |

### 修正的 Bug

**共享内存加载模式错误**（审查命中）：
```
// ❌ 错误：每线程只加载自己行的部分列
for (int d = ti; d < D; d += TILE_N) {
    Q_smem[ti * D + d] = Q[base + global_i * D + d];
}

// ✅ 正确：每线程加载自己行的所有 D 个元素
for (int d = 0; d < D; d++) {
    Q_smem[ti * D + d] = Q[base + global_i * D + d];
}
```

原始 strided 加载模式 `for (int d = ti; d < D; d += TILE_N)` 设计用于跨行分布式加载，但 Q_smem 的行列映射与代码不匹配，导致 Q/K/V 数据加载不全。

### 正确性验证

```
N      | vs Reference  | vs Naive
───────┼───────────────┼─────────────
   32  | 7.15e-07  ✅  | 4.77e-07 ✅
   64  | 5.36e-07  ✅  | 5.36e-07 ✅
  128  | 5.96e-07  ✅  | 6.56e-07 ✅
  256  | 4.77e-07  ✅  | 4.47e-07 ✅
  512  | 7.45e-07  ✅  | 5.66e-07 ✅
 1024  | 7.15e-07  ✅  | 6.56e-07 ✅
```

所有 N 值通过，Naive 与 Tiled 交叉验证一致。

---

## Task 3: 性能基准测试

### 测试条件

- B=2, H=4, D=64, float32
- warmup=5, iters=50
- PyTorch 使用 math backend（禁用 flash/mem-efficient）
- GPU: RTX 4060 Laptop

### 结果

```
N      | Naive (us)  | Tiled (us)  | PyTorch (us)  | Tiled Speedup vs Naive
───────┼─────────────┼─────────────┼───────────────┼───────────────────────
   32  |     158.7   |     166.2   |      173.2    |      0.96x
   64  |     385.4   |     308.6   |      136.3    |      1.25x
  128  |     945.8   |     642.4   |      215.9    |      1.47x
  256  |   2,617.6   |   1,301.6   |      232.3    |      2.01x
  512  |   8,208.3   |   4,850.3   |      211.9    |      1.69x
 1024  |  33,145.3   |  15,114.2   |    1,142.1    |      2.19x
```

### 分析

**Tiled vs Naive (1-2x 加速)**：共享内存 tile 减少了全局内存访问次数，但每个线程仍然用串行 dot product 计算 score，没有用上 Tensor Core 或 cuBLAS。

**PyTorch 比 Tiled 快 7-20x**：PyTorch math backend 实际上调用了 cuBLAS 的矩阵乘法（`cublasGemmEx`），该实现利用了：
- Tensor Cores（SM 8.9 支持）
- Register-level tiling（warp-level matrix multiply）
- 极致的指令级并行

**N=32 时 Tiled ≈ Naive**：只有 1 个 tile，共享内存加载开销抵消了访存收益。

---

## 项目文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `csrc/attention_kernel.cu` | ~250 | Naive + Tiled CUDA kernel 实现 |
| `csrc/attention_bindings.cpp` | 65 | PyTorch 绑定 + C++ 包装 |
| `attention/__init__.py` | 25 | Python API |
| `setup.py` | 28 | 编译配置 |
| `tests/test_naive.py` | 38 | Naive kernel 测试 |
| `tests/test_tiled.py` | 60 | Tiled kernel 测试 |
| `benchmark/benchmark_speed.py` | 75 | 性能基准测试 |

---

## 后续可扩展方向

1. **Float16/Half 支持**：利用 Tensor Cores 的 `wmma` 命名空间
2. **FlashAttention 简化版**：在 tile 外层循环中做 online softmax 的分块计算，减少显存占用
3. **反向传播**：实现 backward kernel，与 `torch.autograd.Function` 集成
4. **Warp-level 优化**：用 warp shuffle 指令做规约，减少 shared memory bank conflict
