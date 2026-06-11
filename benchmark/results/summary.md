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

## 并行计算设计详解

### Self-Attention 的计算公式

Self-Attention 是 Transformer 的核心计算单元：

```
S = Q × K^T            ← 矩阵乘法：计算相似度
P = softmax(S / √D)    ← 按行归一化（数值稳定）
O = P × V              ← 矩阵乘法：加权求和
```

其中 Q、K、V 的形状都是 `[B, H, N, D]`：
- B = batch size（批量大小）
- H = head count（注意力头数）
- N = sequence length（序列长度）
- D = head dimension（每个头的维度，通常 64）

### 为什么需要并行

S 矩阵形状为 `[B, H, N, N]`，计算量 O(B·H·N²·D)。当 N=1024 时：

- 一个 attention head 的 S 矩阵有 1024×1024 ≈ 100 万个元素
- B=2, H=4 时总共约 800 万个 score
- 每个 score 需要一次 D=64 维的点积

串行计算完全不可接受，必须用 GPU 的数千个核心并行计算。

---

### 并行策略对比

#### Naive Kernel：一维 Grid

```
Grid:  (B × H,)          ← 每个 block 处理一个 (batch, head)
Block: (N,)              ← 每个 thread 处理一个 query 位置

Block 内部：每个线程独立完成自己的工作
  for j = 0..N-1:
    dot = Σ(Q[i][d] × K[j][d])    ← 串行做点积
    scores[j] = dot * scale

  然后 softmax + 加权求和
```

**并行度：** B×H×N 个线程同时工作。每个线程完全独立，**无线程间通信**（无共享内存、无同步）。

**问题：** 每个线程都要从全局内存读取 K 的所有行（N 次），Q 的每行也被重复读 N 次。**全局内存带宽成为瓶颈**。

---

#### Tiled Kernel：二维 Grid + 共享内存

```
Grid:  (ceil(N/TILE_N), B × H)    ← 二维：tile 索引 + batch-head
Block: (TILE_N,)                    ← TILE_N = 32

核心思想：将 N×D 矩阵切成 32×D 的 tile，加载到共享内存
```

**优化 1——共享内存 tiling：**

```cuda
// Naive: 每次 dot product 都读全局内存（延迟 ~400 cycles）
dot += Q[base + i*D + d] * K[base + j*D + d];

// Tiled: 先加载到共享内存（一次全局读 + 多次共享内存读）
__shared__ float Q_smem[32][64];   // 加载一次，被 32 个 score 复用
__shared__ float KV_smem[32][64];  // 延迟 ~30 cycles

Q_smem[ti][d] = Q[base + global_i * D + d];
__syncthreads();  // 等待所有线程加载完毕

// 之后的 dot product 从共享内存读取
dot += Q_smem[ti][d] * KV_smem[jl][d];  // 快 10 倍
```

**优化 2——共享内存复用：**

K 和 V 不同时使用，共用同一块共享内存，节省一半空间：

```cuda
float* KV_smem = smem + TILE_N * D;  // 8 KB
// 第一段：存 K_tile，计算 scores
// 第二段：覆盖为 V_tile，累加输出
```

**优化 3——Online Softmax（避免 N×N 矩阵）：**

常规 softmax 需要先算出所有 S[i][j]（N² 个），再逐行做 exp/sum。N=1024 时单个 head 的 S 矩阵需要 1024×1024×4 = 4 MB。

Online softmax 的核心思想是**分块计算，逐 tile 更新 running max 和 running sum**：

```
初始化: m = -inf, d = 0, O = [0]*D

对每个 K_tile:
  1. 算局部 scores: s[j] = Q[i] · K_tile[j]
  2. 找局部最大值: m_local = max(s)
  3. 更新全局最大值: m_new = max(m, m_local)
  4. 缩放旧输出: O *= exp(m - m_new)        ← 适配新最大值
  5. 更新分母: d = d * exp(m - m_new) + Σ exp(s[j] - m_new)
  6. 加载 V_tile
  7. 累加新贡献: O += Σ exp(s[j] - m_new) * V[j]
  8. m = m_new

最终: O /= d
```

只需存储 O(D) 的 running state（256 bytes for D=64），不需要 N×N 注意力矩阵。

---

### 性能数据解读

```
N      | Naive (μs) | Tiled (μs) | PyTorch (μs) | Tiled/Naive
───────┼────────────┼────────────┼──────────────┼────────────
   32  |      158.7 |      166.2 |       173.2  |     0.96×
  256  |    2,617.6 |    1,301.6 |       232.3  |     2.01×
 1024  |   33,145.3 |   15,114.2 |     1,142.1  |     2.19×
```

**Tiled vs Naive（1-2× 加速）：**
- 共享内存让数据复用率从 1/D 提升到接近 1
- 每算一个 score，Naive 要读一次全局内存，Tiled 只需读共享内存
- N 越大，tile 数越多，收益越明显

**PyTorch 比我们快 7-20×：**
- PyTorch math backend 实际调用 cuBLAS 的 `cublasGemmEx`
- cuBLAS 使用 Tensor Cores + warp-level 矩阵乘法 + 指令级并行
- 我们的手写 dot product 是纯标量计算

**N=32 时 Tiled ≈ Naive：**
- 仅需 1 个 tile（TILE_N=32）
- 共享内存加载/同步的开销抵消了访存收益

---

### 使用方法

**编译：**
```bash
cd ~/projects/cuda-self-attention
export CUDA_HOME=~/miniconda3/envs/cuda-attention
export CXX=~/miniconda3/envs/cuda-attention/bin/x86_64-conda-linux-gnu-c++
export CC=~/miniconda3/envs/cuda-attention/bin/x86_64-conda-linux-gnu-cc
python3 setup.py build_ext --inplace
```

**运行测试：**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:~/miniconda3/lib/python3.13/site-packages/torch/lib
PYTHONPATH=. python3 tests/test_naive.py
PYTHONPATH=. python3 tests/test_tiled.py
PYTHONPATH=. python3 benchmark/benchmark_speed.py
```

**在代码中调用：**
```python
from attention import attention_naive, attention_tiled

q = torch.randn(2, 4, 128, 64, device='cuda')
k = torch.randn(2, 4, 128, 64, device='cuda')
v = torch.randn(2, 4, 128, 64, device='cuda')

out1 = attention_naive(q, k, v)   # 基准版本
out2 = attention_tiled(q, k, v)   # 优化版本（推荐）

# 自定义 scale（默认 1/sqrt(D)）
out3 = attention_tiled(q, k, v, scale=0.125)
```

---

### 本项目展示的并行计算概念

| 概念 | 具体体现 |
|------|----------|
| **线程层级** | Grid → Block → Thread，三维映射到 (batch, head, sequence) |
| **内存层级** | 全局内存 → 共享内存 → 寄存器，延迟差 10-100× |
| **数据复用** | Tiling 让共享内存中的数据被多个线程复用 |
| **同步与通信** | `__syncthreads()` 保证线程间数据可见性 |
| **负载均衡** | 2D grid + tile 边界处理均匀覆盖所有查询位置 |
| **算法优化** | Online softmax 将显存复杂度从 O(N²) 降到 O(N) |
| **性能分析** | 理论带宽 vs 实测带宽，识别瓶颈是计算还是访存 |
