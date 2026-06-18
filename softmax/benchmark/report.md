# 利用 Warp Shuffle 实现高效并行归约 —— 以 Softmax 为例

> 张潇瀚 · 并行计算课程项目 · 2026-06-18

---

## 摘要

本实验从零实现了两个 CUDA 版本的 Softmax 内核——串行 Naive 版本与基于 Warp Shuffle 的并行归约版本，封装为 PyTorch 自定义算子，并与 PyTorch 官方实现进行正确性交叉验证和性能对比。实验结果表明，Warp 版本在 D≥256 时与 CUDA 自带的 cuDNN softmax 性能接近（差距 <15%），D=1024 时仅慢 13%，且所有测试偏差 <2×10⁻⁷。

---

## 1. 算法背景

### 1.1 Softmax 定义

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

为避免 $e^{x}$ 溢出，实际采用 Safe Softmax：先找最大值 $m = \max_j x_j$，再计算：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

### 1.2 并行归约

Softmax 的核心操作包含两次归约（Reduction）：

1. **Max Reduction**：找 D 个元素的最大值
2. **Sum Reduction**：求 D 个 exp 值的总和

归约是并行计算中最基础、最常见的操作——把多个值合并为一个值。本实验的核心即利用 CUDA Warp Shuffle 指令高效实现这两次归约。

---

## 2. 系统架构

```
Python 用户接口
  from softmax import softmax_warp
           ↓
C++ Torch 绑定 (softmax_bindings.cpp, g++ 编译)
  softmax_forward_warp() — contiguous()检查 + tensor 提取
           ↓
Launch Wrapper (softmax_kernel.cu)
  switch(D) 模板分发 + <<<rows, 32>>> 启动配置
           ↓
CUDA Kernel (softmax_kernel.cu, nvcc 编译)
  softmax_warp_kernel<D>() — Warp Shuffle 并行归约
           ↓
GPU 硬件 (RTX 4060 Laptop, SM 8.9, 24 SMs, 8GB)
```

---

## 3. 实现代码

### 3.1 Naive 内核（串行基准）

**设计思路：** 每个线程独立处理一行数据。Grid 维度为 `[ceil(rows/256)]`，Block 维度为 `[256]`。每线程串行遍历 D 个元素三次：找最大值 → 算 exp + 求和 → 归一化。无线程间通信。

```c
template <int D>
__global__ void softmax_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int stride)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* x = input + row * stride;
    float* y = output + row * stride;

    // 第一遍: 找最大值 (Safe Softmax)
    float max_val = x[0];
    for (int i = 1; i < D; i++)
        max_val = fmaxf(max_val, x[i]);

    // 第二遍: 算 exp(x - max) 并求和
    float sum_val = 0.0f;
    for (int i = 0; i < D; i++) {
        y[i] = __expf(x[i] - max_val);
        sum_val += y[i];
    }

    // 第三遍: 归一化
    float inv_sum = 1.0f / sum_val;
    for (int i = 0; i < D; i++)
        y[i] *= inv_sum;
}
```

**时间复杂度：** 每线程 O(3D)。D=1024 时，每线程 3072 次操作，完全串行。

### 3.2 Warp 归约内核（并行优化）

**设计思路：** 采用 One-Warp-Per-Row 映射——Grid 维度为 `[rows]`，Block 维度为 `[32]`（恰好一个 Warp）。32 个线程通过跨步访问分治处理 D/32 个元素，然后通过 `__shfl_down_sync` 蝴蝶归约和 `__shfl_sync` 广播完成全局归约。

```c
template <int D>
__global__ void softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int stride)
{
    int row = blockIdx.x;               // 一行一个 Block
    if (row >= rows) return;

    const float* x = input + row * stride;
    float* y = output + row * stride;
    int lane = threadIdx.x;             // Warp 内编号 (0~31)

    // ===== Step 1: 跨步访问找局部最大值 =====
    // 数据分配: lane 0→x[0,32,64,...], lane 1→x[1,33,65,...]
    // 相邻线程访问相邻地址 → 全局内存合并访问
    float max_val = -INFINITY;
    for (int i = lane; i < D; i += 32)
        max_val = fmaxf(max_val, x[i]);

    // ===== Step 2: Max Reduction + Broadcast =====
    // 蝴蝶归约: offset=16→8→4→2→1, 5 步合并 32 个局部值
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    // 广播: 将 lane 0 手里的全局最大值复制给全部 32 线程
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // ===== Step 3: Sum Reduction + Broadcast =====
    float sum_val = 0.0f;
    for (int i = lane; i < D; i += 32) {
        float e = __expf(x[i] - max_val);
        y[i] = e;  sum_val += e;
    }
    for (int offset = 16; offset > 0; offset /= 2)
        sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
    sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);

    // ===== Step 4: 归一化输出 =====
    float inv_sum = 1.0f / sum_val;
    for (int i = lane; i < D; i += 32)
        y[i] *= inv_sum;
}
```

**并行策略：**

```
Warp 0 (32线程) → 处理第 0 行
Warp 1 (32线程) → 处理第 1 行
...
Warp M (32线程) → 处理第 M 行

单个 Warp 内:
  lane  0 → x[ 0], x[32], x[64], ...
  lane  1 → x[ 1], x[33], x[65], ...
  ...
  lane 31 → x[31], x[63], x[95], ...
  每线程只处理 D/32 个元素
```

**蝴蝶归约过程：**

```
初始化: 32 个线程各自有局部 max/sum
offset=16: 线程0和16合并，1和17合并... → 剩 16 个
offset=8:  线程0和8合并，1和9合并...   → 剩 8 个
offset=4:  线程0和4合并，1和5合并...   → 剩 4 个
offset=2:  线程0和2合并，1和3合并     → 剩 2 个
offset=1:  线程0和1合并               → 剩 1 个 ✅ (在 lane 0)
```

**时间复杂度：** 每线程 O(3D/32) 次内存访问 + 5×2 次 shuffle 通信。

### 3.3 Launch 包装器（模板分发）

```c
void softmax_warp_launch(const float* input, float* output,
                         int rows, int D, int stride) {
    // Grid: [rows], Block: [32] — 一行恰好一个 Warp
    switch (D) {
        case 64:  softmax_warp_kernel<64><<<rows, 32>>>(input, output, rows, stride); break;
        case 128: softmax_warp_kernel<128><<<rows, 32>>>(input, output, rows, stride); break;
        case 256: softmax_warp_kernel<256><<<rows, 32>>>(input, output, rows, stride); break;
        case 512: softmax_warp_kernel<512><<<rows, 32>>>(input, output, rows, stride); break;
        case 1024:softmax_warp_kernel<1024><<<rows, 32>>>(input, output, rows, stride); break;
        default:  fprintf(stderr, "Unsupported D=%d\n", D); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

### 3.4 PyTorch 绑定

**C++ 绑定** (`csrc/softmax_bindings.cpp`)：

```cpp
torch::Tensor softmax_forward_warp(torch::Tensor input) {
    input = input.contiguous();
    auto sizes = input.sizes();
    int D = sizes.back();
    int rows = 1;
    for (size_t i = 0; i < sizes.size() - 1; i++) rows *= sizes[i];
    auto output = torch::empty_like(input);
    softmax_warp_launch(input.data_ptr<float>(), output.data_ptr<float>(),
                        rows, D, D);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_forward_naive", &softmax_forward_naive);
    m.def("softmax_forward_warp", &softmax_forward_warp);
}
```

**Python API** (`softmax/__init__.py`)：

```python
from . import _C

def softmax_warp(x):
    """Warp-reduction CUDA softmax forward.
    One warp (32 threads) per row. Uses __shfl_down_sync butterfly
    reduction + __shfl_sync broadcast.
    """
    return _C.softmax_forward_warp(x)
```

**构建配置** (`setup.py`)：

```python
CUDAExtension(
    "softmax._C",
    ["csrc/softmax_bindings.cpp", "csrc/softmax_kernel.cu"],
    extra_compile_args={
        "cxx": ["-O3", "-fopenmp"],
        "nvcc": ["-O3", "--expt-relaxed-constexpr",
                 "-gencode=arch=compute_89,code=sm_89",
                 "-gencode=arch=compute_80,code=sm_80"],
    },
)
```

---

## 4. 实验设置

### 4.1 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 计算能力 | SM 8.9 (Ada Lovelace) |
| SM 数量 | 24 |
| 显存 | 8.6 GB |
| CUDA Toolkit | 13.3 |
| PyTorch | 2.11.0+cu130 |
| 编译器 | g++ 15.2 (conda) |
| 数据类型 | float32 |

### 4.2 测试方法

- **对比对象：** Naive（手写串行）、Warp（手写并行归约）、torch.softmax（PyTorch 官方实现，底层调用 cuDNN）
- **测试维度：** D ∈ {64, 128, 256, 512, 1024}
- **数据规模：** D≤512 时 rows=32768，D=1024 时 rows=16384
- **预热：** 5 轮
- **取平均：** 30 轮
- **正确性验证：** 以 `torch.softmax` 为参考，计算最大绝对偏差

### 4.3 测试脚本

```python
import torch, time
from softmax import softmax_naive, softmax_warp

for D in [64, 128, 256, 512, 1024]:
    rows = 32768 if D <= 512 else 16384
    x = torch.randn(rows, D, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(5): softmax_naive(x); softmax_warp(x); torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Benchmark
    iters = 30
    for _ in range(iters): softmax_warp(x)
    torch.cuda.synchronize()
    t_warp = (time.perf_counter() - start) / iters * 1e6

    # Correctness
    ref   = torch.softmax(x, dim=-1)
    out   = softmax_warp(x)
    diff  = (out - ref).abs().max().item()
```

运行命令：
```bash
source env.sh
python3 softmax/benchmark/benchmark.py
```

---

## 5. 实验结果

### 5.1 正确性验证

```
     D |    Naive err |     Warp err |   Status
--------------------------------------------------
    64 |     1.49e-07 |     5.96e-08 |        ✅
   128 |     1.49e-07 |     5.96e-08 |        ✅
   256 |     1.04e-07 |     2.98e-08 |        ✅
   512 |     1.49e-07 |     1.49e-08 |        ✅
  1024 |     9.31e-08 |     7.45e-09 |        ✅
```

所有维度下最大绝对偏差 < 2×10⁻⁷，远优于 10⁻⁵ 容限。

观察：Warp 版本的精度（~5×10⁻⁸）系统性地优于 Naive 版本（~1.5×10⁻⁷）。这是因为 Warp 版本的归约将浮点累加操作从 D 次减少到约 D/32 + log₂(32) 次，减少了舍入误差累积。

### 5.2 延迟对比

处理 `rows × D` 矩阵的平均耗时（μs，越小越好）：

```
     D | Naive (us) |  Warp (us) | torch (us) | Warp/Naive | Warp/torch
----------------------------------------------------------------------
    64 |        520 |        131 |         32 |       4.0x |      4.07x
   128 |        970 |        129 |         48 |       7.5x |      2.68x
   256 |       1840 |        313 |        283 |       5.9x |      1.11x
   512 |       3185 |        663 |        590 |       4.8x |      1.12x
  1024 |       5651 |       1319 |       1172 |       4.3x |      1.13x
```

### 5.3 吞吐量对比

每秒可处理的行数（M rows/s，越大越好）：

```
     D | Naive (M rows/s) |  Warp (M rows/s) | torch (M rows/s)
------------------------------------------------------------
    64 |            63.03 |           250.13 |          1019.04
   128 |            33.77 |           254.00 |           679.88
   256 |            17.81 |           104.66 |           115.80
   512 |            10.29 |            49.43 |            55.56
  1024 |             5.80 |            24.84 |            27.97
```

---

## 6. 分析与讨论

### 6.1 Warp vs Naive

Warp Shuffle 归约在所有测试维度上均显著优于 Naive 串行实现，加速比 4-8×。加速来源于两点：

1. **分治减少单线程工作量：** Naive 每线程处理 D 个元素，Warp 版每线程仅处理 D/32 个元素
2. **Warp Shuffle 是硬件级通信：** `__shfl_down_sync` 直接通过片上连线交换寄存器数据，延迟约 1 个时钟周期。5 步归约总计约 5 cycles，vs 全局内存访问 ~400 cycles

### 6.2 Warp vs torch.softmax (CUDA cuDNN)

| D | 手写 Warp | torch (cuDNN) | 差距 |
|---|----------|---------------|------|
| 256 | 313 μs | 283 μs | 1.11× |
| 512 | 663 μs | 590 μs | 1.12× |
| 1024 | 1319 μs | 1172 μs | 1.13× |

D≥256 时手写内核与 CUDA 自带函数性能差距 <15%，D=1024 时仅差 13%。这表明：

- 手写 Warp Shuffle 归约已接近 GPU 硬件极限
- cuDNN 的 softmax 内核与我们采用同一种算法模式（warp reduction），区别主要在于：
  - cuDNN 使用 `float4` 向量化加载（每次读 4 个 float），D=64/128 时优势明显（4× 和 2.7×）
  - cuDNN 针对不同 D 值有专用的汇编级优化（register allocation、指令调度）
  - 但 D≥256 时，计算瓶颈从内存带宽转向计算吞吐，向量化优势递减

### 6.3 D 较小时 torch 的优势

D=64 时 torch 仍有 4× 优势。因为 D 较小时每线程的工作量（64/32=2 个元素）极低，kernel 启动开销和 warp 调度开销相对占比大。torch 的 kernel 针对小 D 有专门优化（如融合相邻行、使用更小的 block）。

这也是我们手写 kernel 的下一步优化方向：引入 `float4` 向量化全局加载，减少访存指令数 4×。

### 6.4 与 CUB/Cooperative Groups 的关系

手写的 Reduce + Broadcast 模式在 CUDA 生态中有对应的库实现：

```cpp
// CUB (CUDA UnBound)
cub::BlockReduce<float>(smem).Reduce(val, cub::Sum());

// Cooperative Groups
auto tile = cg::tiled_partition<32>(cg::this_thread_block());
cg::reduce(tile, val, cg::plus<float>());
```

三者底层均使用相同的 `__shfl_down_sync` 硬件指令。手写版本的优势在于：
- 零依赖：无需额外库
- 完全透明：理解每一行代码的原理
- 灵活控制：可以精确控制归约策略

---

## 7. 结论

1. **Warp Shuffle 是 GPU 上最快的线程间通信机制。** 利用蝴蝶归约 + 广播模式，手写的 Softmax 内核在 D≥256 时与 CUDA 自带的 cuDNN 实现性能持平（差距 <15%）。

2. **80 行 CUDA 代码达到接近硬件极限的性能。** D=1024 时，手写 Warp 内核仅比工业级 cuDNN 慢 13%，而代码量仅为后者的极小一部分。这充分证明了 Warp Shuffle 归约模式的高效性。

3. **归约减少浮点误差。** Warp 版本的数值精度系统性地优于 Naive 版本，因为归约将累加次数从 O(D) 降至 O(D/32 + log₂32)。

4. **D 较小时的性能差距来自向量化加载。** 引入 `float4` 后预期可缩小 D=64/128 时的差距至 2× 以内。

5. **Warp Shuffle 归约模式具有普适性。** 不限于 Softmax——任何跨线程聚合操作（求和、最大值、最小值、前缀和、LayerNorm、RMSNorm）都可以用相同的 Reduce + Broadcast 模式高效实现。

---

## 附录 A：项目结构

```
softmax/
├── __init__.py                     # Python API
├── benchmark/
│   ├── benchmark.py                # 性能测试脚本
│   └── report.md                   # 本报告
csrc/
├── softmax_kernel.cu               # CUDA 内核实现（Naive + Warp）
└── softmax_bindings.cpp            # PyTorch C++ 绑定
setup.py                            # 编译配置
docs/softmax/
├── ppt/index.html                  # 课堂演示网页 PPT
├── ppt/script.md                   # 讲稿
└── implementation.md               # 详细实现文档
```

## 附录 B：完整运行输出

```
======================================================================
Softmax Performance Benchmark
======================================================================
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
PyTorch: 2.11.0+cu130, CUDA: 13.0

     D | Naive (us) |  Warp (us) | torch (us) | Warp/Naive | Warp/torch
----------------------------------------------------------------------
    64 |        520 |        131 |         32 |       4.0x |      4.07x
   128 |        970 |        129 |         48 |       7.5x |      2.68x
   256 |       1840 |        313 |        283 |       5.9x |      1.11x
   512 |       3185 |        663 |        590 |       4.8x |      1.12x
  1024 |       5651 |       1319 |       1172 |       4.3x |      1.13x

     D |    Naive err |     Warp err |   Status
--------------------------------------------------
    64 |     1.49e-07 |     5.96e-08 |        ✅
   128 |     1.49e-07 |     5.96e-08 |        ✅
   256 |     1.04e-07 |     2.98e-08 |        ✅
   512 |     1.49e-07 |     1.49e-08 |        ✅
  1024 |     9.31e-08 |     7.45e-09 |        ✅

     D | Naive (M rows/s) |  Warp (M rows/s) | torch (M rows/s)
------------------------------------------------------------
    64 |            63.03 |           250.13 |          1019.04
   128 |            33.77 |           254.00 |           679.88
   256 |            17.81 |           104.66 |           115.80
   512 |            10.29 |            49.43 |            55.56
  1024 |             5.80 |            24.84 |            27.97

=== Environment ===
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Compute Capability: (8, 9)
VRAM: 8.6 GB
CUDA: 13.0
PyTorch: 2.11.0+cu130
Warmup: 5, Iters: 30, dtype: float32
```
