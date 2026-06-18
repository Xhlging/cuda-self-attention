# CUDA Softmax 详细实现

## 目录

1. [算法定义](#1-算法定义)
2. [Naive 实现：每线程串行](#2-naive-实现每线程串行)
3. [Warp 并行归约原理](#3-warp-并行归约原理)
4. [Warp Softmax 实现](#4-warp-softmax-实现)
5. [Launch 配置与模板分发](#5-launch-配置与模板分发)
6. [PyTorch 绑定](#6-pytorch-绑定)
7. [性能数据](#7-性能数据)

---

## 1. 算法定义

Softmax 的数学公式：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Safe Softmax 版本**（防止 exp 溢出）：

```
m = max_j x_j                           // 第一步: 找最大值
softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)   // 第二步: 减最大值再 exp
```

为什么"safe"？因为直接算 `exp(100)` 会超过 float 的最大值（约 3.4×10³⁸）。所有值先减去最大值后，最大值变成 `exp(0) = 1`，永远不会溢出。

---

## 2. Naive 实现：每线程串行

### 设计思路

每个线程独立处理一行数据，串行做三遍扫描。

**并行配置：**
- Grid: `[ceil(rows / 256)]` — 多少个 Block
- Block: `[256]` — 每个 Block 256 个线程
- 每个线程处理一行

### 完整代码

```c
template <int D>
__global__ void softmax_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int stride)
{
    // 1. 确定我处理哪一行
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* x = input + row * stride;   // 输入指针定位
    float* y = output + row * stride;        // 输出指针定位

    // 2. 第一遍: 找该行的最大值 (Safe Softmax 的需要)
    float max_val = x[0];
    for (int i = 1; i < D; i++)
        max_val = fmaxf(max_val, x[i]);

    // 3. 第二遍: 算 exp(x - max) 同时求和
    float sum_val = 0.0f;
    for (int i = 0; i < D; i++) {
        y[i] = __expf(x[i] - max_val);   // __expf 是 CUDA 快速 exp 指令
        sum_val += y[i];
    }

    // 4. 第三遍: 归一化 (除以总和)
    float inv_sum = 1.0f / sum_val;
    for (int i = 0; i < D; i++)
        y[i] *= inv_sum;
}
```

### 时间复杂度

每个线程串行遍历 D 个元素三次 → **O(3D)**。D=1024 时，每线程 3072 次操作。

### 瓶颈

线程之间**完全没有协作**。即便同一行有 32 个线程可用，也没有利用。

---

## 3. Warp 并行归约原理

### 3.1 什么是 Warp

GPU 硬件把 32 条线程编成一组同时调度，称为一个 **Warp**（线程束）。

```
Block [32 线程] = 1 个 Warp
Thread 0 ─┐
Thread 1  ├── 同一个 Warp
...       │    可以互相直接交换数据
Thread 31─┘
```

### 3.2 Warp Shuffle 指令

同一 Warp 内的线程可以通过专用指令**直接交换寄存器数据**，不需要共享内存：

```c
// __shfl_down_sync(mask, val, offset)
// 作用: 线程 i 收到线程 (i+offset) 的 val 值

float my_val = ...;

// offset=16: 线程0←线程16, 线程1←线程17, ..., 线程15←线程31
float merged = __shfl_down_sync(0xFFFFFFFF, my_val, 16);

// __shfl_sync(mask, val, src_lane)
// 作用: 所有线程收到线程 src_lane 的 val 值

float broadcast = __shfl_sync(0xFFFFFFFF, val, 0);  // 广播线程0的值
```

**参数说明：**
- `0xFFFFFFFF`：32 位全是 1 的掩码，表示 Warp 内全部 32 条线程都参与
- `offset`：shuffle 的步长
- `src_lane`：广播源线程的编号

### 3.3 蝴蝶归约（Butterfly Reduction）

目标：把 32 个值合并成 1 个（求和或求最大值），只需 log₂(32) = 5 步。

```
初始:    t0  t1  t2  t3  ...  t16 t17 ... t30 t31    (32 个值)

offset=16: t0=t0+t16  t1=t1+t17 ... t15=t15+t31       (16 个结果)
offset=8:  t0=t0+t8   t1=t1+t9  ... t7=t7+t15         (8 个结果)
offset=4:  t0=t0+t4   ... t3=t3+t7                    (4 个结果)
offset=2:  t0=t0+t2   t1=t1+t3                        (2 个结果)
offset=1:  t0=t0+t1                                    (1 个结果 ✅)
```

每轮 offset 折半，`__shfl_down_sync` 从下家拿值过来和自己合并。5 轮后，只有线程 0 手里有最终结果。

### 3.4 关键一步：广播

归约后只有线程 0 有正确结果。需要把结果广播给所有 32 个线程：

```c
// 归约: 只在 lane 0 产生正确结果
for (int offset = 16; offset > 0; offset /= 2)
    sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);

// 广播: 把 lane 0 的值复制给所有线程
sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);
```

---

## 4. Warp Softmax 实现

### 核心思路

1. **一行一个 Warp**：一行 D 维向量，一个 Block（恰好 32 线程 = 1 Warp）处理
2. **分治**：每个线程处理 D/32 个元素（跨步访问），而非全部 D 个
3. **归约**：用 warp shuffle 把 32 个局部结果合并成 1 个全局结果
4. **广播**：把全局结果分发给所有线程，各自归一化

### 完整代码（带逐行注释）

```c
template <int D>
__global__ void softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int stride)
{
    // ============================================================
    // 初始化: 确定每个线程负责的行和它在 Warp 内的编号
    // ============================================================
    int row = blockIdx.x;               // 一行一个 Block
    if (row >= rows) return;

    const float* x = input + row * stride;  // 指针定位到第 row 行
    float* y = output + row * stride;
    int lane = threadIdx.x;             // 线程在 Warp 内的编号 (0~31)

    // ============================================================
    // Step 1: 每个线程处理 D/32 个元素，找局部最大值
    //
    // 数据分配方式：跨步访问
    //   lane 0  → x[0],  x[32], x[64], ...
    //   lane 1  → x[1],  x[33], x[65], ...
    //   ...
    //   lane 31 → x[31], x[63], x[95], ...
    //
    // 好处：相邻线程访问相邻地址，全局内存合并访问（coalesced）
    // ============================================================
    float max_val = -INFINITY;
    for (int i = lane; i < D; i += 32)
        max_val = fmaxf(max_val, x[i]);

    // ============================================================
    // Step 2: 蝴蝶归约，合并 32 个局部 max → 1 个全局 max
    //
    // offset=16: 线程 0~15 各收到下家(16~31)的值，取 max
    // offset=8:  线程 0~7 各收到下家(8~15)的值，取 max
    // offset=4:  线程 0~3 ...
    // offset=2:  线程 0~1 ...
    // offset=1:  只有线程 0 拿到最终结果
    // ============================================================
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));

    // 广播: 线程 0 手里的全局最大值 → 复制给所有 32 条线程
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // ============================================================
    // Step 3: 每个线程再处理 D/32 个元素，算 exp + 求和
    //
    // 同一线程既算 exp 值（写入输出），又累加局部 sum
    // ============================================================
    float sum_val = 0.0f;
    for (int i = lane; i < D; i += 32) {
        float e = __expf(x[i] - max_val);   // 安全 exp（已减去全局 max）
        y[i] = e;                           // 存到输出数组
        sum_val += e;                       // 累加局部 sum
    }

    // 蝴蝶归约 sum: 同 Step 2
    for (int offset = 16; offset > 0; offset /= 2)
        sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);

    // 广播 sum: 同 Step 2
    sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);

    // ============================================================
    // Step 4: 每个线程归一化自己负责的元素
    // ============================================================
    float inv_sum = 1.0f / sum_val;
    for (int i = lane; i < D; i += 32)
        y[i] *= inv_sum;
}
```

### 每一步详解

#### Step 1: 跨步访问找局部最大值

```
为什么要跨步？
- 不是 lane 0 管 x[0..D/32-1], lane 1 管 x[D/32..2*D/32-1]
- 而是 lane 0 管 x[0],x[32],x[64]..., lane 1 管 x[1],x[33],x[65]...

原因：相邻线程访问相邻全局内存地址 → 一次内存事务能服务多个线程
       这是 GPU 的"合并访问"（coalesced access）原则
```

#### Step 2 & 3: 蝴蝶归约 + 广播

```
归约过程（以求最大值为例）：

初始:  每个 lane 有局部 max
       t0(m0) t1(m1) t2(m2) ... t31(m31)

Round 1 (offset=16):
  t0 = max(m0, __shfl_down(t0, 16)) = max(m0, m16)
  t1 = max(m1, __shfl_down(t1, 16)) = max(m1, m17)
  ...
  t15 = max(m15, __shfl_down(t15, 16)) = max(m15, m31)

Round 2 (offset=8):
  t0 = max(t0, __shfl_down(t0, 8))   // 现在 t0 覆盖了 {m0,m16,m8,m24}
  ...

Round 5 (offset=1):
  t0 = max(t0, __shfl_down(t0, 1))   // t0 覆盖了全部 32 个局部值

最后: __shfl_sync(..., max_val, 0) → 所有人 = t0 的值
```

#### Step 4: 归一化

所有线程现在都有了正确的全局 `sum_val`，各自把第一步存的 `y[i]`（=exp 值）除以 `sum_val` 即可。

### 时间复杂度对比

| | Naive | Warp |
|---|---|---|
| 每线程遍历次数 | D × 3 | (D/32) × 3 + 5×2 次 shuffle |
| D=1024 时 | 3072 次 | 96 次 + 10 次 shuffle |
| 并行度 | 1（串行） | 32（分治） |
| Grid | [rows/256] | [rows] |
| Block | [256] | [32] = 1 Warp |

---

## 5. Launch 配置与模板分发

```c
// ---- Naive Launch ----
void softmax_naive_launch(const float* input, float* output,
                          int rows, int D, int stride) {
    // 每个 Block 256 线程，每个线程处理 1 行
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;  // 向上取整

    // 模板分发: D 必须是编译期常量，用 switch 选择
    switch (D) {
        case 64:  softmax_naive_kernel<64><<<blocks, threads>>>(input, output, rows, stride); break;
        case 128: softmax_naive_kernel<128><<<blocks, threads>>>(input, output, rows, stride); break;
        case 256: softmax_naive_kernel<256><<<blocks, threads>>>(input, output, rows, stride); break;
        case 512: softmax_naive_kernel<512><<<blocks, threads>>>(input, output, rows, stride); break;
        case 1024:softmax_naive_kernel<1024><<<blocks, threads>>>(input, output, rows, stride); break;
        default:  fprintf(stderr, "Unsupported D=%d\n", D); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---- Warp Launch ----
void softmax_warp_launch(const float* input, float* output,
                         int rows, int D, int stride) {
    // 一行一个 Block，每个 Block 恰好 32 线程 = 1 Warp
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

**为什么用模板 `template <int D>`？**
- D 成为编译期常量，`for (int i = 0; i < D; i++)` 可以被编译器完全展开
- `#pragma unroll` 配合模板常量 → 最优代码生成

**为什么 Wrapper 用 switch？**
- 模板参数必须编译期确定，运行时 D 不能直接传入模板
- switch 分发已知的 D 值到对应的模板实例化

---

## 6. PyTorch 绑定

### C++ 绑定 (`csrc/attention_bindings.cpp`)

```cpp
// 前向声明 (实现在 .cu 文件)
void softmax_naive_launch(const float* input, float* output,
                          int rows, int D, int stride);
void softmax_warp_launch(const float* input, float* output,
                         int rows, int D, int stride);

// Torch 包装函数 - Naive
torch::Tensor softmax_forward_naive(torch::Tensor input) {
    input = input.contiguous();          // 确保内存连续
    auto sizes = input.sizes();
    int D = sizes.back();                // 最后一维 = 特征维度
    // 把前面所有维压平为 rows
    int rows = 1;
    for (size_t i = 0; i < sizes.size() - 1; i++)
        rows *= sizes[i];

    auto output = torch::empty_like(input);
    softmax_naive_launch(input.data_ptr<float>(), output.data_ptr<float>(),
                         rows, D, D);     // stride = D (无 padding)
    return output;
}

// Torch 包装函数 - Warp (结构相同)
torch::Tensor softmax_forward_warp(torch::Tensor input) { /* ... */ }

// 注册到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_forward_naive", &softmax_forward_naive);
    m.def("softmax_forward_warp", &softmax_forward_warp);
}
```

### Python API (`attention/__init__.py`)

```python
def softmax_naive(x):
    """Naive CUDA softmax (serial per-row, correctness baseline)."""
    return _C.softmax_forward_naive(x)


def softmax_warp(x):
    """Warp-reduction CUDA softmax.

    One warp per row. Uses __shfl_down_sync for parallel reduction
    across 32 threads, with __shfl_sync broadcast after reduction.

    Args:
        x: [*, D] float32 tensor, D ∈ {64, 128, 256, 512, 1024}
    Returns:
        [*, D] float32, softmax along last dim
    """
    return _C.softmax_forward_warp(x)
```

### 使用方式

```python
import torch
from attention import softmax_warp

# 支持任意维度: [B, D] / [B, N, D] / [B, H, N, D]
x = torch.randn(2, 4, 1024, 512, device='cuda')
out = softmax_warp(x)  # 沿最后一维做 softmax

# 与 PyTorch 对比
ref = torch.softmax(x, dim=-1)
assert (out - ref).abs().max() < 1e-5  # 误差 < 0.00001
```

---

## 7. 性能数据

**测试环境：** RTX 4060 Laptop · CUDA 13.3 · PyTorch 2.11 · rows=32768 · float32

| D | Naive (μs) | Warp (μs) | torch (μs) | Warp/Naive |
|---|-----------|-----------|------------|------------|
| 64 | 626 | 158 | 33 | 4.0× |
| 128 | 1110 | 150 | 53 | 7.4× |
| 256 | 2045 | 477 | 347 | 4.3× |
| 512 | 3117 | 727 | 602 | 4.3× |
| 1024 | 3179 | 657 | 594 | 4.8× |

**分析：**
- Warp 版本在 D≥128 时稳定比 Naive 快 4~7 倍
- 与 PyTorch 官方实现（torch.softmax）性能接近（D≥256 时基本持平）
- 加速来源：D 次操作分给 32 个线程 → 每线程只需 D/32 次
- D 较小时（≤64），torch 仍有优势（底层用了更激进的向量化优化）
