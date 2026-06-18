# CUDA Self-Attention 内核设计与优化 —— PPT 内容资料

> 每页内容按标题、要点、代码/图表组织，可直接用于制作 PPT。

---

## 第 1 页：封面

**标题：** 从零实现 CUDA Self-Attention 内核 —— 设计、优化与性能分析

**副标题：** 共享内存分块 · Online Softmax · Bank Conflict 消除

**信息：** 并行计算课程项目 | GPU: NVIDIA RTX 4060 | CUDA 13.3

---

## 第 2 页：项目目标

**我们要做什么：**
- 手写 CUDA C++ 内核，实现 Transformer 的核心算子 Self-Attention 前向计算
- 封装为 PyTorch 自定义算子（`attention_naive` / `attention_tiled`）
- 正确性交叉验证 + 性能基准测试

**输入输出：**
```
输入:  Q, K, V  ∈ ℝ^{B×H×N×D}    (B=2, H=4, N≤1024, D=64)
输出:  O         ∈ ℝ^{B×H×N×D}
```

**Self-Attention 公式（背景，一页带过）：**
```
S = Q × K^T                    →  [N × N] 相似度矩阵
P = softmax(S / √D)            →  逐行归一化为概率
O = P × V                      →  加权求和输出
```

**框内标注：** 本项目的核心不是公式，而是——**如何用 GPU 高效并行计算这个过程**。

---

## 第 3 页：GPU 并行计算基础（上）

**核心问题：** CPU 有 8-16 个大核心，GPU 有数千个小核心。怎么把计算任务拆给几千个线程？

**CUDA 线程层级：**
```
Grid (网格) ── 一次 kernel 启动的所有线程
├── Block 0 ── 一组可以协作的线程（共享内存、同步）
│   ├── Thread 0
│   ├── Thread 1
│   └── ...  (最多 1024 个)
├── Block 1
│   └── ...
└── Block M-1

关键: 32 个线程 = 1 个 Warp（GPU 调度基本单位）
```

**GPU 内存三层级：**
```
寄存器    → 1 cycle    → 仅本线程   → 容量: 255/线程
共享内存  → ~30 cycles → Block 内   → 容量: 48 KB/Block ← 我们的主角！
全局内存  → ~400 cycles → 所有线程   → 容量: 8 GB
```

**一句话设计哲学：** 把频繁复用的数据从全局内存搬到共享内存，让数据离计算单元尽可能近。

---

## 第 4 页：GPU 并行计算基础（下）—— 关键规则

**Block 内线程同步：**
```c
__syncthreads();  // 屏障：所有线程都到这才能继续
// 用处：确保共享内存数据全部加载完毕后，才开始读取
```

**Bank Conflict（铺垫后续核心优化）：**
- 共享内存分为 32 个 Bank，每 Bank 4 字节
- 同一 Warp 内多个线程访问同一 Bank → 串行化 → 32 倍慢
- `bank = (byte_addr / 4) % 32`

**示例（后面要用的知识点）：**
```
地址 0-31： Bank 0,1,2,...,31  (一行)
地址 32-63: Bank 0,1,2,...,31  (又一轮)
访问规则: 地址差 32 的倍数 = 同一个 Bank
```

---

## 第 5 页：内核一 —— Naive Kernel 设计原理

**设计定位：** 不追求性能，只保证正确。作为性能基准线。

**并行策略：**
```
Grid:  [B × H]             每个 Block 独立处理一个 (batch, head)
Block: [N]                 每个 Thread 独立处理一个 query 位置
```

**为什么这样分配？**
- 不同 (batch, head) 之间零数据依赖 → 一个 Block 管一个，完美并行
- 同一 head 内 N 个 query 相互独立 → 一个 Thread 管一个，继续并行
- 完全无线程间通信，无共享内存，实现最简单

**每个线程的三阶段计算：**
```
Phase 1:  for j=0..N-1:  scores[j] = (Q[i]·K[j]) / √D
Phase 2:  scores = safe_softmax(scores)
Phase 3:  for d=0..D-1:  O[i][d] = Σ_j scores[j] × V[j][d]
```

---

## 第 6 页：Naive Kernel 代码实现

```c
template <int D>
__global__ void attention_naive_kernel(
    const float* __restrict__ Q,    // [B,H,N,D]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, float scale)
{
    int bh = blockIdx.x;            // 我是哪个 (batch,head)?
    int i  = threadIdx.x;           // 我是哪个 query 位置?
    if (i >= N) return;
    int base = bh * N * D;          // 我的 Q/K/V 数据起始地址

    // == Phase 1: 算 N 个点积，存到局部数组 ==
    float scores[1024];
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < D; d++)
            dot += Q[base + i*D + d] * K[base + j*D + d];
        scores[j] = dot * scale;
    }

    // == Phase 2: Safe Softmax ==
    float max_val = scores[0];
    for (int j = 1; j < N; j++)
        max_val = fmaxf(max_val, scores[j]);
    float sum_val = 0.0f;
    for (int j = 0; j < N; j++) {
        scores[j] = __expf(scores[j] - max_val);
        sum_val += scores[j];
    }
    float inv_sum = 1.0f / sum_val;
    for (int j = 0; j < N; j++) scores[j] *= inv_sum;

    // == Phase 3: 加权求和 ==
    for (int d = 0; d < D; d++) {
        float out = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < N; j++)
            out += scores[j] * V[base + j*D + d];
        O[base + i*D + d] = out;
    }
}
```

**关键特征：** 无 `__shared__`、无 `__syncthreads()`、每个数据元素被重复读 N 次全局内存。

---

## 第 7 页：Naive Kernel 的性能瓶颈分析

**数据访问量分析（N=1024, D=64，以单个线程为例）：**

```
Q[i][d]:  每个点积读一次，j 循环 N 次 → N×D = 65,536 次全局内存读
K[j][d]:  同上                              → 65,536 次
V[j][d]:  输出 D 维，每维遍历 N 个 score  → D×N = 65,536 次
─────────────────────────────────────────────────────
单线程总计 ≈ 20 万次全局内存访问
B×H=8, N=1024 → 8192 个线程 → 总共 ≈ 16 亿次全局访问
```

**根本问题：** 每个数据元素（Q/K/V 的每个浮点数）被反复从全局内存读取，没有任何复用。全局内存延迟 ~400 cycles，计算能力完全浪费在等待数据上。

**图示建议：** 画一个线程反复从"全局内存"方块拉箭头的示意图，标注"每个元素被读 N 次"。

---

## 第 8 页：内核二 —— Tiled Kernel 设计原理（核心章节开始）

**核心思想：分块 (Tiling)**

把 N×D 的大矩阵切成 32×D 的小块（Tile），每次把一个 Tile 加载到共享内存，Block 内的 32 个线程共用这份数据。

```
         D=64
    ┌──────────────┐
    │  Tile 0      │ ← 32 行，加载到共享内存
    ├──────────────┤   32 个线程各加载 1 行
  N │  Tile 1      │
    ├──────────────┤
    │  ...         │
    └──────────────┘
```

**并行策略变化：**
```
Naive:                    Tiled:
Grid:  [B×H]              Grid:  [ceil(N/32), B×H]   ← 多了一维！
Block: [N]                Block: [32]                ← 固定 32 线程
```

**总结：**
- x 维度 = 处理哪个 Tile 区间的 query
- y 维度 = 处理哪个 (batch, head)
- 32 个线程协作：一起加载数据到共享内存，然后各自算自己的 query

---

## 第 9 页：Tiled Kernel 共享内存布局

**共享内存分配（每个 Block 内）：**
```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Q_smem              │ K_smem              │ V_smem              │
│ [32 × 65] float     │ [32 × 65] float     │ [32 × 65] float     │
│ = 8,320 bytes       │ = 8,320 bytes       │ = 8,320 bytes       │
└─────────────────────┴─────────────────────┴─────────────────────┘
                总大小: 24,960 bytes ≈ 24.4 KB < 48 KB ✅

注意: 为什么是 65 不是 64？← 这是后面 Bank Conflict 优化的关键！
```

**线程分工：**
```
我是线程 ti (0-31)：
1. 加载 Q 的第 global_i 行 → Q_smem[ti]
2. 主循环（遍历所有 K 的 Tile）：
   a. 加载 K 的第 (j_start+ti) 行 → K_smem[ti]
   b. 加载 V 的第 (j_start+ti) 行 → V_smem[ti]
   c. __syncthreads()  ← 等 32 个线程都加载完
   d. 我的 Q_smem[ti] 与 K_smem 每一行做点积
   e. Online Softmax 更新中间状态
   f. __syncthreads()  ← 等全部算完，准备下个 Tile
3. 最终结果 O_local / d_ 写到全局内存
```

---

## 第 10 页：技术一 —— Online Softmax 算法

**问题：** 分块后不能一次性算 Softmax——Tile 1 不知道 Tile 2 的最大值。

**方案：** 维护 running state：`(m, d, O)`，每来一个新 Tile 增量更新。

**算法推导（以 3 个数分两批为例）：**

```
初始: m = -∞,  d = 0,  O = [0, 0]

Tile 0: [2.0, 5.0]
  m_new = max(-∞, 5.0) = 5.0
  scale = exp(-∞ - 5.0) = 0
  sum = exp(2.0-5.0) + exp(5.0-5.0) = 0.0498 + 1.0 = 1.0498
  d = 1.0498,  m = 5.0

Tile 1: [3.0]
  m_new = max(5.0, 3.0) = 5.0
  scale = exp(5.0 - 5.0) = 1.0            ← 最大值没变!
  sum = exp(3.0-5.0) = 0.1353
  d = 1.0498×1.0 + 0.1353 = 1.1851

最终: [0.0498, 1.0, 0.1353] / 1.1851 = [0.042, 0.844, 0.114]
验证: softmax([2,5,3])                    = [0.042, 0.844, 0.114] ✅
```

**关键洞察：** `scale = exp(m_old - m_new)` 只在**最大值发生变化**时才起作用；最大值不变时 scale=1，直接追加。

---

## 第 11 页：Online Softmax 代码实现

```c
// 每个 Tile 迭代中的 Online Softmax 更新（核心代码块）
float m_new = fmaxf(m, local_max);          // ① 更新全局最大值

float scale_old = __expf(m - m_new);        // ② 缩放因子
// m - m_new ≤ 0 → scale_old ∈ (0, 1]

if (m != -INFINITY) {                       // ③ 缩放旧输出（首次跳过）
    for (int d = 0; d < D; d++)
        O_local[d] *= scale_old;
}

float sum_exp = 0.0f;
for (int jl = 0; jl < TILE_N; jl++) {       // ④ 融合：exp + 累加
    float p = __expf(scores[jl] - m_new);   // 新 Tile 的概率
    sum_exp += p;
    #pragma unroll
    for (int d = 0; d < D; d++)
        O_local[d] += p * V_smem[jl * D_PAD + d];
}

d_ = d_ * scale_old + sum_exp;              // ⑤ 更新分母
m = m_new;
```

**对比常规 Softmax：**
| 方面 | 常规 Softmax | Online Softmax |
|------|-------------|----------------|
| 内存 | 需存完整 N×N scores | 只存 D 维 running state |
| 计算 | 一次性，不可分块 | 支持分块增量计算 |
| Tile 要求 | 不可拆分 | 天然适配 Tiling |

---

## 第 12 页：技术二 —— 共享内存 Bank Conflict 分析（本项目最核心优化）

**什么是 Bank Conflict？**

共享内存有 32 个 Bank（每 Bank 4 字节 = 1 float）。同一 Warp 的 32 个线程：
- 访问 32 个不同 Bank → 1 cycle ✅
- ≥2 个线程访问同一 Bank → 串行化，多倍延迟 ❌

**我们的代码：**

```c
// Thread ti 访问 Q_smem[ti * D + d]  (D=64)
```

**逐线程 Bank 编号计算：**
```
bank = (地址_in_float) % 32 = (ti × 64 + d) % 32
     = (ti × 0 + d) % 32                    ← 因为 64 % 32 = 0 !
     = d % 32

Thread 0:  bank =  d % 32 ─┐
Thread 1:  bank =  d % 32  ├── 全部相同！
Thread 2:  bank =  d % 32  │
  ...                       │
Thread 31: bank =  d % 32 ─┘   32-way conflict！最坏情况！
```

**后果：** 本应 1 cycle 的共享内存读写，实际需要 32 cycles——单这一项损失 32 倍性能。

**图示建议：** 画 32 个线程的箭头全部指向同一个 Bank 格子的图。

---

## 第 13 页：Bank Conflict 修复 —— D_PAD 技术

**方案：** 共享内存行宽度从 64 改为 65（D_PAD = D + 1）

```c
constexpr int D_PAD = D + 1;   // 65，多了 1 列 padding
```

**修复后的 Bank 编号：**
```
bank = (ti × 65 + d) % 32
     = (ti × (64+1) + d) % 32
     = (ti × 1 + d) % 32                    ← 因为 64%32=0, 65%32=1
     = (ti + d) % 32

Thread 0:  bank = (0+d) % 32 = d % 32        → Bank d
Thread 1:  bank = (1+d) % 32 = (d+1) % 32    → Bank d+1  (不同！)
Thread 2:  bank = (2+d) % 32 = (d+2) % 32    → Bank d+2  (不同！)
  ...
Thread 31: bank = (31+d) % 32 = (d+31) % 32  → Bank d+31 (不同！)

全部 32 个线程命中不同 Bank → 0 conflict → 1 cycle 完成 ✅
```

**为什么要 65 而不是其他？**
- 核心条件是 `stride % 32 ≠ 0`（不能被 32 整除）
- D=64 是 32 的倍数 → conflict；D+1=65 与 32 互质 → 零 conflict
- 其他可选值：33, 34,... 但 D+1 最省内存（仅多 1 列）

---

## 第 14 页：Bank Conflict 修复的代码改动

**改动点 1：共享内存布局**

```c
// 修复前                                       // 修复后
float* Q_smem  = smem;                          float* Q_smem  = smem;
float* KV_smem = smem + TILE_N * D;     →       float* K_smem  = smem + TILE_N * D_PAD;
                                                float* V_smem  = smem + 2 * TILE_N * D_PAD;
```

**改动点 2：所有共享内存访问**

```c
// 修复前                                       // 修复后
Q_smem[ti * D + d]                      →       Q_smem[ti * D_PAD + d]
KV_smem[jl * D + d]                     →       K_smem[jl * D_PAD + d]
//                                               V_smem[jl * D_PAD + d]

// 注意：全局内存访问不需要改！
Q[base + global_i * D + d]              →       不变（全局内存布局仍是 [N, D=64]）
```

**改动点 3：Launch 配置**

```c
// 修复前                    // 修复后
2 * TILE_N * D * sizeof(float)  →  3 * TILE_N * D_PAD * sizeof(float)
// 16,384 bytes                     // 24,960 bytes（仍 < 48 KB）
```

**关键设计要点：** 共享内存 stride (65) 与全局内存 stride (64) 是**独立的**——padding 只加在共享内存，不改数据语义（循环 `d < D` 只遍历前 64 列有效数据）。

---

## 第 15 页：技术三 —— 其他工程优化

**优化 1：K/V 同时加载，减少同步点**

```c
// 优化前（每次 Tile 迭代 4 个 __syncthreads）:
K_load → __syncthreads() → compute → __syncthreads()
→ V_load → __syncthreads() → accumulate → __syncthreads()

// 优化后（每次 Tile 迭代 2 个 __syncthreads）:
K_load + V_load（并行写不同区域）→ __syncthreads()
→ compute + accumulate（exp 与 O 累加融合为单循环）→ __syncthreads()

// 每个 Tile 节省 2 次同步 ≈ 节省 600+ cycles
```

**优化 2：Template D 编译期常量**

```c
template <int D>               // D 成为编译期常量
__global__ void attention_tiled_kernel(...) {
    // 内层循环 #pragma unroll 完全展开，零循环开销
    for (int d = 0; d < D; d++) { ... }
}
// Launch 时: attention_tiled_kernel<64><<<grid, block, smem>>>(...);
```

**优化 3：循环融合**

```c
// 优化前: 3 次遍历 jl
// Pass 1: dot product → scores[jl]
// Pass 2: exp(scores[jl] - m_new) → scores[jl]
// Pass 3: O_local += scores[jl] * V

// 优化后: Pass 2+3 融合
for (int jl = 0; jl < TILE_N; jl++) {
    float p = __expf(scores[jl] - m_new);  // 即时 exp
    sum_exp += p;
    for (int d = 0; d < D; d++)
        O_local[d] += p * V_smem[jl][d];   // 即时累加，无需存回
}
```

---

## 第 16 页：Tiled Kernel 完整代码（主循环）

```c
template <int D>
__global__ void attention_tiled_kernel(
    const float* __restrict__ Q, K, V, float* __restrict__ O,
    int N, float scale)
{
    constexpr int D_PAD = D + 1;        // ← Bank Conflict 修复
    int bh = blockIdx.y, ti = threadIdx.x;
    int global_i = blockIdx.x * TILE_N + ti;
    if (global_i >= N) return;
    int base = bh * N * D;

    extern __shared__ float smem[];
    float* Q_smem = smem;                              // [TILE_N, D_PAD]
    float* K_smem = smem + TILE_N * D_PAD;             // [TILE_N, D_PAD]
    float* V_smem = smem + 2 * TILE_N * D_PAD;         // [TILE_N, D_PAD]

    // 加载 Q tile
    for (int d = 0; d < D; d++)
        Q_smem[ti * D_PAD + d] = Q[base + global_i * D + d];
    __syncthreads();

    float m = -INFINITY, d_ = 0.0f;
    float O_local[D] = {0};

    // ============ 主循环：逐 Tile 处理 K/V ============
    for (int j_start = 0; j_start < N; j_start += TILE_N) {
        int j_global = j_start + ti;

        // Step 1: 同时加载 K 和 V tile（写入不同共享内存区域，无需中间同步）
        if (j_global < N) {
            for (int d = 0; d < D; d++) {
                K_smem[ti * D_PAD + d] = K[base + j_global * D + d];
                V_smem[ti * D_PAD + d] = V[base + j_global * D + d];
            }
        }
        __syncthreads();  // ← 唯一的数据加载同步点

        // Step 2: 计算当前 Tile 的局部 scores
        float local_max = -INFINITY, scores[TILE_N];
        for (int jl = 0; jl < TILE_N; jl++) {
            if (j_start + jl >= N) { scores[jl] = -INFINITY; continue; }
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++)
                dot += Q_smem[ti*D_PAD + d] * K_smem[jl*D_PAD + d];
            scores[jl] = dot * scale;
            local_max = fmaxf(local_max, scores[jl]);
        }

        // Step 3: Online Softmax 更新 + O 累加（融合）
        float m_new = fmaxf(m, local_max);
        float scale_old = __expf(m - m_new);
        if (m != -INFINITY)
            for (int d = 0; d < D; d++) O_local[d] *= scale_old;
        float sum_exp = 0.0f;
        for (int jl = 0; jl < TILE_N; jl++) {
            if (j_start + jl >= N) break;
            float p = __expf(scores[jl] - m_new);  sum_exp += p;
            #pragma unroll
            for (int d = 0; d < D; d++)
                O_local[d] += p * V_smem[jl*D_PAD + d];
        }
        d_ = d_ * scale_old + sum_exp;  m = m_new;
        __syncthreads();  // ← 准备下个 Tile
    }

    // 最终归一化输出
    float inv_d = 1.0f / d_;
    for (int d = 0; d < D; d++)
        O[base + global_i*D + d] = O_local[d] * inv_d;
}
```

---

## 第 17 页：PyTorch 封装 —— 从 CUDA 到 Python

**C++ 绑定层 (`csrc/attention_bindings.cpp`)：**
```cpp
torch::Tensor attention_forward_tiled(
    torch::Tensor Q, K, V, float scale)
{
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
    auto dims = Q.sizes();
    int B=dims[0], H=dims[1], N=dims[2], D=dims[3];
    auto O = torch::empty_like(Q);

    // 模板分发 → 调用编译期优化的 kernel
    attention_tiled_kernel_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(),
        N, D, scale, B * H);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward_tiled", &attention_forward_tiled);
}
```

**Python API (`attention/__init__.py`)：**
```python
from . import _C

def attention_tiled(q, k, v, scale=None):
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5      # 默认 1/√D
    return _C.attention_forward_tiled(q, k, v, float(scale))
```

**用户调用：**
```python
import torch
from attention import attention_tiled
q = torch.randn(2, 4, 1024, 64, device='cuda')
out = attention_tiled(q, q, q)   # 就像调 PyTorch 原生函数一样
```

---

## 第 18 页：性能实验数据

**测试环境：** RTX 4060 Laptop (SM 8.9, 24 SM, 8GB) | B=2, H=4, D=64, float32

### 表 1：绝对耗时（μs，越小越好）

| N | Naive (μs) | Tiled (μs) | PyTorch (μs) | Tiled/Naive |
|---|------------|------------|--------------|-------------|
| 32 | 316.5 | 139.0 | 243.3 | 2.28× |
| 64 | 544.1 | 306.5 | 222.1 | 1.78× |
| 128 | 1,030.5 | 374.6 | 159.7 | 2.75× |
| 256 | 2,891.9 | 465.4 | 148.2 | 6.21× |
| 512 | 8,327.5 | 1,351.0 | 206.3 | 6.16× |
| 1024 | 47,779.8 | 4,684.5 | 1,464.3 | 10.20× |

### 表 2：Tiled/Naive 加速比（优化前后对比）

| N | 原始 Tiled | 优化后 Tiled | 提升 |
|---|-----------|-------------|------|
| 256 | 2.01× | **6.21×** | 3.1× |
| 512 | 1.69× | **6.16×** | 3.6× |
| 1024 | 2.19× | **10.20×** | 4.7× |

**建议：** 用柱状图展示 N=1024 的三组数据（Naive / 原始 Tiled / 优化后 Tiled）。

---

## 第 19 页：优化历程与各技术贡献

**逐步优化效果追踪：**

| 优化阶段 | N=1024 Tiled 耗时 | vs Naive 加速比 |
|----------|-------------------|----------------|
| 原始版本（无优化） | 15,114 μs | 2.19× |
| + Bank Conflict 修复 | 4,543 μs | 7.30× |
| + K/V 同时加载 + 循环融合 | 4,685 μs | 10.20× |

**各技术贡献分析：**
- **Bank Conflict 修复**：最大单一收益项（3.3×），消除了 32-way 串行化
- **K/V 同时加载**：中等 N 提升 10-20%，大 N 时 occupancy 限制收益递减
- **Template 编译期展开**：消除内层循环开销，但对标量计算收益有限
- **循环融合**：减少中间数组读写，间接降低寄存器压力

**为什么与 PyTorch 仍有 3-4× 差距？**
- PyTorch 底层调 cuBLAS 矩阵乘法，使用 **Tensor Cores**（专用矩阵乘法硬件）
- cuBLAS 是 **Warp 级别** 协同计算（32 线程合作算一块），我们是各线程独立做点积
- 手写标量点积 vs 硬件 Tensor Core MMA 指令 ≈ 8× 单精度算力差距

---

## 第 20 页：正确性验证

**验证方法：** 双重对比
1. 与 PyTorch `scaled_dot_product_attention`（math backend，禁用 FlashAttention）对比
2. Tiled 与 Naive 交叉验证

| N | Naive vs PyTorch | Tiled vs PyTorch | Tiled vs Naive |
|---|-----------------|------------------|----------------|
| 32 | 4.77×10⁻⁷ ✅ | 7.15×10⁻⁷ ✅ | 4.77×10⁻⁷ ✅ |
| 64 | 4.77×10⁻⁷ ✅ | 5.36×10⁻⁷ ✅ | 5.36×10⁻⁷ ✅ |
| 128 | 4.77×10⁻⁷ ✅ | 5.96×10⁻⁷ ✅ | 6.56×10⁻⁷ ✅ |
| 256 | 4.17×10⁻⁷ ✅ | 4.77×10⁻⁷ ✅ | 4.47×10⁻⁷ ✅ |
| 512 | 4.32×10⁻⁷ ✅ | 7.45×10⁻⁷ ✅ | 5.66×10⁻⁷ ✅ |
| 1024 | 4.17×10⁻⁷ ✅ | 7.15×10⁻⁷ ✅ | 6.56×10⁻⁷ ✅ |

**结论：** 所有偏差 < 10⁻⁶，远优于 10⁻³ 容限。优化未引入任何精度损失。

---

## 第 21 页：总结 —— 三条核心经验

**1. 共享内存是 GPU 编程最重要的优化杠杆**
- 把频繁复用的数据从"图书馆"（全局内存，400 cycles）搬到"课桌"（共享内存，30 cycles）
- 数据复用率从 1/N 提升到接近 1，减少 90%+ 的全局内存访问

**2. Bank Conflict 是隐形的性能杀手**
- 代码完全正确，输出完全一致，编译零警告
- 但 32 个线程排队等同一个 Bank → 实际慢 32 倍
- 诊断方法：分析 stride 是否被 32 整除 → `D_PAD = D + 1` 消除
- **教训：理解硬件才能写出高性能代码**

**3. 高性能 GPU 内核 = 算法 + 硬件特性的共同设计**
- Online Softmax：算法创新让分块计算成为可能
- D_PAD Padding：硬件知识（32 banks × 4 bytes）指导内存布局
- 模板编译期展开：语言特性服务于代码生成质量

---

## 第 22 页：致谢 & Q&A

- 项目仓库：`github.com/Xhlging/cuda-self-attention`
- 完整教程：`docs/tutorial.md`
- 本 PPT 源：`docs/slides.md`

**欢迎提问！**
