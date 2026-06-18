# CUDA Self-Attention 并行计算教程

> 面向大二计算机专业学生的课堂讲解材料  
> 前置知识：C/C++ 基础、线性代数基础、对 GPU 有初步概念

---

## 目录

1. [背景：为什么 Self-Attention 需要 GPU 加速](#1-背景为什么-self-attention-需要-gpu-加速)
2. [GPU 并行计算基础](#2-gpu-并行计算基础)
3. [Self-Attention 的数学公式](#3-self-attention-的数学公式)
4. [方案一：Naive Kernel（朴素实现）](#4-方案一naive-kernel朴素实现)
5. [方案二：Tiled Kernel（分块优化）](#5-方案二tiled-kernel分块优化)
6. [Online Softmax 算法详解](#6-online-softmax-算法详解)
7. [共享内存 Bank Conflict 与修复](#7-共享内存-bank-conflict-与修复)
8. [性能分析与对比](#8-性能分析与对比)
9. [课堂演示要点](#9-课堂演示要点)

---

## 1. 背景：为什么 Self-Attention 需要 GPU 加速

### 1.1 Self-Attention 是什么

Self-Attention（自注意力）是 Transformer 架构的核心，被 ChatGPT、Claude 等大语言模型广泛使用。它的作用是：**让序列中的每个词都能"看到"其他所有词，计算它们之间的关联强度**。

举个例子，句子 "The cat sat on the mat because it was tired" 中，"it" 指的是 "cat" 还是 "mat"？Self-Attention 通过计算词与词之间的相似度，自动学会 "it" 应该更关注 "cat"。

### 1.2 计算量有多大

输入是一个形状为 `[B, H, N, D]` 的张量：
- **B (Batch)**：一次处理多少句话
- **H (Head)**：多少个注意力头（每个头关注不同方面）
- **N (Sequence Length)**：句子长度（词的数量）
- **D (Head Dimension)**：每个词用多少维向量表示

Self-Attention 需要计算一个 `N × N` 的相似度矩阵。当 N=1024, D=64 时：
- 一个头就有 1024 × 1024 ≈ **100 万个**相似度要算
- 每个相似度需要 64 次乘加 → **6400 万次浮点运算**（仅一个头）
- B=2, H=4 时总共约 **5 亿次运算**

如果用 CPU 单线程串行计算，这需要几十毫秒甚至更久。而大模型（如 GPT-4）的 N 可以达到数万甚至数十万，必须用 GPU 并行加速。

---

## 2. GPU 并行计算基础

### 2.1 CPU vs GPU 的哲学差异

```
CPU: 少量强大核心（8-16 核），擅长复杂逻辑、分支预测
GPU: 数千个简单核心（RTX 4060 有 3072 个 CUDA 核心），擅长大规模数据并行
```

CPU 像一个博士团队（人少但个个精英），GPU 像一个小学生方阵（人多但每人只会简单运算）。Self-Attention 的计算恰好是"大量简单运算"，天然适合 GPU。

### 2.2 CUDA 的线程层级

CUDA 把线程组织成三个层级：

```
Grid（网格）
├── Block 0（线程块 0）
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   └── ...
└── Block M-1
```

- **Thread（线程）**：最小执行单元，每个线程执行相同的代码（kernel 函数）
- **Block（线程块）**：一组线程，可以通过**共享内存**通信和同步
- **Grid（网格）**：所有 Block 的集合，一次 kernel 启动的所有线程

**关键约束**：
- 一个 Block 最多 1024 个线程
- 同一个 Block 内的线程可以通过 `__syncthreads()` 同步
- 不同 Block 之间**不能**直接同步

### 2.3 GPU 的内存层级

这是理解本项目的**最关键知识点**：

| 内存类型 | 容量 | 延迟 | 作用域 | 类比 |
|----------|------|------|--------|------|
| **全局内存 (Global Memory)** | 大 (8GB) | ~400 cycles | 所有线程 | 图书馆（大但远） |
| **共享内存 (Shared Memory)** | 小 (48KB/Block) | ~30 cycles | 同一 Block | 课桌（小但近） |
| **寄存器 (Register)** | 极少 (255个/线程) | ~1 cycle | 单个线程 | 大脑（即刻访问） |

**核心优化思想**：把数据从"图书馆"（全局内存）搬到"课桌"（共享内存），让所有同学（线程）可以快速复用，而不是每次都跑去图书馆。

> 🔑 **类比理解**：假设 32 个同学要分别计算自己与 32 本书中每个章节的相似度。
> - **不用共享内存**：每个同学跑图书馆 32 次，每次翻一本书 → 32×32 = 1024 次图书馆往返
> - **用共享内存**：32 个同学一起把 32 本书搬到课桌上，每人只需去 1 次图书馆 → 32 次往返，之后都在课桌上快速翻阅

### 2.4 什么时候用共享内存

✅ **应该用**：同一份数据被 Block 内多个线程反复读取  
❌ **不应该用**：每个线程只访问自己的数据（没有复用）  
⚠️ **注意**：共享内存只有 48KB/Block，超出会导致 kernel 启动失败

---

## 3. Self-Attention 的数学公式

### 3.1 三步计算

给定输入 Q (Query), K (Key), V (Value)，形状都是 `[N, D]`（简化到一个头）：

```
Step 1: 计算相似度矩阵 S
  S = Q × K^T           → 形状 [N, N]
  S[i][j] = Q[i] · K[j] （向量点积）

Step 2: Softmax 归一化
  P[i][j] = exp(S[i][j] / √D) / Σ_k exp(S[i][k] / √D)
  目的：把相似度转成概率（每行加起来 = 1）

Step 3: 加权求和
  O = P × V             → 形状 [N, D]
  O[i] = Σ_j P[i][j] · V[j]
```

**为什么要除以 √D？** 当 D 很大时，点积的方差会很大，导致 softmax 的梯度消失。除以 √D 可以稳定训练。

### 3.2 Safe Softmax

直接计算 `exp(large_number)` 会溢出（float 最大约 3.4e38）。Safe Softmax 的做法：

```
m = max_j S[i][j]                        // 先找最大值
P[i][j] = exp(S[i][j] - m)               // 所有值减去最大值再 exp
P[i][j] /= Σ_k exp(S[i][k] - m)         // 归一化
```

减去最大值后，最大的 exp 值 = exp(0) = 1，永远不会溢出。

---

## 4. 方案一：Naive Kernel（朴素实现）

### 4.1 并行策略

```
Grid:  [B × H]           ← 每个 Block 处理一个 (batch, head)
Block: [N]               ← 每个 Thread 处理一个 query 位置
```

**问：为什么这样设计？**

因为不同的 (batch, head) 之间完全独立——处理第 0 个 batch 的第 0 个头，和处理第 1 个 batch 的第 3 个头，没有任何数据依赖。所以每个 Block 各管一个 (batch, head)，互不干扰。

而在一个 (batch, head) 内部，N 个 query 位置也是独立的——计算第 i 个 query 的输出不需要等第 j 个完成。所以每个线程各管一个 query 位置。

### 4.2 每个线程的工作

```c
// Thread i 的完整工作流程
float scores[1024];       // 局部数组，存 N 个相似度

// 阶段 1：计算 Q[i] 与所有 K[j] 的点积
for (j = 0; j < N; j++) {
    dot = 0;
    for (d = 0; d < D; d++) {
        dot += Q[i][d] * K[j][d];   // 每次乘加都读全局内存！
    }
    scores[j] = dot * scale;
}

// 阶段 2：Safe Softmax
max_val = max(scores[0..N-1]);
sum_val = 0;
for (j = 0; j < N; j++) {
    scores[j] = exp(scores[j] - max_val);
    sum_val += scores[j];
}

// 阶段 3：加权求和 O[i] = Σ scores[j] * V[j]
for (d = 0; d < D; d++) {
    out = 0;
    for (j = 0; j < N; j++) {
        out += scores[j] * V[j][d];
    }
    O[i][d] = out;
}
```

### 4.3 瓶颈分析

**问题：每个数据元素被重复读了太多次。**

```
Q[i][d]：被读了 N 次（内层 j 循环每次都要读）
K[j][d]：被读了 N 次（每个线程 i 都要读一遍所有 K）
V[j][d]：被读了 D 次（每个输出维度 d 都要读一遍所有 V）
```

以 N=1024, D=64 为例，一个线程要做：
- 读 Q：1024 × 64 = 65,536 次全局内存访问
- 读 K：1024 × 64 = 65,536 次
- 读 V：64 × 1024 × 64 ≈ 4,194,304 次

**总计约 430 万次全局内存访问**，每次约 400 个时钟周期。这就是为什么 Naive kernel 很慢。

> 🐌 **类比**：32 个同学，每个人都要跑图书馆 32 次。一天下来腿都跑断了，时间全花在路上了。

---

## 5. 方案二：Tiled Kernel（分块优化）

### 5.1 核心思想

把 `N × D` 的大矩阵切成 `32 × D` 的小块（tile），每个 tile 一次性加载到共享内存，然后在共享内存里反复使用。

```
原始 N×D 矩阵：
┌─────────────────────┐
│ Tile 0  (32×D)      │  ← 加载到共享内存
├─────────────────────┤
│ Tile 1  (32×D)      │  ← 覆盖或换区，加载下一个 tile
├─────────────────────┤
│ ...                 │
└─────────────────────┘
```

### 5.2 并行策略的变化

```
Naive:                    Tiled:
Grid:  [B×H]              Grid:  [ceil(N/32), B×H]   ← 多了一个维度！
Block: [N]                Block: [32]
```

**为什么 Grid 要多一个维度？**

Naive 中一个 Block 处理整个 N，一个线程处理一个 query。现在改用 tile，每个 Block 只处理 32 个 query，所以需要 `ceil(N/32)` 个 Block 来覆盖所有 N 个位置。

Block 的 x 维度对应 "哪个 tile 区间的 query"，y 维度对应 "哪个 (batch, head)"。

### 5.3 共享内存布局

```
共享内存 (一个 Block 内):
┌──────────────────┬──────────────────┬──────────────────┐
│ Q_smem           │ K_smem           │ V_smem           │
│ [32 × 65]        │ [32 × 65]        │ [32 × 65]        │
│ = 8,320 bytes    │ = 8,320 bytes    │ = 8,320 bytes    │
└──────────────────┴──────────────────┴──────────────────┘
总大小: 24,960 bytes < 48 KB ✅
```

**注意 65 而不是 64！** 这就引出了我们最重要的优化——Bank Conflict 修复（第 7 节详述）。

### 5.4 线程的分工

Block 内有 32 个线程。在线程的视角里：

```
我是线程 ti (0-31)，我的工作：
1. 把 Q 的第 global_i 行加载到 Q_smem[ti]（共享内存）
2. 主循环：遍历 K 的所有 tile：
   a. 把 K 的第 (tile_start + ti) 行加载到 K_smem[ti]
   b. 把 V 的第 (tile_start + ti) 行加载到 V_smem[ti]
   c. __syncthreads() ← 等所有 32 个线程都加载完
   d. 用 Q_smem[ti] 与 K_smem 的每一行做点积
   e. 用 Online Softmax 累加到 O_local
   f. __syncthreads() ← 等所有线程都算完，准备下一轮
3. 把 O_local 写回全局内存
```

### 5.5 数据流图

```
全局内存                          共享内存                   寄存器
─────────                        ────────                   ──────
Q [B,H,N,D]  ──加载──→  Q_smem [32,65]  ──读取──→  dot product
K [B,H,N,D]  ──加载──→  K_smem [32,65]  ──读取──→  dot product
                                              ↓
V [B,H,N,D]  ──加载──→  V_smem [32,65]  ──读取──→  O_local[D]
                                              ↓
                                          O_local[D] ──写回──→ O [B,H,N,D]
```

---

## 6. Online Softmax 算法详解

### 6.1 为什么需要 Online Softmax

常规 softmax 需要两遍扫描：
1. 第一遍找最大值
2. 第二遍算 exp 和归一化

但这需要把所有 N 个 score 都算出来并存着。N=1024 时一个线程要存 1024 个 float（4KB），在 shared memory 里太占地方。如果分 tile 计算，那常规 softmax 就没法做了——因为第一个 tile 不知道后面 tile 的最大值是多少。

**Online Softmax** 解决了这个问题：逐 tile 计算，同时维护"到目前为止"的最大值和归一化分母，每来一个新 tile 就更新一次。

### 6.2 算法推导

假设已经处理了前 k 个 score，维护了：
- `m`：当前最大值
- `d`：当前分母 = Σ exp(s_j - m)

来了一个新的 tile，里面有新 score `s'`：

```
Step 1: 更新最大值
  m_new = max(m, max(s'))

Step 2: 缩放旧状态（因为最大值变了，exp 的基准也变了）
  scale = exp(m - m_new)      // m - m_new ≤ 0，所以 scale ∈ (0, 1]
  d_scaled = d × scale         // 旧分母按新最大值缩放
  O_scaled = O × scale         // 旧输出按新最大值缩放

Step 3: 加入新 tile 的贡献
  d_new = d_scaled + Σ exp(s'_j - m_new)
  O_new = O_scaled + Σ exp(s'_j - m_new) × V'_j

Step 4: 更新状态，准备下一个 tile
  m = m_new, d = d_new, O = O_new
```

### 6.3 为什么等价于常规 softmax

设最终最大值 = M，处理完所有 tile 后：
- `d = Σ_j exp(s_j - M)`  （所有 score 的统一分母）
- `O = Σ_j exp(s_j - M) × V_j`

最后 `O = O / d`，结果与一次性 softmax 完全一致。

### 6.4 数值例子

假设有 3 个 score: [2.0, 5.0, 3.0]，分两批处理：

```
初始: m = -∞, d = 0, O = 0

Tile 1: [2.0, 5.0]
  m_new = max(-∞, 5.0) = 5.0
  scale = exp(-∞ - 5.0) = 0
  sum_exp = exp(2.0-5.0) + exp(5.0-5.0) = 0.0498 + 1.0 = 1.0498
  d = 0 × 0 + 1.0498 = 1.0498
  m = 5.0

Tile 2: [3.0]
  m_new = max(5.0, 3.0) = 5.0     // 最大值没变
  scale = exp(5.0 - 5.0) = 1.0
  sum_exp = exp(3.0 - 5.0) = 0.1353
  d = 1.0498 × 1.0 + 0.1353 = 1.1851
  m = 5.0

最终 softmax: [0.0498, 1.0, 0.1353] / 1.1851 = [0.042, 0.844, 0.114]
验证常规 softmax: softmax([2,5,3]) = [0.042, 0.844, 0.114] ✅
```

### 6.5 在代码中的体现

```c
// 每个 tile 迭代中的 online softmax 更新
float m_new = fmaxf(m, local_max);           // 更新全局最大值
float scale_old = __expf(m - m_new);          // 缩放因子

for (int d = 0; d < D; d++) {
    O_local[d] *= scale_old;                  // 缩放旧输出
}

float sum_exp = 0.0f;
for (int jl = 0; jl < TILE_N; jl++) {
    float p = __expf(scores[jl] - m_new);     // 新 tile 的概率
    sum_exp += p;
    for (int d = 0; d < D; d++) {
        O_local[d] += p * V_smem[jl][d];      // 累加新贡献
    }
}

d_ = d_ * scale_old + sum_exp;                // 更新分母
m = m_new;
```

---

## 7. 共享内存 Bank Conflict 与修复

这是本项目**最重要、最精彩**的优化！

### 7.1 什么是 Bank Conflict

CUDA 的共享内存被分成 **32 个 bank**，每个 bank 4 字节（一个 float）。32 个 bank 可以同时服务 32 个线程（一个 warp）。

```
Bank 布局（每一列是一个 bank）：
Bank:  0    1    2   ...   31
      ┌────┬────┬────┬────┬────┐
      │ 0  │ 1  │ 2  │... │ 31 │  ← 地址 0-31
      │ 32 │ 33 │ 34 │... │ 63 │  ← 地址 32-63
      │ 64 │ 65 │ 66 │... │ 95 │  ← ...
      └────┴────┴────┴────┴────┘
```

**规则**：bank = (字节地址 / 4) % 32

- 如果 32 个线程访问**不同** bank → 1 个周期完成 ✅
- 如果多个线程访问**相同** bank → 串行化，32 个周期 ❌（这就是 Bank Conflict！）

### 7.2 原始代码的 Bank Conflict

我们的共享内存是一个 2D 数组 `smem[32][64]`（32 行，64 列）。每个线程访问自己那一行的一列：

```c
// Thread ti (0-31) 访问 smem[ti][d]
// 地址 = ti * 64 + d
bank = (ti * 64 + d) % 32
```

**关键发现**：`64 % 32 == 0`，所以 `(ti * 64) % 32 == 0`

```
bank = (0 + d) % 32 = d % 32
```

**所有 32 个线程命中同一个 bank！** 这是 **32-way bank conflict**——最坏的情况。

```
线程 0: 地址 0*64+d → bank (0+d)%32
线程 1: 地址 1*64+d → bank (64+d)%32 = d%32   ← 同一个 bank！
线程 2: 地址 2*64+d → bank (128+d)%32 = d%32   ← 同一个 bank！
...
线程31: 地址 31*64+d → bank (1984+d)%32 = d%32 ← 同一个 bank！
```

**后果**：本来 1 个周期能完成的共享内存读写，现在需要 32 个周期——32 倍的性能损失！

### 7.3 修复：加一列 Padding

把行宽度从 64 改成 **65**（D_PAD = D + 1 = 65）。

```
bank = (ti * 65 + d) % 32
     = (ti * (64 + 1) + d) % 32
     = (ti * 64 + ti + d) % 32
     = (0 + ti + d) % 32          ← 因为 64 % 32 == 0
     = (ti + d) % 32
```

现在每个线程命中不同的 bank！

```
线程 0: bank = (0+d) % 32 = d % 32
线程 1: bank = (1+d) % 32 = (d+1) % 32     ← 不同！
线程 2: bank = (2+d) % 32 = (d+2) % 32     ← 不同！
...
线程31: bank = (31+d) % 32 = (d+31) % 32   ← 不同！
```

**32 个线程命中 32 个不同的 bank → 零冲突，1 个周期完成！** 

### 7.4 代价与收益

**代价**：
- 共享内存从 32×64 = 2048 → 32×65 = 2080 个 float（多了 1.6%）
- 第 64 列在共享内存中存在但从不使用（padding column）

**收益**：
- N=1024 时 Tiled kernel 从 15,114 μs → 4,543 μs（**3.3 倍加速**）
- 仅此一项优化！

### 7.5 为什么 65 可以而 64 不行

因为 65 是奇数，不被 32 整除，而 64 被 32 整除。

通用的规则：要让 M 列的数据避免 bank conflict，列数不能是 32 的倍数。常见的 padding 技巧：[N][M] → [N][M+1] 或 [N+1][M]。

---

## 8. 性能分析与对比

### 8.1 测试环境

- GPU: NVIDIA GeForce RTX 4060 Laptop (SM 8.9, 24 SMs, 8GB VRAM)
- 配置: B=2, H=4, D=64, float32
- 基准: PyTorch `scaled_dot_product_attention` (math backend, 禁用 FlashAttention)

### 8.2 性能数据

| 序列长度 N | Naive (μs) | Tiled (μs) | PyTorch (μs) | Tiled/Naive 加速比 |
|-----------|------------|------------|--------------|-------------------|
| 32 | 316.5 | 139.0 | 243.3 | 2.28× |
| 64 | 544.1 | 306.5 | 222.1 | 1.78× |
| 128 | 1,030.5 | 374.6 | 159.7 | 2.75× |
| 256 | 2,891.9 | 465.4 | 148.2 | 6.21× |
| 512 | 8,327.5 | 1,351.0 | 206.3 | 6.16× |
| 1024 | 47,779.8 | 4,684.5 | 1,464.3 | 10.20× |

### 8.3 数据解读

**Naive kernel 随 N 增长急剧恶化**：
N 从 32 → 1024（32 倍），耗时从 317 μs → 47,780 μs（150 倍）。因为每个线程的工作量是 O(N²×D)，全局内存访问量也等比例增长。

**Tiled kernel 增长平稳得多**：
N 从 32 → 1024，耗时从 139 μs → 4,685 μs（34 倍）。共享内存复用了 K/V 数据，减少了全局内存访问。

**为什么 PyTorch 还是更快（3-4×）**：
- PyTorch 底层调用 cuBLAS 的矩阵乘法，使用了 **Tensor Cores**（专门的矩阵乘法硬件）
- cuBLAS 在 warp 级别做矩阵乘法，每个 warp 协同计算，而非我们的"每个线程独立做点积"
- 我们的手写 kernel 本质上是标量计算，没有利用 Tensor Cores

### 8.4 优化历程

```
原始版本:  Tiled/Naive 加速比 = 1-2×
  ↓
+ Bank Conflict 修复 (D→D_PAD):  加速比 = 3-6×  ← 最关键的优化
  ↓
+ K/V 同时加载 + 循环融合:       加速比 = 6-10×
```

---

## 9. 课堂演示要点

### 9.1 建议的讲解结构（15-20 分钟）

**Part 1: 问题引入（3 分钟）**
- 展示 Self-Attention 公式，画出 N×N 矩阵
- 抛出问题：N=1024 时有多少计算量？为什么 CPU 不行？
- 引出 GPU 并行计算

**Part 2: GPU 基础知识（3 分钟）**
- 用"博士 vs 小学生"的类比讲 CPU vs GPU
- 画出线程层级图（Grid → Block → Thread）
- 重点讲内存层级（用"图书馆 vs 课桌"类比）

**Part 3: Naive 实现（3 分钟）**
- 展示并行策略：Grid=[B×H], Block=[N]
- 每个线程的三阶段工作流程
- 指出瓶颈：数据被反复读

**Part 4: Tiled 实现（5 分钟）**
- 引入 Tile 概念（画图展示如何切分矩阵）
- 共享内存布局
- **重点展示 Bank Conflict 问题**（用具体的 bank 编号表格演示 32 个线程命中同一个 bank）
- 展示 D_PAD=65 的修复（同样用表格展示）
- 性能对比图（优化前 vs 优化后）

**Part 5: Online Softmax（2 分钟）**
- 为什么要分块算 softmax
- 用 3 个数的例子手算一遍
- 展示代码对应关系

**Part 6: 总结与收获（2 分钟）**
- 三条核心经验
- 实际性能数据

### 9.2 核心实验结论

1. **共享内存是 GPU 编程最重要的优化手段**——数据复用能减少 90%+ 的全局内存访问
2. **Bank Conflict 是隐形的性能杀手**——代码完全正确，但比预期慢 32 倍，编译器不报任何警告
3. **硬件特性决定优化方向**——理解 GPU 的 bank 布局、warp 调度才能写出高性能 kernel

### 9.3 可现场演示

```bash
# 编译
cd ~/projects/cuda-self-attention
source env.sh
python3 setup.py build_ext --inplace

# 正确性验证（展示正确性）
python3 tests/test_naive.py
python3 tests/test_tiled.py

# 性能对比（展示加速效果）
python3 benchmark/benchmark_speed.py
```

### 9.4 可能被问到的问题

**Q: 为什么 TILE_N 选 32 而不是 64？**
A: 因为一个 warp 是 32 个线程。TILE_N=32 时一个 warp 恰好覆盖一行。如果用 64，共享内存翻倍（可能超限），而且同步开销更大。128 虽然更大但共享内存不够（48KB 上限）。

**Q: Bank Conflict 为什么编译器不优化？**
A: nvcc 不会自动加 padding，因为编译器不知道你的 2D 数组"语义上"有多少列。它只看到一个扁平的 float 数组。

**Q: 能不能直接用 cuBLAS 做矩阵乘法？**
A: 可以，而且会更快（因为用 Tensor Cores）。但本项目的目标是**从零手写 CUDA kernel**，学习并行编程的底层原理，所以故意不调库。

**Q: D=64 够用吗？实际模型用多少？**
A: 实际 Transformer 模型通常用 D=64 或 D=128（每个头），所以我们的配置是贴近实际的。

---

## 附录：关键术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| 自注意力 | Self-Attention | 序列中每个元素关注所有其他元素的机制 |
| 线程束 | Warp | GPU 中 32 个线程为一组的调度单位 |
| 共享内存 | Shared Memory | Block 内线程共享的高速缓存（~30 cycles） |
| 全局内存 | Global Memory | GPU 显存，所有线程可访问但慢（~400 cycles） |
| 分块 | Tiling | 把大矩阵切成小块分别处理的技术 |
| Bank Conflict | Bank Conflict | 同一 warp 多个线程访问同一 bank 导致的串行化 |
| 在线 Softmax | Online Softmax | 分块维护 running max/sum 的数值稳定算法 |
| 占用率 | Occupancy | 每个 SM 上同时活跃的 warp 数量与理论最大值的比率 |
