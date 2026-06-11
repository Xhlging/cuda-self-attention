# CUDA Self-Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 从零实现一个 CUDA 版本的 Scaled Dot-Product Self-Attention kernel（Naive 基准 + Tiled 优化），封装为 PyTorch 自定义算子，与 PyTorch 原生实现做正确性和性能对比。

**Architecture:**
- 项目根：`~/projects/cuda-self-attention/`
- 纯 CUDA C++ 实现 Attention 前向 kernel（Naive 版本 + Tiling 优化版本）
- 通过 PyTorch `CUDAExtension` 封装为自定义算子
- Python 脚本进行正确性验证 + 性能 Benchmark

**Tech Stack:**（根据审查结论修正）
- **CUDA 13.3** + g++ (via conda env `cuda-attention`) — 确认为官方发布版本
- **PyTorch 2.5.x**（稳定版）— 非 2.11.0 开发版，确保兼容性
- **Python 3.12** — 非 3.13，确保 PyTorch 官方支持
- **Ninja** — 编译加速
- **GPU:** RTX 4060 Laptop (SM 8.9, 24 SMs, 8GB VRAM)

**环境准备（Task 0 前置步骤）：**

```bash
# 1. 创建专用 conda 环境
conda create -n cuda-attention python=3.12 -y
conda activate cuda-attention

# 2. 安装 CUDA toolkit 和 PyTorch
conda install -c nvidia cuda-nvcc cuda-cudart -y
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja

# 3. 安装课程项目依赖
pip install einops matplotlib pandas

# 4. 验证环境
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
nvcc --version
```

**运行时环境变量（每次运行前设置）：**

```bash
export CUDA_HOME=$CONDA_PREFIX                              # 指向 conda 环境
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # 运行时库路径
```

---

## Task 0: 项目骨架与编译验证

**Objective:** 建立项目目录结构，编写 `setup.py`，确认 PyTorch CUDA 扩展可以成功编译。

**Files:**
- Create: `~/projects/cuda-self-attention/setup.py`
- Create: `~/projects/cuda-self-attention/csrc/attention_kernel.cu`（占位 kernel）
- Create: `~/projects/cuda-self-attention/attention/__init__.py`（Python 封装）
- Create: `~/projects/cuda-self-attention/tests/__init__.py`

**Step 1: 创建目录结构**

```bash
mkdir -p ~/projects/cuda-self-attention/{csrc,attention,tests}
```

**Step 2: 创建 setup.py**

修正要点：
- 移除 `no_python_abi_suffix=True`，保留默认 .so 后缀
- g++ 追加 `-fopenmp` 以利用多核编译
- 明确 `.contiguous()` 在 C++ 入口处理，Python 侧不重复

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# CUDA_HOME 从环境变量获取，运行时必须设置
# export CUDA_HOME=$CONDA_PREFIX

ext_modules = [
    CUDAExtension(
        "attention._C",
        ["csrc/attention_kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-fopenmp"],
            "nvcc": [
                "-O3",
                "--expt-relaxed-constexpr",
                # SM 8.9 = RTX 4060 (Ada Lovelace)
                "-gencode=arch=compute_89,code=sm_89",
                # 回退到 SM 8.0 (Ampere) 保证兼容性
                "-gencode=arch=compute_80,code=sm_80",
            ],
        },
    ),
]

setup(
    name="cuda-self-attention",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["attention"],
    package_dir={"attention": "attention"},
)
```

**Step 3: 创建占位 kernel**

`csrc/attention_kernel.cu`（仅含 PYBIND11 绑定入口，无实际 kernel）：

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// ========== 前向声明 ==========
torch::Tensor attention_forward_naive(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale);

torch::Tensor attention_forward_tiled(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale);

// ========== PYBIND11 绑定 ==========
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward_naive", &attention_forward_naive,
          "Naive Scaled Dot-Product Attention Forward");
    m.def("attention_forward_tiled", &attention_forward_tiled,
          "Tiled Scaled Dot-Product Attention Forward");
}
```

**Step 4: 创建 Python 封装**

`attention/__init__.py` — 注意：`contiguous()` 在 C++ 入口做，Python 侧不再重复：

```python
from . import _C

def attention_naive(q, k, v, scale=None):
    """Naive CUDA attention forward (no optimizations)."""
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    return _C.attention_forward_naive(q, k, v, float(scale))


def attention_tiled(q, k, v, scale=None):
    """Tiled + shared memory optimized CUDA attention forward."""
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    return _C.attention_forward_tiled(q, k, v, float(scale))
```

**Step 5: 编译验证**

```bash
cd ~/projects/cuda-self-attention
CUDA_HOME=$CONDA_PREFIX LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
    python3 setup.py build_ext --inplace
```

验证：
- 编译成功，无报错
- 生成 `attention/_C.cpython-312-x86_64-linux-gnu.so`
- `python3 -c "from attention._C import attention_forward_naive; print('OK')"` 成功

---

## Task 1: Naive Self-Attention Kernel（基准实现）

**Objective:** 实现最直接的 Self-Attention 前向 kernel——**不使用共享内存**，每个线程用局部变量直接在全局内存上计算。作为性能基准，不追求速度，只保证正确性。

> **设计决策（审查结论）：** Naive kernel 定位为"无优化基准线"，不应使用共享内存。原计划中共享内存存储 scores 方案存在数据竞争（所有线程同时写 `scores[j]`）。改为每个线程独立用寄存器/局部内存计算 scores，避免竞态，保持实现简单。

### 算法

```
S[i][j] = (Q[i,:] · K[j,:]) / sqrt(D),  ∀i,j ∈ [0,N)
P[i][j] = exp(S[i][j] - max_j S[i][j]) / sum_j exp(S[i][j] - max_j S[i][j])
O[i,:]  = Σ_j P[i][j] · V[j,:],        ∀i ∈ [0,N)
```

### Kernel 设计

- **Grid:** `(B*H,)` — 每个 block 处理一个 `(batch, head)`
- **Block:** `(N,)` — 每个 thread 处理一个 query 位置 i
- **每个 thread 的工作：** 串行遍历 j=0..N-1，用局部 `float` 数组存所有 scores，然后做 softmax 和加权求和
- **限制:** N ≤ 1024（block 最大线程数 1024；超过时改用 2D grid）

**Files:**
- Modify: `csrc/attention_kernel.cu`（添加 kernel + C++ 入口）
- Create: `tests/test_naive.py`

### Step 1: 写 kernel

```cpp
// ============================================================
// Naive Self-Attention Forward Kernel
// 每个 thread 独立处理一个 query 位置
// 不使用共享内存 → 无数据竞争
// 限制: N <= 1024 (CUDA block 线程上限)
// ============================================================

__global__ void attention_naive_kernel(
    const float* __restrict__ Q,    // [B, H, N, D]
    const float* __restrict__ K,    // [B, H, N, D]
    const float* __restrict__ V,    // [B, H, N, D]
    float* __restrict__ O,          // [B, H, N, D]
    int N, int D, float scale
) {
    // blockIdx.x = batch * H + head
    int bh = blockIdx.x;              // batch-head index
    int i = threadIdx.x;              // query position (i from 0 to N-1)

    if (i >= N) return;

    // 每个 (batch, head) 的 QKV 起始偏移
    int base = bh * N * D;

    // ----------------------------------------------------------
    // Phase 1: 计算 scores S[i][j] = (Q[i] · K[j]) * scale
    // 每个 thread 的局部数组，无共享内存竞争
    // ----------------------------------------------------------
    // 由于 N <= 1024, scores 最多 1024 个 float = 4 KB
    // 在寄存器不够时编译器会 spill 到 local memory
    float scores[1024];  // 运行时 N 已知，用 const 或 VLA

    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += Q[base + i * D + d] * K[base + j * D + d];
        }
        scores[j] = dot * scale;
    }

    // ----------------------------------------------------------
    // Phase 2: Safe Softmax
    //   max_val = max_j scores[j]
    //   sum_val = sum_j exp(scores[j] - max_val)
    //   P[j]    = exp(scores[j] - max_val) / sum_val
    // ----------------------------------------------------------
    float max_val = scores[0];
    for (int j = 1; j < N; j++) {
        max_val = fmaxf(max_val, scores[j]);
    }

    float sum_val = 0.0f;
    for (int j = 0; j < N; j++) {
        scores[j] = __expf(scores[j] - max_val);
        sum_val += scores[j];
    }

    // ----------------------------------------------------------
    // Phase 3: 加权求和 O[i] = Σ_j P[j] * V[j]
    // ----------------------------------------------------------
    for (int d = 0; d < D; d++) {
        float out = 0.0f;
        for (int j = 0; j < N; j++) {
            out += (scores[j] / sum_val) * V[base + j * D + d];
        }
        O[base + i * D + d] = out;
    }
}
```

### Step 2: C++ 入口函数

```cpp
torch::Tensor attention_forward_naive(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale) {

    // 保证输入连续（审查结论）
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    auto dims = Q.sizes();
    int B = dims[0], H = dims[1], N = dims[2], D = dims[3];
    int total_heads = B * H;

    auto O = torch::empty_like(Q);

    // 限制: N <= 1024 (block 线程上限)
    TORCH_CHECK(N <= 1024, "Naive kernel requires N <= 1024, got ", N);

    attention_naive_kernel<<<total_heads, N>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, D, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return O;
}
```

### Step 3: 写测试脚本

`tests/test_naive.py` — 注意测试中**强制禁用 PyTorch 的 FlashAttention/MemEfficientAttention**，确保对比公平：

```python
import torch
import sys
sys.path.insert(0, '..')
from attention import attention_naive

def test_naive():
    """验证 Naive kernel 结果与 PyTorch math backend 一致"""
    B, H, N, D = 2, 4, 64, 64
    torch.manual_seed(42)

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    scale = D ** -0.5

    # 强制 PyTorch 使用纯数学实现（不调用 FlashAttention）
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False
    ):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    out = attention_naive(q, k, v, scale=scale)

    diff = (out - ref).abs().max().item()
    print(f"Naive kernel max diff vs PyTorch math backend: {diff:.6f}")

    assert diff < 1e-3, f"Too large difference: {diff}"
    print("PASS: Naive attention matches PyTorch reference")

    # 额外：N=1024 压力测试
    N_big = 1024
    q2 = torch.randn(B, H, N_big, D, device='cuda')
    k2 = torch.randn(B, H, N_big, D, device='cuda')
    v2 = torch.randn(B, H, N_big, D, device='cuda')
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        ref2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, scale=scale)
    out2 = attention_naive(q2, k2, v2, scale=scale)
    diff2 = (out2 - ref2).abs().max().item()
    print(f"Naive kernel (N=1024) max diff: {diff2:.6f}")
    assert diff2 < 1e-3, f"Too large difference at N=1024: {diff2}"
    print("PASS: Naive attention N=1024 passes")

if __name__ == '__main__':
    test_naive()
```

### Step 4: 运行验证

```bash
cd ~/projects/cuda-self-attention
CUDA_HOME=$CONDA_PREFIX LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
    python3 tests/test_naive.py
```

Expected: `Max diff: < 1e-3, PASS`

---

## Task 2: Tiling + Shared Memory 优化 Kernel

**Objective:** 实现分块 (tiling) 的优化 kernel，将 Q/K/V 分块加载到共享内存，配合 online (safe) softmax 避免显存存储整个 N×N 注意力矩阵。

**Files:**
- Modify: `csrc/attention_kernel.cu`（添加 tiled kernel + C++ 入口）
- Create: `tests/test_tiled.py`

### Kernel 设计

| 项目 | 值 | 说明 |
|---|---|---|
| Block 维度 | `(TILE_N,)` | TILE_N = 32，每个线程处理 1 个 query |
| Grid 维度 | `(B*H*ceil(N/TILE_N),)` | 多 block 覆盖所有 query 位置 |
| **TILE_N** | **32** | **审查修正：原 TILE_N=64 内存超限，32 确保安全** |

### 共享内存布局与复用（审查修正要点）

**修正前：** Q_tile + K_tile + V_tile 同时驻留 → TILE_N=64 时每 tile 64×64×4 = 16 KB，三个 tile 共 48 KB，加上 stats 等超出 48KB 限制。

**修正后：** K_tile 与 V_tile 分时复用同一块共享内存区域。

```
Shared memory layout:
  [0 .. TILE_N*D)     → Q_tile    (TILE_N × D floats)    = 32*64*4 = 8 KB
  [TILE_N*D .. 2*TILE_N*D) → K_or_V_tile (复用区域)      = 8 KB
  [2*TILE_N*D .. 2*TILE_N*D + TILE_N] → stats            = 128 B
  Total: ~16.2 KB << 48 KB ✅
```

### Online Softmax 算法（审查补充）

核心思想：在外循环中逐步维护 running max 和 running sum，避免一次性存储整个 N×N scores 矩阵。

```
初始化: m = -inf, d = 0, O = [0]*D

对于每个 K-tile [j_start, j_end):
  1. 加载 Q_tile, K_tile → 计算局部 scores: S_local[i][j] = Q[i]·K[j]
  2. 更新 running max: m_new = max(m, max_j S_local)
  3. 更新 running sum:
     d_new = d * exp(m - m_new) + Σ_j exp(S_local[j] - m_new)
  4. 更新 output: O *= exp(m - m_new) / ... 
     O += Σ_j exp(S_local[j] - m_new) * V[j]
  5. m = m_new, d = d_new
  6. 重复直到所有 K 列处理完毕

最终: O /= d
```

### Step 1: 编写 tiled kernel

```cpp
// ============================================================
// Tiled Self-Attention Forward Kernel
// 分块加载 Q/K/V 到共享内存 + Online softmax
// TILE_N = 32, K_tile 与 V_tile 复用共享内存
// ============================================================

constexpr int TILE_N = 32;

// 共享内存布局 (每个 block 一份):
//   Q_tile:   [TILE_N, D]     — 当前 query tile
//   KV_tile:  [TILE_N, D]     — 当前 key/value tile (复用)
//   stats:    [TILE_N, 2]     — 每个 thread 的 max_val 和 sum_exp (临时)

__global__ void attention_tiled_kernel(
    const float* __restrict__ Q,    // [B, H, N, D]
    const float* __restrict__ K,    // [B, H, N, D]
    const float* __restrict__ V,    // [B, H, N, D]
    float* __restrict__ O,          // [B, H, N, D]
    int N, int D, float scale
) {
    // blockIdx = tiled block 索引
    // batch 和 head 由 grid 维度编码
    int bh = blockIdx.y;             // batch * H + head
    int tile_start = blockIdx.x * TILE_N;  // 这个 block 处理的 query 起始位置
    int ti = threadIdx.x;            // tile 内位置 (0..TILE_N-1)

    int global_i = tile_start + ti;  // 全局 query 位置
    int base = bh * N * D;

    if (global_i >= N) return;

    // 共享内存指针
    extern __shared__ float smem[];
    float* Q_tile  = smem;                         // [TILE_N, D]
    float* KV_tile = smem + TILE_N * D;            // [TILE_N, D]  K 与 V 复用

    // ----------------------------------------------------------
    // 加载 Q_tile: Q[base + global_i * D : D] → smem
    // ----------------------------------------------------------
    for (int d = ti; d < D; d += TILE_N) {
        Q_tile[ti * D + d] = Q[base + global_i * D + d];
    }
    __syncthreads();

    // ----------------------------------------------------------
    // Online softmax running state
    // ----------------------------------------------------------
    float m = -INFINITY;   // running max
    float d_ = 0.0f;       // running sum of exp(m_j - m)
    float O_local[D];      // 局部累加输出 (每个 thread 独立)
    #pragma unroll
    for (int d = 0; d < D; d++) O_local[d] = 0.0f;

    // ----------------------------------------------------------
    // 主循环: 遍历 K 的 tiles (j_start = 0, TILE_N, 2*TILE_N, ...)
    // ----------------------------------------------------------
    for (int j_start = 0; j_start < N; j_start += TILE_N) {
        // ---- 加载 K_tile: K[base + j*D + d] → KV_tile ----
        int j_global = j_start + ti;
        if (j_global < N) {
            for (int d = 0; d < D; d++) {
                KV_tile[ti * D + d] = K[base + j_global * D + d];
            }
        }
        __syncthreads();

        // ---- 计算局部 scores: S = Q_tile[i] · K_tile[j] ----
        // ti 代表当前 thread 处理的 query 位置 (tile 内)
        // 遍历这一 tile 中的 key 位置 (j_local = 0..TILE_N-1)
        float local_max = -INFINITY;
        float scores[TILE_N];  // TILE_N=32 → 128 bytes, 寄存器可容纳

        for (int jl = 0; jl < TILE_N; jl++) {
            int j_global2 = j_start + jl;
            if (j_global2 >= N) {
                scores[jl] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q_tile[ti * D + d] * KV_tile[jl * D + d];
            }
            scores[jl] = dot * scale;
            local_max = fmaxf(local_max, scores[jl]);
        }

        // ---- 更新 running max ----
        float m_new = fmaxf(m, local_max);

        // ---- 计算 exp(scores - m_new) 和 sum ----
        float sum_exp = 0.0f;
        for (int jl = 0; jl < TILE_N; jl++) {
            if (j_start + jl >= N) break;
            scores[jl] = __expf(scores[jl] - m_new);
            sum_exp += scores[jl];
        }

        // ---- 缩放旧输出: O *= exp(m - m_new) ----
        float scale_old = __expf(m - m_new);
        for (int d = 0; d < D; d++) {
            O_local[d] *= scale_old;
        }

        // ---- 加载 V_tile 并累加新贡献 ----
        // 复用 KV_tile, 需要重新加载
        __syncthreads();
        if (j_global < N) {
            for (int d = 0; d < D; d++) {
                KV_tile[ti * D + d] = V[base + j_global * D + d];
            }
        }
        __syncthreads();

        for (int jl = 0; jl < TILE_N; jl++) {
            int jg = j_start + jl;
            if (jg >= N) break;
            float p = scores[jl];  // p = exp(score - m_new)
            for (int d = 0; d < D; d++) {
                O_local[d] += p * KV_tile[jl * D + d];
            }
        }

        // ---- 更新 running sum ----
        d_ = d_ * scale_old + sum_exp;
        m = m_new;

        __syncthreads();  // 保证下一轮加载前 KV_tile 读取完毕
    }

    // ---- 最终归一化: O /= d_ ----
    for (int d = 0; d < D; d++) {
        O[base + global_i * D + d] = O_local[d] / d_;
    }
}
```

### Step 2: C++ 入口函数

```cpp
// 辅助宏: CUDA 错误检查
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

torch::Tensor attention_forward_tiled(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale) {

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    auto dims = Q.sizes();
    int B = dims[0], H = dims[1], N = dims[2], D = dims[3];

    auto O = torch::empty_like(Q);

    int total_heads = B * H;
    int grid_x = (N + TILE_N - 1) / TILE_N;

    // 共享内存大小: Q_tile + KV_tile + padding
    size_t smem_bytes = 2 * TILE_N * D * sizeof(float);

    dim3 grid(grid_x, total_heads);
    dim3 block(TILE_N);

    attention_tiled_kernel<<<grid, block, smem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, D, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return O;
}
```

### Step 3: 共享内存 Bank Conflict 缓解（审查补充）

在 `Q_tile` 和 `KV_tile` 的 last 维度加 **1 float padding**，将 `D` 扩展为 `D + 1`，打破 bank conflict：

```cpp
// 在 kernel 中: 将共享内存声明为
// float Q_tile[TILE_N][D + 1];   // D+1 而不是 D
// float KV_tile[TILE_N][D + 1];

// 访问方式:
// Q_tile[ti][d] 而不是 Q_tile[ti * D + d]
// 这样同一个 warp 内相邻线程访问相邻 bank →
// stride = 1, 无 bank conflict
```

实现时需动态计算共享内存大小，或使用模板特化。

> 注意：上面的 kernel 代码为了清晰使用了线性索引。实际实现时为 bank conflict 优化应改用二维索引 + 末尾 padding。这可以作为可选项在实现时决定。

### Step 4: Tiled 测试与验证

`tests/test_tiled.py`:

```python
import torch
import sys
sys.path.insert(0, '..')
from attention import attention_naive, attention_tiled

def test_tiled():
    B, H, D = 2, 4, 64

    for N in [32, 64, 128, 256, 512, 1024]:
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='cuda')
        k = torch.randn(B, H, N, D, device='cuda')
        v = torch.randn(B, H, N, D, device='cuda')
        scale = D ** -0.5

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

        out = attention_tiled(q, k, v, scale=scale)

        diff = (out - ref).abs().max().item()
        print(f"N={N:5d} | Tiled max diff: {diff:.6f}", end='')
        assert diff < 1e-3, f"FAIL at N={N}: {diff}"
        print("  ✅ PASS")

    print("\nAll tests passed! Tiled kernel is correct.")

def test_tiled_vs_naive():
    """验证 Tiled kernel 和 Naive kernel 结果一致（交叉验证）"""
    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')
    scale = D ** -0.5

    out_naive = attention_naive(q, k, v, scale=scale)
    out_tiled = attention_tiled(q, k, v, scale=scale)

    diff = (out_naive - out_tiled).abs().max().item()
    print(f"Naive vs Tiled diff: {diff:.6f}")
    assert diff < 1e-5, f"Two implementations disagree: {diff}"
    print("PASS: Naive and Tiled produce identical results")

if __name__ == '__main__':
    test_tiled()
    test_tiled_vs_naive()
```

---

## Task 3: 性能 Benchmark 与报告

**Objective:** 系统对比三版本（Naive, Tiled, PyTorch math backend）的延迟与带宽。

**Files:**
- Create: `benchmark/benchmark_speed.py`
- Create: `benchmark/plot_results.py`
- Output: `benchmark/results/` 目录下的数据和图表

### Step 1: Benchmark 设计

修正要点：
- **禁用 PyTorch 自动优化：** 使用 `torch.backends.cuda.sdp_kernel()` 强制 math backend
- **环境变量屏蔽：** 设置 `TORCH_BLAS_PREFER_CUBLAS=1` 控制底层 BLAS 选择
- **公平对比：** 所有 kernel 使用相同的 warmup 策略和 timing 方法

```python
import torch
import time
import sys
sys.path.insert(0, '..')
from attention import attention_naive, attention_tiled

def benchmark():
    B, H, D = 2, 4, 64
    Ns = [32, 64, 128, 256, 512, 1024]
    WARMUP = 5
    ITERS = 50

    results = []

    for N in Ns:
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='cuda')
        k = torch.randn(B, H, N, D, device='cuda')
        v = torch.randn(B, H, N, D, device='cuda')
        scale = D ** -0.5

        # Naive
        for _ in range(WARMUP):
            _ = attention_naive(q, k, v, scale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            _ = attention_naive(q, k, v, scale)
        torch.cuda.synchronize()
        naive_time = (time.perf_counter() - start) / ITERS * 1e6  # us

        # Tiled
        for _ in range(WARMUP):
            _ = attention_tiled(q, k, v, scale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            _ = attention_tiled(q, k, v, scale)
        torch.cuda.synchronize()
        tiled_time = (time.perf_counter() - start) / ITERS * 1e6

        # PyTorch math backend (禁用 Flash/MemEff)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(ITERS):
                fn()
            torch.cuda.synchronize()
            pt_time = (time.perf_counter() - start) / ITERS * 1e6

        print(f"N={N:5d} | Naive: {naive_time:8.1f} us | Tiled: {tiled_time:8.1f} us | "
              f"PyTorch(math): {pt_time:8.1f} us | Tiled/Naive: {naive_time/tiled_time:.2f}x")

        results.append({
            'N': N, 'naive_us': naive_time, 'tiled_us': tiled_time, 'pytorch_us': pt_time
        })

    return results
```

### Step 2: Benchmark 输出格式

```
Benchmark: B=2, H=4, D=64, GPU=RTX4060
N        Naive(us)    Tiled(us)    PyTorch(us)   Speedup
───      ─────────    ─────────    ──────────    ───────
32        xxx.x        xxx.x        xxx.x          x.xx
64        xxx.x        xxx.x        xxx.x          x.xx
128       xxx.x        xxx.x        xxx.x          x.xx
256       xxx.x        xxx.x        xxx.x          x.xx
512       xxx.x        xxx.x        xxx.x          x.xx
1024      xxx.x        xxx.x        xxx.x          x.xx
```

### Step 3: 性能分析报告要点

报告应包含：
1. **Naive 瓶颈分析：** 每个 element 被反复从 global memory 读取 O(N) 次 → 带宽受限
2. **Tiled 优化分析：** 共享内存 tile 使数据复用率从 1/D 提升到接近 1，全局访存减少 O(N/tile_size)
3. **与 PyTorch math 的差距分析：** 为什么还有差距？PyTorch 使用了 cuBLAS 矩阵乘法（高度优化的 warp-level 实现），而我们的 kernel 是手写 dot product
4. **可改进方向：** 如果能用 cuBLAS 做 tile 内的矩阵乘法会更快；但这偏离了"手写 kernel"的课程目标

---

## 课程交付物 Checklist

| # | 交付物 | 说明 |
|---|--------|------|
| 1 | `csrc/attention_kernel.cu` | 全部 CUDA kernel 代码（Naive + Tiled） |
| 2 | `setup.py` | 编译配置 |
| 3 | `attention/__init__.py` | Python API 封装 |
| 4 | `tests/test_naive.py` | Naive 正确性验证 |
| 5 | `tests/test_tiled.py` | Tiled 正确性验证（覆盖多 N） |
| 6 | `benchmark/benchmark_speed.py` | 性能对比脚本 |
| 7 | `benchmark/results/report.md` | 报告 + 图表 |
| 8 | 课程演示材料 | Python notebook 或幻灯片（可选） |

---

## 风险与权衡（更新版）

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| **共享内存不足** | kernel 启动失败 | TILE_N=32, K/V tile 复用 → 总共享内存 ~16 KB << 48 KB |
| **Naive kernel 大 N 超时** | N>1024 无法使用 | 设定 N≤1024 硬限制 + 2D grid 方案预留（不实现） |
| **Tiled kernel bank conflict** | 性能损失 20-30% | 共享内存末尾加 1 float padding 缓解 |
| **Online softmax 精度** | 大 N 时浮点误差累积 | 使用 `fmathf` + `__expf` 保证精度；测试覆盖 N=1024 |
| **PyTorch 对比不公平** | 对比数据无意义 | 强制 `enable_math=True`, 禁用 Flash/MemEff |

---

## 验证清单

- [x] Task 0: `setup.py` 编译成功，`attention._C` 可正常导入
- [x] Task 1: Naive kernel 在 N=64..1024 范围内与 PyTorch math 偏差 < 1e-3
- [x] Task 2: Tiled kernel 在所有 N 上正确；与 Naive 结果一致（交叉验证）
- [x] Task 3: Benchmark 图表完整，报告包含环境、方法、数据和分析
- [ ] 所有中间产物可被 `git clean -fdx` 清理
- [ ] 代码风格统一（注释英文，命名蛇形/驼峰）

---

## 实现完成记录（2026-06-11）

### Task 0: 项目骨架与编译验证 ✅

**最终环境配置：**
- CUDA Toolkit: 13.3 (conda env `cuda-attention`, `conda install -c nvidia cuda-nvcc cuda-cudart`)
- PyTorch: 2.11.0+cu130 (base conda env)
- Host Compiler: conda g++ 15.2 (系统 g++ 13.3 与 PyTorch 头文件不兼容)
- Python: 3.13

**编译关键变量：**
```bash
export CUDA_HOME=~/miniconda3/envs/cuda-attention
export CXX=~/miniconda3/envs/cuda-attention/bin/x86_64-conda-linux-gnu-c++
export CC=~/miniconda3/envs/cuda-attention/bin/x86_64-conda-linux-gnu-cc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:~/miniconda3/lib/python3.13/site-packages/torch/lib
```

**文件拆分：** 将 Torch 绑定（.cpp）与 CUDA kernel（.cu）分离，避免 nvcc 使用系统 g++ 编译 PyTorch 头文件时的兼容性问题。

### Task 1: Naive Self-Attention Kernel ✅

**实际实现：**
- 每个 block 处理一个 (batch, head)，grid = `(B*H,)`
- 每个 thread 处理一个 query 位置，block = `(N,)`，N ≤ 1024
- 局部数组 `float scores[1024]`，无共享内存（避免数据竞争）
- 三阶段计算：score → safe softmax → 加权求和

**测试结果：**
```
N=   64 | max_diff=4.77e-07 | PASS
N=  128 | max_diff=4.77e-07 | PASS
N=  256 | max_diff=4.17e-07 | PASS
N=  512 | max_diff=4.32e-07 | PASS
N= 1024 | max_diff=4.17e-07 | PASS
```

### Task 2: Tiled Self-Attention Kernel ✅

**实际实现：**
- TILE_N = 32, K/V tile 复用同一共享内存区域
- 共享内存: Q_smem [32, 64] + KV_smem [32, 64] = 16 KB
- Online softmax（running max + running denom）
- Grid: `(ceil(N/TILE_N), B*H)`, Block: `(TILE_N,)`

**修复的 Bug（共享内存加载）：**
原始 strided 加载 `for (int d = ti; d < D; d += TILE_N)` 导致 Q/K/V 加载不全——每线程只加载了部分列而非全部 D 个元素。修正为每个线程直接加载自己行的所有 D 个元素。

**测试结果：**
```
N=   32 | vs_ref=7.15e-07 ✅ | vs_naive=4.77e-07 ✅
N=   64 | vs_ref=5.36e-07 ✅ | vs_naive=5.36e-07 ✅
N=  128 | vs_ref=5.96e-07 ✅ | vs_naive=6.56e-07 ✅
N=  256 | vs_ref=4.77e-07 ✅ | vs_naive=4.47e-07 ✅
N=  512 | vs_ref=7.45e-07 ✅ | vs_naive=5.66e-07 ✅
N= 1024 | vs_ref=7.15e-07 ✅ | vs_naive=6.56e-07 ✅
```

### Task 3: 性能基准测试 ✅

| N | Naive (us) | Tiled (us) | PyTorch (us) | Speedup (Tiled/Naive) |
|---|------------|------------|--------------|----------------------|
| 32 | 158.7 | 166.2 | 173.2 | 0.96x |
| 64 | 385.4 | 308.6 | 136.3 | 1.25x |
| 128 | 945.8 | 642.4 | 215.9 | 1.47x |
| 256 | 2,617.6 | 1,301.6 | 232.3 | 2.01x |
| 512 | 8,208.3 | 4,850.3 | 211.9 | 1.69x |
| 1024 | 33,145.3 | 15,114.2 | 1,142.1 | 2.19x |

**分析：**
- Tiled 相比 Naive 有 1-2x 加速，N 越大收益越明显
- PyTorch 比 Tiled 快 7-20x（底层 cuBLAS + Tensor Cores）
- N=32 时 Tiled ≈ Naive（1 tile，共享内存开销抵消收益）

### 改写计划中修正的内容

| 修正项 | 原方案 | 实际实现 |
|--------|--------|----------|
| PyTorch 版本 | 2.5.x | 2.11.0（已安装的 base env） |
| Python 版本 | 3.12 | 3.13（PyTorch 2.11.0 支持） |
| 编译器 | 系统 g++ | conda g++ 15.2（兼容 PyTorch 头文件） |
| Naive 共享内存 | ❌ 共享内存有数据竞争 | ✅ 局部数组，无竞争 |
| TILE_N | 64（共享内存超限） | 32（安全 16 KB） |
| 编译架构 | setup.py 单一源文件 | .cpp + .cu 分离编译 |