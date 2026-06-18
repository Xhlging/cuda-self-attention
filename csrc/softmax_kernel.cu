#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

constexpr int WARP_SIZE = 32;
constexpr int MAX_DIM = 1024;

// ============================================================
// Naive Softmax — 每线程串行两遍扫描
// ============================================================
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

    // 第一遍: 找最大值 (safe softmax)
    float max_val = x[0];
    for (int i = 1; i < D; i++) max_val = fmaxf(max_val, x[i]);

    // 第二遍: 算 exp 和 sum
    float sum_val = 0.0f;
    for (int i = 0; i < D; i++) {
        y[i] = __expf(x[i] - max_val);
        sum_val += y[i];
    }

    // 第三遍: 归一化
    float inv_sum = 1.0f / sum_val;
    for (int i = 0; i < D; i++) y[i] *= inv_sum;
}

// ============================================================
// Warp Reduction Softmax — 用 __shfl_down_sync 做归约
//
// 核心思想: 一个 warp 32 线程协作算 sum。
// 每线程先算自己的局部 sum，然后用 warp shuffle
// 把 32 个局部 sum 合并成一个总和。
//
// __shfl_down_sync(mask, val, offset):
//   把本线程的 val 发给 offset 步之后的线程
//   类似于: 线程 i 收到线程 i+offset 的值
//   反复折半, log2(32)=5 步后所有人拿到总和
// ============================================================
template <int D>
__global__ void softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int stride)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input + row * stride;
    float* y = output + row * stride;
    int lane = threadIdx.x;  // 我在 warp 里的编号 (0-31)

    // --- Step 1: 找全局最大值 (warp reduction) ---
    float max_val = -INFINITY;
    // 每个线程处理 D/WARP_SIZE 个元素 (跨步访问)
    for (int i = lane; i < D; i += WARP_SIZE)
        max_val = fmaxf(max_val, x[i]);

    // Warp shuffle: 把 32 个局部 max 归约成线程 0 手里的全局 max
    // 然后广播给所有 32 个线程（归约只产生在 lane 0，其他 lane 只有部分结果）
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);  // 广播 lane 0 → 所有人

    // --- Step 2: 算 exp + sum (warp reduction) ---
    float sum_val = 0.0f;
    for (int i = lane; i < D; i += WARP_SIZE) {
        float e = __expf(x[i] - max_val);
        y[i] = e;          // 先存 exp 值
        sum_val += e;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
    sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);    // 广播 lane 0 → 所有人

    // --- Step 3: 归一化 ---
    float inv_sum = 1.0f / sum_val;
    for (int i = lane; i < D; i += WARP_SIZE)
        y[i] *= inv_sum;
}

// ============================================================
// Launch wrappers
// ============================================================
void softmax_naive_launch(const float* input, float* output,
                          int rows, int D, int stride) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    switch (D) {
        case 64:  softmax_naive_kernel<64><<<blocks, threads>>>(input, output, rows, stride); break;
        case 128: softmax_naive_kernel<128><<<blocks, threads>>>(input, output, rows, stride); break;
        case 256: softmax_naive_kernel<256><<<blocks, threads>>>(input, output, rows, stride); break;
        case 512: softmax_naive_kernel<512><<<blocks, threads>>>(input, output, rows, stride); break;
        case 1024:softmax_naive_kernel<1024><<<blocks, threads>>>(input, output, rows, stride); break;
        default:
            fprintf(stderr, "Softmax naive: unsupported D=%d\n", D); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void softmax_warp_launch(const float* input, float* output,
                         int rows, int D, int stride) {
    if (D > MAX_DIM) { fprintf(stderr, "D=%d > MAX_DIM\n", D); exit(1); }
    switch (D) {
        case 64:  softmax_warp_kernel<64><<<rows, WARP_SIZE>>>(input, output, rows, stride); break;
        case 128: softmax_warp_kernel<128><<<rows, WARP_SIZE>>>(input, output, rows, stride); break;
        case 256: softmax_warp_kernel<256><<<rows, WARP_SIZE>>>(input, output, rows, stride); break;
        case 512: softmax_warp_kernel<512><<<rows, WARP_SIZE>>>(input, output, rows, stride); break;
        case 1024:softmax_warp_kernel<1024><<<rows, WARP_SIZE>>>(input, output, rows, stride); break;
        default:
            fprintf(stderr, "Softmax warp: unsupported D=%d\n", D); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
