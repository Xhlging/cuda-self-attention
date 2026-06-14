#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// ============================================================
// Kernel declarations
// ============================================================

template <int D>
__global__ void attention_naive_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, float scale);

template <int D>
__global__ void attention_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, float scale);

// ============================================================
// CUDA error checking helper
// ============================================================
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ============================================================
// Constants
// ============================================================
constexpr int TILE_N = 32;  // Tile size for tiled kernel

// ============================================================
// Naive Self-Attention Forward Kernel (Task 1)
//
// Each block handles one (batch, head), each thread handles one query.
// No shared memory — intentionally naive baseline.
// ============================================================

template <int D>
__global__ void attention_naive_kernel(
    const float* __restrict__ Q,    // [B, H, N, D]
    const float* __restrict__ K,    // [B, H, N, D]
    const float* __restrict__ V,    // [B, H, N, D]
    float* __restrict__ O,          // [B, H, N, D]
    int N, float scale
) {
    int bh = blockIdx.x;
    int i = threadIdx.x;
    if (i >= N) return;

    int base = bh * N * D;

    // Phase 1: scores[i][j] = Q[i]·K[j] * scale
    float scores[1024];
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < D; d++) {
            dot += Q[base + i * D + d] * K[base + j * D + d];
        }
        scores[j] = dot * scale;
    }

    // Phase 2: Safe softmax
    float max_val = scores[0];
    for (int j = 1; j < N; j++) {
        max_val = fmaxf(max_val, scores[j]);
    }
    float sum_val = 0.0f;
    for (int j = 0; j < N; j++) {
        scores[j] = __expf(scores[j] - max_val);
        sum_val += scores[j];
    }

    // Pre-normalize softmax scores to save N*(D-1) multiplies in Phase 3
    float inv_sum = 1.0f / sum_val;
    for (int j = 0; j < N; j++) {
        scores[j] *= inv_sum;
    }

    // Phase 3: Weighted sum (scores already normalized)
    for (int d = 0; d < D; d++) {
        float out = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < N; j++) {
            out += scores[j] * V[base + j * D + d];
        }
        O[base + i * D + d] = out;
    }
}


// ============================================================
// Tiled Self-Attention Forward Kernel (Task 2)
//
// Key optimizations over naive:
//   1. Tile Q and K/V into shared memory for data reuse
//   2. Online (safe) softmax — maintain running max/sum across tiles
//   3. K and V loaded simultaneously into separate smem regions
//   4. D_PAD = D + 1 row stride eliminates 32-way bank conflicts
//
// Block layout: (TILE_N,) threads, each processes one query in the tile
// Grid layout:  (ceil(N/TILE_N), total_heads)
// Shared memory: Q_smem + K_smem + V_smem (3 regions)
//   Total: 3 * 32 * 65 * 4 = 24,960 bytes < 48 KB ✅
// ============================================================

template <int D>
__global__ void attention_tiled_kernel(
    const float* __restrict__ Q,    // [B, H, N, D]
    const float* __restrict__ K,    // [B, H, N, D]
    const float* __restrict__ V,    // [B, H, N, D]
    float* __restrict__ O,          // [B, H, N, D]
    int N, float scale
) {
    constexpr int D_PAD = D + 1;   // +1 padding to avoid shared memory bank conflicts

    int bh = blockIdx.y;
    int tile_start = blockIdx.x * TILE_N;
    int ti = threadIdx.x;
    int global_i = tile_start + ti;

    if (global_i >= N) return;

    int base = bh * N * D;

    // Shared memory: 3 separate regions (K and V no longer reused)
    // Row stride = D_PAD (not D) to avoid 32-way bank conflicts
    extern __shared__ float smem[];
    float* Q_smem = smem;                           // [TILE_N, D_PAD]
    float* K_smem = smem + TILE_N * D_PAD;          // [TILE_N, D_PAD]
    float* V_smem = smem + 2 * TILE_N * D_PAD;      // [TILE_N, D_PAD]

    // ----------------------------------------------------------
    // Load Q tile: each thread loads ALL D elements of its own row
    // ----------------------------------------------------------
    for (int d = 0; d < D; d++) {
        Q_smem[ti * D_PAD + d] = Q[base + global_i * D + d];
    }
    __syncthreads();

    // ----------------------------------------------------------
    // Online softmax running state (per thread, registers)
    // ----------------------------------------------------------
    float m = -INFINITY;
    float d_ = 0.0f;
    float O_local[D];  // D elements in registers
    for (int d = 0; d < D; d++) O_local[d] = 0.0f;

    // ----------------------------------------------------------
    // Main loop: iterate over K tiles
    // ----------------------------------------------------------
    for (int j_start = 0; j_start < N; j_start += TILE_N) {
        // --- Load K and V tiles simultaneously (no sync between them) ---
        int j_global = j_start + ti;
        if (j_global < N) {
            for (int d = 0; d < D; d++) {
                K_smem[ti * D_PAD + d] = K[base + j_global * D + d];
            }
            for (int d = 0; d < D; d++) {
                V_smem[ti * D_PAD + d] = V[base + j_global * D + d];
            }
        }
        __syncthreads();  // single sync after K+V loads (was 2)

        // --- Compute local scores: Q_smem[ti] · K_smem[jl] * scale ---
        float local_max = -INFINITY;
        float scores[TILE_N];

        for (int jl = 0; jl < TILE_N; jl++) {
            int jg = j_start + jl;
            if (jg >= N) {
                scores[jl] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                dot += Q_smem[ti * D_PAD + d] * K_smem[jl * D_PAD + d];
            }
            scores[jl] = dot * scale;
            local_max = fmaxf(local_max, scores[jl]);
        }

        // --- Online softmax update ---
        float m_new = fmaxf(m, local_max);
        float scale_old = __expf(m - m_new);

        // Scale old output (skip on first tile when m == -INFINITY)
        if (m != -INFINITY) {
            for (int d = 0; d < D; d++) {
                O_local[d] *= scale_old;
            }
        }

        // --- Fused: exp(scores - m_new) + O accumulation with V ---
        float sum_exp = 0.0f;
        for (int jl = 0; jl < TILE_N; jl++) {
            int jg = j_start + jl;
            if (jg >= N) break;
            float p = __expf(scores[jl] - m_new);
            sum_exp += p;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                O_local[d] += p * V_smem[jl * D_PAD + d];
            }
        }

        // Update running denominator and max
        d_ = d_ * scale_old + sum_exp;
        m = m_new;

        __syncthreads();  // ready for next iteration
    }

    // --- Final normalization: O /= d_ ---
    float inv_d = 1.0f / d_;
    for (int d = 0; d < D; d++) {
        O[base + global_i * D + d] = O_local[d] * inv_d;
    }
}


// ============================================================
// Launch wrappers (called from C++ bindings)
// ============================================================

void attention_naive_kernel_launch(
    const float* Q, const float* K, const float* V,
    float* O, int N, int D, float scale,
    int total_heads) {

    switch (D) {
        case 64:
            attention_naive_kernel<64><<<total_heads, N>>>(
                Q, K, V, O, N, scale
            );
            break;
        default:
            fprintf(stderr, "Naive kernel: unsupported D=%d (only D=64 built)\n", D);
            exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void attention_tiled_kernel_launch(
    const float* Q, const float* K, const float* V,
    float* O, int N, int D, float scale,
    int total_heads) {

    int grid_x = (N + TILE_N - 1) / TILE_N;
    dim3 grid(grid_x, total_heads);
    dim3 block(TILE_N);

    switch (D) {
        case 64: {
            constexpr int D_PAD = 65;  // D + 1 for bank-conflict-free padding
            size_t smem_bytes = 3 * TILE_N * D_PAD * sizeof(float);
            attention_tiled_kernel<64><<<grid, block, smem_bytes>>>(
                Q, K, V, O, N, scale
            );
            break;
        }
        default:
            fprintf(stderr, "Tiled kernel: unsupported D=%d (only D=64 built)\n", D);
            exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
