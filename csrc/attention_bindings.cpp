#include <torch/extension.h>

// ========== CUDA kernel declarations (implemented in .cu) ==========
void attention_naive_kernel_launch(
    const float* Q, const float* K, const float* V,
    float* O, int N, int D, float scale,
    int total_heads);

void attention_tiled_kernel_launch(
    const float* Q, const float* K, const float* V,
    float* O, int N, int D, float scale,
    int total_heads);

// ========== C++ wrappers with Torch tensor handling ==========

torch::Tensor attention_forward_naive(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    auto dims = Q.sizes();
    int B = dims[0], H = dims[1], N = dims[2], D = dims[3];
    int total_heads = B * H;

    TORCH_CHECK(N <= 1024, "Naive kernel requires N <= 1024, got ", N);

    auto O = torch::empty_like(Q);

    attention_naive_kernel_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), N, D, scale, total_heads);

    return O;
}


torch::Tensor attention_forward_tiled(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    auto dims = Q.sizes();
    int B = dims[0], H = dims[1], N = dims[2], D = dims[3];
    int total_heads = B * H;

    auto O = torch::empty_like(Q);

    attention_tiled_kernel_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), N, D, scale, total_heads);

    return O;
}


// ========== PYBIND11 bindings ==========
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward_naive", &attention_forward_naive,
          "Naive Scaled Dot-Product Attention Forward");
    m.def("attention_forward_tiled", &attention_forward_tiled,
          "Tiled Scaled Dot-Product Attention Forward");
}
