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


// ========== Softmax kernel declarations ==========
void softmax_naive_launch(const float* input, float* output,
                          int rows, int D, int stride);
void softmax_warp_launch(const float* input, float* output,
                         int rows, int D, int stride);

// ========== Softmax C++ wrappers ==========
torch::Tensor softmax_forward_naive(torch::Tensor input) {
    input = input.contiguous();
    auto sizes = input.sizes();
    int D = sizes.back();
    int rows = 1;
    for (size_t i = 0; i < sizes.size() - 1; i++) rows *= sizes[i];
    auto output = torch::empty_like(input);
    softmax_naive_launch(input.data_ptr<float>(), output.data_ptr<float>(),
                         rows, D, D);
    return output;
}

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

// ========== PYBIND11 bindings ==========
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward_naive", &attention_forward_naive,
          "Naive Scaled Dot-Product Attention Forward");
    m.def("attention_forward_tiled", &attention_forward_tiled,
          "Tiled Scaled Dot-Product Attention Forward");
    m.def("softmax_forward_naive", &softmax_forward_naive,
          "Naive Softmax Forward");
    m.def("softmax_forward_warp", &softmax_forward_warp,
          "Warp-Reduction Softmax Forward");
}
