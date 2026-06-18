#include <torch/extension.h>

// ========== CUDA kernel declarations (implemented in .cu) ==========
void softmax_naive_launch(const float* input, float* output,
                          int rows, int D, int stride);
void softmax_warp_launch(const float* input, float* output,
                         int rows, int D, int stride);

// ========== C++ wrappers with Torch tensor handling ==========

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
    m.def("softmax_forward_naive", &softmax_forward_naive,
          "Naive Softmax Forward (serial per-row)");
    m.def("softmax_forward_warp", &softmax_forward_warp,
          "Warp-Reduction Softmax Forward");
}
