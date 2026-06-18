from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

nvcc_args = [
    "-O3",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=sm_80",
]

ext_modules = [
    # ---- Self-Attention extension ----
    CUDAExtension(
        "attention._C",
        [
            "csrc/attention_bindings.cpp",
            "csrc/attention_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-fopenmp"],
            "nvcc": nvcc_args,
        },
    ),
    # ---- Softmax extension ----
    CUDAExtension(
        "softmax._C",
        [
            "csrc/softmax_bindings.cpp",
            "csrc/softmax_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-fopenmp"],
            "nvcc": nvcc_args,
        },
    ),
]

setup(
    name="cuda-parallel",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["attention", "softmax"],
    package_dir={"attention": "attention", "softmax": "softmax"},
)
