from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = [
    CUDAExtension(
        "attention._C",
        [
            "csrc/attention_bindings.cpp",  # Torch bindings (g++)
            "csrc/attention_kernel.cu",     # CUDA kernels (nvcc)
        ],
        extra_compile_args={
            "cxx": ["-O3", "-fopenmp"],
            "nvcc": [
                "-O3",
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_89,code=sm_89",
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
