<<<<<<< HEAD
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_arch_flags():
    """
    Determine -gencode flags in priority order:
      1. TORCH_CUDA_ARCH_LIST env var  (e.g. "8.6" or "7.5 8.0 8.6")
      2. The GPU(s) visible at build time
      3. A broad safe fallback covering Maxwell … Hopper
    """
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()

    if arch_list:
        archs = [a.strip() for a in arch_list.replace(",", " ").split() if a.strip()]
    elif torch.cuda.is_available():
        caps = set()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            caps.add(f"{major}.{minor}")
        archs = sorted(caps)
        print(f"[pointrope] Auto-detected GPU arch(s): {archs}")
    else:
        archs = ["7.5", "8.0", "8.6", "8.9", "9.0"]
        print(f"[pointrope] No GPU detected — building for fallback archs: {archs}")

    flags = []
    for arch in archs:
        code = arch.replace(".", "")
        flags += ["-gencode", f"arch=compute_{code},code=sm_{code}"]
    return flags


setup(
    name="pointrope",
    ext_modules=[
        CUDAExtension(
            name="pointrope",
            sources=["pointrope.cpp", "kernels.cu"],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + get_cuda_arch_flags(),
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
=======

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
# all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# check https://developer.nvidia.com/cuda-gpus to find your arch
all_cuda_archs = [
    '-gencode', 'arch=compute_90,code=sm_90',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    # '-gencode', 'arch=compute_86,code=sm_86'
]

setup(
    name = 'pointrope',
    ext_modules = [
        CUDAExtension(
                name='pointrope',
                sources=[
                    "pointrope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
>>>>>>> origin/main
