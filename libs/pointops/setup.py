import os
<<<<<<< HEAD
import torch
=======
>>>>>>> origin/main
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

<<<<<<< HEAD

=======
>>>>>>> origin/main
(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

<<<<<<< HEAD

def get_cuda_arch_flags():
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()

    if arch_list:
        archs = [a.strip() for a in arch_list.replace(",", " ").split() if a.strip()]
    elif torch.cuda.is_available():
        caps = set()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            caps.add(f"{major}.{minor}")
        archs = sorted(caps)
        print(f"[pointops] Auto-detected GPU arch(s): {archs}")
    else:
        archs = ["7.5", "8.0", "8.6", "8.9", "9.0"]
        print(f"[pointops] No GPU detected — building for fallback archs: {archs}")

    return [f"-gencode=arch=compute_{a.replace('.','')},code=sm_{a.replace('.','')}"
            for a in archs]


src = "src"
sources = [
    os.path.join(root, f)
    for root, _, files in os.walk(src)
    for f in files
    if f.endswith(".cpp") or f.endswith(".cu")
=======
src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
>>>>>>> origin/main
]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
<<<<<<< HEAD
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2"] + get_cuda_arch_flags(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
=======
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
>>>>>>> origin/main
