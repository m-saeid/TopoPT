import os
<<<<<<< HEAD
import torch
=======
>>>>>>> origin/main
from sys import argv
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
        print(f"[pointgroup_ops] Auto-detected GPU arch(s): {archs}")
    else:
        archs = ["7.5", "8.0", "8.6", "8.9", "9.0"]
        print(f"[pointgroup_ops] No GPU detected — building for fallback archs: {archs}")

    return [f"-gencode=arch=compute_{a.replace('.','')},code=sm_{a.replace('.','')}"
            for a in archs]


=======
>>>>>>> origin/main
def _argparse(pattern, argv, is_flag=True, is_list=False):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern == arg.split("=")[0]]
        if is_list:
<<<<<<< HEAD
            if len(arr) == 0:
                return False, argv
            assert "=" in arr[0], f"{arr[0]} requires a value."
            argv.remove(arr[0])
            val = arr[0].split("=")[1]
            return (val.split(",") if "," in val else [val]), argv
        else:
            if len(arr) == 0:
                return False, argv
            assert "=" in arr[0], f"{arr[0]} requires a value."
            argv.remove(arr[0])
            return arr[0].split("=")[1], argv


INCLUDE_DIRS, argv = _argparse("--include_dirs", argv, False, is_list=True)
include_dirs = [] if INCLUDE_DIRS is False else list(INCLUDE_DIRS)
=======
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                val = arr[0].split("=")[1]
                if "," in val:
                    return val.split(","), argv
                else:
                    return [val], argv
        else:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                return arr[0].split("=")[1], argv


INCLUDE_DIRS, argv = _argparse("--include_dirs", argv, False, is_list=True)
include_dirs = []
if not (INCLUDE_DIRS is False):
    include_dirs += INCLUDE_DIRS
>>>>>>> origin/main

setup(
    name="pointgroup_ops",
    packages=["pointgroup_ops"],
    package_dir={"pointgroup_ops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointgroup_ops_cuda",
            sources=["src/bfs_cluster.cpp", "src/bfs_cluster_kernel.cu"],
<<<<<<< HEAD
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2"] + get_cuda_arch_flags(),
            },
        )
    ],
    include_dirs=include_dirs,
    cmdclass={"build_ext": BuildExtension},
)
=======
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    include_dirs=[*include_dirs],
    cmdclass={"build_ext": BuildExtension},
)
>>>>>>> origin/main
