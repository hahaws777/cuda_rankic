import os
import sys
from pathlib import Path
from setuptools import Extension, setup
# PEP 517 executes setup.py without putting the project root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cuda import CUDABuildExt

setup(
    ext_modules=[Extension("cuda_rankic._native",
        sources=["csrc/python_module.c", "csrc/rankic_backend.cu"],
        include_dirs=["csrc"],
        depends=["csrc/rankic.h", "csrc/warp_rankic.cuh", "csrc/scan_rankic.cuh"],
        extra_compile_args={"c": ["/O2"] if os.name == "nt" else ["-O3"],
                            "nvcc": ["-lineinfo"]})],
    cmdclass={"build_ext": CUDABuildExt},
)
