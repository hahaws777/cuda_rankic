# cuda_rankic

[PyPI](https://pypi.org/project/cuda-rankic/) · [Download files](https://pypi.org/project/cuda-rankic/#files) · [MIT license](LICENSE)

GPU cross-sectional Spearman RankIC with a C interface and CUDA/CUB kernels.
Inputs are `[dates, assets]`; output contains one RankIC per date. Supports
average ranks for ties, pairwise exclusion of nonfinite values, bounded GPU
memory and pipelined file processing for inputs larger than RAM.

## Install

Install a CUDA-enabled PyTorch build appropriate for your GPU, then:

```bash
pip install cuda_rankic
```

The initial release is a source distribution: building requires the NVIDIA
CUDA toolkit (`nvcc`) and a compatible C/C++ compiler. Python 3.10+ is required.
Tested on Windows, Python 3.12, CUDA 13, RTX 5070 Ti. Linux remains unverified.

## Usage

```python
import torch
from cuda_rankic import rank_ic, rank_ic_multi_files

factor = torch.randn(252, 5000)
returns = torch.randn_like(factor)
ic = rank_ic(factor, returns, max_memory_fraction=0.5)

# Larger-than-RAM input; returns are read/uploaded once per date chunk.
ic = rank_ic_multi_files(
    ["momentum.npy", "value.npy"], "returns.npy",
    max_memory_fraction=0.5,
    max_host_memory_bytes=2 * 1024**3,
)
# CPU result: [2, dates]
```

Files must be aligned, matching native float32 C-contiguous NPY arrays.
Input staging is bounded; the complete CPU output must fit in memory.
The GPU fraction caps this process's PyTorch allocator, not other programs
or CUDA context memory. Windows users can select `io_backend="windows_direct"`.

## Documentation

- [Complete Python API and installation](PYPI_README.md)
- [Native C interface](C_API.md)

## License

MIT. Copyright (c) 2026 cuda_rankic contributors.
