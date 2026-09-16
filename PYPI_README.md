# cuda_rankic

GPU cross-sectional Spearman rank correlation with a **C interface** and CUDA/CUB kernels. Python uses a small C extension, with no PyTorch C++ ABI dependency. Inputs have shape `[dates, assets]`; output is one float32 RankIC per date. Nonfinite pairs are excluded, tied values receive average ranks, and rows with fewer than two valid pairs or constant ranks return NaN.

## Installation

This initial distribution is a **source package**. Building requires Python 3.10+, the NVIDIA CUDA toolkit with `nvcc` (CUDA 12+ recommended), and a compatible C/C++ compiler (MSVC on Windows, GCC on Linux). Runtime requires a CUDA-enabled PyTorch installation providing `torch.cuda.get_per_process_memory_fraction`. Install the appropriate PyTorch build for your GPU first.

```powershell
# Example for an RTX 5070 Ti with CUDA 13 / MSVC installed:
$env:TORCH_CUDA_ARCH_LIST = "12.0"
python -m pip install .
```

Set `TORCH_CUDA_ARCH_LIST` for your GPU when building (e.g. `8.6` or `8.6;12.0+PTX`); otherwise the source build targets `7.5+PTX`. This default allows driver JIT on newer supported devices but native architecture builds are preferable. An architecture requires toolkit support. Linux builds accept the same variable, or `CUDAARCHS`. The local Windows wheel is specific to its Python/platform/CUDA environment and is not a universal wheel.

## Python

```python
import torch
from cuda_rankic import rank_ic

# CPU input is streamed through GPU chunks, with a CPU result.
factor = torch.randn(1000, 3000)
future_return = torch.randn_like(factor)
ic = rank_ic(factor, future_return, max_memory_fraction=0.5)

# GPU input returns a GPU result. Inputs must use the same device and dtype.
ic = rank_ic(factor.cuda(), future_return.cuda(), max_memory_fraction=0.4)
```

`max_memory_fraction` must be in `(0, 1]` and defaults to `0.5`. The wrapper temporarily caps the current process's PyTorch allocator at that fraction of total device memory, preserves any stricter existing cap, and restores the prior setting afterward. It counts existing allocations, output, staging buffers and CUB workspace and leaves a safety margin. Actual free VRAM can reduce the chunk size further. A process already reserving more than the cap gets a clear error; the wrapper does not silently release its cache. The cap excludes CUDA driver/context memory, allocations outside PyTorch, and other processes. It does not keep all programs combined under 50% or impose a cap on future unrelated allocations after the call. Calls through this package are serialized while changing the process-wide allocator setting; unrelated concurrent allocations are outside its control.

CPU inputs allow data larger than GPU memory. CUDA inputs already occupy GPU memory and count against the budget, so load large datasets on CPU first. `chunk_rows` optionally limits rows per launch. Smaller chunks can reduce throughput. Float32, float16 and bfloat16 inputs are supported; the latter two are promoted to float32 in reusable staging buffers. Strided inputs work; gradients and mixed input devices/dtypes are rejected. `strategy="auto"` selects the tuned implementation; `"block"` supports up to 6144 assets; `"segmented"` forces CUB segmented sorting. The budgeted Python wrapper does not support CUDA graph capture; the C interface can launch into a graph with workspace allocated beforehand.

## Overlap file loading and GPU computation

For data larger than RAM, use the file API directly:

```python
from cuda_rankic import rank_ic_files

ic = rank_ic_files(
    "factor.npy", "returns.npy",
    max_memory_fraction=0.5,
    chunk_rows=8192,
    pipeline_depth=3,
)
```

Both files must have the same `[dates, assets]` shape and contain native-endian,
C-contiguous float32 arrays. The result is a ready CPU float32 tensor. The
reader fills a bounded set of pinned CPU buffers while independent CUDA streams
upload, compute, and download earlier batches. CUDA events prevent a buffer
from being reused before its previous batch finishes. All slots and one shared
CUDA workspace count toward `max_memory_fraction`; the planner can reduce
`chunk_rows`. Host input staging is bounded by
`pipeline_depth * chunk_rows * assets * 8` bytes, plus `dates * 4` bytes for the
CPU output. Pinned host memory is separate from the GPU allocator limit.

Set `pipeline_depth=1` for the serialized reference path. `return_stats=True`
returns `(ic, stats)`, including stage durations and per-batch timelines. Stages
overlap, so their durations do not add up to total elapsed time. Cross-domain
CPU/GPU timeline alignment is approximate. Memory peak counters may include
earlier process activity: this API never resets them or clears allocator caches.
Profiling events/timelines are omitted by default.

This file API adds overlapped I/O; the existing tensor `rank_ic` API is unchanged.

### Windows disk-reading options

On Windows, optionally use queued disk requests independently of CUDA batch size:

```python
ic = rank_ic_files(
    "factor.npy", "returns.npy",
    max_memory_fraction=0.5,
    chunk_rows=8192, pipeline_depth=3,
    io_backend="windows_direct",
    io_block_bytes=4 * 1024**2,
    io_queue_depth=4,
    max_host_memory_bytes=2 * 1024**3,
)
```

`io_backend="python"` remains the portable default. `"windows"` uses Windows
overlapped reads and a sequential-access hint with OS caching. `"windows_direct"`
bypasses the OS data cache, reads aligned blocks into bounded temporary buffers,
and copies payload bytes into pinned staging; NPY headers and partial tails are
handled without modifying the files. It can suit scans larger than RAM; cached
reads may be faster for repeated small datasets. Neither option changes ranks or
input precision. The Windows modes require Windows 8+ and a filesystem/device
supporting the requested operations; failures are reported without silent fallback.

The queue depth (1–64) bounds outstanding requests across both files within one
batch. Block size (1 byte–1 GiB) is separate from `chunk_rows`; these two options
are unused by the Python backend. `max_host_memory_bytes` optionally limits live
input/output and I/O staging buffers by reducing batch rows. For direct I/O, the
planner reserves `io_queue_depth * (io_block_bytes + 3 * 65536)` bytes for aligned
scratch, a conservative bound. This host budget excludes allocator-retained
memory, Python/CUDA runtime memory, OS cache, other calls and other processes;
it is not a process RSS limit. The ready output still needs `dates * 4` bytes.

## Multiple factors sharing returns

```python
from cuda_rankic import rank_ic_multi_files

ic = rank_ic_multi_files(
    ["momentum.npy", "value.npy", "quality.npy"],
    "returns.npy",
    max_memory_fraction=0.5,
    max_host_memory_bytes=2 * 1024**3,
    chunk_rows=8192, pipeline_depth=3,
    io_backend="windows_direct",  # Windows; use "python" on other platforms
    io_block_bytes=4 * 1024**2, io_queue_depth=4,
)
# ic.shape == (3, dates); ic[0] corresponds to momentum.npy
```

The files must have matching `[dates, assets]` shapes, C-contiguous native float32
NPY storage, and the same row/column identities. The caller aligns dates and
stocks; the API validates array metadata. Each date chunk's raw returns are read
and uploaded exactly once, then retained on the GPU until every factor finishes.
Factors stream through reusable slots, so input staging does not grow with factor
count. Pairwise masks and ranks are computed separately for each factor, keeping
the same missing-value and tie behavior as independent `rank_ic_files` calls.

The ready CPU output is `[factor, date]` in input order and uses
`factor_count * dates * 4` bytes. It must fit the host buffer budget along with
staging; the API does not stream outputs to disk. GPU input staging uses
`chunk_rows * assets * 4 * (pipeline_depth + 1)` bytes plus output slots and one
workspace. CPU input staging and direct-I/O scratch follow the single-factor
budget. `return_stats=True` provides logical factor/return read and upload byte
counts and per-factor batch timings; profiling storage grows with the number of
work items. One file handle remains open per factor plus returns, so very large
factor lists are also subject to OS file-handle limits.

For F equal-sized factors and one return matrix of B bytes, independent calls
read `2*F*B` bytes; the shared path reads `(F+1)*B`. This saves return reads and
uploads, not the required reads or computation of each factor.

## Native C interface

See `csrc/rankic.h` and `C_API.md`. The CUDA backend accepts pointers, a
caller-owned workspace and a CUDA stream, and performs no GPU allocations.
The implementation requires nvcc/C++; callers can use ordinary C.

## Validation status

Development hardware: Windows, Python 3.12, CUDA 13.0, RTX 5070 Ti. Linux packaging has not yet been executed on a Linux host. No performance claim is made for untested GPUs.

Source code and releases: [GitHub](https://github.com/hahaws777/cuda_rankic).
Report bugs through [GitHub Issues](https://github.com/hahaws777/cuda_rankic/issues).

## License

MIT. See LICENSE. Copyright (c) 2026 cuda_rankic contributors.
