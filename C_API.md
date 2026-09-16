# C ABI

Include `csrc/rankic.h` from a C or C++ caller. `rankic_workspace_size(rows, cols, strategy, &bytes, error, sizeof(error))` queries scratch requirements on the current CUDA device. Allocate device float32 row-major input arrays, a float32 output vector of length `rows`, and `bytes` of workspace using `cudaMalloc` (256-byte aligned). Call `rankic_cuda_f32` with those pointers and the caller's `cudaStream_t` cast to `void *`.

The library never allocates GPU memory. The caller owns the memory budget, device selection, buffer lifetimes and stream synchronization. Memory must not overlap. A successful return means work was enqueued; synchronize to detect asynchronous execution errors. Error buffers are caller-owned. `rows`, `cols`, and `rows*cols` must each be at most `INT_MAX/2` per launch; split larger matrices by rows. Strategies are 0 (auto), 1 (block, width <=6144), 2 (segmented). Empty batches succeed; widths 0 or 1 produce NaN. NaN and infinity pairs are excluded; ties receive average ranks.

Minimal launch sequence, after selecting a device and allocating x/y/output:

```c
#include "rankic.h"
#include <cuda_runtime_api.h>
size_t bytes = 0;
char error[512];
int rc = rankic_workspace_size(rows, cols, 0, &bytes, error, sizeof(error));
/* Check rc; then cudaMalloc workspace if bytes > 0. */
rc = rankic_cuda_f32(x, y, output, rows, cols, workspace, bytes,
                     0, (void *)stream, error, sizeof(error));
/* Check rc and cudaStreamSynchronize(stream) before consuming output. */
```

Build a standalone library without Python or PyTorch, for example on Linux:

```sh
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch=sm_86 \
  csrc/rankic_backend.cu -o librankic.so
```

On Windows run nvcc from a configured MSVC x64 environment, omit `-Xcompiler -fPIC`, and output `rankic.dll`. Choose an architecture supported by your GPU and CUDA toolkit. The Python extension also exports the two C functions, but loading that extension requires its Python runtime; the standalone library avoids that dependency. Only the caller-facing interface and CPython glue are C; CUDA/CUB kernels remain CUDA C++.
