#ifndef CUDA_RANKIC_H
#define CUDA_RANKIC_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define RANKIC_API __declspec(dllexport)
#else
#define RANKIC_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
enum rankic_status {
    RANKIC_SUCCESS = 0,
    RANKIC_INVALID_ARGUMENT = 1,
    RANKIC_CUDA_ERROR = 2,
    RANKIC_WORKSPACE_TOO_SMALL = 3
};
/* Strategies: 0=auto, 1=block (cols<=6144), 2=segmented.
 * Select the CUDA device before calling. All arrays are device pointers.
 * Matrices are contiguous row-major float32; output contains rows floats.
 * Workspace must be 256-byte aligned (cudaMalloc satisfies this), with size
 * returned by rankic_workspace_size. No GPU allocations occur in either call.
 * Launch is asynchronous on the caller's cudaStream_t cast to void*.
 * Buffers must remain alive until the stream completes and must not overlap.
 * NaN/Inf pairs are excluded, ties use average ranks, degenerate rows yield NaN.
 * Error text is optional, caller-owned and always NUL-terminated when size>0.
 */
RANKIC_API int rankic_workspace_size(int64_t rows, int64_t cols, int strategy,
    size_t *bytes, char *error, size_t error_size);
RANKIC_API int rankic_cuda_f32(const float *x, const float *y, float *out,
    int64_t rows, int64_t cols, void *workspace, size_t workspace_bytes,
    int strategy, void *stream, char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
