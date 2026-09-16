"""PyTorch tensor frontend for the standalone CUDA C ABI backend.

The native module does not depend on PyTorch. This frontend uses PyTorch for
tensor ownership, transfers, stream selection, and a scoped allocator limit.
"""

from contextlib import contextmanager
import importlib
import operator
import os
import threading

import torch

from .memory import plan_chunk_rows, validate_max_memory_fraction


_STRATEGIES={"auto": 0, "block": 1, "segmented": 2}
_MEMORY_LOCK=threading.RLock()
_NATIVE=None
_DLL_HANDLES=[]


def _load_native():
    global _NATIVE
    with _MEMORY_LOCK:
        if _NATIVE is None:
            if os.name=="nt" and hasattr(os, "add_dll_directory"):
                cuda_path=os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                if cuda_path:
                    directory=os.path.join(cuda_path, "bin")
                    if os.path.isdir(directory):
                        _DLL_HANDLES.append(os.add_dll_directory(directory))
            try:
                _NATIVE=importlib.import_module("._native", __package__)
            except ImportError as exc:
                raise ImportError(
                    "cuda_rankic's native CUDA backend could not be loaded. "
                    "Build/install from source with `python -m pip install .` "
                    "using a CUDA toolkit and compatible C/C++ compiler; ensure "
                    "its CUDA runtime libraries are available."
                ) from exc
        return _NATIVE


def build_info():
    """Return the native backend's build metadata without importing PyTorch C++."""
    return _load_native().build_info()


def _requested_device(device):
    if device is None:
        return None
    try:
        if isinstance(device, bool):
            raise TypeError
        if isinstance(device, (str, torch.device)):
            selected=torch.device(device)
        else:
            selected=torch.device("cuda", operator.index(device))
    except (TypeError, ValueError, RuntimeError):
        raise ValueError("device must name a CUDA device") from None
    if selected.type!="cuda":
        raise ValueError("device must name a CUDA device")
    return selected


def _validate_inputs(factor, future_return, strategy, chunk_rows):
    if not isinstance(factor, torch.Tensor) or not isinstance(future_return, torch.Tensor):
        raise TypeError("factor and future_return must be torch.Tensor objects")
    if factor.ndim!=2 or factor.shape!=future_return.shape:
        raise ValueError("inputs must have the same two-dimensional [T, N] shape")
    if factor.device.type not in ("cpu", "cuda") or future_return.device.type not in ("cpu", "cuda"):
        raise ValueError("inputs must be CPU or CUDA tensors")
    if factor.device!=future_return.device:
        raise ValueError("inputs must be on the same device")
    if factor.layout!=torch.strided or future_return.layout!=torch.strided:
        raise TypeError("dense strided tensor layout required")
    allowed=(torch.float32, torch.float16, torch.bfloat16)
    if factor.dtype!=future_return.dtype or factor.dtype not in allowed:
        raise TypeError("matching float32, float16 or bfloat16 dtype required")
    if factor.requires_grad or future_return.requires_grad:
        raise ValueError("RankIC has no backward; detach inputs explicitly")
    if not isinstance(strategy, str) or strategy not in _STRATEGIES:
        raise ValueError("strategy must be 'auto', 'block', or 'segmented'")
    if strategy=="block" and factor.shape[1]>6144:
        raise ValueError("block strategy supports at most 6144 assets")
    if chunk_rows is not None:
        if isinstance(chunk_rows, bool):
            raise TypeError("chunk_rows must be a positive integer")
        try:
            chunk_rows=operator.index(chunk_rows)
        except TypeError:
            raise TypeError("chunk_rows must be a positive integer") from None
        if chunk_rows<1:
            raise ValueError("chunk_rows must be a positive integer")
    return chunk_rows


@contextmanager
def _allocator_budget(device, requested_fraction):
    """Scope our process-wide allocator change and expose incremental capacity."""
    with _MEMORY_LOCK:
        getter=getattr(torch.cuda, "get_per_process_memory_fraction", None)
        if getter is None:
            raise RuntimeError(
                "This frontend requires torch.cuda.get_per_process_memory_fraction "
                "to preserve and restore the existing allocator limit. Upgrade PyTorch."
            )
        previous=float(getter(device))
        effective=min(requested_fraction, previous)
        _, total_bytes=torch.cuda.mem_get_info(device)
        reserved=torch.cuda.memory_reserved(device)
        cap=int(total_bytes*effective)
        if reserved>cap:
            raise MemoryError(
                f"Existing PyTorch reserved GPU memory ({reserved} bytes) exceeds "
                f"the RankIC allocator budget ({cap} bytes, fraction {effective:g}). "
                "Release existing tensors or unused cache explicitly, or increase "
                "max_memory_fraction. RankIC does not clear the cache automatically."
            )
        torch.cuda.set_per_process_memory_fraction(float(effective), device)
        try:
            # Other callers do not share our lock. Recheck after installing the
            # cap so an allocation between the initial snapshot and the setter
            # cannot silently leave existing reservation above this call's cap.
            free_bytes, _=torch.cuda.mem_get_info(device)
            allocated=torch.cuda.memory_allocated(device)
            reserved=torch.cuda.memory_reserved(device)
            if reserved>cap:
                raise MemoryError(
                    f"PyTorch reserved GPU memory changed to {reserved} bytes "
                    f"while installing the RankIC allocator budget ({cap} bytes). "
                    "Concurrent allocations exceeded the requested budget."
                )
            unused_cache=max(0, reserved-allocated)
            available=max(0, min(cap-allocated, free_bytes+unused_cache))
            yield available
        except torch.cuda.OutOfMemoryError as exc:
            raise MemoryError(
                "RankIC could not allocate within the scoped GPU memory budget; "
                "allocator fragmentation or concurrent allocations may reduce "
                "the space available after planning."
            ) from exc
        finally:
            torch.cuda.set_per_process_memory_fraction(previous, device)


def rank_ic(factor, future_return, *, strategy="auto", max_memory_fraction=0.5,
            device=None, chunk_rows=None):
    """Compute per-row Spearman correlation with exact average ranks for ties.

    Inputs must be matching two-dimensional CPU or CUDA tensors with float32,
    float16, or bfloat16 dtype, on the same device, and without gradients. Only
    observations where both values are finite participate. Signed zeros tie;
    constant rows or fewer than two valid observations produce NaN. Returns
    float32 on the input device: CPU inputs are staged in GPU chunks and return
    a CPU tensor; CUDA inputs return a CUDA tensor without a host round trip.

    ``device`` selects the CUDA device for CPU inputs; for CUDA inputs it must
    match their device. ``chunk_rows`` is an optional upper bound on rows per
    launch. The planner may choose a smaller chunk. Half/bfloat16 and strided
    inputs are converted into reusable float32 chunk buffers.

    ``max_memory_fraction`` limits the current process's **PyTorch CUDA
    allocator** during this call, including its existing allocations and cache.
    A stricter previous allocator limit is preserved and restored on exit.
    Other CUDA libraries, driver allocations, and other applications are
    outside this cap. The planner also checks currently free GPU memory and
    reserves 64 MiB for allocation overhead. Existing reserved memory above the
    cap raises MemoryError without clearing the cache. Calls through this module
    serialize allocator-limit changes; unrelated threads using PyTorch see the
    temporary process-wide limit while a call is active.

    The current CUDA stream is respected. CPU results are ready on return;
    CUDA results are asynchronous. CUDA graph capture is explicitly unsupported
    by this frontend because its per-call budget checks and allocator changes
    cannot be enforced on graph replay. The native C ABI accepts CUDA streams.
    """
    fraction=validate_max_memory_fraction(max_memory_fraction)
    chunk_rows=_validate_inputs(factor, future_return, strategy, chunk_rows)
    target=_requested_device(device)
    rows, assets=map(int, factor.shape)
    cpu_input=factor.device.type=="cpu"

    # These CPU results need no CUDA initialization, native module, or staging.
    if cpu_input and (rows==0 or assets<2):
        return torch.full((rows,), float("nan"), dtype=torch.float32, device="cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required to compute RankIC")
    if target is None:
        target=factor.device if not cpu_input else torch.device("cuda", torch.cuda.current_device())
    elif target.index is None:
        target=torch.device("cuda", torch.cuda.current_device())
    if not cpu_input and target!=factor.device:
        raise ValueError("device must match the CUDA input device")

    with torch.cuda.device(target):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA graph capture is not supported by the memory-budgeted rank_ic frontend")
        if rows==0:
            return torch.empty((0,), dtype=torch.float32, device=target)
        native=_load_native() if assets>=2 else None
        stream=torch.cuda.current_stream(target)
        native_strategy=_STRATEGIES[strategy]
        convert_x=factor.dtype!=torch.float32 or not factor.is_contiguous() or factor.is_neg()
        convert_y=future_return.dtype!=torch.float32 or not future_return.is_contiguous() or future_return.is_neg()
        stage_x=cpu_input or convert_x
        stage_y=cpu_input or convert_y
        # Degenerate GPU rows require only the full output, without input copies.
        input_bytes=4*assets*(int(stage_x)+int(stage_y)) if assets>=2 else 0
        limit_rows=min(rows, chunk_rows) if chunk_rows is not None else rows
        workspace_size=(lambda r, n: native.workspace_size(r, n, native_strategy)) if native else (lambda r, n: 0)

        with _allocator_budget(target, fraction) as available:
            plan=plan_chunk_rows(
                limit_rows, assets, available, workspace_size,
                input_bytes_per_row=input_bytes,
                output_bytes_per_row=4 if cpu_input else 0,
                fixed_bytes=0 if cpu_input else rows*4,
            )
            output=torch.empty((rows,), dtype=torch.float32,
                               device="cpu" if cpu_input else target)
            if assets<2:
                output.fill_(float("nan"))
                return output
            capacity=plan.chunk_rows
            gx=torch.empty((capacity, assets), dtype=torch.float32, device=target) if stage_x else None
            gy=torch.empty((capacity, assets), dtype=torch.float32, device=target) if stage_y else None
            scratch=torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=target) if plan.workspace_bytes else None
            scratch_ptr=scratch.data_ptr() if scratch is not None else 0
            gpu_output=torch.empty((capacity,), dtype=torch.float32, device=target) if cpu_input else None

            # Convert on CPU before uploading: cross-device dtype conversion
            # must not create an unplanned temporary allocation on the GPU.
            host_x=(torch.empty((capacity, assets), dtype=torch.float32, device="cpu")
                    if cpu_input and convert_x else None)
            host_y=(torch.empty((capacity, assets), dtype=torch.float32, device="cpu")
                    if cpu_input and convert_y else None)
            if not cpu_input:
                factor.record_stream(stream)
                future_return.record_stream(stream)
            for start in range(0, rows, capacity):
                stop=min(rows, start+capacity)
                count=stop-start
                ax=factor[start:stop]
                ay=future_return[start:stop]
                if host_x is not None:
                    host_x[:count].copy_(ax)
                    ax=host_x[:count]
                if host_y is not None:
                    host_y[:count].copy_(ay)
                    ay=host_y[:count]
                if gx is not None:
                    gx[:count].copy_(ax, non_blocking=not cpu_input)
                    ax=gx[:count]
                if gy is not None:
                    gy[:count].copy_(ay, non_blocking=not cpu_input)
                    ay=gy[:count]
                destination=gpu_output[:count] if cpu_input else output[start:stop]
                native.run(ax.data_ptr(), ay.data_ptr(), destination.data_ptr(),
                           count, assets, scratch_ptr, plan.workspace_bytes,
                           native_strategy, stream.cuda_stream)
                if cpu_input:
                    # Blocking D2H also finishes this chunk before its staging
                    # buffers are reused and makes the returned CPU data ready.
                    output[start:stop].copy_(destination, non_blocking=False)
            return output
