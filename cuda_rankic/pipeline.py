"""Bounded file-to-GPU RankIC with overlapping reads, copies, and computation."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import operator
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch

from . import api
from .memory import plan_chunk_rows, validate_max_memory_fraction


@dataclass(frozen=True)
class FileInput:
    """Validated file metadata; no mapped input pages or open handles are kept."""

    path: Path
    offset: int
    rows: int
    assets: int


def _read_input_pair(paths):
    """Read two NPY headers, validate their layout, and close the header maps."""
    paths = tuple(paths)
    if len(paths) != 2:
        raise ValueError("exactly two input file paths are required")
    return _read_inputs(paths)


def _read_inputs(paths):
    """Validate same-shaped file headers without touching payload pages."""
    inputs = []
    for path in paths:
        path = Path(path).resolve()
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if not isinstance(array, np.memmap):
                raise TypeError("inputs must be .npy arrays, not an archive")
            if array.ndim != 2:
                raise ValueError("inputs must have the same two-dimensional [T, N] shape")
            if array.dtype != np.dtype("float32") or not array.dtype.isnative:
                raise TypeError("inputs must be native-endian float32 .npy arrays")
            if not array.flags.c_contiguous:
                raise TypeError("inputs must be C-contiguous .npy arrays")
            rows, assets = map(int, array.shape)
            inputs.append(FileInput(path, int(array.offset), rows, assets))
        finally:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
            elif hasattr(array, "close"):
                array.close()
    if not inputs:
        raise ValueError("at least one input file is required")
    if any((i.rows, i.assets) != (inputs[0].rows, inputs[0].assets) for i in inputs[1:]):
        raise ValueError("inputs must have the same two-dimensional [T, N] shape")
    return tuple(inputs)


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_exact_into(handle, destination):
    """Fill a contiguous array directly, including files that return short reads."""
    target = memoryview(destination).cast("B")
    try:
        offset = 0
        while offset < len(target):
            count = handle.readinto(target[offset:])
            if count is None:
                raise OSError("file read returned no data without reaching EOF")
            if count == 0:
                raise EOFError(f"input ended after {offset} of {len(target)} required chunk bytes")
            offset += count
    finally:
        target.release()


@dataclass
class _Slot:
    host_x: torch.Tensor
    host_y: torch.Tensor
    host_x_array: np.ndarray
    host_y_array: np.ndarray
    gpu_x: torch.Tensor
    gpu_y: torch.Tensor
    gpu_output: torch.Tensor
    uploaded: object
    computed: object
    downloaded: object
    h2d_start: object = None
    compute_start: object = None
    d2h_start: object = None


@dataclass(frozen=True)
class _Batch:
    slot: int
    start: int
    count: int
    load_start_ms: float = 0.0
    load_end_ms: float = 0.0
    factor: int = 0


def _reader(inputs, slots, capacity, free, ready, stop, finished, errors, begin, timed,
            io_backend="python", io_block_bytes=16 * 1024**2, io_queue_depth=4):
    """Own both file handles and touch a slot only while the CPU owns it."""
    try:
        with ExitStack() as resources:
            if io_backend == "python":
                handles = [resources.enter_context(item.path.open("rb", buffering=0)) for item in inputs]
                for handle, item in zip(handles, inputs):
                    handle.seek(item.offset)
            else:
                from ._windows_io import WindowsReader
                windows = resources.enter_context(WindowsReader(
                    [item.path for item in inputs], block_bytes=io_block_bytes,
                    queue_depth=io_queue_depth, direct=io_backend == "windows_direct"))
            for start in range(0, inputs[0].rows, capacity):
                while not stop.is_set():
                    try:
                        index = free.get(timeout=0.05)
                        break
                    except queue.Empty:
                        continue
                else:
                    return
                if stop.is_set():
                    return
                slot = slots[index]
                count = min(capacity, inputs[0].rows - start)
                load_start = (time.perf_counter() - begin) * 1000 if timed else 0.0
                if io_backend == "python":
                    _read_exact_into(handles[0], slot.host_x_array[:count])
                    if stop.is_set():
                        return
                    _read_exact_into(handles[1], slot.host_y_array[:count])
                else:
                    windows.read_pair(
                        [item.offset + start * item.assets * 4 for item in inputs],
                        [slot.host_x_array[:count], slot.host_y_array[:count]], stop=stop)
                load_end = (time.perf_counter() - begin) * 1000 if timed else 0.0
                batch = _Batch(index, start, count, load_start, load_end)
                while not stop.is_set():
                    try:
                        ready.put(batch, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                else:
                    return
    except BaseException as exc:
        if not (isinstance(exc, InterruptedError) and stop.is_set()):
            errors.append(exc)
    finally:
        finished.set()


def _stop_and_drain(reader, stop, streams):
    """Keep draining other streams if a reader join or CUDA sync itself fails."""
    stop.set()
    errors = []
    if reader is not None and reader.ident is not None:
        try:
            reader.join()
        except BaseException as exc:
            errors.append(exc)
    for stream in streams:
        try:
            stream.synchronize()
        except BaseException as exc:
            errors.append(exc)
    return errors


def _empty_stats(rows, assets, depth, elapsed_ms):
    return {
        "rows": rows, "assets": assets, "chunk_rows": 0, "chunk_count": 0,
        "pipeline_depth": depth, "workspace_bytes": 0, "pinned_input_bytes": 0,
        "pinned_output_bytes": 0, "pinned_cpu_bytes": 0, "gpu_buffer_bytes": 0,
        "load_cpu_ms": 0.0, "h2d_ms": 0.0, "kernel_ms": 0.0, "d2h_ms": 0.0,
        "setup_ms": elapsed_ms, "total_ms": elapsed_ms, "map_open_ms": 0.0,
        "baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0,
        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
        "observed_allocated_bytes": 0, "observed_reserved_bytes": 0,
        "total_gpu_bytes": None, "effective_memory_fraction": None,
        "allocator_cap_bytes": None, "available_before_plan_bytes": 0,
        "planned_bytes_including_safety": 0, "anchor_wall_ms": None,
        "timeline": [], "peak_stats_include_prior_work": True,
        "notes": ["Degenerate input needs no file payload reads or CUDA operations."],
    }


def _validate_io(io_backend, io_block_bytes, io_queue_depth, max_host_memory_bytes):
    if io_backend not in ("python", "windows", "windows_direct"):
        raise ValueError("io_backend must be python, windows, or windows_direct")
    block = _positive_integer(io_block_bytes, "io_block_bytes")
    depth = _positive_integer(io_queue_depth, "io_queue_depth")
    if block > 1024**3 or depth > 64:
        raise ValueError("I/O block must be <=1 GiB and queue depth <=64")
    if io_backend != "python" and os.name != "nt":
        raise RuntimeError("Windows file I/O requires Windows")
    host = (None if max_host_memory_bytes is None else
            _positive_integer(max_host_memory_bytes, "max_host_memory_bytes"))
    from ._windows_io import scratch_bytes
    return block, depth, host, scratch_bytes(block, depth, io_backend == "windows_direct")


def _run_pipeline(inputs, *, max_memory_fraction=0.5, chunk_rows=8192,
                  pipeline_depth=3, device=None, return_stats=False,
                  io_backend="python", io_block_bytes=16 * 1024**2,
                  io_queue_depth=4, max_host_memory_bytes=None, _shared_returns=False):
    """Run validated inputs; private callers may provide synthetic payload views."""
    begin = time.perf_counter() if return_stats else 0.0
    fraction = validate_max_memory_fraction(max_memory_fraction)
    requested_rows = _positive_integer(chunk_rows, "chunk_rows")
    depth = _positive_integer(pipeline_depth, "pipeline_depth")
    io_block_bytes, io_queue_depth, host_cap, io_scratch = _validate_io(
        io_backend, io_block_bytes, io_queue_depth, max_host_memory_bytes)
    target = api._requested_device(device)
    inputs = tuple(inputs)
    if ((len(inputs) < 2 if _shared_returns else len(inputs) != 2)
            or not all(isinstance(item, FileInput) for item in inputs)):
        raise TypeError("inputs must contain factor(s) followed by returns as FileInput objects")
    factor_count = len(inputs) - 1
    rows, assets = inputs[0].rows, inputs[0].assets
    if any((rows, assets) != (i.rows, i.assets) for i in inputs[1:]) or rows < 0 or assets < 0:
        raise ValueError("inputs must have the same nonnegative two-dimensional shape")
    output_shape = (factor_count, rows) if _shared_returns else (rows,)
    output_bytes = factor_count * rows * 4
    gpu_input_row_bytes = assets * 4 * (depth + 1 if _shared_returns else 2 * depth)
    if rows == 0 or assets < 2:
        if host_cap is not None and output_bytes > host_cap:
            raise MemoryError("host memory budget cannot hold the output")
        output = torch.full(output_shape, float("nan"), dtype=torch.float32, device="cpu")
        if return_stats:
            stats = _empty_stats(rows, assets, depth, (time.perf_counter() - begin) * 1000)
            stats.update(io_backend=io_backend, io_block_bytes=io_block_bytes,
                         io_queue_depth=io_queue_depth, io_scratch_budget_bytes=0,
                         managed_host_buffer_bytes=output_bytes, max_host_memory_bytes=host_cap,
                         factor_count=factor_count, work_item_count=0, factor_read_bytes=0,
                         return_read_bytes=0, factor_upload_bytes=0, return_upload_bytes=0)
            return output, stats
        return output
    if host_cap is not None:
        host_rows = (host_cap - output_bytes - io_scratch) // (assets * 8 * depth)
        if host_rows < 1:
            raise MemoryError("host memory budget cannot hold output, I/O scratch and one input row per slot")
        requested_rows = min(requested_rows, host_rows)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required to compute RankIC")
    if target is None or target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())

    with torch.cuda.device(target):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA graph capture is not supported by rank_ic_files")
        native = api._load_native()
        with api._allocator_budget(target, fraction) as available:
            plan = plan_chunk_rows(
                min(rows, requested_rows), assets, available,
                lambda r, n: native.workspace_size(r, n, 0),
                input_bytes_per_row=gpu_input_row_bytes,
                output_bytes_per_row=4 * depth,
            )
            capacity = plan.chunk_rows
            chunk_count = (rows + capacity - 1) // capacity
            streams, slots = [], []
            reader = None
            stop = threading.Event()
            failure = None
            stats = None
            try:
                if return_stats:
                    baseline_allocated = torch.cuda.memory_allocated(target)
                    baseline_reserved = torch.cuda.memory_reserved(target)
                    _, total_gpu = torch.cuda.mem_get_info(target)
                    effective = torch.cuda.get_per_process_memory_fraction(target)
                current = torch.cuda.current_stream(target)
                for _ in range(3):
                    streams.append(torch.cuda.Stream(device=target))
                h2d, compute, d2h = streams
                output = torch.empty(output_shape, dtype=torch.float32, device="cpu", pin_memory=True)
                scratch = (torch.empty((plan.workspace_bytes,), dtype=torch.uint8, device=target)
                           if plan.workspace_bytes else None)
                scratch_ptr = scratch.data_ptr() if scratch is not None else 0
                shared_y = (torch.empty((capacity, assets), dtype=torch.float32, device=target)
                            if _shared_returns else None)
                returns_consumed = torch.cuda.Event() if _shared_returns else None
                for _ in range(depth):
                    host_x = torch.empty((capacity, assets), dtype=torch.float32,
                                         device="cpu", pin_memory=True)
                    host_y = torch.empty_like(host_x, pin_memory=True)
                    slots.append(_Slot(
                        host_x, host_y, host_x.numpy(), host_y.numpy(),
                        torch.empty((capacity, assets), dtype=torch.float32, device=target),
                        shared_y if _shared_returns else torch.empty(
                            (capacity, assets), dtype=torch.float32, device=target),
                        torch.empty((capacity,), dtype=torch.float32, device=target),
                        torch.cuda.Event(enable_timing=return_stats),
                        torch.cuda.Event(enable_timing=return_stats),
                        torch.cuda.Event(enable_timing=return_stats),
                        *([torch.cuda.Event(enable_timing=True) for _ in range(3)]
                          if return_stats else [None, None, None]),
                    ))

                # Empty allocations can reuse blocks with outstanding work on
                # the caller's stream. This dependency is required even without
                # timing; retaining buffers alone only protects their lifetime.
                anchor = torch.cuda.Event(enable_timing=return_stats)
                anchor.record(current)
                if return_stats:
                    anchor.synchronize()
                    anchor_wall_ms = (time.perf_counter() - begin) * 1000
                for stream in streams:
                    stream.wait_event(anchor)
                if return_stats:
                    stats = {
                        "rows": rows, "assets": assets, "chunk_rows": capacity,
                        "chunk_count": chunk_count, "pipeline_depth": depth,
                        "factor_count": factor_count, "work_item_count": chunk_count * factor_count,
                        "factor_read_bytes": 0, "return_read_bytes": 0,
                        "factor_upload_bytes": 0, "return_upload_bytes": 0,
                        "workspace_bytes": plan.workspace_bytes,
                        "pinned_input_bytes": capacity * assets * 8 * depth,
                        "pinned_output_bytes": output_bytes,
                        "pinned_cpu_bytes": capacity * assets * 8 * depth + output_bytes,
                        "gpu_buffer_bytes": capacity * (gpu_input_row_bytes + 4 * depth),
                        "load_cpu_ms": 0.0, "h2d_ms": 0.0,
                        "kernel_ms": 0.0, "d2h_ms": 0.0,
                        "baseline_allocated_bytes": baseline_allocated,
                        "baseline_reserved_bytes": baseline_reserved,
                        "observed_allocated_bytes": torch.cuda.memory_allocated(target),
                        "observed_reserved_bytes": torch.cuda.memory_reserved(target),
                        "total_gpu_bytes": total_gpu,
                        "effective_memory_fraction": effective,
                        "allocator_cap_bytes": int(total_gpu * effective),
                        "available_before_plan_bytes": available,
                        "planned_bytes_including_safety": plan.estimated_bytes,
                        "anchor_wall_ms": anchor_wall_ms,
                        "map_open_ms": 0.0, "timeline": [],
                        "io_backend": io_backend, "io_block_bytes": io_block_bytes,
                        "io_queue_depth": io_queue_depth,
                        "io_scratch_budget_bytes": io_scratch,
                        "max_host_memory_bytes": host_cap,
                        "managed_host_buffer_bytes": capacity * assets * 8 * depth + output_bytes + io_scratch,
                        "peak_stats_include_prior_work": True,
                        "notes": [
                            "Stage durations overlap, so their sum is not total elapsed time.",
                            "In shared-return mode H2D intervals include any wait for the previous date chunk's return consumers.",
                            "load_cpu_ms covers both inputs, including direct-I/O bounce copies when selected. OS file cache is bypassed only by windows_direct.",
                            "Timeline intervals use milliseconds from call start. GPU intervals add anchor_wall_ms to a common CUDA-event anchor for approximate CPU/GPU clock alignment.",
                            "Global allocator peak counters are read without resetting and may include earlier work or concurrent callers.",
                            "The allocator cap includes this process's PyTorch allocations and cache; driver allocations and other CUDA allocators are outside it.",
                        ],
                    }
                free, ready = queue.Queue(maxsize=depth), queue.Queue(maxsize=depth)
                for index in range(depth):
                    free.put_nowait(index)
                finished = threading.Event()
                reader_errors = []
                if _shared_returns:
                    from .multi import _multi_reader
                reader = threading.Thread(
                    target=_multi_reader if _shared_returns else _reader,
                    name="cuda-rankic-reader", daemon=False,
                    args=(inputs, slots, capacity, free, ready, stop, finished,
                          reader_errors, begin, return_stats),
                    kwargs=({"io_backend": io_backend, "io_block_bytes": io_block_bytes,
                             "io_queue_depth": io_queue_depth} if io_backend != "python" else {}),
                )
                if return_stats:
                    stats["setup_ms"] = (time.perf_counter() - begin) * 1000
                reader.start()
                inflight = deque()

                def retire_oldest():
                    batch = inflight.popleft()
                    slot = slots[batch.slot]
                    slot.downloaded.synchronize()
                    if return_stats:
                        interval = {
                            "chunk_index": batch.start // capacity,
                            "slot": batch.slot, "row_start": batch.start,
                            "row_count": batch.count,
                            "factor_index": batch.factor,
                            "load_start_ms": batch.load_start_ms,
                            "load_end_ms": batch.load_end_ms,
                        }
                        for stage, start, end, total_key in (
                            ("h2d", slot.h2d_start, slot.uploaded, "h2d_ms"),
                            ("compute", slot.compute_start, slot.computed, "kernel_ms"),
                            ("d2h", slot.d2h_start, slot.downloaded, "d2h_ms"),
                        ):
                            interval[f"{stage}_start_ms"] = anchor_wall_ms + anchor.elapsed_time(start)
                            interval[f"{stage}_end_ms"] = anchor_wall_ms + anchor.elapsed_time(end)
                            stats[total_key] += start.elapsed_time(end)
                        stats["load_cpu_ms"] += batch.load_end_ms - batch.load_start_ms
                        payload_bytes = batch.count * assets * 4
                        stats["factor_read_bytes"] += payload_bytes
                        stats["factor_upload_bytes"] += payload_bytes
                        if batch.factor == 0:
                            stats["return_read_bytes"] += payload_bytes
                            stats["return_upload_bytes"] += payload_bytes
                        stats["timeline"].append(interval)
                    # No read or copy may reuse this slot until its D2H ended.
                    free.put_nowait(batch.slot)

                for _ in range(chunk_count * factor_count):
                    if reader_errors:
                        raise reader_errors[0]
                    while inflight and slots[inflight[0].slot].downloaded.query():
                        retire_oldest()
                    # With every slot on the GPU, the reader cannot produce a
                    # ready item until we return a completed slot. Drain first.
                    if len(inflight) == depth:
                        retire_oldest()
                    while True:
                        if reader_errors:
                            raise reader_errors[0]
                        try:
                            batch = ready.get(timeout=0.05)
                            break
                        except queue.Empty:
                            if finished.is_set():
                                if reader_errors:
                                    raise reader_errors[0]
                                # The final put can race with the timeout. Once
                                # finished is visible, check that put once more.
                                try:
                                    batch = ready.get_nowait()
                                    break
                                except queue.Empty:
                                    raise RuntimeError(
                                        "file reader ended before all chunks were received") from None
                    slot = slots[batch.slot]
                    count = batch.count
                    with torch.cuda.stream(h2d):
                        if return_stats:
                            slot.h2d_start.record(h2d)
                        slot.gpu_x[:count].copy_(slot.host_x[:count], non_blocking=True)
                        if batch.factor == 0:
                            # The prior date chunk's final factor must finish
                            # consuming shared_y before its next upload starts.
                            if _shared_returns and batch.start > 0:
                                h2d.wait_event(returns_consumed)
                            slot.gpu_y[:count].copy_(slot.host_y[:count], non_blocking=True)
                        slot.uploaded.record(h2d)
                    with torch.cuda.stream(compute):
                        compute.wait_event(slot.uploaded)
                        if return_stats:
                            slot.compute_start.record(compute)
                        native.run(slot.gpu_x.data_ptr(), slot.gpu_y.data_ptr(),
                                   slot.gpu_output.data_ptr(), count, assets,
                                   scratch_ptr, plan.workspace_bytes, 0, compute.cuda_stream)
                        slot.computed.record(compute)
                        if _shared_returns and batch.factor == factor_count - 1:
                            returns_consumed.record(compute)
                    with torch.cuda.stream(d2h):
                        d2h.wait_event(slot.computed)
                        if return_stats:
                            slot.d2h_start.record(d2h)
                        destination = output[batch.factor] if _shared_returns else output
                        destination[batch.start:batch.start + count].copy_(
                            slot.gpu_output[:count], non_blocking=True)
                        slot.downloaded.record(d2h)
                    inflight.append(batch)
                    if depth == 1:
                        retire_oldest()
                while inflight:
                    retire_oldest()
                if reader_errors:
                    raise reader_errors[0]
            except BaseException as exc:
                failure = exc
                raise
            finally:
                # Reader, slots, output and shared scratch remain strongly held
                # through every owned stream sync, including partial launches.
                cleanup_errors = _stop_and_drain(reader, stop, streams)
                if cleanup_errors:
                    if failure is None:
                        raise cleanup_errors[0]
                    if hasattr(failure, "add_note"):
                        failure.add_note("Additional errors during pipeline cleanup: " +
                                         "; ".join(str(error) for error in cleanup_errors))
            # The worker can fail while closing its files after publishing the
            # last batch. Only joining it makes this final check authoritative.
            if reader_errors:
                raise reader_errors[0]
            if return_stats:
                stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(target)
                stats["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(target)
                stats["observed_allocated_bytes"] = max(
                    stats["observed_allocated_bytes"], torch.cuda.memory_allocated(target))
                stats["observed_reserved_bytes"] = max(
                    stats["observed_reserved_bytes"], torch.cuda.memory_reserved(target))
        if return_stats:
            stats["total_ms"] = (time.perf_counter() - begin) * 1000
            return output, stats
        return output


def rank_ic_files(factor_path, future_return_path, *, max_memory_fraction=0.5,
                  chunk_rows=8192, pipeline_depth=3, device=None, return_stats=False,
                  io_backend="python", io_block_bytes=16 * 1024**2,
                  io_queue_depth=4, max_host_memory_bytes=None):
    """Compute RankIC from matching native-float32, C-contiguous 2D NPY files.

    Returns a ready CPU float32 tensor, one value per input row. Ties, missing
    values, signed zeros and degenerate rows follow :func:`rank_ic`. Input
    files must remain unchanged for the duration of the call. Header-only
    read-only mappings validate metadata; a single worker reads payloads
    directly into a bounded set of pinned buffers without loading full files.

    ``pipeline_depth=3`` overlaps file reads, uploads, native GPU computation,
    and result downloads. ``pipeline_depth=1`` is fully serialized. The memory
    planner may reduce ``chunk_rows`` to fit all slots, one shared workspace,
    and a 64 MiB safety reserve within the scoped PyTorch allocator cap. It
    preserves stricter existing caps and restores the previous cap on exit.
    Pinned input storage is bounded by depth * chunk_rows * assets * 8 bytes;
    the ready pinned CPU output additionally uses rows * 4 bytes.

    The caller's current CUDA stream is respected. Only this call's streams
    are synchronized; the API neither clears cache nor resets global memory
    peak counters. CUDA graph capture is unsupported. CPU-only degenerate
    shapes (zero rows or fewer than two assets) require no CUDA operations.

    Set ``return_stats=True`` for ``(output, stats)`` with phase durations and
    per-chunk timelines. All timeline intervals use call start as their origin;
    GPU intervals already include ``anchor_wall_ms`` to approximately align
    their shared CUDA-event anchor with CPU time. Durations overlap and do not add up to
    total time. Memory peaks can include prior or concurrent process activity.
    Timing events and per-chunk timeline storage are omitted by default.

    ``io_backend='python'`` preserves synchronous readinto behavior (I/O block
    and queue options are unused). On Windows, ``'windows'`` uses overlapped
    reads with a sequential cache hint; ``'windows_direct'`` also bypasses the
    OS data cache via sector-aligned bounce buffers, handling NPY headers and
    file tails. Hardware/controller caches remain enabled. ``io_queue_depth``
    bounds outstanding requests across both files within a batch; it is
    independent of pipeline depth. ``io_block_bytes`` bounds each logical read.
    ``max_host_memory_bytes`` optionally caps this call's live input/output and
    I/O staging storage, reducing chunk rows as needed. It excludes Python,
    CUDA context, allocator-retained memory, OS cache, and other calls/processes.
    This is a buffer budget, not a process RSS or system RAM limit.
    """
    begin = time.perf_counter() if return_stats else 0.0
    # Reject configuration errors before opening files or initializing CUDA.
    fraction = validate_max_memory_fraction(max_memory_fraction)
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    pipeline_depth = _positive_integer(pipeline_depth, "pipeline_depth")
    _validate_io(io_backend, io_block_bytes, io_queue_depth, max_host_memory_bytes)
    api._requested_device(device)
    inputs = _read_input_pair((factor_path, future_return_path))
    prefix_ms = (time.perf_counter() - begin) * 1000 if return_stats else 0.0
    result = _run_pipeline(
        inputs, max_memory_fraction=fraction, chunk_rows=chunk_rows,
        pipeline_depth=pipeline_depth, device=device, return_stats=return_stats,
        io_backend=io_backend, io_block_bytes=io_block_bytes,
        io_queue_depth=io_queue_depth, max_host_memory_bytes=max_host_memory_bytes,
    )
    if return_stats:
        output, stats = result
        stats["map_open_ms"] = prefix_ms
        stats["total_ms"] = (time.perf_counter() - begin) * 1000
        if stats["anchor_wall_ms"] is not None:
            stats["anchor_wall_ms"] += prefix_ms
        for interval in stats["timeline"]:
            for stage in ("load", "h2d", "compute", "d2h"):
                interval[f"{stage}_start_ms"] += prefix_ms
                interval[f"{stage}_end_ms"] += prefix_ms
        return output, stats
    return result
