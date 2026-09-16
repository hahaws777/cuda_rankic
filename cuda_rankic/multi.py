"""Multiple file-backed factors sharing one raw return block at a time."""
from collections.abc import Mapping
from contextlib import ExitStack
import os
import queue
import time

from . import api, pipeline
from .memory import validate_max_memory_fraction


def _multi_reader(inputs, slots, capacity, free, ready, stop, finished, errors, begin, timed,
                  io_backend="python", io_block_bytes=16 * 1024**2, io_queue_depth=4):
    """Date-major reads: factor zero + returns, then each remaining factor."""
    factor_count = len(inputs) - 1
    try:
        with ExitStack() as resources:
            if io_backend == "python":
                handles = [resources.enter_context(i.path.open("rb", buffering=0)) for i in inputs]
                for handle, item in zip(handles, inputs):
                    handle.seek(item.offset)
            else:
                from ._windows_io import WindowsReader
                windows = resources.enter_context(WindowsReader(
                    [i.path for i in inputs], block_bytes=io_block_bytes,
                    queue_depth=io_queue_depth, direct=io_backend == "windows_direct"))
            for start in range(0, inputs[0].rows, capacity):
                count = min(capacity, inputs[0].rows - start)
                for factor in range(factor_count):
                    while not stop.is_set():
                        try:
                            index = free.get(timeout=.05)
                            break
                        except queue.Empty:
                            continue
                    else:
                        return
                    if stop.is_set():
                        return
                    slot = slots[index]
                    indices = [factor, factor_count] if factor == 0 else [factor]
                    destinations = [slot.host_x_array[:count]]
                    if factor == 0:
                        destinations.append(slot.host_y_array[:count])
                    start_ms = (time.perf_counter() - begin) * 1000 if timed else 0.
                    if io_backend == "python":
                        for file_index, destination in zip(indices, destinations):
                            if stop.is_set():
                                return
                            pipeline._read_exact_into(handles[file_index], destination)
                    else:
                        offsets = [inputs[i].offset + start * inputs[i].assets * 4 for i in indices]
                        windows.read_many(offsets, destinations, file_indices=indices, stop=stop)
                    end_ms = (time.perf_counter() - begin) * 1000 if timed else 0.
                    batch = pipeline._Batch(index, start, count, start_ms, end_ms, factor)
                    while not stop.is_set():
                        try:
                            ready.put(batch, timeout=.05)
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


def rank_ic_multi_files(factor_paths, future_return_path, *, max_memory_fraction=0.5,
                       chunk_rows=8192, pipeline_depth=3, device=None, return_stats=False,
                       io_backend="python", io_block_bytes=16 * 1024**2,
                       io_queue_depth=4, max_host_memory_bytes=None):
    """Compute [factors, dates] CPU RankIC, reading/uploading returns once.

    factor_paths is a nonempty iterable of file paths, in output order. All
    files, including returns, must have identical [dates, assets] shapes and
    native-endian C-contiguous float32 NPY storage. Files must remain unchanged
    during the call. Metadata validates shapes, not date/stock identities:
    callers must align every file's rows and columns before using this API.

    One date chunk stays in a shared GPU return buffer while factors stream
    through bounded slots. Raw returns are reused; ranks are recomputed per
    factor so factor-specific missing values retain pairwise-finite semantics.
    An event protects returns from overwrite until the last factor completes.
    CUDA kernels and all numeric rules are the same as rank_ic_files.

    GPU input bytes per chunk row are assets*4*(pipeline_depth+1); GPU staging
    output is 4*pipeline_depth bytes/row plus one native workspace. CPU input
    storage is assets*8*pipeline_depth bytes/row, independent of factor count;
    CPU output is factors*dates*4 bytes. max_host_memory_bytes includes these
    live buffers and I/O scratch, not allocator caches, runtime or process RSS.
    It may reduce chunk_rows; output alone exceeding the budget raises an error.

    Options and synchronization follow rank_ic_files. Defaults keep the portable
    Python reader; Windows backends share one I/O request pool across files.
    return_stats=True returns (output, stats). Logical read/upload counters
    exclude file headers, alignment padding and OS readahead. Stage times may
    overlap; timeline storage when requested grows with factors*chunk_count.
    One handle per factor plus returns stays open, subject to OS handle limits.
    """
    begin = time.perf_counter() if return_stats else 0.
    fraction = validate_max_memory_fraction(max_memory_fraction)
    chunk_rows = pipeline._positive_integer(chunk_rows, "chunk_rows")
    pipeline_depth = pipeline._positive_integer(pipeline_depth, "pipeline_depth")
    pipeline._validate_io(io_backend, io_block_bytes, io_queue_depth, max_host_memory_bytes)
    api._requested_device(device)
    if isinstance(factor_paths, (str, bytes, os.PathLike, Mapping)):
        raise TypeError("factor_paths must be a nonempty iterable of paths, not a single path or mapping")
    paths = tuple(factor_paths)
    if not paths:
        raise ValueError("at least one factor path is required")
    inputs = pipeline._read_inputs((*paths, future_return_path))
    prefix_ms = (time.perf_counter() - begin) * 1000 if return_stats else 0.
    result = pipeline._run_pipeline(
        inputs, max_memory_fraction=fraction, chunk_rows=chunk_rows,
        pipeline_depth=pipeline_depth, device=device, return_stats=return_stats,
        io_backend=io_backend, io_block_bytes=io_block_bytes, io_queue_depth=io_queue_depth,
        max_host_memory_bytes=max_host_memory_bytes, _shared_returns=True)
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
