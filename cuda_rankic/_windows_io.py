"""Bounded Win32 overlapped reads; optional aligned, uncached disk I/O.

Each instance belongs to one reader thread. All requests finish or are cancelled
and drained before returning, so callers may safely recycle their destinations.
"""
from collections import deque
import ctypes as c
from ctypes import wintypes as w
import os
import operator

import numpy as np


# Reserve an upper bound before opening files; reject unusual larger sectors.
MAX_ALIGNMENT = 65536


def scratch_bytes(block_bytes, queue_depth, direct):
    return queue_depth * (block_bytes + 3 * MAX_ALIGNMENT) if direct else 0


class _Overlapped(c.Structure):
    _fields_ = [("Internal", c.c_size_t), ("InternalHigh", c.c_size_t),
                ("Offset", w.DWORD), ("OffsetHigh", w.DWORD), ("hEvent", w.HANDLE)]


def _kernel32():
    if os.name != "nt":
        raise RuntimeError("Windows file I/O requires Windows")
    k = c.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateFileW": ([w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE], w.HANDLE),
        "CreateEventW": ([c.c_void_p, w.BOOL, w.BOOL, w.LPCWSTR], w.HANDLE),
        "ReadFile": ([w.HANDLE, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.POINTER(_Overlapped)], w.BOOL),
        "GetOverlappedResult": ([w.HANDLE, c.POINTER(_Overlapped), c.POINTER(w.DWORD), w.BOOL], w.BOOL),
        "CancelIoEx": ([w.HANDLE, c.POINTER(_Overlapped)], w.BOOL),
        "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        "GetFileInformationByHandleEx": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
        "CloseHandle": ([w.HANDLE], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(k, name)
        fn.argtypes, fn.restype = args, result
    return k


class WindowsReader:
    def __init__(self, paths, *, block_bytes=16 * 1024**2, queue_depth=4,
                 direct=False, sequential=True):
        self.k = _kernel32()
        if not 0 < block_bytes <= 1024**3 or not 0 < queue_depth <= 64:
            raise ValueError("I/O block must be 1..1 GiB and queue depth 1..64")
        self.block_bytes, self.direct = block_bytes, direct
        self.handles, self.requests = [], []
        self.pending = deque()
        self.closed = False
        self.alignment = 1
        self.peak_outstanding = 0
        self.buffer_bytes = 0
        try:
            flags = 0x40000000 | (0x20000000 if direct else (0x08000000 if sequential else 0))
            for path in paths:
                # Read sharing only: source mutation is outside the API contract.
                handle = self.k.CreateFileW(str(path), 0x80000000, 1, None, 3, flags, None)
                if handle == c.c_void_p(-1).value:
                    raise c.WinError(c.get_last_error())
                self.handles.append(handle)
                if direct:
                    storage = (w.DWORD * 7)()
                    alignment_mask = w.DWORD()
                    if not self.k.GetFileInformationByHandleEx(handle, 16, storage, c.sizeof(storage)):
                        raise c.WinError(c.get_last_error())
                    if not self.k.GetFileInformationByHandleEx(handle, 17, c.byref(alignment_mask), 4):
                        raise c.WinError(c.get_last_error())
                    alignment = max(*storage[:4], alignment_mask.value + 1)
                    if alignment <= 0 or alignment > MAX_ALIGNMENT or alignment & (alignment - 1):
                        raise OSError(f"unsupported storage alignment: {alignment}")
                    self.alignment = max(self.alignment, alignment)
            if not self.handles:
                raise ValueError("at least one file required")
            for _ in range(queue_depth):
                event = self.k.CreateEventW(None, True, False, None)
                if not event:
                    raise c.WinError(c.get_last_error())
                request = {"ov": _Overlapped(hEvent=event), "raw": None}
                self.requests.append(request)
                if direct:
                    raw = np.empty(block_bytes + 3 * self.alignment, dtype=np.uint8)
                    pad = (-raw.ctypes.data) % self.alignment
                    request["raw"] = raw
                    request["buffer"] = raw[pad:]
                    self.buffer_bytes += raw.nbytes
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _submit(self, request, handle, offset, destination):
        n = destination.nbytes
        skip = offset % self.alignment if self.direct else 0
        amount = ((skip + n + self.alignment - 1) // self.alignment * self.alignment
                  if self.direct else n)
        pointer = request["buffer"].ctypes.data if self.direct else destination.ctypes.data
        position = offset - skip
        ov = request["ov"]
        ov.Internal = ov.InternalHigh = 0
        ov.Offset, ov.OffsetHigh = position & 0xffffffff, position >> 32
        request.update(handle=handle, destination=destination, skip=skip, size=n)
        # Track before crossing into native code: an asynchronous exception
        # just after ReadFile accepts the request must still cancel and drain it.
        # Zero status also makes a failure before native submission nonpending.
        self.pending.append(request)
        ok = self.k.ReadFile(handle, pointer, amount, None, c.byref(ov))
        error = c.get_last_error() if not ok else 0
        if not ok and error != 997:  # ERROR_IO_PENDING
            if error == 38:
                raise EOFError("input ended before requested bytes")
            raise c.WinError(error)
        self.peak_outstanding = max(self.peak_outstanding, len(self.pending))

    def _complete(self, request, stop):
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("file reading stopped")
            status = self.k.WaitForSingleObject(request["ov"].hEvent, 50)
            if status == 0:
                break
            if status != 258:  # WAIT_TIMEOUT
                raise c.WinError(c.get_last_error())
        count = w.DWORD()
        if not self.k.GetOverlappedResult(request["handle"], c.byref(request["ov"]), c.byref(count), True):
            error = c.get_last_error()
            if error == 38:
                raise EOFError("input ended before requested bytes")
            raise c.WinError(error)
        needed = request["skip"] + request["size"]
        if count.value < needed:
            raise EOFError(f"input ended after {count.value} of {needed} required bytes")
        if self.direct:
            np.copyto(request["destination"], request["buffer"][request["skip"]:needed])

    def _drain(self):
        # CancelIoEx does not wait: always collect completion before releasing
        # OVERLAPPED structures, events, or source/destination storage.
        for request in self.pending:
            self.k.CancelIoEx(request["handle"], c.byref(request["ov"]))
        for request in self.pending:
            count = w.DWORD()
            self.k.GetOverlappedResult(request["handle"], c.byref(request["ov"]), c.byref(count), True)
        self.pending.clear()

    def read_pair(self, offsets, destinations, *, stop=None):
        if len(self.handles) != 2 or len(offsets) != 2 or len(destinations) != 2:
            raise ValueError("two files, offsets and destinations required")
        return self.read_many(offsets, destinations, stop=stop)

    def read_many(self, offsets, destinations, *, file_indices=None, stop=None):
        """Read selected files into disjoint buffers using one shared queue."""
        if self.closed:
            raise RuntimeError("reader is closed")
        indices = list(range(len(self.handles))) if file_indices is None else list(file_indices)
        indices = [operator.index(i) for i in indices]
        offsets = [operator.index(o) for o in offsets]
        if (not indices or len(offsets) != len(indices) or len(destinations) != len(indices)
                or any(o < 0 for o in offsets)
                or any(i < 0 or i >= len(self.handles) for i in indices)):
            raise ValueError("matching valid file indices, nonnegative offsets and destinations required")
        handles = [self.handles[i] for i in indices]
        arrays = []
        for destination in destinations:
            array = np.asarray(destination)
            if not array.flags.c_contiguous or not array.flags.writeable or array.dtype.hasobject:
                raise TypeError("destinations must be writable contiguous numeric arrays")
            arrays.append(array.reshape(-1).view(np.uint8))
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1:]):
            raise ValueError("read destinations must not overlap")

        def jobs():
            for start in range(0, max(a.nbytes for a in arrays), self.block_bytes):
                for handle, offset, array in zip(handles, offsets, arrays):
                    if start < array.nbytes:
                        yield handle, offset + start, array[start:start + self.block_bytes]

        available = deque(self.requests)
        try:
            for handle, offset, destination in jobs():
                if stop is not None and stop.is_set():
                    raise InterruptedError("file reading stopped")
                if not available:
                    request = self.pending[0]
                    self._complete(request, stop)
                    self.pending.popleft()
                    available.append(request)
                self._submit(available.popleft(), handle, offset, destination)
            while self.pending:
                self._complete(self.pending[0], stop)
                self.pending.popleft()
        finally:
            self._drain()
            for request in self.requests:
                request.pop("destination", None)

    def close(self):
        if self.closed:
            return
        self._drain()
        for request in self.requests:
            self.k.CloseHandle(request["ov"].hEvent)
        for handle in self.handles:
            self.k.CloseHandle(handle)
        self.requests.clear()
        self.handles.clear()
        self.closed = True
