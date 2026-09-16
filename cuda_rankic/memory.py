"""Pure-Python planning for the additional memory used by a RankIC chunk.

The budget is supplied by the caller. This helper does not inspect a GPU or
limit process-wide memory, and its estimate cannot account for concurrent
allocations. The safety reserve covers allocator rounding and other overhead;
it is separate from the exact workspace bytes returned by the backend.
"""

from dataclasses import dataclass
import math
from numbers import Real
import operator
from typing import Callable


# Each launch uses signed 32-bit indices, with two packed sort keys per value.
_MAX_ELEMENTS = (2**31 - 1) // 2


@dataclass(frozen=True)
class MemoryPlan:
    """A chunk size and its scratch allocation and reserved memory budget."""

    chunk_rows: int
    workspace_bytes: int
    estimated_bytes: int


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be a nonnegative integer") from None
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_max_memory_fraction(value) -> float:
    """Validate and normalize a finite real fraction in the interval (0, 1]."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("max_memory_fraction must be a real number in (0, 1]")
    if not 0 < value <= 1:
        raise ValueError("max_memory_fraction must be finite and in (0, 1]")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("max_memory_fraction must be finite and in (0, 1]")
    return result


def plan_chunk_rows(
    total_rows: int,
    n_assets: int,
    available_bytes: int,
    workspace_size: Callable[[int, int], int],
    *,
    input_bytes_per_row: int,
    output_bytes_per_row: int = 4,
    fixed_bytes: int = 0,
    safety_bytes: int = 64 * 1024**2,
) -> MemoryPlan:
    """Find the largest chunk that fits the caller's available byte budget.

    ``workspace_size(rows, n_assets)`` returns exact scratch bytes; bind any
    backend strategy before passing it here. The total cost must be monotone
    in ``rows`` so binary search can find the largest fitting chunk.

    ``input_bytes_per_row`` counts all additional live input/conversion buffers,
    and ``output_bytes_per_row`` counts the chunk output. ``fixed_bytes`` counts
    any other allocations charged to this workload. Existing buffers already
    deducted from ``available_bytes`` must not be counted again. The returned
    ``estimated_bytes`` includes ``safety_bytes``, which is reserved, not itself
    allocated. A zero-row dataset requires no allocations or reserve.

    An individual chunk also respects the backend's 32-bit indexing limits.
    If even one row cannot fit, raise ``MemoryError`` before GPU allocation.
    """
    total_rows = _nonnegative_integer(total_rows, "total_rows")
    n_assets = _nonnegative_integer(n_assets, "n_assets")
    available_bytes = _nonnegative_integer(available_bytes, "available_bytes")
    input_bytes_per_row = _nonnegative_integer(input_bytes_per_row, "input_bytes_per_row")
    output_bytes_per_row = _nonnegative_integer(output_bytes_per_row, "output_bytes_per_row")
    fixed_bytes = _nonnegative_integer(fixed_bytes, "fixed_bytes")
    safety_bytes = _nonnegative_integer(safety_bytes, "safety_bytes")
    if not callable(workspace_size):
        raise TypeError("workspace_size must be callable")
    if n_assets > _MAX_ELEMENTS:
        raise ValueError("n_assets exceeds the backend's 32-bit indexing limit")
    if total_rows == 0:
        return MemoryPlan(0, 0, 0)

    max_rows = min(total_rows, _MAX_ELEMENTS // max(1, n_assets))
    bytes_per_row = input_bytes_per_row + output_bytes_per_row
    reserve = fixed_bytes + safety_bytes

    def estimate(rows: int) -> MemoryPlan:
        workspace_bytes = (
            _nonnegative_integer(workspace_size(rows, n_assets), "workspace_size result")
            if n_assets else 0
        )
        return MemoryPlan(rows, workspace_bytes, reserve + rows * bytes_per_row + workspace_bytes)

    best = estimate(1)
    if best.estimated_bytes > available_bytes:
        raise MemoryError(
            "RankIC memory budget cannot fit one row: "
            f"requires {best.estimated_bytes} bytes including safety reserve, "
            f"but {available_bytes} bytes are available"
        )

    low, high = 2, max_rows
    while low <= high:
        middle = (low + high) // 2
        candidate = estimate(middle)
        if candidate.estimated_bytes <= available_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best
