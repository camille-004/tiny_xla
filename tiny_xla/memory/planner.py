from __future__ import annotations

import dataclasses
import enum

from ..core.graph import Graph
from ..core.types import NodeId
from ..utils.errors import ValidationError
from ..utils.logger import XLALogger

logger = XLALogger.get_logger(__name__)


class AllocationStrategy(enum.StrEnum):
    """Memory allocation strategies."""

    GREEDY = "greedy"
    OPTIMAL = "optimal"


@dataclasses.dataclass
class Buffer:
    """Represents a memory buffer requirement."""

    size: int
    alignment: int = 8

    def __post_init__(self) -> None:
        """Validate buffer parameters."""
        if self.size <= 0:
            raise ValidationError(f"Invalid buffer size: {self.size}")
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValidationError(
                f"Alignment must be positive power of 2, got: {self.alignment}"
            )

    def aligned_size(self) -> int:
        """Get size including alignment padding."""
        return (self.size + self.alignment - 1) & ~(self.alignment - 1)


@dataclasses.dataclass
class Lifetime:
    """Represents the lifetime of a buffer."""

    start: int
    end: int

    def overlaps(self, other: Lifetime) -> bool:
        """Check if this lifetime overlaps with another."""
        return self.start < other.end and other.start < self.end


@dataclasses.dataclass
class BufferAllocation:
    """Represents an allocated buffer in memory."""

    offset: int
    buffer: Buffer

    def __post_init__(self) -> None:
        """Validate allocation."""
        if self.offset < 0:
            raise ValidationError(f"Invalid negative offset: {self.offset}")
        if self.offset % self.buffer.alignment != 0:
            raise ValidationError(
                f"Offset {self.offset} violates alignment "
                f"{self.buffer.alignment}"
            )

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self.buffer.size

    @property
    def alignment(self) -> int:
        """Get buffer alignment."""
        return self.buffer.alignment

    @property
    def end_offset(self) -> int:
        """Get end offset of this allocation."""
        return self.offset + self.buffer.aligned_size()


class LifetimeAnalysis:
    """Analyzes buffer lifetimes in a computation graph."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def compute_lifetimes(self) -> dict[NodeId, Lifetime]:
        order = self.graph.topological_sort()
        node_order = {node_id: idx for idx, node_id in enumerate(order)}
        lifetimes = {}
        for node_id, op in self.graph.ops.items():
            start = node_order[node_id]
            users = self.graph.get_users(node_id)
            end = (
                max((node_order[user_id] for user_id in users), default=start)
                + 1
            )
            lifetimes[node_id] = Lifetime(start, end)
        return lifetimes


class MemoryPlanner:
    """Plans memory allocation for buffers."""

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.GREEDY,
        memory_limit: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.memory_limit = memory_limit

    def _free_expired_allocations(
        self, active_allocations: list[tuple[int, int, int]], curr_start: int
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
        """Free allocations whose lifetime has ended.

        Returns a tuple (remaining_active, freed_blocks),
        where freed_blocks is a list of (offset, size) from allocations
        that have expired.
        """
        remaining = []
        freed = []
        for end, offset, size in active_allocations:
            if end <= curr_start:
                freed.append((offset, size))
            else:
                remaining.append((end, offset, size))
        return remaining, freed

    def _select_offset(
        self, free_blocks: list[tuple[int, int]], bs: int, peak: int
    ) -> tuple[int, int]:
        """Select an offset for a buffer of size bs.

        If a free block is available (i.e., its size >= bs), return its
        offset and leave peak unchanged. Otherwise, use the current peak as
        the candidate, update peak, and return.
        """
        for i, (offset, size) in enumerate(free_blocks):
            if size >= bs:
                free_blocks.pop(i)
                if (
                    self.memory_limit is not None
                    and offset + bs > self.memory_limit
                ):
                    raise ValidationError(
                        f"Memory limit {self.memory_limit} exceeded"
                    )
                return offset, peak
        # No free block available
        candidate_offset = peak
        if (
            self.memory_limit is not None
            and candidate_offset + bs > self.memory_limit
        ):
            raise ValidationError(f"Memory limit {self.memory_limit} exceeded")
        return candidate_offset, candidate_offset + bs

    def allocate(
        self, buffers: list[Buffer], lifetimes: list[Lifetime] | None = None
    ) -> list[BufferAllocation]:
        """Allocate memory for buffers."""
        if self.strategy != AllocationStrategy.GREEDY:
            raise NotImplementedError("Only GREEDY strategy is supported.")

        if lifetimes is None:
            # Every buffer gets offset 0.
            for buffer in buffers:
                if (
                    self.memory_limit is not None
                    and buffer.aligned_size() > self.memory_limit
                ):
                    raise ValidationError(
                        f"Buffer of size {buffer.aligned_size()} "
                        f"exceeds memory limit {self.memory_limit}"
                    )
            return [
                BufferAllocation(offset=0, buffer=buffer) for buffer in buffers
            ]

        # Perform lifetime-based allocation.
        items: list[tuple[Buffer, Lifetime]] = list(zip(buffers, lifetimes))
        items.sort(key=lambda x: x[1].start)

        allocations: list[BufferAllocation] = []
        free_blocks: list[tuple[int, int]] = []
        active_allocations: list[tuple[int, int, int]] = []
        peak = 0

        for buffer, lifetime in items:
            bs = buffer.aligned_size()

            active_allocations, freed = self._free_expired_allocations(
                active_allocations, lifetime.start
            )
            free_blocks.extend(freed)

            candidate_offset, new_peak = self._select_offset(
                free_blocks, bs, peak
            )
            peak = new_peak

            alloc = BufferAllocation(offset=candidate_offset, buffer=buffer)
            allocations.append(alloc)
            active_allocations.append((lifetime.end, candidate_offset, bs))
            active_allocations.sort(key=lambda x: x[0])
        return allocations


def calc_buffer_size(buffer_type: str, shape: tuple[int, ...]) -> int:
    type_sizes = {
        "float32": 4,
        "float64": 8,
        "int32": 4,
        "int64": 8,
        "bool": 1,
    }

    if buffer_type not in type_sizes:
        raise ValidationError(f"Unknown buffer type: {buffer_type}")

    total_elements = 1
    for dim in shape:
        if dim <= 0:
            raise ValidationError(f"Invalid dimension size: {dim}")
        total_elements *= dim

    return total_elements * type_sizes[buffer_type]
