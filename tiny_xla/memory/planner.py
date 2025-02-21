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
        return self.buffer.sizes

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
        self._node_order: dict[NodeId, int] = {}
        self._compute_node_order()

    def _compute_node_order(self) -> None:
        """Compute execution order of nodes."""
        order = self.graph.topological_sort()
        self._node_order = {node_id: idx for idx, node_id in enumerate(order)}

    def compute_lifetimes(self) -> dict[NodeId, Lifetime]:
        """Compute buffer lifetimes for all operations."""
        lifetimes: dict[NodeId, Lifetime] = {}

        for node_id, op in self.graph.ops.items():
            start = self._node_order[node_id]
            users = self.graph.get_users(node_id)
            end = (
                max(self._node_order[user_id] for user_id in users)
                if users
                else start + 1
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

    def allocate(self, buffers: list[Buffer]) -> list[BufferAllocation]:
        """Allocate memory for buffers."""
        if self.strategy == AllocationStrategy.GREEDY:
            return self._allocate_greedy(buffers)
        else:
            return self._allocate_optimal(buffers)

    def _allocate_greedy(
        self, buffers: list[Buffer]
    ) -> list[BufferAllocation]:
        """Simple first-fit allocation strategy."""
        allocations: list[BufferAllocation] = []
        used_regions: list[BufferAllocation] = []

        for buffer in buffers:
            offset = 0
            while True:
                aligned_offset = (offset + buffer.alignment - 1) & ~(
                    buffer.alignment - 1
                )

                end_offset = aligned_offset + buffer.aligned_size()

                if (
                    self.memory_limit is not None
                    and end_offset > self.memory_limit
                ):
                    raise ValidationError(
                        f"Memory limit {self.memory_limit} exceeded"
                    )

                overlap = False
                for used in used_regions:
                    if (
                        aligned_offset < used.end_offset
                        and used.offset < end_offset
                    ):
                        overlap = True
                        offset = used.end_offset
                        break

                if not overlap:
                    allocation = BufferAllocation(
                        offset=aligned_offset, buffer=buffer
                    )
                    allocations.append(allocation)
                    used_regions.append(allocation)
                    break

        return allocations

    def _allocate_optimal(
        self, buffers: list[Buffer]
    ) -> list[BufferAllocation]:
        """Optimal allocation trategy.

        TODO: Implement optimal allocation.
        """
        return self._allocate_greedy(buffers)


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
