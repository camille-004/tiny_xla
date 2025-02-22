from __future__ import annotations

import dataclasses
from typing import Sequence

from ..core.context import XLAContext
from ..core.graph import Graph
from ..core.operation import Operation
from ..core.types import NodeId
from ..memory.planner import (
    AllocationStrategy,
    Buffer,
    BufferAllocation,
    Lifetime,
    LifetimeAnalysis,
    MemoryPlanner,
    calc_buffer_size,
)
from ..utils.logger import XLALogger
from .pass_base import Pass, PassResult

logger = XLALogger.get_logger(__name__)


@dataclasses.dataclass
class MemoryAnalysisStats:
    """Statistics from memory analysis."""

    total_memory: int = 0
    peak_memory: int = 0
    num_buffers: int = 0
    reused_buffers: int = 0

    def __str__(self) -> str:
        return (
            f"Memory Analysis: {self.num_buffers} buffers, "
            f"{self.reused_buffers} reused, "
            f"peak memory {self.peak_memory} bytes"
        )


class MemoryAnalysis(Pass):
    """Analyzes memory requirements for a computation graph."""

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.GREEDY,
        memory_limit: int | None = None,
    ) -> None:
        super().__init__("memory_analysis")
        self.strategy = strategy
        self.memory_limit = memory_limit
        self._buffer_map: dict[NodeId, Buffer] = {}
        self._allocation_map: dict[NodeId, BufferAllocation] = {}

    def _analyze_buffer_requirements(self, op: Operation) -> Buffer:
        """Determine buffer requirements for an operation."""
        # Operations that don't need buffers
        if op.op_type.name in {"CONSTANT", "PLACEHOLDER", "PARAMETER"}:
            return Buffer(size=0, alignment=8)

        output_type = op.output_type
        size = calc_buffer_size(str(output_type.dtype), output_type.shape)
        alignment = 8
        if op.op_type.name in {"CONVOLUTION", "MATRIX_MULTIPLY"}:
            alignment = 32  # For vectorization

        return Buffer(size=size, alignment=alignment)

    def _collect_buffer_info(self, graph: Graph) -> None:
        """Collect buffer requirements for all operations."""
        self._buffer_map.clear()

        for node_id, op in graph.ops.items():
            buffer = self._analyze_buffer_requirements(op)
            self._buffer_map[node_id] = buffer

    def _plan_memory_allocation(
        self, graph: Graph, lifetimes: dict[NodeId, Lifetime]
    ) -> list[BufferAllocation]:
        """Plan memory allocation for all buffers."""
        planner = MemoryPlanner(
            strategy=self.strategy, memory_limit=self.memory_limit
        )
        nodes_ordered = sorted(
            lifetimes.keys(), key=lambda n: lifetimes[n].start
        )
        buffers = [self._buffer_map[node_id] for node_id in nodes_ordered]
        allocations = planner.allocate(
            buffers, [lifetimes[node_id] for node_id in nodes_ordered]
        )

        self._allocation_map.clear()
        for node_id, allocation in zip(nodes_ordered, allocations):
            self._allocation_map[node_id] = allocation

        return allocations

    def _compute_statistics(
        self, allocations: list[BufferAllocation]
    ) -> MemoryAnalysisStats:
        """Compute memory usage statistics."""
        stats = MemoryAnalysisStats()
        stats.num_buffers = len(allocations)
        stats.total_memory = sum(alloc.size for alloc in allocations)
        peak_memory = 0
        offset_set = set()

        for alloc in allocations:
            peak_memory = max(peak_memory, alloc.end_offset)
            offset_set.add(alloc.offset)

        stats.peak_memory = peak_memory
        stats.reused_buffers = len(allocations) - len(offset_set)

        return stats

    def _annotate_operations(self, graph: Graph) -> None:
        """Annotate operations with memory information."""
        for node_id, op in graph.ops.items():
            if node_id in self._allocation_map:
                alloc = self._allocation_map[node_id]
                # Store memory info in operation metadata
                op.metadata.memory_offset = alloc.offset
                op.metadata.memory_size = alloc.size
                op.metadata.memory_alignment = alloc.alignment
            else:
                # For ops that were not allocated
                # (parameters, constants, placeholders)
                op.metadata.memory_offset = 0
                op.metadata.memory_size = 0
                op.metadata.memory_alignment = 8

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        try:
            self._collect_buffer_info(graph)
            lifetime_analysis = LifetimeAnalysis(graph)
            lifetimes = lifetime_analysis.compute_lifetimes()
            allocations = self._plan_memory_allocation(graph, lifetimes)
            stats = self._compute_statistics(allocations)
            if (
                self.memory_limit is not None
                and stats.total_memory > self.memory_limit
            ):
                error_msg = (
                    f"Total memory {stats.total_memory} exceeds memory "
                )
                "limit {self.memory_limit}"
                logger.error(error_msg)
                return PassResult(self.name, changed=False, errors=[error_msg])
            self._annotate_operations(graph)
            logger.info(str(stats))
            return PassResult(self.name, changed=True, stats=stats)
        except Exception as e:
            logger.error("Memory analysis failed: %s", e)
            return PassResult(self.name, changed=False, errors=[str(e)])

    def verify(self, graph: Graph) -> bool:
        try:
            for node_id, op in graph.ops.items():
                offset = (
                    op.metadata.memory_offset
                    if op.metadata.memory_offset is not None
                    else 0
                )
                size = (
                    op.metadata.memory_size
                    if op.metadata.memory_size is not None
                    else 0
                )
                alignment = (
                    op.metadata.memory_alignment
                    if op.metadata.memory_alignment is not None
                    else 8
                )
                # For ops that received no allocation, size is zero.
                if size == 0:
                    continue
                if alignment == 0 or offset % alignment != 0:
                    return False
            return True
        except Exception as e:
            logger.error("Memory analysis verification failed: %s", e)
            return False


def create_memory_analysis_pass(
    strategy: AllocationStrategy = AllocationStrategy.GREEDY,
    memory_limit: int | None = None,
) -> Pass:
    """Create a memory analysis pass."""
    return MemoryAnalysis(strategy, memory_limit)
