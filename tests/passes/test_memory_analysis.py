import pytest

from tiny_xla.core.graph import Graph
from tiny_xla.core.operation import Operation, OpType
from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.memory.planner import LifetimeAnalysis
from tiny_xla.passes.memory_analysis import (
    MemoryAnalysisStats,
    create_memory_analysis_pass,
)


class TestOp(Operation):
    """Test operation with configurable output size."""

    __test__ = False

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        inputs: list[Operation] | None = None,
    ):
        super().__init__(OpType.ELEMENT_WISE, name, inputs or [])
        self._shape = shape

    def compute_output_type(self) -> Tensor:
        return Tensor(XLAType.FLOAT32, self._shape)

    def validate(self) -> bool:
        return True

    def lower(self) -> Operation:
        return self


@pytest.fixture
def linear_graph():
    """Create a linear graph with known sizes."""
    graph = Graph()

    # Create a -> b -> c chain
    a = graph.add_op(TestOp("a", (100,)))  # 400 bytes
    b = graph.add_op(TestOp("b", (200,), [graph.get_op(a)]))  # 800 bytes
    c = graph.add_op(TestOp("c", (50,), [graph.get_op(b)]))  # 200 bytes

    return graph, {"a": a, "b": b, "c": c}


@pytest.fixture
def diamond_graph():
    """Create a diamond-shaped graph with parallel paths."""
    graph = Graph()

    # Create a diamond: a -> (b,c) -> d
    a = graph.add_op(TestOp("a", (100,)))
    b = graph.add_op(TestOp("b", (200,), [graph.get_op(a)]))
    c = graph.add_op(TestOp("c", (150,), [graph.get_op(a)]))
    d = graph.add_op(TestOp("d", (50,), [graph.get_op(b), graph.get_op(c)]))

    return graph, {"a": a, "b": b, "c": c, "d": d}


def test_basic_analysis(linear_graph):
    """Test basic memory analysis on linear graph."""
    graph, ops = linear_graph

    pass_obj = create_memory_analysis_pass()
    result = pass_obj.run(graph)

    assert result.changed
    assert isinstance(result.stats, MemoryAnalysisStats)

    # Check that all operations are annotated
    for op_id in ops.values():
        op = graph.get_op(op_id)
        assert hasattr(op.metadata, "memory_offset")
        assert hasattr(op.metadata, "memory_size")
        assert hasattr(op.metadata, "memory_alignment")


def test_memory_reuse(diamond_graph):
    """Test memory reuse in diamond pattern."""
    graph, ops = diamond_graph

    pass_obj = create_memory_analysis_pass()
    result = pass_obj.run(graph)

    stats = result.stats
    assert stats.reused_buffers > 0  # Should reuse some memory

    # Parallel paths b and c should not overlap in memory
    op_b = graph.get_op(ops["b"])
    op_c = graph.get_op(ops["c"])

    b_range = (
        op_b.metadata.memory_offset,
        op_b.metadata.memory_offset + op_b.metadata.memory_size,
    )
    c_range = (
        op_c.metadata.memory_offset,
        op_c.metadata.memory_offset + op_c.metadata.memory_size,
    )

    # Check for no overlap
    assert not (b_range[0] < c_range[1] and c_range[0] < b_range[1])


def test_memory_limit():
    """Test memory limit enforcement."""
    graph = Graph()

    # Create operations requiring 2KB total
    a = graph.add_op(TestOp("a", (256,)))  # noqa:F841
    b = graph.add_op(TestOp("b", (256,)))  # noqa:F841

    # Set a 1KB limit.
    pass_obj = create_memory_analysis_pass(memory_limit=1024)
    result = pass_obj.run(graph)

    assert not result.changed
    assert result.errors is not None
    assert any("exceeds memory limit" in err.lower() for err in result.errors)


def test_alignment_requirements():
    """Test handling of alignment requirements."""
    graph = Graph()

    # Create operation with vector data (needs alignment)
    op = graph.add_op(TestOp("vector_op", (64,)))

    pass_obj = create_memory_analysis_pass()
    result = pass_obj.run(graph)

    assert result.changed

    # Check alignment
    op = graph.get_op(op)
    assert op.metadata.memory_offset % op.metadata.memory_alignment == 0


def test_statistics_tracking(linear_graph):
    """Test memory statistics collection."""
    graph, ops = linear_graph

    pass_obj = create_memory_analysis_pass()
    result = pass_obj.run(graph)

    stats = result.stats
    assert stats.num_buffers == len(ops)
    assert stats.total_memory > 0
    assert stats.peak_memory > 0
    assert str(stats).startswith("Memory Analysis")


def test_verification(linear_graph):
    """Test memory analysis verification."""
    graph, _ = linear_graph

    pass_obj = create_memory_analysis_pass()
    result = pass_obj.run(graph)

    assert result.changed
    assert pass_obj.verify(graph)


def test_error_handling():
    """Test handling of invalid operations."""
    graph = Graph()

    # Create operation with invalid shape
    class InvalidOp(TestOp):
        def compute_output_type(self) -> Tensor:
            return Tensor(XLAType.FLOAT32, (-1,))  # Invalid shape

    graph.add_op(InvalidOp("invalid", (-1,)))

    pass_obj = create_memory_analysis_pass()
    result = pass_obj.run(graph)

    assert not result.changed
    assert result.errors is not None


def test_metadata_consistency(linear_graph):
    """Test consistency of operation metadata."""
    graph, ops = linear_graph

    pass_obj = create_memory_analysis_pass()
    pass_obj.run(graph)

    # Recompute lifetimes here.
    lifetime_analysis = LifetimeAnalysis(graph)
    lifetimes = lifetime_analysis.compute_lifetimes()

    used_ranges = []
    for op_id in ops.values():
        op = graph.get_op(op_id)
        mem_range = (
            op.metadata.memory_offset,
            op.metadata.memory_offset + op.metadata.memory_size,
        )

        for other_id, other_range in used_ranges:
            if lifetimes[op_id].overlaps(lifetimes[other_id]):
                assert not (
                    mem_range[0] < other_range[1]
                    and other_range[0] < mem_range[1]
                )
        used_ranges.append((op_id, mem_range))
