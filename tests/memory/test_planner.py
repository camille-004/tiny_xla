import pytest

from tiny_xla.core.graph import Graph
from tiny_xla.core.operation import Operation, OpType
from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.memory.planner import (
    AllocationStrategy,
    Buffer,
    BufferAllocation,
    Lifetime,
    LifetimeAnalysis,
    MemoryPlanner,
    calc_buffer_size,
)
from tiny_xla.utils.errors import ValidationError


class DummyOp(Operation):
    def __init__(
        self, name: str, inputs: list[Operation] | None = None
    ) -> None:
        super().__init__(OpType.ELEMENT_WISE, name, inputs or [])

    def compute_output_type(self) -> Tensor:
        return Tensor(XLAType.FLOAT32, (16,))

    def validate(self) -> bool:
        return True

    def lower(self) -> Operation:
        return self


def test_buffer_creation():
    """Test buffer creation and validation."""
    buffer = Buffer(size=1024, alignment=8)
    assert buffer.size == 1024
    assert buffer.alignment == 8

    with pytest.raises(ValidationError):
        Buffer(size=0)
    with pytest.raises(ValidationError):
        Buffer(size=-1024)

    with pytest.raises(ValidationError):
        Buffer(size=1024, alignment=0)
    with pytest.raises(ValidationError):
        Buffer(size=1024, alignment=7)


def test_buffer_aligned_size():
    """Test buffer size alignment calculations."""
    buffer = Buffer(size=1000, alignment=16)
    aligned_size = buffer.aligned_size()

    assert aligned_size >= buffer.size
    assert aligned_size % buffer.alignment == 0
    assert aligned_size == 1008


def test_lifetime_overlap():
    """Test lifetime overlap detection."""
    life1 = Lifetime(start=0, end=2)
    life2 = Lifetime(start=1, end=3)
    life3 = Lifetime(start=2, end=4)
    life4 = Lifetime(start=4, end=6)

    assert life1.overlaps(life2)
    assert life2.overlaps(life3)

    assert not life1.overlaps(life4)
    assert not life4.overlaps(life1)


def test_buffer_allocation():
    """Test buffer allocation creation and validation."""
    buffer = Buffer(size=1024, alignment=8)

    alloc = BufferAllocation(offset=16, buffer=buffer)
    assert alloc.offset == 16
    assert alloc.size == 1024
    assert alloc.end_offset == 1024 + 16

    with pytest.raises(ValidationError):
        BufferAllocation(offset=-8, buffer=buffer)

    with pytest.raises(ValidationError):
        BufferAllocation(offset=3, buffer=buffer)


def test_lifetime_analysis():
    """Test lifetime analysis on a graph."""
    graph = Graph()

    a = graph.add_op(DummyOp("a"))
    b = graph.add_op(DummyOp("b", [graph.get_op(a)]))
    c = graph.add_op(DummyOp("c", [graph.get_op(b)]))

    analysis = LifetimeAnalysis(graph)
    lifetimes = analysis.compute_lifetimes()

    assert lifetimes[a].start < lifetimes[b].start
    assert lifetimes[b].start < lifetimes[c].start

    assert lifetimes[a].end > lifetimes[b].start
    assert lifetimes[b].end > lifetimes[c].start


def test_memory_planner_greedy():
    """Test greedy memory allocation strategy."""
    buffers = [Buffer(size=1024), Buffer(size=512), Buffer(size=2048)]

    planner = MemoryPlanner(strategy=AllocationStrategy.GREEDY)
    allocations = planner.allocate(buffers)

    assert len(allocations) == len(buffers)

    for alloc in allocations:
        assert alloc.offset % alloc.alignment == 0


def test_memory_planner_memory_limit():
    """Test memory planner with memory limit."""
    buffers = [
        Buffer(size=1024),
        Buffer(size=2048),
    ]

    planner = MemoryPlanner(
        strategy=AllocationStrategy.GREEDY, memory_limit=1500
    )

    with pytest.raises(ValidationError):
        planner.allocate(buffers)

    planner = MemoryPlanner(memory_limit=4096)
    allocations = planner.allocate(buffers)
    assert len(allocations) == len(buffers)


def test_calc_buffer_size():
    """Test buffer size calculation."""
    assert calc_buffer_size("float32", (10,)) == 40
    assert calc_buffer_size("float64", (10,)) == 80
    assert calc_buffer_size("int32", (10,)) == 40

    assert calc_buffer_size("float32", (2, 3)) == 24  # 4 bytes * 6
    assert calc_buffer_size("int32", (2, 2, 2)) == 32  # 4 bytes * 8

    with pytest.raises(ValidationError):
        calc_buffer_size("unknown_type", (10,))

    with pytest.raises(ValidationError):
        calc_buffer_size("float32", (-1, 10))


def test_memory_reuse():
    """Test memory reuse for non-overlapping buffers."""
    buffers = [Buffer(size=1024), Buffer(size=1024), Buffer(size=1024)]

    planner = MemoryPlanner()
    allocations = planner.allocate(buffers)

    # Check if memory is being reused (same offsets)
    offsets = {alloc.offset for alloc in allocations}
    assert len(offsets) < len(allocations)


def test_complex_graph_analysis():
    """Test lifetime analysis on a more complex graph."""
    graph = Graph()
    a = graph.add_op(DummyOp("a"))
    b = graph.add_op(DummyOp("b", [graph.get_op(a)]))
    c = graph.add_op(DummyOp("c", [graph.get_op(a)]))
    d = graph.add_op(DummyOp("d", [graph.get_op(b), graph.get_op(c)]))

    analysis = LifetimeAnalysis(graph)
    lifetimes = analysis.compute_lifetimes()

    a_lifetime = lifetimes[a]
    b_lifetime = lifetimes[b]
    c_lifetime = lifetimes[c]

    assert a_lifetime.end >= max(b_lifetime.start, c_lifetime.start)
    d_lifetime = lifetimes[d]
    assert b_lifetime.end >= d_lifetime.start
    assert c_lifetime.end >= d_lifetime.start
