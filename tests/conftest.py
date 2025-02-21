import pytest

from tiny_xla.core.context import XLAContext
from tiny_xla.core.graph import Graph
from tiny_xla.core.operation import Operation, OpType
from tiny_xla.core.types import Tensor, XLAType


@pytest.fixture
def reset_context():
    """Reset compiler context before each test."""
    XLAContext.reset()
    yield
    XLAContext.reset()


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return Graph()


class DummyOp(Operation):
    """Dummy operation for testing."""

    def __init__(self, name: str, inputs: list[Operation] | None = None):
        super().__init__(OpType.ELEMENT_WISE, name, inputs or [])

    def compute_output_type(self) -> Tensor:
        return Tensor(XLAType.FLOAT32, ())

    def validate(self) -> bool:
        return True

    def lower(self) -> Operation:
        return self


@pytest.fixture
def dummy_op():
    """Create a dummy operation."""
    return lambda name, inputs=None: DummyOp(name, inputs)


@pytest.fixture
def simple_graph(dummy_op):
    """Create a simple graph with a few operations."""
    graph = Graph()

    # a -> b -> c
    a = graph.add_op(dummy_op("a"))
    b = graph.add_op(dummy_op("b", [graph.get_op(a)]))
    c = graph.add_op(dummy_op("c", [graph.get_op(b)]))

    return graph, {"a": a, "b": b, "c": c}
