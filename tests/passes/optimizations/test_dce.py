import pytest

from tiny_xla.core.graph import Graph
from tiny_xla.core.operation import Operation, OpType
from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.passes.optimizations.dead_code_elimination import (
    DCEStats,
    create_dce_pass,
)


class TestOp(Operation):
    """Test operation for DCE."""

    __test__ = False

    def __init__(
        self,
        name: str,
        inputs: list[Operation] | None = None,
        is_stateless: bool = True,
    ):
        super().__init__(OpType.ELEMENT_WISE, name, inputs or [])
        self.metadata.is_stateless = is_stateless

    def compute_output_type(self) -> Tensor:
        return Tensor(XLAType.FLOAT32, ())

    def validate(self) -> bool:
        return True

    def lower(self) -> Operation:
        return self


@pytest.fixture
def linear_graph():
    """Create a linear graph: a -> b -> c."""
    graph = Graph()

    a = graph.add_op(TestOp("a"))
    b = graph.add_op(TestOp("b", [graph.get_op(a)]))
    c = graph.add_op(TestOp("c", [graph.get_op(b)]))

    return graph, {"a": a, "b": b, "c": c}


@pytest.fixture
def branching_graph():
    graph = Graph()

    a = graph.add_op(TestOp("a"))
    b = graph.add_op(TestOp("b", [graph.get_op(a)]))
    c = graph.add_op(TestOp("c", [graph.get_op(a)]))
    d = graph.add_op(TestOp("d", [graph.get_op(b), graph.get_op(c)]))

    return graph, {"a": a, "b": b, "c": c, "d": d}


def test_dce_empty_graph():
    """Test DCE on empty graph."""
    graph = Graph()
    dce = create_dce_pass()
    result = dce.run(graph)

    assert not result.changed
    assert isinstance(result.stats, DCEStats)
    assert result.stats.removed_ops == 0


def test_dce_no_dead_code(linear_graph):
    """Test DCE when all operations are used."""
    graph, ops = linear_graph
    dce = create_dce_pass()

    # Mark last op as output
    result = dce.run(graph, outputs=[ops["c"]])

    # Nothing should be removed
    assert not result.changed
    assert result.stats.removed_ops == 0
    assert len(graph.ops) == 3


def test_dce_unused_branch(branching_graph):
    """Test DCE with an unused branch."""
    graph, ops = branching_graph
    dce = create_dce_pass()

    # Mark only b->d path as used
    result = dce.run(graph, outputs=[ops["b"]])

    # Node c should be removed
    assert result.changed
    assert result.stats.removed_ops == 2  # c and d should be removed
    assert ops["c"] not in graph.ops
    assert ops["d"] not in graph.ops


def test_dce_preserve_stateful(linear_graph):
    """Test DCE preserves stateful operations."""
    graph, ops = linear_graph

    # Make operation b stateful
    stateful_op = TestOp("b", [graph.get_op(ops["a"])], is_stateless=False)
    graph.remove_op(ops["b"])
    new_b = graph.add_op(stateful_op)
    graph.add_edge(ops["a"], new_b)

    dce = create_dce_pass()
    result = dce.run(graph)  # noqa:F841

    assert new_b in graph.ops
    assert not graph.get_op(new_b).metadata.is_stateless


def test_dce_multiple_outputs(branching_graph):
    """Test DCE with multiple outputs."""
    graph, ops = branching_graph
    dce = create_dce_pass()

    # Mark both b and c as outputs
    result = dce.run(graph, outputs=[ops["b"], ops["c"]])

    # Only d should be removed
    assert result.changed
    assert result.stats.removed_ops == 1
    assert ops["d"] not in graph.ops
    assert all(ops[x] in graph.ops for x in ["a", "b", "c"])


def test_dce_no_outputs():
    """Test DCE behavior when no outputs specified."""
    graph = Graph()

    # Add some operations
    ops = []
    for name in ["a", "b", "c"]:
        ops.append(graph.add_op(TestOp(name)))

    dce = create_dce_pass()
    result = dce.run(graph)

    # All operations should be removed (no outputs specified)
    assert result.changed
    assert result.stats.removed_ops == 3
    assert len(graph.ops) == 0


def test_dce_stats():
    """Test DCE statistics tracking."""
    graph = Graph()
    a = graph.add_op(TestOp("a"))
    b = graph.add_op(TestOp("b", [graph.get_op(a)]))  # noqa:F841

    dce = create_dce_pass()
    result = dce.run(graph, outputs=[a])

    stats = result.stats
    assert stats.removed_ops == 1  # b should be removed
    assert stats.total_ops == 2
    assert stats.removed_edges == 1
    assert str(stats).startswith("DCE removed")


def test_dce_graph_validation(branching_graph):
    """Test DCE maintains graph validity."""
    graph, ops = branching_graph
    dce = create_dce_pass()

    # Remove c and d
    result = dce.run(graph, outputs=[ops["b"]])  # noqa:F841

    # Check graph remains valid
    assert graph.validate()
    # Check edge consistency
    assert not graph.get_users(ops["b"])  # b should have no users
    assert ops["a"] in graph.ops  # a should be preserved


def test_dce_cyclic_references():
    """Test DCE handles cycles properly."""
    graph = Graph()

    # Create operations
    a = graph.add_op(TestOp("a"))
    b = graph.add_op(TestOp("b", [graph.get_op(a)]))

    # Create a cycle
    graph.add_edge(b, a)

    dce = create_dce_pass()
    result = dce.run(graph, outputs=[b])

    # Both operations should be preserved due to cycle
    assert not result.changed
    assert len(graph.ops) == 2


def test_dce_partial_removal(branching_graph):
    """Test DCE removes only unreachable parts."""
    graph, ops = branching_graph

    # Add another branch that's clearly dead
    e = graph.add_op(TestOp("e"))
    f = graph.add_op(TestOp("f", [graph.get_op(e)]))

    dce = create_dce_pass()
    result = dce.run(graph, outputs=[ops["d"]])

    # e and f should be removed, but main graph preserved
    assert result.changed
    assert result.stats.removed_ops == 2
    assert e not in graph.ops
    assert f not in graph.ops
    assert all(ops[x] in graph.ops for x in ["a", "b", "c", "d"])
