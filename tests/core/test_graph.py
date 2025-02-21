import pytest

from tiny_xla.utils.errors import ValidationError


def test_empty_graph(empty_graph):
    """Test empty graph creation."""
    assert len(empty_graph.ops) == 0
    assert empty_graph.validate()


def test_add_op(empty_graph, dummy_op):
    """Test adding ops to graph."""
    node_id = empty_graph.add_op(dummy_op("test"))
    assert node_id in empty_graph.ops
    assert len(empty_graph.ops) == 1

    op1 = empty_graph.get_op(node_id)
    node_id2 = empty_graph.add_op(dummy_op("test2", [op1]))
    assert len(empty_graph.ops) == 2
    assert empty_graph.get_users(node_id) == {node_id2}


def test_remove_op(simple_graph):
    """Test removing operations from graph."""
    graph, ops = simple_graph

    graph.remove_op(ops["b"])
    assert ops["b"] not in graph.ops
    assert ops["a"] in graph.ops
    assert ops["c"] in graph.ops
    assert not graph.get_users(ops["a"])


def test_graph_edges(simple_graph):
    """Test graph edge handling."""
    graph, ops = simple_graph

    assert graph.get_users(ops["a"]) == {ops["b"]}
    assert graph.get_users(ops["b"]) == {ops["c"]}
    assert not graph.get_users(ops["c"])

    assert not graph.get_inputs(ops["a"])
    assert graph.get_inputs(ops["b"]) == {ops["a"]}
    assert graph.get_inputs(ops["c"]) == {ops["b"]}


def test_topological_sort(simple_graph):
    """Test topological sorting."""
    graph, ops = simple_graph

    ops_sorted = graph.topological_sort()

    assert ops_sorted.index(ops["a"]) < ops_sorted.index(ops["b"])
    assert ops_sorted.index(ops["b"]) < ops_sorted.index(ops["c"])


def test_graph_iteration(simple_graph):
    """Test graph iteration methods."""
    graph, ops = simple_graph

    bfs_order = list(graph.iter_bfs())
    assert len(bfs_order) == 3
    assert bfs_order[0].name == "a"

    dfs_order = list(graph.iter_dfs())
    assert len(dfs_order) == 3
    assert dfs_order[0].name == "a"


def test_graph_validation(simple_graph, dummy_op):
    """Test graph validation."""
    graph, ops = simple_graph

    assert graph.validate()

    c_op = graph.get_op(ops["c"])  # noqa:F841
    a_op = graph.get_op(ops["a"])  # noqa:F841
    graph.add_edge(ops["c"], ops["a"])
    assert not graph.validate()


def test_graph_stats(simple_graph):
    """Test graph statistics."""
    graph, ops = simple_graph

    stats = graph.compute_stats()
    assert stats.node_count == 3
    assert stats.edge_count == 2
    assert stats.max_depth == 2


def test_invalid_ops(empty_graph, dummy_op):
    """Test handling of invalid operations."""

    class InvalidOp(dummy_op("invalid").__class__):
        def validate(self) -> bool:
            return False

    with pytest.raises(ValidationError):
        empty_graph.add_op(InvalidOp("invalid"))


def test_graph_edge_cases(empty_graph, dummy_op):
    """Test graph edge cases."""
    with pytest.raises(ValidationError):
        empty_graph.remove_op(-1)

    with pytest.raises(ValidationError):
        empty_graph.get_op(-1)

    op = dummy_op("test")
    node_id = empty_graph.add_op(op)
    with pytest.raises(ValidationError):
        empty_graph.add_edge(node_id, -1)
