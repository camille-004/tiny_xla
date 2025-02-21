from __future__ import annotations

import dataclasses
from collections import defaultdict, deque
from typing import Callable, Generator, Iterator, TypeVar

from ..utils.errors import InternalError, ValidationError
from ..utils.logger import XLALogger
from .operation import Operation, OpType
from .types import NodeId

logger = XLALogger.get_logger(__name__)

T = TypeVar("T", bound=Operation)


@dataclasses.dataclass(slots=True)
class GraphStats:
    """Statistics about the computation graph."""

    node_count: int = 0
    edge_count: int = 0
    max_depth: int = 0
    op_counts: dict[OpType, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )


class Graph:
    """Represents a computation graph of operations.

    Maintains both forward (dependencies) and backward (users) edges.
    """

    def __init__(self) -> None:
        self._ops: dict[NodeId, Operation] = {}
        self._forward_edges: dict[NodeId, set[NodeId]] = defaultdict(set)
        self._backward_edges: dict[NodeId, set[NodeId]] = defaultdict(set)
        self._root_nodes: set[NodeId] = set()
        self._leaf_nodes: set[NodeId] = set()

    @property
    def ops(self) -> dict[NodeId, Operation]:
        """Get all operations in the graph."""
        return self._ops

    def add_op(self, op: Operation) -> NodeId:
        """Add an operation to the graph."""
        if not op.validate():
            raise ValidationError(
                f"Operation {op.name} failed validation", op.id
            )

        node_id = op.id
        if node_id in self._ops:
            raise ValidationError(f"Duplicate node ID: {node_id}", node_id)

        self._ops[node_id] = op
        self._forward_edges[node_id] = set()
        self._backward_edges[node_id] = set()

        # If no inputs, it's a root node
        if not op.inputs:
            self._root_nodes.add(node_id)
        else:
            for input_op in op.inputs:
                self.add_edge(input_op.id, node_id)

        self._leaf_nodes.add(node_id)

        logger.debug(
            "Added operation %s (id=%d, type=%s)", op.name, node_id, op.op_type
        )
        return node_id

    def add_edge(self, from_id: NodeId, to_id: NodeId) -> None:
        """Add a directed edge between operations."""
        if from_id not in self._ops:
            raise ValidationError(f"Invalid source node ID: {from_id}")
        if to_id not in self._ops:
            raise ValidationError(f"Invalid target node ID: {to_id}")

        # Removed cycle detection so that cycles are allowed.
        self._forward_edges[from_id].add(to_id)
        self._backward_edges[to_id].add(from_id)

        self._root_nodes.discard(to_id)
        self._leaf_nodes.discard(from_id)

    def _has_path(self, start: NodeId, end: NodeId) -> bool:
        """Check if there is a path from start to end using DFS."""
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            if current not in visited:
                visited.add(current)
                stack.extend(self._forward_edges[current])
        return False

    def remove_op(self, node_id: NodeId) -> None:
        """Remove an operation and all its edges from the graph."""
        if node_id not in self._ops:
            raise ValidationError(f"Invalid node ID: {node_id}")

        # Remove forward edges
        for target in self._forward_edges[node_id]:
            self._backward_edges[target].remove(node_id)
            if not self._backward_edges[target]:
                self._root_nodes.add(target)

        # Remove backward edges
        for source in self._backward_edges[node_id]:
            self._forward_edges[source].remove(node_id)
            if not self._forward_edges[source]:
                self._leaf_nodes.add(source)

        # Clean up
        del self._ops[node_id]
        del self._forward_edges[node_id]
        del self._backward_edges[node_id]
        self._root_nodes.discard(node_id)
        self._leaf_nodes.discard(node_id)

    def get_op(self, node_id: NodeId) -> Operation:
        """Get an operation by its ID."""
        if node_id not in self._ops:
            raise ValidationError(f"Invalid node ID: {node_id}")
        return self._ops[node_id]

    def get_users(self, node_id: NodeId) -> set[NodeId]:
        """Get all operations that use the output of this operation."""
        return self._forward_edges[node_id]

    def get_inputs(self, node_id: NodeId) -> set[NodeId]:
        """Get all input operations forthis operation."""
        return self._backward_edges[node_id]

    def topological_sort(self) -> list[NodeId]:
        """Return operations in topological order."""
        visited: set[NodeId] = set()
        temp_mark: set[NodeId] = set()
        order: list[NodeId] = []

        def visit(node_id: NodeId) -> None:
            if node_id in temp_mark:
                raise InternalError("Graph contains cycles")
            if node_id in visited:
                return

            temp_mark.add(node_id)
            for target in self._forward_edges[node_id]:
                visit(target)
            temp_mark.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        for root in sorted(self._root_nodes):
            if root not in visited:
                visit(root)

        return order[::-1]

    def iter_bfs(self) -> Generator[Operation, None, None]:
        """Iterate through operations in breadth-first order."""
        visited: set[NodeId] = set()
        queue = deque(sorted(self._root_nodes))

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue

            visited.add(node_id)
            yield self._ops[node_id]

            for target in sorted(self._forward_edges[node_id]):
                if target not in visited:
                    queue.append(target)

    def iter_dfs(self) -> Generator[Operation, None, None]:
        """Iterate through operations in depth-first order."""
        visited: set[NodeId] = set()

        def dfs(node_id: NodeId) -> Generator[Operation, None, None]:
            if node_id in visited:
                return
            visited.add(node_id)
            yield self._ops[node_id]
            for target in sorted(self._forward_edges[node_id]):
                yield from dfs(target)

        for root in sorted(self._root_nodes):
            yield from dfs(root)

    def collect_subgraph(
        self, predicate: Callable[[Operation], bool]
    ) -> set[NodeId]:
        """Collect all operations that satisfy the predicate."""
        result: set[NodeId] = set()

        def collect(node_id: NodeId) -> None:
            if node_id in result:
                return
            op = self._ops[node_id]
            if predicate(op):
                result.add(node_id)
                for target in self._forward_edges[node_id]:
                    collect(target)

        for root in self._root_nodes:
            collect(root)

        return result

    def validate(self) -> bool:
        """Validate the entire graph.

        Checks:
        - All operations are valid
        - No cycles
        - Edge consistency
        - Type compatibility
        """
        try:
            for op in self._ops.values():
                if not op.validate():
                    return False

            # Check for cycles
            self.topological_sort()

            for node_id, op in self._ops.items():
                input_ids = {input_op.id for input_op in op.inputs}
                if input_ids != self._backward_edges[node_id]:
                    return False

            return True

        except Exception as e:
            logger.error("Graph validation failed: %s", e)
            return False

    def compute_stats(self) -> GraphStats:
        """Compute statistics about the graph."""
        stats = GraphStats()
        stats.node_count = len(self._ops)
        stats.edge_count = sum(
            len(edges) for edges in self._forward_edges.values()
        )

        for op in self._ops.values():
            stats.op_counts[op.op_type] += 1

        visited: dict[NodeId, int] = {}

        def compute_depth(node_id: NodeId) -> int:
            if node_id in visited:
                return visited[node_id]

            if not self._backward_edges[node_id]:
                depth = 0
            else:
                depth = 1 + max(
                    compute_depth(pred)
                    for pred in self._backward_edges[node_id]
                )

            visited[node_id] = depth
            stats.max_depth = max(stats.max_depth, depth)
            return depth

        for leaf in self._leaf_nodes:
            compute_depth(leaf)

        return stats

    def __str__(self) -> str:
        """Get string representation of the graph."""
        stats = self.compute_stats()
        return (
            f"Graph(nodes={stats.node_count}, "
            f"edges={stats.edge_count}, "
            f"depth={stats.max_depth})"
        )


class GraphView:
    """Provides a view of a subgraph of operations."""

    def __init__(self, graph: Graph, nodes: set[NodeId]) -> None:
        self._graph = graph
        self._nodes = nodes

    def __iter__(self) -> Iterator[Operation]:
        """Iterate over operations in the view."""
        return (self._graph.get_op(node_id) for node_id in self._nodes)

    def filter(self, predicate: Callable[[Operation], bool]) -> GraphView:
        """Create a new view.

        The view will only haveoperations that satisfy the predicate.
        """
        filtered = {
            node_id
            for node_id in self._nodes
            if predicate(self._graph.get_op(node_id))
        }
        return GraphView(self._graph, filtered)
