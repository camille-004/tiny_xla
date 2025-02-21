from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from ...core.graph import Graph
from ...core.operation import Operation
from ...core.types import NodeId
from ...utils.logger import XLALogger
from ..pass_base import Pass, PassResult

logger = XLALogger.get_logger(__name__)


@dataclasses.dataclass
class DCEStats:
    removed_ops: int = 0
    removed_edges: int = 0
    total_ops: int = 0

    def __str__(self) -> str:
        return (
            f"DCE removed {self.removed_ops}/{self.total_ops} operations "
            f"({self.removed_edges} edges)"
        )


class DeadCodeElimination(Pass):
    """Eliminates operations that don't contribute to the output."""

    def __init__(self) -> None:
        super().__init__("dead_code_elimination")

    def _find_live_nodes(
        self, graph: Graph, outputs: Sequence[NodeId]
    ) -> set[NodeId]:
        live_nodes: set[NodeId] = set()
        work_list = list(outputs)
        while work_list:
            node_id = work_list.pop()
            if node_id in live_nodes:
                continue
            live_nodes.add(node_id)
            op = graph.get_op(node_id)
            for input_op in op.inputs:
                work_list.append(input_op.id)
        return live_nodes

    def _remove_dead_nodes(
        self, graph: Graph, live_nodes: set[NodeId]
    ) -> DCEStats:
        stats = DCEStats(total_ops=len(graph.ops))
        dead_nodes = set(graph.ops.keys()) - live_nodes
        for node_id in dead_nodes:
            stats.removed_edges += len(graph.get_users(node_id))
            stats.removed_edges += len(graph.get_inputs(node_id))
        for node_id in dead_nodes:
            graph.remove_op(node_id)
            stats.removed_ops += 1
        return stats

    def _should_preserve(self, op: Operation) -> bool:
        # Preserve stateful operations and resource variables.
        if not op.metadata.is_stateless:
            return True
        if op.name.startswith("resource"):
            return True
        return False

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: Any | None = None,
    ) -> PassResult:
        # If no outputs are specified, remove all operations.
        if outputs is None:
            outputs = []

        live_nodes = self._find_live_nodes(graph, outputs)
        for node_id, op in graph.ops.items():
            if self._should_preserve(op):
                live_nodes.add(node_id)
        stats = self._remove_dead_nodes(graph, live_nodes)
        changed = stats.removed_ops > 0
        if changed:
            logger.info(str(stats))
            return PassResult(self.name, changed=True, stats=stats)
        return PassResult(self.name, changed=False, stats=stats)

    def verify(self, graph: Graph) -> bool:
        try:
            for node_id, op in graph.ops.items():
                for user_id in graph.get_users(node_id):
                    user_op = graph.get_op(user_id)
                    if op not in user_op.inputs:
                        return False
                for input_op in op.inputs:
                    if node_id not in graph.get_users(input_op.id):
                        return False
            for op in graph.ops.values():
                if not op.validate():
                    return False
            return True
        except Exception as e:
            logger.error("Graph verification failed: %s", e)
            return False


def create_dce_pass() -> Pass:
    return DeadCodeElimination()
