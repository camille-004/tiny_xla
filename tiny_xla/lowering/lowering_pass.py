from __future__ import annotations

import abc
import dataclasses
from typing import Sequence

from ..core.context import XLAContext
from ..core.graph import Graph
from ..core.operation import Operation, OpType
from ..core.types import NodeId
from ..passes.pass_base import Pass, PassResult
from ..utils.logger import XLALogger

logger = XLALogger.get_logger(__name__)


@dataclasses.dataclass
class LoweringContext:
    """Context for operation lowering."""

    target: str
    graph: Graph
    replacements: dict[NodeId, NodeId]
    new_ops: list[Operation]


class LoweringRule(abc.ABC):
    """Base class to operation lowering rules."""

    @abc.abstractmethod
    def matches(self, op: Operation) -> bool:
        """Check if this rule applies to the operation."""
        raise NotImplementedError

    @abc.abstractmethod
    def lower(
        self, op: Operation, ctx: LoweringContext
    ) -> list[Operation] | None:
        """Lower the operation to simpler operations."""
        raise NotImplementedError


class TargetInfo:
    """Information about target platform capabilities."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.supported_ops: set[OpType] = set()
        self.vector_size: int = 1
        self.max_unroll: int = 1

    def _init_target_info(self) -> None:
        """Initialize target-specific information."""
        if self.target == "cpu":
            self.supported_ops = {
                OpType.ELEMENT_WISE,
                OpType.RESHAPE,
                OpType.BINARY,
            }
            self.vector_size = 4
            self.max_unroll = 8
        else:
            logger.warning(f"Unknown target: {self.target}")


class LoweringPass(Pass):
    """Pass that lowers operations to target-specific implementations."""

    def __init__(
        self, target: str = "cpu", rules: list[LoweringRule] | None = None
    ) -> None:
        super().__init__("lowering")
        self.target = target
        self.target_info = TargetInfo(target)
        self.rules = rules or []
        self._ctx: LoweringContext | None = None

    def needs_lowering(self, op: Operation) -> bool:
        """Check if operation needs to be lowered."""
        if op.op_type not in self.target_info.supported_ops:
            return True

        # Check if operation has its own lowering logic
        try:
            lowered = op.lower()
            return lowered is not op
        except NotImplementedError:
            return True

    def _create_context(self, graph: Graph) -> LoweringContext:
        """Create lowering context."""
        return LoweringContext(
            target=self.target, graph=graph, replacements={}, new_ops=[]
        )

    def _apply_rules(
        self, op: Operation, ctx: LoweringContext
    ) -> list[Operation] | None:
        """Apply lowering rules to an operation."""
        for rule in self.rules:
            if rule.matches(op):
                return rule.lower(op, ctx)
        return None

    def _update_graph(self, graph: Graph, ctx: LoweringContext) -> None:
        for op in ctx.new_ops:
            graph.add_op(op)

        for old_id, new_id in ctx.replacements.items():
            if old_id in graph.ops:
                for user_id in graph.get_users(old_id):
                    user_op = graph.get_op(user_id)
                    new_inputs = [
                        graph.get_op(ctx.replacements.get(inp.id, inp.id))
                        for inp in user_op.inputs
                    ]
                    user_op._inputs = new_inputs

                graph.remove_op(old_id)

    def run(
        self,
        graph: Graph,
        outputs: Sequence[int] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        """Run lowering pass."""
        try:
            changed = False
            self._ctx = self._create_context(graph)

            for node_id in graph.topological_sort():
                op = graph.get_op(node_id)

                if self.needs_lowering(op):
                    try:
                        lowered = op.lower()
                        if lowered is not op:
                            self._ctx.new_ops.append(lowered)
                            self._ctx.replacements[node_id] = lowered.id
                            changed = True
                            continue
                    except NotImplementedError:
                        pass

                    ops_lowered = self._apply_rules(op, self._ctx)
                    if ops_lowered:
                        self._ctx.new_ops.extend(ops_lowered)
                        self._ctx.replacements[node_id] = ops_lowered[-1].id
                        changed = True

            if changed:
                self._update_graph(graph, self._ctx)

            return PassResult(self.name, changed=changed)

        except Exception as e:
            logger.error("Lowering failed: %s", e)
            return PassResult(self.name, changed=False, errors=[str(e)])

        finally:
            self._ctx = None

    def verify(self, graph: Graph) -> bool:
        """Verify graph after lowering."""
        try:
            for op in graph.ops.values():
                if op.op_type not in self.target_info.supported_ops:
                    logger.error(
                        "Operation %s not supported by target %s",
                        op.name,
                        self.target,
                    )
                    return False

            return graph.validate()

        except Exception as e:
            logger.error("Lowering verification failed: %s", e)
            return False


def create_cpu_lowering_pass() -> LoweringPass:
    """Create a lowering pass for CPU target."""
    return LoweringPass(target="cpu")
