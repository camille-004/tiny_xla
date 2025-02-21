from __future__ import annotations

import abc
import dataclasses
from typing import Any, Sequence

from ..core.context import XLAContext
from ..core.graph import Graph
from ..core.types import NodeId
from ..utils.logger import XLALogger

logger = XLALogger.get_logger(__name__)


@dataclasses.dataclass
class PassResult:
    """Result of running an optimization pass."""

    name: str
    changed: bool
    stats: Any | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None

    def __str__(self) -> str:
        result = (
            f"Pass {self.name}: {'changed' if self.changed else 'no change'}"
        )
        if self.stats:
            result += f"\n  Stats: {self.stats}"
        if self.errors:
            result += f"\n  Errors: {len(self.errors)}"
        if self.warnings:
            result += f"\n  Warnings: {len(self.warnings)}"
        return result


@dataclasses.dataclass
class PassConfig:
    """Configuration for an optimization pass."""

    enabled: bool = True
    verify_after: bool = True
    max_iter: int = 1
    debug: bool = False


class Pass(abc.ABC):
    """Base class for all optimization passes.

    Each pass takes a Graph and applies transformations to optimize it.
    Passes should:
    1. Verify graph integrity before and after
    2. Track and report statistics
    3. Handle errors
    """

    def __init__(self, name: str, config: PassConfig | None = None) -> None:
        self.name = name
        self.config = config or PassConfig()
        self._iteration = 0

    @property
    def iteration(self) -> int:
        """Get current iteration count."""
        return self._iteration

    @abc.abstractmethod
    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        """Run the optimization pass on the graph."""
        raise NotImplementedError

    @abc.abstractmethod
    def verify(self, graph: Graph) -> bool:
        """Verify graph integrity after optimization."""
        raise NotImplementedError

    def run_with_verify(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        """Run the pass with verification before and after."""
        if not self.config.enabled:
            return PassResult(self.name, changed=False)

        if not self.verify(graph):
            logger.error("Graph verification failed before %s pass", self.name)
            return PassResult(
                self.name,
                changed=False,
                errors=["Initial graph verification failed"],
            )

        try:
            self._iteration += 1
            result = self.run(graph, outputs, context)

            if result.changed and self.config.verify_after:
                if not self.verify(graph):
                    logger.error(
                        "Graph verification failed after %s pass", self.name
                    )
                    result.errors = result.errors or []
                    result.errors.append("Final graph verification failed")

            return result

        except Exception as e:
            logger.error("Pass %s failed: %s", self.name, e)
            return PassResult(self.name, changed=False, errors=[str(e)])

    def reset(self) -> None:
        """Reset pass state."""
        self._iteration = 0

    def supports_partial_graph(self) -> bool:
        """Whether pass can optimize a subset of the graph."""
        return False

    def __str__(self) -> str:
        return f"{self.name} (iteration {self._iteration})"


class AnalysisPass(Pass):
    """Base class for analysis passes that don't modify the graph."""

    def verify(self, graph: Graph) -> bool:
        return True


class FunctionPass(Pass):
    """Base class for passes that operate on individual functions."""

    def supports_partial_graph(self) -> bool:
        return True


class GraphPass(Pass):
    """Base class for passes that operate on the whole graph."""

    def supports_partial_graph(self) -> bool:
        return False
