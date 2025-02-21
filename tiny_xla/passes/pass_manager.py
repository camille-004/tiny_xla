from __future__ import annotations

import dataclasses
import enum
from typing import Any, Sequence

from ..core.context import XLAContext
from ..core.graph import Graph
from ..core.types import NodeId
from ..utils.logger import XLALogger
from .pass_base import Pass, PassResult

logger = XLALogger.get_logger(__name__)


class OptimizationLevel(enum.IntEnum):
    NONE = 0
    BASIC = 1
    NORMAL = 2
    AGGRESSIVE = 3


@dataclasses.dataclass
class PassGroupConfig:
    name: str
    enabled: bool = True
    repeat_until_stable: bool = True
    max_iter: int = 10
    verify_after: bool = True


@dataclasses.dataclass
class PassManagerConfig:
    optimization_level: OptimizationLevel = OptimizationLevel.NORMAL
    debug: bool = False
    verify_each_pass: bool = True
    max_iter_per_pass: int = 1
    groups: list[PassGroupConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PassManagerStats:
    total_passes: int = 0
    successful_passes: int = 0
    failed_passes: int = 0
    total_changes: int = 0
    total_iterations: int = 0
    pass_stats: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Ran {self.total_passes} passes "
            f"({self.successful_passes} successful, "
            f"{self.failed_passes} failed) with "
            f"{self.total_changes} total changes in "
            f"{self.total_iterations} iterations"
        )


class PassGroup:
    def __init__(
        self,
        name: str,
        passes: list[Pass],
        config: PassGroupConfig | None = None,
    ) -> None:
        self.name = name
        self.passes = passes
        self.config = config or PassGroupConfig(name)

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> tuple[bool, list[PassResult]]:
        if not self.config.enabled:
            return False, []
        iteration = 0
        group_changed = False
        all_results: list[PassResult] = []
        while iteration < self.config.max_iter:
            iteration += 1
            iteration_changed = False
            for optimization_pass in self.passes:
                result = optimization_pass.run_with_verify(
                    graph, outputs, context
                )
                all_results.append(result)
                if result.errors:
                    logger.error(
                        "Pass %s failed in group %s: %s",
                        optimization_pass.name,
                        self.name,
                        result.errors[0],
                    )
                    continue
                iteration_changed |= result.changed
                group_changed |= result.changed
            if not iteration_changed and iteration > 1:
                break
            if not self.config.repeat_until_stable:
                break
        if group_changed and self.config.verify_after:
            for optimization_pass in self.passes:
                if not optimization_pass.verify(graph):
                    logger.error(
                        "Graph verification failed after group %s", self.name
                    )
                    return False, all_results
        return group_changed, all_results


class PassManager:
    def __init__(self, config: PassManagerConfig | None = None) -> None:
        self.config = config or PassManagerConfig()
        self._groups: list[PassGroup] = []
        self._passes: dict[str, Pass] = {}

    def add_pass(self, pass_obj: Pass, group_name: str = "default") -> None:
        self._passes[pass_obj.name] = pass_obj
        group = next((g for g in self._groups if g.name == group_name), None)
        if group is None:
            group = PassGroup(
                group_name,
                [],
                PassGroupConfig(
                    name=group_name,
                    verify_after=self.config.verify_each_pass,
                    max_iter=self.config.max_iter_per_pass,
                ),
            )
            self._groups.append(group)
        group.passes.append(pass_obj)

    def add_group(
        self,
        name: str,
        passes: list[Pass],
        config: PassGroupConfig | None = None,
    ) -> None:
        group = PassGroup(name, passes, config)
        self._groups.append(group)
        for pass_obj in passes:
            self._passes[pass_obj.name] = pass_obj

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassManagerStats:
        stats = PassManagerStats()
        if self.config.optimization_level == OptimizationLevel.NONE:
            logger.info("Skipping optimizations (level=None)")
            return stats
        for group in self._groups:
            if not group.config.enabled:
                continue
            logger.info("Running pass group: %s", group.name)
            changed, results = group.run(graph, outputs, context)
            for result in results:
                stats.total_passes += 1
                if result.errors:
                    stats.failed_passes += 1
                else:
                    stats.successful_passes += 1
                    if result.changed:
                        stats.total_changes += 1
                    if result.stats:
                        stats.pass_stats[result.name] = result.stats
            stats.total_iterations += 1
            if changed:
                logger.info("Pass group %s modified the graph", group.name)
            if self.config.debug:
                logger.debug("Graph after group %s: %s", group.name, graph)
        logger.info("Optimization complete: %s", stats)
        return stats
