from dataclasses import dataclass

import pytest

from tiny_xla.core.context import XLAContext
from tiny_xla.core.graph import Graph
from tiny_xla.passes.pass_base import Pass, PassResult
from tiny_xla.passes.pass_manager import (
    OptimizationLevel,
    PassGroupConfig,
    PassManager,
    PassManagerConfig,
)


@dataclass
class CounterStats:
    runs: int = 0
    changes: int = 0


class CountingPass(Pass):
    def __init__(self, name: str, should_change: bool = False):
        super().__init__(name)
        self.stats = CounterStats()
        self.should_change = should_change

    def run(
        self, graph: Graph, outputs=None, context: XLAContext | None = None
    ) -> PassResult:
        self.stats.runs += 1
        if self.should_change:
            self.stats.changes += 1
        return PassResult(
            self.name, changed=self.should_change, stats=self.stats
        )

    def verify(self, graph: Graph) -> bool:
        return True

    def reset(self) -> None:
        self.stats.runs = 0
        self.stats.changes = 0


class FailingPass(Pass):
    """Pass that fails verification."""

    def __init__(self, name: str = "failing_pass"):
        super().__init__(name)

    def run(self, graph: Graph, outputs=None, context=None) -> PassResult:
        return PassResult(self.name, changed=True)

    def verify(self, graph: Graph) -> bool:
        return False


@pytest.fixture
def reset_context():
    """Reset compiler context before and after tests."""
    XLAContext.reset()
    yield
    XLAContext.reset()


@pytest.fixture(scope="function")
def empty_manager():
    """Create an empty pass manager."""
    from tiny_xla.passes.pass_manager import PassManager, PassManagerConfig

    return PassManager(PassManagerConfig())


@pytest.fixture
def basic_setup_factory():
    def create_setup():
        manager = PassManager(PassManagerConfig(max_iter_per_pass=1))
        passes = [
            CountingPass("pass1"),
            CountingPass("pass2", should_change=True),
            CountingPass("pass3"),
        ]
        for p in passes:
            manager.add_pass(p)
        return manager, passes

    return create_setup


def test_pass_manager_creation():
    """Test pass manager creation and configuration."""
    config = PassManagerConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        debug=True,
        verify_each_pass=True,
        max_iter_per_pass=5,
    )
    pm = PassManager(config)

    assert pm.config.optimization_level == OptimizationLevel.AGGRESSIVE
    assert pm.config.debug
    assert pm.config.verify_each_pass
    assert pm.config.max_iter_per_pass == 5


def test_add_single_pass(empty_manager):
    """Test adding a single pass."""
    pass_obj = CountingPass("test_pass")
    empty_manager.add_pass(pass_obj)

    # Should be in default group
    default_group = empty_manager._groups[0]
    assert len(default_group.passes) == 1
    assert default_group.passes[0].name == "test_pass"


def test_add_pass_group(empty_manager):
    """Test adding a pass group."""
    passes = [CountingPass("group_pass1"), CountingPass("group_pass2")]
    config = PassGroupConfig(
        name="test_group", repeat_until_stable=True, max_iter=3
    )

    empty_manager.add_group("test_group", passes, config)

    # Find the group
    group = next(g for g in empty_manager._groups if g.name == "test_group")
    assert len(group.passes) == 2
    assert group.config.max_iter == 3
    assert group.config.repeat_until_stable


def test_run_passes(basic_setup_factory, empty_graph):
    """Test running all passes."""
    manager, passes = basic_setup_factory()
    stats = manager.run(empty_graph)

    # Check pass execution
    assert all(p.stats.runs == 1 for p in passes)
    assert stats.total_passes == 3
    assert stats.total_changes == 1  # Only pass2 changes
    assert stats.successful_passes == 3
    assert stats.failed_passes == 0


def test_optimization_levels(empty_graph):
    """Test different optimization levels."""
    pass_obj = CountingPass("test_pass")

    # No optimization
    config = PassManagerConfig(optimization_level=OptimizationLevel.NONE)
    pm = PassManager(config)
    pm.add_pass(pass_obj)
    stats = pm.run(empty_graph)
    assert pass_obj.stats.runs == 0

    # Basic optimization
    config = PassManagerConfig(optimization_level=OptimizationLevel.BASIC)
    pm = PassManager(config)
    pm.add_pass(pass_obj)
    stats = pm.run(empty_graph)  # noqa:F841
    assert pass_obj.stats.runs == 1


def test_pass_group_iteration(empty_graph):
    """Test pass group iteration behavior."""
    # Create passes that always change
    passes = [
        CountingPass("iter_pass1", should_change=True),
        CountingPass("iter_pass2", should_change=True),
    ]

    config = PassGroupConfig(
        name="iter_group", repeat_until_stable=True, max_iter=3
    )

    pm = PassManager()
    pm.add_group("iter_group", passes, config)
    stats = pm.run(empty_graph)

    # Should run max_iter times
    assert all(p.stats.runs == 3 for p in passes)
    assert stats.total_iterations > 0


def test_failing_pass(empty_graph):
    """Test handling of failing passes."""
    pm = PassManager()
    pm.add_pass(FailingPass())
    stats = pm.run(empty_graph)

    assert stats.failed_passes == 1
    assert stats.successful_passes == 0


def test_pass_manager_reset(basic_setup_factory, empty_graph):
    manager1, passes1 = basic_setup_factory()
    manager1.run(empty_graph)

    manager2, passes2 = basic_setup_factory()
    manager2.run(empty_graph)

    # Each new setup should have passes run exactly once.
    assert all(p.stats.runs == 1 for p in passes2)


def test_pass_execution_order(empty_graph):
    """Test passes execute in correct order."""
    execution_order = []

    class OrderTracking(Pass):
        def run(self, graph: Graph, outputs=None, context=None) -> PassResult:
            execution_order.append(self.name)
            return PassResult(self.name, changed=False)

        def verify(self, graph: Graph) -> bool:
            return True

    pm = PassManager()
    passes = ["first", "second", "third"]
    for name in passes:
        pm.add_pass(OrderTracking(name))

    pm.run(empty_graph)
    assert execution_order == passes


def test_pass_group_dependencies(empty_graph):
    """Test pass group dependencies."""
    group1_executed = False
    group2_executed = False

    class Group1Pass(Pass):
        def run(self, graph: Graph, outputs=None, context=None) -> PassResult:
            nonlocal group1_executed
            group1_executed = True
            return PassResult(self.name, changed=True)

        def verify(self, graph: Graph) -> bool:
            return True

    class Group2Pass(Pass):
        def run(self, graph: Graph, outputs=None, context=None) -> PassResult:
            nonlocal group2_executed
            assert group1_executed  # Group 1 should run first
            group2_executed = True
            return PassResult(self.name, changed=False)

        def verify(self, graph: Graph) -> bool:
            return True

    pm = PassManager()
    pm.add_pass(Group1Pass("g1"), group_name="group1")
    pm.add_pass(Group2Pass("g2"), group_name="group2")

    pm.run(empty_graph)
    assert group1_executed and group2_executed


def test_pass_manager_stats_tracking(basic_setup_factory, empty_graph):
    """Test detailed statistics tracking."""
    manager, passes = basic_setup_factory()
    stats = manager.run(empty_graph)

    assert stats.total_passes == len(passes)
    assert stats.total_changes == 1  # One pass changes
    assert stats.successful_passes == len(passes)
    assert stats.failed_passes == 0

    # Check individual pass stats
    for pass_obj in passes:
        assert pass_obj.name in stats.pass_stats
        pass_stats = stats.pass_stats[pass_obj.name]
        assert pass_stats.runs == 1
