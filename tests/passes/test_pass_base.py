from dataclasses import dataclass
from typing import Sequence

from tiny_xla.core.context import XLAContext
from tiny_xla.core.graph import Graph
from tiny_xla.core.types import NodeId
from tiny_xla.passes.pass_base import (
    AnalysisPass,
    FunctionPass,
    GraphPass,
    Pass,
    PassConfig,
    PassResult,
)


@dataclass
class TestPassStats:
    """Stats for test passes."""

    __test__ = False

    count: int = 0
    changed: bool = False


class TestPass(Pass):
    """Simple test pass that counts iterations."""

    __test__ = False

    def __init__(
        self,
        name: str = "test_pass",
        always_change: bool = False,
        config: PassConfig | None = None,
    ):
        super().__init__(name, config=config)
        self.always_change = always_change
        self.stats = TestPassStats()

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        self.stats.count += 1
        return PassResult(self.name, changed=self.always_change)

    def verify(self, graph: Graph) -> bool:
        return True


class FailingPass(Pass):
    """Pass that fails verification."""

    def __init__(self):
        super().__init__("failing_pass")

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        return PassResult(self.name, changed=True)

    def verify(self, graph: Graph) -> bool:
        return False


class ErrorPass(Pass):
    """Pass that raises an exception."""

    def __init__(self):
        super().__init__("error_pass")

    def run(
        self,
        graph: Graph,
        outputs: Sequence[NodeId] | None = None,
        context: XLAContext | None = None,
    ) -> PassResult:
        raise RuntimeError("Test error")

    def verify(self, graph: Graph) -> bool:
        return True


def test_pass_config():
    """Test pass configuration."""
    config = PassConfig(
        enabled=True, verify_after=True, max_iter=5, debug=True
    )
    test_pass = TestPass(config=config)
    assert test_pass.config.enabled
    assert test_pass.config.verify_after
    assert test_pass.config.max_iter == 5
    assert test_pass.config.debug


def test_pass_disabled(empty_graph):
    """Test disabled pass behavior."""
    config = PassConfig(enabled=False)
    test_pass = TestPass(config=config)
    result = test_pass.run_with_verify(empty_graph)
    assert not result.changed
    assert test_pass.stats.count == 0


def test_pass_verification(empty_graph):
    """Test pass verification."""
    failing_pass = FailingPass()
    result = failing_pass.run_with_verify(empty_graph)
    assert not result.changed
    assert result.errors is not None
    assert "verification failed" in result.errors[0].lower()


def test_pass_error_handling(empty_graph):
    """Test pass error handling."""
    error_pass = ErrorPass()
    result = error_pass.run_with_verify(empty_graph)
    assert not result.changed
    assert result.errors is not None
    assert "Test error" in result.errors[0]


def test_pass_iteration_tracking(empty_graph):
    """Test pass iteration tracking."""
    test_pass = TestPass(always_change=True)
    assert test_pass.iteration == 0
    test_pass.run_with_verify(empty_graph)
    assert test_pass.iteration == 1
    test_pass.run_with_verify(empty_graph)
    assert test_pass.iteration == 2
    test_pass.reset()
    assert test_pass.iteration == 0


def test_analysis_pass(empty_graph):
    """Test analysis pass behavior."""

    class TestAnalysis(AnalysisPass):
        def run(
            self,
            graph: Graph,
            outputs: Sequence[NodeId] | None = None,
            context: XLAContext | None = None,
        ) -> PassResult:
            return PassResult(self.name, changed=False)

        def verify(self, graph: Graph) -> bool:
            return True

    analysis = TestAnalysis("test_analysis")
    result = analysis.run_with_verify(empty_graph)
    assert not result.changed
    assert analysis.verify(empty_graph)


def test_function_pass(empty_graph):
    """Test function pass behavior."""

    class TestFunction(FunctionPass):
        def run(
            self,
            graph: Graph,
            outputs: Sequence[NodeId] | None = None,
            context: XLAContext | None = None,
        ) -> PassResult:
            return PassResult(self.name, changed=False)

        def verify(self, graph: Graph) -> bool:
            return True

    func_pass = TestFunction("test_function")
    assert func_pass.supports_partial_graph()


def test_graph_pass(empty_graph):
    """Test graph pass behavior."""

    class TestGraph(GraphPass):
        def run(
            self,
            graph: Graph,
            outputs: Sequence[NodeId] | None = None,
            context: XLAContext | None = None,
        ) -> PassResult:
            return PassResult(self.name, changed=False)

        def verify(self, graph: Graph) -> bool:
            return True

    graph_pass = TestGraph("test_graph")
    assert not graph_pass.supports_partial_graph()


def test_pass_result_creation():
    """Test pass result creation and string representation."""
    result = PassResult(
        name="test",
        changed=True,
        stats={"count": 1},
        errors=["error1"],
        warnings=["warning1"],
    )
    assert result.changed
    assert result.stats == {"count": 1}
    assert "error1" in result.errors
    assert "warning1" in result.warnings
    result_str = str(result)
    assert "test" in result_str
    assert "changed" in result_str
    assert "Errors" in result_str
    assert "Warnings" in result_str


def test_pass_context_handling(empty_graph):
    """Test pass handling of compiler context."""
    from tiny_xla.core.context import XLAContext

    ctx = XLAContext.get_context()

    class ContextTestPass(Pass):
        def run(
            self,
            graph: Graph,
            outputs: Sequence[NodeId] | None = None,
            context: XLAContext | None = None,
        ) -> PassResult:
            assert context is ctx
            return PassResult(self.name, changed=False)

        def verify(self, graph: Graph) -> bool:
            return True

    test_pass = ContextTestPass("context_test")
    test_pass.run_with_verify(empty_graph, context=ctx)
