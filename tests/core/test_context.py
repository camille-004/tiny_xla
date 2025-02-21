import threading
from pathlib import Path

import pytest

from tiny_xla.core.context import XLAContext, XLAOptions
from tiny_xla.core.types import XLAPhase


def test_singleton_pattern(reset_context):
    """Test that XLAContext behaves as a singleton."""
    ctx1 = XLAContext.get_context()
    ctx2 = XLAContext.get_context()

    assert ctx1 is ctx2
    assert id(ctx1) == id(ctx2)


def test_xla_options():
    """Test compiler options management."""
    ctx = XLAContext.get_context()

    assert ctx.options.optimization_level == 0
    assert ctx.options.target == "cpu"
    assert not ctx.options.debug
    assert ctx.options.output_dir == Path("./output")

    new_options = XLAOptions(
        optimization_level=2,
        target="gpu",
        debug=True,
        output_dir=Path("/tmp/output"),
    )
    ctx.options = new_options

    assert ctx.options.optimization_level == 2
    assert ctx.options.target == "gpu"
    assert ctx.options.debug
    assert ctx.options.output_dir == Path("/tmp/output")


def test_phase_management():
    """Test compilation phase management."""
    ctx = XLAContext.get_context()

    assert ctx.curr_phase == XLAPhase.PARSING

    ctx.enter_phase(XLAPhase.OPTIMIZATION)
    assert ctx.curr_phase == XLAPhase.OPTIMIZATION

    ctx.exit_phase()
    assert ctx.curr_phase == XLAPhase.PARSING


def test_nested_phases():
    """Test nested phase handling."""
    ctx = XLAContext.get_context()

    ctx.enter_phase(XLAPhase.OPTIMIZATION)
    ctx.enter_phase(XLAPhase.LOWERING)
    assert ctx.curr_phase == XLAPhase.LOWERING

    ctx.exit_phase()
    assert ctx.curr_phase == XLAPhase.OPTIMIZATION

    ctx.exit_phase()
    assert ctx.curr_phase == XLAPhase.PARSING


def test_node_id_generation():
    """Test unique node ID generation."""
    ctx = XLAContext.get_context()

    id1 = ctx.get_next_node_id()
    id2 = ctx.get_next_node_id()
    id3 = ctx.get_next_node_id()

    assert id1 != id2 != id3
    assert id2 > id1
    assert id3 > id2


def test_thread_safety():
    """Test thread-safe node ID generation."""
    ctx = XLAContext.get_context()
    ids = set()
    lock = threading.Lock()

    def generate_ids():
        for _ in range(100):
            node_id = ctx.get_next_node_id()
            with lock:
                ids.add(node_id)

    threads = [threading.Thread(target=generate_ids) for _ in range(10)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == 1000


def test_thread_local_storage():
    """Test thread-local storage."""
    ctx = XLAContext.get_context()
    results = {}
    lock = threading.Lock()

    def thread_func(thread_id):
        ctx.set_thread_local("key", f"value_{thread_id}")
        value = ctx.get_thread_local("key")
        with lock:
            results[thread_id] = value

    threads = [
        threading.Thread(target=thread_func, args=(i,)) for i in range(5)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 5
    for i in range(5):
        assert results[i] == f"value_{i}"


def test_context_scope():
    """Test context manager for phases."""
    ctx = XLAContext.get_context()

    with ctx.scope(XLAPhase.OPTIMIZATION):
        assert ctx.curr_phase == XLAPhase.OPTIMIZATION

        with ctx.scope(XLAPhase.LOWERING):
            assert ctx.curr_phase == XLAPhase.LOWERING

        assert ctx.curr_phase == XLAPhase.OPTIMIZATION

    assert ctx.curr_phase == XLAPhase.PARSING


def test_reset_behavior():
    """Test context reset behavior."""
    ctx = XLAContext.get_context()

    ctx.enter_phase(XLAPhase.OPTIMIZATION)
    ctx.set_metadata("test", "value")
    ctx.set_thread_local("test", "value")

    XLAContext.reset()
    new_ctx = XLAContext.get_context()

    assert new_ctx.curr_phase == XLAPhase.PARSING
    assert new_ctx.get_metadata("test") is None
    assert new_ctx.get_thread_local("test") is None


def test_error_handling():
    """Test error handling in context operations."""
    ctx = XLAContext.get_context()

    with pytest.raises(ValueError):
        ctx.exit_phase()  # No phase to exit

    assert ctx.get_thread_local("non_existent") is None
    assert ctx.get_thread_local("non_existent", default="default") == "default"


def test_default_values():
    """Test default value handling."""
    ctx = XLAContext.get_context()

    assert ctx.get_thread_local("key", "default") == "default"

    assert ctx.get_metadata("key") is None

    assert isinstance(ctx.options, XLAOptions)
