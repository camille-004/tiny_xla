from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any, ClassVar

from ..core.types import NodeId, XLAPhase
from ..utils.logger import XLALogger

logger = XLALogger.get_logger(__name__)


@dataclasses.dataclass(slots=True)
class XLAOptions:
    """Configuration options for the compiler."""

    optimization_level: int = 0
    target: str = "cpu"
    debug: bool = False
    output_dir: Path = Path("./output")
    max_unroll_factor: int = 8
    vectorize: bool = True
    enable_fast_math: bool = False
    inline_threshold: int = 225
    memory_limit_mb: int = 4096


class XLAContext:
    """Thread-safe singleton managing global compiler state."""

    _instance: ClassVar[XLAContext | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._options: XLAOptions | None = None
        self._curr_phase = XLAPhase.PARSING
        self._node_counter = 0
        self._node_counter_lock = threading.Lock()
        self._thread_local = threading.local()
        self._metadata: dict[str, Any] = {}
        self._phase_stack: list[XLAPhase] = []

    @classmethod
    def get_context(cls) -> XLAContext:
        """Get or create the singleton compiler context."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("Created new Context instance")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the compiler context (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                logger.debug("Resetting Context")
                cls._instance = None

    @property
    def options(self) -> XLAOptions:
        """Get the current compiler options."""
        if self._options is None:
            self._options = XLAOptions()
        return self._options

    @options.setter
    def options(self, options: XLAOptions) -> None:
        """Set new compiler options."""
        self._options = options
        logger.debug("Updated compiler options: %s", options)

    @property
    def curr_phase(self) -> XLAPhase:
        """Get the current compilation phase."""
        if not self._phase_stack:
            return self._curr_phase
        return self._phase_stack[-1]

    def enter_phase(self, phase: XLAPhase) -> None:
        """Enter a new compilation phase."""
        self._phase_stack.append(phase)
        logger.info("Entering compiler phase: %s", phase)

    def exit_phase(self) -> None:
        """Exit the current compilation phase."""
        if self._phase_stack:
            phase = self._phase_stack.pop()
            logger.info("Exiting compiler phase: %s", phase)
        else:
            raise ValueError("No phase to exit")

    def get_next_node_id(self) -> NodeId:
        """Generate a unique node ID in a thread-safe manner."""
        with self._node_counter_lock:
            self._node_counter += 1
            return self._node_counter

    def set_metadata(self, key: str, value: Any) -> None:
        """Set compiler metadata."""
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get compiler metadata."""
        return self._metadata.get(key)

    def set_thread_local(self, key: str, value: Any) -> None:
        """Set thread-local data."""
        setattr(self._thread_local, key, value)

    def get_thread_local(self, key: str, default: Any = None) -> Any:
        """Get thread-local data."""
        return getattr(self._thread_local, key, default)

    def scope(self, phase: XLAPhase | None = None) -> XLAContextScope:
        """Create a new compiler context scope."""
        return XLAContextScope(self, phase)


class XLAContextScope:
    """Context manager for compiler phases."""

    def __init__(
        self, context: XLAContext, phase: XLAPhase | None = None
    ) -> None:
        self.context = context
        self.phase = phase

    def __enter__(self) -> XLAContext:
        if self.phase is not None:
            self.context.enter_phase(self.phase)
        return self.context

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.phase is not None:
            self.context.exit_phase()
