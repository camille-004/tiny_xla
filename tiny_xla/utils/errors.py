from __future__ import annotations

from pathlib import Path

from tiny_xla.core.types import NodeId, Shape, XLAType


class CompilerError(Exception):
    """Base class for all compiler-related errors."""


class ValidationError(CompilerError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        node_id: int | None = None,
        operation: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.operation = operation
        super().__init__(
            f"{message} "
            f"[Node: {node_id if node_id is not None else 'unknown'}] "
            f"[Op: {operation if operation is not None else 'unknown'}]"
        )


class TypeCheckingError(ValidationError):
    """Raised when type checking fails."""

    def __init__(
        self,
        expected_type: XLAType,
        actual_type: XLAType,
        node_id: NodeId | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(
            f"Type mismatch: expected {expected_type}, got {actual_type}",
            node_id,
            operation,
        )
        self.expected_type = expected_type
        self.actual_type = actual_type


class ShapeError(ValidationError):
    """Raised when tensor shape validation fails."""

    def __init__(
        self,
        expected_shape: Shape,
        actual_shape: Shape,
        node_id: NodeId | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}",
            node_id,
            operation,
        )
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class ParseError(CompilerError):
    """Raised when parsing input code fails."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        file: Path | None = None,
    ) -> None:
        location = ""
        if file is not None:
            location += f" in {file}"
        if line is not None:
            location += f" at line {line}"
            if column is not None:
                location += f", column {column}"

        super().__init__(f"Parse error: {message}{location}")
        self.line = line
        self.column = column
        self.file = file


class LoweringError(CompilerError):
    """Raised when operation lowering fails."""

    def __init__(
        self, message: str, operation: str, node_id: NodeId | None = None
    ) -> None:
        super().__init__(
            f"Failed to lower operation {operation}: {message} "
            f"[Node: {node_id if node_id is not None else 'unknown'}]"
        )
        self.operation = operation
        self.node_id = node_id


class OptimizationError(CompilerError):
    """Raised when an optimization pass fails."""

    def __init__(
        self, message: str, pass_name: str, node_id: NodeId | None = None
    ) -> None:
        super().__init__(
            f"Optimization pass {pass_name} failed: {message} "
            f"[Node: {node_id if node_id is not None else 'unknown'}]"
        )
        self.pass_name = pass_name
        self.node_id = node_id


class CodegenError(CompilerError):
    """Raised when code generation fails."""

    def __init__(
        self,
        message: str,
        target: str,
        node_id: NodeId | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(
            f"Code generation failed for target {target}: {message} "
            f"[Node: {node_id if node_id is not None else 'unknown'}] "
            f"[Op: {operation if operation is not None else 'unknown'}]"
        )
        self.target = target
        self.node_id = node_id
        self.operation = operation


class ResourceError(CompilerError):
    """Raised when resource limits are exceeded."""

    def __init__(
        self,
        resource_type: str,
        limit: int,
        actual: int,
        node_id: NodeId | None = None,
    ) -> None:
        super().__init__(
            f"Resource limit exceeded for {resource_type}: "
            f"limit {limit}, actual {actual}"
            f"[Node: {node_id if node_id is not None else 'unknown'}]"
        )
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual
        self.node_id = node_id


class InternalError(CompilerError):
    """Raised when an internal invariant is violated.

    Usually indicates a bug in the compiler itself.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"Internal compiler error: {message}")
