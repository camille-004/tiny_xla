from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Any, Callable, ClassVar, Protocol, TypeVar, cast

from ..utils.errors import (
    ShapeError,
    TypeCheckingError,
    ValidationError,
)
from ..utils.logger import XLALogger
from .context import XLAContext
from .types import NodeId, Shape, Tensor

logger = XLALogger.get_logger(__name__)

T = TypeVar("T", bound="Operation")


class OpType(enum.StrEnum):
    """Types of operations supported."""

    BINARY = "binary"
    UNARY = "unary"
    REDUCTION = "reduction"
    RESHAPE = "reshape"
    ELEMENT_WISE = "element_wise"
    CONVOLUTION = "convolution"
    MATRIX_MULTIPLY = "matrix_multiply"
    BROADCAST = "broadcast"
    CONCAT = "concat"
    SLICE = "slice"
    CUSTOM = "custom"


@dataclasses.dataclass(slots=True)
class OpMetadata:
    """Metadata associated with an operation."""

    source_location: str | None = None
    original_node_id: NodeId | None = None
    is_stateless: bool = True
    is_commutative: bool = False
    is_constant: bool = False
    required_memory: int | None = False
    compute_cost: int | None = None
    memory_offset: int | None = None
    memory_size: int | None = None
    memory_alignment: int | None = None


class OpAttributes(Protocol):
    """Protocol defining the interface for operation attributes."""

    def validate(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpAttributes: ...


@dataclasses.dataclass(frozen=True, slots=True)
class BaseAttributes(OpAttributes):
    """Base implementation of operation attributes."""

    def validate(self) -> bool:
        """Validate attribute values."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert attributes to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpAttributes:
        """Create attributes from dictionary."""
        return cls(**data)


class Operation(abc.ABC):
    """Abstract base class for all XLA operations."""

    _registry: ClassVar[dict[str, type[Operation]]] = {}

    def __init__(
        self,
        op_type: OpType,
        name: str,
        inputs: list[Operation] | None = None,
        attributes: OpAttributes | None = None,
        metadata: OpMetadata | None = None,
    ) -> None:
        self._op_type = op_type
        self._name = name
        self._inputs = inputs or []
        self._attributes = attributes or BaseAttributes()
        self._metadata = metadata or OpMetadata()
        self._id: NodeId = XLAContext.get_context().get_next_node_id()
        self._output_type: Tensor | None = None

        if name not in self._registry:
            self._registry[name] = self.__class__

    @property
    def id(self) -> NodeId:
        """Get the unique node ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get the operation name."""
        return self._name

    @property
    def op_type(self) -> OpType:
        """Get the operation type."""
        return self._op_type

    @property
    def inputs(self) -> list[Operation]:
        """Get input operations."""
        return self._inputs

    @property
    def attributes(self) -> OpAttributes:
        """Get operation attributes."""
        return self._attributes

    @property
    def metadata(self) -> OpMetadata:
        """Get operation metadata."""
        return self._metadata

    @property
    def output_type(self) -> Tensor:
        """Get the output tensor type."""
        if self._output_type is None:
            self._output_type = self.compute_output_type()
        return self._output_type

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register operation implementation."""

        def decorator(op_class: type[T]) -> type[T]:
            cls._registry[name] = op_class
            return op_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        inputs: list[Operation] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Operation:
        """Create an operation by name."""
        if name not in cls._registry:
            raise ValidationError(f"Unknown operation: {name}")

        op_class = cls._registry[name]
        if attributes:
            return op_class(inputs or [], **attributes)
        return op_class(inputs or [])

    def add_input(self, op: Operation) -> None:
        """Add an input operation."""
        self._inputs.append(op)
        # Invalidate cached output type
        self._output_type = None

    def validate_input_types(self, expected: list[Tensor]) -> bool:
        """Validate input operation types."""
        if len(self._inputs) != len(expected):
            raise ValidationError(
                f"Expected {len(expected)} inputs, got {len(self._inputs)}",
                self._id,
                self._name,
            )

        for i, (input_op, expected_type) in enumerate(
            zip(self._inputs, expected)
        ):
            input_type = input_op.output_type
            if input_type != expected_type:
                raise TypeCheckingError(
                    expected_type,
                    input_type,
                    self._id,
                    f"{self._name} input {i}",
                )

        return True

    def validate_shapes(self, expected: list[Shape]) -> bool:
        """Validate input operation shapes."""
        for i, (input_op, expected_shape) in enumerate(
            zip(self._inputs, expected)
        ):
            input_shape = input_op.output_type.shape
            if input_shape != expected_shape:
                raise ShapeError(
                    expected_shape,
                    input_shape,
                    self._i,
                    f"{self._name} input {i}",
                )
        return True

    @abc.abstractmethod
    def compute_output_type(self) -> Tensor:
        """Compute the output tensor type."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate the operation."""
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self) -> Operation:
        """Lower this operation to simpler operations."""
        raise NotImplementedError

    def clone(self: T, new_inputs: list[Operation] | None = None) -> T:
        """Create a copy of this operation with optional new inputs."""
        return cast(
            T,
            self.__class__(
                new_inputs or self._inputs,
                attributes=self._attributes,
                metadata=self._metadata,
            ),
        )

    def __str__(self) -> str:
        """Get string representation."""
        inputs_str = ", ".join(str(op.id) for op in self._inputs)
        return f"{self._name}({inputs_str}) -> {self.output_type}"

    def __repr__(self) -> str:
        """Get detailed string representation."""
        return (
            f"{self.__class__.__name__}(id={self._id}, "
            f"name='{self._name}', type={self._op_type}, "
            f"inputs=[{', '.join(str(op.id) for op in self._inputs)}]"
        )
