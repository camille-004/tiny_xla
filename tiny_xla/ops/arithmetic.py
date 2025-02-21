from __future__ import annotations

import dataclasses
from typing import ClassVar, Final

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Shape, Tensor, XLAType
from ..utils.errors import ShapeError, TypeCheckingError, ValidationError


@dataclasses.dataclass(frozen=True, slots=True)
class ArithmeticAttributes(OpAttributes):
    """Attributes for arithmetic operations."""

    broadcast_dimensions: tuple[int, ...] = ()

    def validate(self) -> bool:
        return all(dim >= 0 for dim in self.broadcast_dimensions)


class ArithmeticOp(Operation):
    """Base class for arithmetic operations."""

    # Promotion rules for mixing different types
    TYPE_PROMOTION: ClassVar[dict[tuple[XLAType, XLAType], XLAType]] = {
        (XLAType.INT32, XLAType.INT64): XLAType.INT64,
        (XLAType.INT64, XLAType.INT32): XLAType.INT64,
        (XLAType.FLOAT32, XLAType.FLOAT64): XLAType.FLOAT64,
        (XLAType.FLOAT64, XLAType.FLOAT32): XLAType.FLOAT64,
        (XLAType.INT32, XLAType.FLOAT32): XLAType.FLOAT32,
        (XLAType.FLOAT32, XLAType.INT32): XLAType.FLOAT32,
        (XLAType.INT64, XLAType.FLOAT64): XLAType.FLOAT64,
        (XLAType.FLOAT64, XLAType.INT64): XLAType.FLOAT64,
    }

    def __init__(
        self,
        name: str,
        x: Operation,
        y: Operation,
        attributes: ArithmeticAttributes | None = None,
    ) -> None:
        super().__init__(
            OpType.BINARY, name, [x, y], attributes or ArithmeticAttributes()
        )
        self._output_type = self.compute_output_type()

    def _promote_types(self, x_type: XLAType, y_type: XLAType) -> XLAType:
        """Determine output type based on input types."""
        if x_type == y_type:
            return x_type
        key = (x_type, y_type)
        if key not in self.TYPE_PROMOTION:
            raise TypeCheckingError(
                x_type, y_type, self.id, f"Incompatible types for {self.name}"
            )
        return self.TYPE_PROMOTION[key]

    def _broadcast_shapes(self, x_shape: Shape, y_shape: Shape) -> Shape:
        """Compute output shape after broadcasting."""
        # Handle scalar cases.
        if not x_shape:
            return y_shape
        if not y_shape:
            return x_shape

        max_rank = max(len(x_shape), len(y_shape))
        x_padded = (1,) * (max_rank - len(x_shape)) + x_shape
        y_padded = (1,) * (max_rank - len(y_shape)) + y_shape

        result = []
        for x_dim, y_dim in zip(x_padded, y_padded):
            if x_dim == 1:
                result.append(y_dim)
            elif y_dim == 1:
                result.append(x_dim)
            elif x_dim == y_dim:
                result.append(x_dim)
            else:
                raise ShapeError(
                    x_padded,
                    y_padded,
                    self.id,
                    f"Incompatible shapes for broadcasting in {self.name}",
                )
        return tuple(result)

    def compute_output_type(self) -> Tensor:
        x_type = self.inputs[0].output_type
        y_type = self.inputs[1].output_type

        output_dtype = self._promote_types(x_type.dtype, y_type.dtype)
        output_shape = self._broadcast_shapes(x_type.shape, y_type.shape)

        if all(dim != -1 for dim in output_shape) and any(
            dim == 0 for dim in output_shape
        ):
            raise ValidationError(
                f"Zero-size tensor not allowed in {self.name}",
                self.id,
                self.name,
            )

        return Tensor(output_dtype, output_shape)

    def validate(self) -> bool:
        _ = self.compute_output_type()
        return True


@Operation.register("add")
class Add(ArithmeticOp):
    """Element-wise addition with broadcasting."""

    def __init__(
        self,
        x: Operation,
        y: Operation,
        attributes: ArithmeticAttributes | None = None,
    ) -> None:
        super().__init__("add", x, y, attributes)

    def lower(self) -> Operation:
        """Add is already a basic operation."""
        return self


@Operation.register("subtract")
class Subtract(ArithmeticOp):
    """Element-wise subtraction with broadcasting."""

    def __init__(
        self,
        x: Operation,
        y: Operation,
        attributes: ArithmeticAttributes | None = None,
    ) -> None:
        super().__init__("subtract", x, y, attributes)

    def lower(self) -> Operation:
        """Subtract is already a basic operation."""
        return self


@Operation.register("multiply")
class Multiply(ArithmeticOp):
    """Element-wise multiplication with broadcasting."""

    def __init__(
        self,
        x: Operation,
        y: Operation,
        attributes: ArithmeticAttributes | None = None,
    ) -> None:
        super().__init__("multiply", x, y, attributes)

    def lower(self) -> Operation:
        """Multiply is already a basic operation."""
        return self


@Operation.register("divide")
class Divide(ArithmeticOp):
    """Element-wise division with broadcasting."""

    VALID_FLOAT_TYPES: Final[set[XLAType]] = {XLAType.FLOAT32, XLAType.FLOAT64}

    def __init__(
        self,
        x: Operation,
        y: Operation,
        attributes: ArithmeticAttributes | None = None,
    ) -> None:
        super().__init__("divide", x, y, attributes)

    def compute_output_type(self) -> Tensor:
        t = super().compute_output_type()
        if t.dtype not in self.VALID_FLOAT_TYPES:
            raise ValidationError(
                f"Division only supported for types {self.VALID_FLOAT_TYPES}, "
                f"got {t.dtype}",
                self.id,
                self.name,
            )
        return t

    def lower(self) -> Operation:
        """Divide is already a basic operation."""
        return self


def add(
    x: Operation, y: Operation, broadcast_dims: tuple[int, ...] = ()
) -> Operation:
    """Create an addition operation."""
    return Add(x, y, ArithmeticAttributes(broadcast_dims))


def subtract(
    x: Operation, y: Operation, broadcast_dims: tuple[int, ...] = ()
) -> Operation:
    """Create a subtraction operation."""
    return Subtract(x, y, ArithmeticAttributes(broadcast_dims))


def multiply(
    x: Operation, y: Operation, broadcast_dims: tuple[int, ...] = ()
) -> Operation:
    """Create a multiplication operation."""
    return Multiply(x, y, ArithmeticAttributes(broadcast_dims))


def divide(
    x: Operation, y: Operation, broadcast_dims: tuple[int, ...] = ()
) -> Operation:
    """Create a division operation."""
    return Divide(x, y, ArithmeticAttributes(broadcast_dims))
