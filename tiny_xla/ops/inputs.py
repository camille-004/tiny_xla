from __future__ import annotations

import dataclasses
import enum
from typing import Final

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Shape, Tensor, XLAType
from ..utils.errors import ValidationError


class InputType(enum.StrEnum):
    """Types of input operations."""

    PLACEHOLDER = "placeholder"  # For runtime inputs
    PARAMETER = "parameter"  # Trainable parameters
    RESOURCE = "resource"  # Resource variables


@dataclasses.dataclass(frozen=True, slots=True)
class InputAttributes(OpAttributes):
    """Attributes for input operations."""

    shape: Shape
    dtype: XLAType
    input_type: InputType
    name: str
    is_trainable: bool = False
    min_value: float | None = None
    max_value: float | None = None

    def validate(self) -> bool:
        # All dimensions must be >= -1.
        if any(dim < -1 for dim in self.shape):
            return False
        # Only placeholders may have dynamic dimensions (-1).
        if (
            any(dim == -1 for dim in self.shape)
            and self.input_type != InputType.PLACEHOLDER
        ):
            return False
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                return False
        return True


@Operation.register("input")
class Input(Operation):
    """Base class for input operations."""

    MAX_DIM_SIZE: Final[int] = 10**6

    def __init__(
        self,
        shape: Shape,
        dtype: XLAType,
        input_type: InputType,
        name: str,
        *,
        is_trainable: bool = False,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        if not name:
            raise ValueError("Input name must be non-empty")
        super().__init__(
            OpType.ELEMENT_WISE,
            f"input_{name}",
            [],
            InputAttributes(
                shape=shape,
                dtype=dtype,
                input_type=input_type,
                name=name,
                is_trainable=is_trainable,
                min_value=min_value,
                max_value=max_value,
            ),
        )
        self._output_type = self.compute_output_type()
        self.validate()

    def compute_output_type(self) -> Tensor:
        shape = self.attributes.shape
        dtype = self.attributes.dtype
        try:
            return Tensor(dtype, shape)
        except ValueError as e:
            if "Invalid dtype:" in str(e):
                raise e
            raise ValidationError(str(e), self.id) from e

    def validate(self) -> bool:
        if not self.attributes.validate():
            raise ValidationError("Invalid input attributes", self.id)
        for i, dim in enumerate(self.attributes.shape):
            if dim != -1 and (dim < 1 or dim > self.MAX_DIM_SIZE):
                raise ValidationError(
                    f"Invalid dimension size {dim} at index {i}. "
                    f"Must be between 1 and {self.MAX_DIM_SIZE} or -1 for "
                    f"dynamic dimensions.",
                    self.id,
                )
        if (
            any(dim == -1 for dim in self.attributes.shape)
            and self.attributes.input_type != InputType.PLACEHOLDER
        ):
            raise ValidationError(
                "Only placeholders can have dynamic dimensions (-1)", self.id
            )
        if (
            self.attributes.min_value is not None
            and self.attributes.max_value is not None
        ) and (self.attributes.min_value > self.attributes.max_value):
            raise ValidationError(
                "Invalid constraints: min_value > max_value", self.id
            )
        if (
            self.attributes.input_type == InputType.PARAMETER
            and not self.attributes.is_trainable
        ):
            raise ValidationError(
                "Parameters must be marked as trainable.", self.id
            )
        return True

    def lower(self) -> Operation:
        return self


@Operation.register("parameter")
class Parameter(Input):
    """Convenience class for creating trainable parameters."""

    def __init__(
        self,
        shape: Shape,
        dtype: XLAType,
        name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dtype=dtype,
            input_type=InputType.PARAMETER,
            name=name,
            is_trainable=True,
            min_value=min_value,
            max_value=max_value,
        )


@Operation.register("placeholder")
class Placeholder(Input):
    """Convenience class for creating input placeholders."""

    def __init__(
        self,
        shape: Shape,
        dtype: XLAType,
        name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dtype=dtype,
            input_type=InputType.PLACEHOLDER,
            name=name,
            is_trainable=False,
            min_value=min_value,
            max_value=max_value,
        )


@Operation.register("resource")
class Resource(Input):
    """Convenience class for creating resource variables."""

    def __init__(
        self,
        shape: Shape,
        dtype: XLAType,
        name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dtype=dtype,
            input_type=InputType.RESOURCE,
            name=name,
            is_trainable=False,
            min_value=min_value,
            max_value=max_value,
        )
        self.metadata.is_stateless = False


def placeholder(
    shape: Shape,
    dtype: XLAType,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Operation:
    """Create an input placeholder."""
    return Placeholder(shape, dtype, name, min_value, max_value)


def parameter(
    shape: Shape,
    dtype: XLAType,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
):
    """Create a trainable parameter."""
    return Parameter(shape, dtype, name, min_value, max_value)


def resource(
    shape: Shape,
    dtype: XLAType,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
):
    """Create a resource variable."""
    return Resource(shape, dtype, name, min_value, max_value)
