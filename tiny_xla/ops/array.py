from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Shape, Tensor, XLAType
from ..utils.errors import ShapeError, ValidationError


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantAttributes(OpAttributes):
    """Attributes for constant operations."""

    value: list[Any]  # Raw value data
    shape: Shape
    dtype: XLAType

    def _flatten(self, x: Any) -> list[Any]:
        if isinstance(x, list):
            result = []
            for item in x:
                result.extend(self._flatten(item))
            return result
        else:
            return [x]

    def validate(self) -> bool:
        flat = self._flatten(self.value)
        expected = 1
        for d in self.shape:
            expected *= d
        if len(flat) != expected:
            raise ValidationError(
                "Constant value size does not match shape", None, "constant"
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "shape": self.shape, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstantAttributes:
        return cls(data["value"], tuple(data["shape"]), XLAType(data["dtype"]))


@dataclasses.dataclass(frozen=True, slots=True)
class ReshapeAttributes(OpAttributes):
    """Attributes for reshape operation."""

    new_shape: Shape

    def validate(self) -> bool:
        return all(dim > 0 for dim in self.new_shape)


@dataclasses.dataclass(frozen=True, slots=True)
class TransposeAttributes(OpAttributes):
    """Attributes for transpose operation."""

    permutation: tuple[int, ...]

    def validate(self) -> bool:
        if not self.permutation:
            return True
        return sorted(self.permutation) == list(range(len(self.permutation)))


@dataclasses.dataclass(frozen=True, slots=True)
class SliceAttributes(OpAttributes):
    """Attributes for slice operation."""

    start_indices: tuple[int, ...]
    end_indices: tuple[int, ...]
    strides: tuple[int, ...] = tuple[int, ...]


@Operation.register("constant")
class Constant(Operation):
    """Represents a constant tensor."""

    def __init__(self, value: list[Any], shape: Shape, dtype: XLAType) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            "constant",
            [],
            ConstantAttributes(value, shape, dtype),
        )
        self.attributes.validate()
        self._output_type = self.compute_output_type()

    def compute_output_type(self) -> Tensor:
        attrs = self.attributes
        return Tensor(attrs.dtype, attrs.shape)

    def validate(self) -> bool:
        _ = self.attributes.validate()
        return True

    def lower(self) -> Operation:
        return self


@Operation.register("reshape")
class Reshape(Operation):
    """Reshape input tensor to new dimensions."""

    def __init__(self, x: Operation, new_shape: Shape) -> None:
        super().__init__(
            OpType.RESHAPE, "reshape", [x], ReshapeAttributes(new_shape)
        )
        self._output_type = self.compute_output_type()

    def compute_output_type(self) -> Tensor:
        input_type = self.inputs[0].output_type
        new_shape = self.attributes.new_shape

        input_size = 1
        for dim in input_type.shape:
            input_size *= dim

        output_size = 1
        for dim in new_shape:
            output_size *= dim

        if input_size != output_size:
            raise ShapeError(
                input_type.shape,
                new_shape,
                self.id,
                "Reshape total size must match",
            )

        return Tensor(input_type.dtype, new_shape)

    def validate(self) -> bool:
        if len(self.inputs) != 1:
            raise ValidationError(
                "Reshape requires exactly one input", self.id
            )
        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        return self


@Operation.register("transpose")
class Transpose(Operation):
    """Transposes input tensor dimensions."""

    def __init__(
        self, x: Operation, permutation: Sequence[int] | None = None
    ) -> None:
        perm = tuple(permutation) if permutation is not None else ()
        super().__init__(
            OpType.RESHAPE, "transpose", [x], TransposeAttributes(perm)
        )
        self._output_type = self.compute_output_type()

    def compute_output_type(self) -> Tensor:
        input_type = self.inputs[0].output_type
        perm = self.attributes.permutation

        if not perm:
            perm = tuple(range(len(input_type.shape)))[::-1]

        if len(perm) != len(input_type.shape):
            raise ShapeError(
                input_type.shape,
                perm,
                self.id,
                "Permutation must match input rank",
            )

        new_shape = tuple(input_type.shape[i] for i in perm)
        return Tensor(input_type.dtype, new_shape)

    def validate(self) -> bool:
        if len(self.inputs) != 1:
            raise ValidationError(
                "Transpose requires exactly one input", self.id
            )
        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        return self


@Operation.register("slice")
class Slice(Operation):
    """Extracts a slice of the input tensor."""

    def __init__(
        self,
        x: Operation,
        start_indices: Sequence[int],
        end_indices: Sequence[int],
        strides: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            OpType.SLICE,
            "slice",
            [x],
            SliceAttributes(
                tuple(start_indices),
                tuple(end_indices),
                tuple(strides)
                if strides is not None
                else tuple([1] * len(start_indices)),
            ),
        )
        self._output_type = self.compute_output_type()
        self.validate()

    def compute_output_type(self) -> Tensor:
        input_type = self.inputs[0].output_type
        attrs = self.attributes

        new_shape = []
        for start, end, stride in zip(
            attrs.start_indices, attrs.end_indices, attrs.strides
        ):
            size = end - start
            if size < 0:
                raise ValidationError(f"Invalid slice size {size}", self.id)
            new_shape.append((size + stride - 1) // stride)

        return Tensor(input_type.dtype, tuple(new_shape))

    def validate(self) -> bool:
        if len(self.inputs) != 1:
            raise ValidationError("Slice requires exactly one input", self.id)

        input_shape = self.inputs[0].output_type.shape
        attrs = self.attributes

        for i, (start, end, dim_size) in enumerate(
            zip(attrs.start_indices, attrs.end_indices, input_shape)
        ):
            if not (0 <= start < end <= dim_size):
                raise ValidationError(
                    f"Slice index out of bounds in dimension {i}", self.id
                )

        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        return self


def constant(value: list[Any], shape: Shape, dtype: XLAType) -> Operation:
    """Create a constant operation."""
    return Constant(value, shape, dtype)


def reshape(x: Operation, new_shape: Shape) -> Operation:
    """Create a reshape operation."""
    return Reshape(x, new_shape)


def transpose(x: Operation, perm: Sequence[int] | None = None) -> Operation:
    """Create a transpose operation."""
    return Transpose(x, perm)


def slice_array(
    x: Operation,
    start_indices: Sequence[int],
    end_indices: Sequence[int],
    strides: Sequence[int] | None = None,
) -> Operation:
    """Create a slice operation."""
    return Slice(x, start_indices, end_indices, strides)
