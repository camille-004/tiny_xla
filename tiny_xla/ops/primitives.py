from __future__ import annotations

import dataclasses
from typing import Sequence

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Tensor
from ..utils.errors import ShapeError, ValidationError


@dataclasses.dataclass(frozen=True, slots=True)
class Im2ColAttributes(OpAttributes):
    """Attributes for im2col operation."""

    kernel_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str
    dilation: tuple[int, int] = (1, 1)

    def validate(self) -> bool:
        return (
            all(k > 0 for k in self.kernel_size)
            and all(s > 0 for s in self.strides)
            and all(d > 0 for d in self.dilation)
            and self.padding in {"same", "valid"}
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReduceAttributes(OpAttributes):
    """Attributes for reduction operations."""

    axes: tuple[int, ...]
    keep_dims: bool = False

    def validate(self) -> bool:
        return all(isinstance(axis, int) for axis in self.axes)


@Operation.register("im2col")
class Im2Col(Operation):
    """Convert image patches to columns for efficient convolution.

    Takes a 4D input tensor [N, H, W, C] and extracts pathces to form
    a 3D tensor [N, P, K*K*C] where P is the number of patches and K
    is the kernel size.
    """

    def __init__(
        self,
        input_op: Operation,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: str,
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__(
            OpType.RESHAPE,
            "im2col",
            [input_op],
            Im2ColAttributes(kernel_size, strides, padding, dilation),
        )

    def compute_output_type(self) -> Tensor:
        input_type = self.inputs[0].output_type
        N, H, W, C = input_type.shape
        K_h, K_w = self.attributes.kernel_size
        s_h, s_w = self.attributes.strides
        d_h, d_w = self.attributes.dilation

        if self.attributes.padding == "same":
            out_h = (H + s_h - 1) // s_h
            out_w = (W + s_w - 1) // s_w
        else:
            out_h = (H - (K_h - 1) * d_h + s_h - 1) // s_h
            out_w = (W - (K_w - 1) * d_w + s_w - 1) // s_w

        patches = out_h * out_w
        return Tensor(input_type.dtype, (N, patches, K_h * K_w * C))

    def validate(self) -> bool:
        if len(self.inputs) != 1:
            raise ValidationError("Im2Col requires exactly one input", self.id)

        input_type = self.inputs[0].output_type
        if len(input_type.shape) != 4:
            raise ShapeError(
                input_type.shape,
                (None, None, None, None),
                self.id,
                "Im2Col input must be 4D [N, H, W, C]",
            )

        return True

    def lower(self) -> Operation:
        """Im2Col is already a primitive operation."""
        return self


@Operation.register("reduce")
class Reduce(Operation):
    """Base class for reduction operations.

    Performs reduction along specified axes. Subclasses implement
    specific reduction operations (max, sum, etc).
    """

    def __init__(
        self, input_op: Operation, axes: Sequence[int], keep_dims: bool = False
    ) -> None:
        super().__init__(
            OpType.REDUCTION,
            "reduce",
            [input_op],
            ReduceAttributes(tuple(axes), keep_dims),
        )

    def compute_output_type(self) -> Tensor:
        input_type = self.inputs[0].output_type
        input_shape = list(input_type.shape)
        ndim = len(input_shape)

        axes = [
            axis if axis >= 0 else ndim + axis for axis in self.attributes.axes
        ]

        if any(axis >= ndim or axis < -ndim for axis in axes):
            raise ValidationError(
                f"Invalid reduction axes {axes} for shape in {input_shape}",
                self.id,
            )

        if self.attributes.keep_dims:
            # Replace reduced dimensions with 1
            for axis in sorted(axes):
                input_shape[axis] = 1
        else:
            # Reomve reduced dimensions
            for axis in sorted(axes, reverse=True):
                del input_shape[axis]

        return Tensor(input_type.dtype, tuple(input_shape))

    def validate(self) -> bool:
        if len(self.input) != 1:
            raise ValidationError(
                "Reduce operation requires exactly one input", self.id
            )

        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        """Reduce is already a primitive operation."""
        return self


@Operation.register("reduce_max")
class ReduceMax(Reduce):
    """Maximum reduction along specified axes."""

    def __init__(
        self, input_op: Operation, axes: Sequence[int], keep_dims: bool = False
    ) -> None:
        super().__init__(input_op, axes, keep_dims)
        self._name = "reduce_max"


@Operation.register("reduce_sum")
class ReduceSum(Reduce):
    """Maximum reduction along specified axes."""

    def __init__(
        self, input_op: Operation, axes: Sequence[int], keep_dims: bool = False
    ) -> None:
        super().__init__(input_op, axes, keep_dims)
        self._name = "reduce_sum"


def im2col(
    input_op: Operation,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    dilation: tuple[int, int] = (1, 1),
) -> Operation:
    """Create an im2col operation."""
    return Im2Col(input_op, kernel_size, strides, padding, dilation)


def reduce_max(
    input_op: Operation, axes: Sequence[int], keep_dims: bool = False
) -> Operation:
    """Create a reduce_max operation."""
    return ReduceMax(input_op, axes, keep_dims)


def reduce_sum(
    input_op: Operation, axes: Sequence[int], keep_dims: bool = False
) -> Operation:
    """Create a reduce_sum operation."""
    return ReduceSum(input_op, axes, keep_dims)
