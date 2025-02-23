from __future__ import annotations

import dataclasses
import enum

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Shape, Tensor
from ..utils.errors import ShapeError, ValidationError


class PadMode(enum.StrEnum):
    """Padding modes for convolution and pooling."""

    SAME = "same"  # Output size same as input
    VALID = "valid"  # No padding
    EXPLICIT = "explicit"  # User-specified


class PoolMode(enum.StrEnum):
    """Pooling operation modes."""

    MAX = "max"
    AVG = "average"
    MIN = "min"


@dataclasses.dataclass(frozen=True, slots=True)
class Conv2DAttributes(OpAttributes):
    """Attributes for 2D convolution."""

    kernel_size: tuple[int, int]
    strides: tuple[int, int]
    padding: PadMode
    channels_out: int
    dilation_rate: tuple[int, int] = (1, 1)
    groups: int = 1
    explicit_padding: tuple[int, int, int, int] | None = (
        None  # (top, bottom, left, right)
    )

    def validate(self) -> bool:
        """Validate convolution attributes."""
        if not all(k > 0 for k in self.kernel_size):
            return False
        if not all(s > 0 for s in self.strides):
            return False
        if not all(d > 0 for d in self.dilation_rate):
            return False
        if self.channels_out <= 0:
            return False
        if self.groups <= 0:
            return False
        if self.padding == PadMode.EXPLICIT and self.explicit_padding is None:
            return False
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class Pool2DAttributes(OpAttributes):
    """Attributes for 2D pooling."""

    pool_size: tuple[int, int]
    strides: tuple[int, int]
    padding: PadMode
    pool_mode: PoolMode
    explicit_padding: tuple[int, int, int, int] | None = None

    def validate(self) -> bool:
        """Validate pooling attributes."""
        # Check that each pooling dimension is > 0.
        if not all(p > 0 for p in self.pool_size):
            return False
        if not all(s > 0 for s in self.strides):
            return False
        if self.padding == PadMode.EXPLICIT and self.explicit_padding is None:
            return False
        return True


@Operation.register("conv2d")
class Conv2D(Operation):
    """2D Convolution operation."""

    def __init__(
        self,
        input_op: Operation,
        kernel: Operation,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: PadMode | str,
        channels_out: int,
        *,
        dilation_rate: tuple[int, int] = (1, 1),
        groups: int = 1,
        explicit_padding: tuple[int, int, int, int] | None = None,
    ) -> None:
        if len(input_op.output_type.shape) != 4:
            raise ValueError(
                "Input must be a 4D tensor [batch, height, width, channels]"
            )

        input_channels = input_op.output_type.shape[3]

        if input_channels % groups != 0:
            raise ValueError(
                f"Input channels {input_channels} not divisible by "
                f"groups {groups}"
            )

        channels_per_group = input_channels // groups
        k_h, k_w = kernel_size
        expected_kernel_shape = (k_h, k_w, channels_per_group, channels_out)
        actual_kernel_shape = kernel.output_type.shape

        if actual_kernel_shape != expected_kernel_shape:
            raise ValueError(
                f"Expected kernel shape {expected_kernel_shape} but got "
                f"kernel shape {actual_kernel_shape}"
            )

        super().__init__(
            OpType.CONVOLUTION,
            "conv2d",
            [input_op, kernel],
            Conv2DAttributes(
                kernel_size=kernel_size,
                strides=strides,
                padding=PadMode(padding),
                channels_out=channels_out,
                dilation_rate=dilation_rate,
                groups=groups,
                explicit_padding=explicit_padding,
            ),
        )

    def _compute_output_shape(
        self, input_shape: Shape, kernel_shape: Shape
    ) -> Shape:
        """Compute output shape for convolution."""
        batch, in_h, in_w, in_c = input_shape
        k_h, k_w = self.attributes.kernel_size
        s_h, s_w = self.attributes.strides
        d_h, d_w = self.attributes.dilation_rate

        eff_k_h = k_h + (k_h - 1) * (d_h - 1)
        eff_k_w = k_w + (k_w - 1) * (d_w - 1)

        if self.attributes.padding == PadMode.SAME:
            out_h = (in_h + s_h - 1) // s_h
            out_w = (in_w + s_w - 1) // s_w
        elif self.attributes.padding == PadMode.VALID:
            out_h = (in_h - eff_k_h + s_h) // s_h
            out_w = (in_w - eff_k_w + s_w) // s_h
        else:
            if self.attributes.explicit_padding is None:
                raise ValidationError("Explicit padding required", self.id)
            pad_t, pad_b, pad_l, pad_r = self.attributes.explicit_padding
            out_h = (in_h + pad_t + pad_b - eff_k_h + s_h) // s_h
            out_w = (in_w + pad_l + pad_r - eff_k_w + s_w) // s_w

        return (batch, out_h, out_w, self.attributes.channels_out)

    def compute_output_type(self) -> Tensor:
        input_type = self.inputs[0].output_type
        kernel_type = self.inputs[1].output_type

        if len(input_type.shape) != 4:
            raise ShapeError(
                (None, None, None, None),
                input_type.shape,
                self.id,
                "Conv2D input must be 4D " "[batch, height, width, channels]",
            )
        if len(kernel_type.shape) != 4:
            raise ShapeError(
                (None, None, None, None),
                kernel_type.shape,
                self.id,
                "Conv2D kernel must be 4D "
                "[height, width, in_channels, out_channels]",
            )

        _, _, _, in_c = input_type.shape
        _, _, k_c_in, k_c_out = kernel_type.shape

        if k_c_out != self.attributes.channels_out:
            raise ValidationError(
                f"Kernel output channels {k_c_out} doesn't "
                "match specified channels_out {self.attributes.channels_out}",
                self.id,
            )

        if in_c != k_c_in * self.attributes.groups:
            raise ValidationError(
                f"Input channels {in_c} not compatible with "
                f"kernel input channels {k_c_in} and groups "
                f"{self.attributes.groups}",
                self.id,
            )

        output_shape = self._compute_output_shape(
            input_type.shape, kernel_type.shape
        )
        return Tensor(input_type.dtype, output_shape)

    def validate(self) -> bool:
        if len(self.inputs) != 2:
            raise ValidationError(
                "Conv2D requires input and kernel tensors", self.id
            )
        if not self.attributes.validate():
            raise ValidationError("Invalid convolution attributes", self.id)
        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        """Conv2D is a basic operation."""
        return self


@Operation.register("pool2d")
class Pool2D(Operation):
    """2D Pooling operation."""

    def __init__(
        self,
        input_op: Operation,
        pool_size: tuple[int, int],
        strides: tuple[int, int],
        padding: PadMode | str,
        pool_mode: PoolMode | str,
        explicit_padding: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            f"pool2d_{pool_mode}",
            [input_op],
            Pool2DAttributes(
                pool_size=pool_size,
                strides=strides,
                padding=PadMode(padding),
                pool_mode=PoolMode(pool_mode),
                explicit_padding=explicit_padding,
            ),
        )

    def _compute_output_shape(self, input_shape: Shape) -> Shape:
        """Compute output shape for pooling."""
        batch, in_h, in_w, channels = input_shape
        p_h, p_w = self.attributes.pool_size
        s_h, s_w = self.attributes.strides

        if self.attributes.padding == PadMode.SAME:
            out_h = (in_h + s_h - 1) // s_h
            out_w = (in_w + s_w - 1) // s_w
        elif self.attributes.padding == PadMode.VALID:
            out_h = (in_h - p_h + s_h) // s_h
            out_w = (in_w - p_w + s_w) // s_h
        else:
            if self.attributes.explicit_padding is None:
                raise ValidationError("Explicit padding required", self.id)
            pad_t, pad_b, pad_l, pad_r = self.attributes.explicit_padding
            out_h = (in_h + pad_t + pad_b - p_h + s_h) // s_h
            out_w = (in_w + pad_l + pad_r - p_w + s_w) // s_w

        return (batch, out_h, out_w, channels)

    def compute_output_type(self) -> Tensor:
        """Compute output type for pooling."""
        input_type = self.inputs[0].output_type

        if len(input_type.shape) != 4:
            raise ShapeError(
                (None, None, None, None),
                input_type.shape,
                self.id,
                "Pool2D input must be 4D [batch, height, width, channels]",
            )

        output_shape = self._compute_output_shape(input_type.shape)
        return Tensor(input_type.dtype, output_shape)

    def validate(self) -> bool:
        if len(self.inputs) != 1:
            raise ValidationError("Pool2D requires one input tensor", self.id)
        if not self.attributes.validate():
            raise ValidationError("Invalid pooling attributes", self.id)
        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        """Pool2D is a basic operation."""
        return self


def conv2d(
    input_op: Operation,
    kernel: Operation,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    channels_out: int,
    *,
    dilation_rate: tuple[int, int] = (1, 1),
    groups: int = 1,
    explicit_padding: tuple[int, int, int, int] | None = None,
) -> Operation:
    """Create a 2D convolution operation."""
    return Conv2D(
        input_op,
        kernel,
        kernel_size,
        strides,
        padding,
        channels_out,
        dilation_rate=dilation_rate,
        groups=groups,
        explicit_padding=explicit_padding,
    )


def max_pool2d(
    input_op: Operation,
    pool_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    explicit_padding: tuple[int, int, int, int] | None = None,
) -> Operation:
    """Create a max pooling operation."""
    return Pool2D(
        input_op, pool_size, strides, padding, PoolMode.MAX, explicit_padding
    )


def avg_pool2d(
    input_op: Operation,
    pool_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    explicit_padding: tuple[int, int, int, int] | None = None,
) -> Operation:
    """Create an average pooling operation."""
    return Pool2D(
        input_op, pool_size, strides, padding, PoolMode.AVG, explicit_padding
    )
