from __future__ import annotations

import dataclasses
import enum

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Tensor
from ..utils.errors import ShapeError, ValidationError


class Phase(enum.StrEnum):
    """Operation phase for training vs inference."""

    TRAIN = "train"
    INFERENCE = "inference"


class ActivationType(enum.StrEnum):
    """Supported activation functions."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    ELU = "elu"
    SELU = "selu"
    SOFTMAX = "softmax"
    GELU = "gelu"


@dataclasses.dataclass(frozen=True, slots=True)
class BatchNormAttributes(OpAttributes):
    """Attributes for batch normalization."""

    momentum: float = 0.99
    epsilon: float = 1e-5
    phase: Phase = Phase.TRAIN
    axis: int = -1

    def validate(self) -> bool:
        """Validate batch norm attribute."""
        return 0.0 <= self.momentum <= 1.0 and self.epsilon > 0


@dataclasses.dataclass(frozen=True, slots=True)
class DropoutAttributes(OpAttributes):
    """Attributes for dropout."""

    rate: float
    phase: Phase = Phase.TRAIN
    seed: int | None = None

    def validate(self) -> bool:
        """Validate dropout attributes."""
        return 0.0 <= self.rate < 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class ActivationAttributes(OpAttributes):
    """Attributes for activaton functions."""

    activation_type: ActivationType
    alpha: float | None = None  # For ELU/SELU
    axis: int = -1  # For softmax

    def validate(self) -> bool:
        return True


@Operation.register("batch_norm")
class BatchNorm(Operation):
    """Batch Normalization operation."""

    def __init__(
        self,
        input_op: Operation,
        scale: Operation,
        offset: Operation,
        mean: Operation,
        variance: Operation,
        *,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        phase: Phase | str = Phase.TRAIN,
        axis: int = -1,
    ) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            "batch_norm",
            [input_op, scale, offset, mean, variance],
            BatchNormAttributes(
                momentum=momentum,
                epsilon=epsilon,
                phase=Phase(phase),
                axis=axis,
            ),
        )
        self.validate()

    def compute_output_type(self) -> Tensor:
        """Output type matches input type."""
        input_type = self.inputs[0].output_type
        axis = (
            self.attributes.axis
            if self.attributes.axis >= 0
            else len(input_type.shape) + self.attributes.axis
        )
        stats_shape = tuple(
            input_type.shape[i] if i == axis else 1
            for i in range(len(input_type.shape))
        )
        for i, param in enumerate(self.inputs[1:], 1):
            param_shape = param.output_type.shape
            if param_shape != stats_shape:
                if not (
                    len(param_shape) == 1 and param_shape[0] == stats_shape[-1]
                ):
                    raise ShapeError(
                        stats_shape,
                        param_shape,
                        self.id,
                        f"BatchNorm parameter {i} has wrong shape",
                    )
        return input_type

    def validate(self) -> bool:
        if len(self.inputs) != 5:
            raise ValidationError(
                "BatchNorm requires input, scale, offset, mean, and variance",
                self.id,
            )
        if not self.attributes.validate():
            raise ValidationError("Invalid BatchNorm attributes", self.id)
        input_type = self.inputs[0].output_type
        if self.attributes.axis < -len(
            input_type.shape
        ) or self.attributes.axis >= len(input_type.shape):
            raise ValidationError(
                f"Invalid axis {self.attributes.axis} for shape "
                f"{input_type.shape}",
                self.id,
            )
        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        """BatchNorm is a basic operation."""
        return self


@Operation.register("dropout")
class Dropout(Operation):
    """Dropout operation.

    Randomly zeroes elements with probability rate during training.
    Does nothing during inference.
    """

    def __init__(
        self,
        input_op: Operation,
        rate: float,
        *,
        phase: Phase | str = Phase.TRAIN,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            "dropout",
            [input_op],
            DropoutAttributes(rate=rate, phase=Phase(phase), seed=seed),
        )
        self.validate()

    def compute_output_type(self) -> Tensor:
        """Output type matches input type."""
        return self.inputs[0].output_type

    def validate(self) -> bool:
        """Validate dropout operation."""
        if len(self.inputs) != 1:
            raise ValidationError("Dropout requires one input tensor", self.id)
        if not self.attributes.validate():
            raise ValidationError("Invalid dropout attributes", self.id)
        return True

    def lower(self) -> Operation:
        """Dropout is a basic operation."""
        return self


@Operation.register("activation")
class Activation(Operation):
    """Activation function operation."""

    def __init__(
        self,
        input_op: Operation,
        activation_type: ActivationType | str,
        *,
        alpha: float | None = None,
        axis: int = -1,
    ) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            f"activation_{activation_type}",
            [input_op],
            ActivationAttributes(
                activation_type=ActivationType(activation_type),
                alpha=alpha,
                axis=axis,
            ),
        )
        self.validate()

    def compute_output_type(self) -> Tensor:
        """Output type matches input type."""
        input_type = self.inputs[0].output_type

        if self.attributes.activation_type == ActivationType.SOFTMAX:
            axis = self.attributes.axis
            if axis < -len(input_type.shape) or axis >= len(input_type.shape):
                raise ValidationError(
                    f"Invalid softmax axis {axis} for shape "
                    f"{input_type.shape}",
                    self.id,
                )

        return input_type

    def validate(self) -> bool:
        """Validate activation operation."""
        if len(self.inputs) != 1:
            raise ValidationError(
                "Activation requires one input tensor", self.id
            )

        if self.attributes.activation_type in {
            ActivationType.ELU,
            ActivationType.SELU,
        }:
            if self.attributes.alpha is None:
                raise ValidationError(
                    f"{self.attributes.activation_type} requires "
                    f"alpha parameter",
                    self.id,
                )

        _ = self.compute_output_type()
        return True

    def lower(self) -> Operation:
        """Activation is a basic operation."""
        return self


def batch_norm(
    input_op: Operation,
    scale: Operation,
    offset: Operation,
    mean: Operation,
    variance: Operation,
    *,
    momentum: float = 0.99,
    epsilon: float = 1e-5,
    phase: str = "train",
    axis: int = -1,
) -> Operation:
    """Create a batch normalization operation."""
    return BatchNorm(
        input_op,
        scale,
        offset,
        mean,
        variance,
        momentum=momentum,
        epsilon=epsilon,
        phase=phase,
        axis=axis,
    )


def dropout(
    input_op: Operation,
    rate: float,
    phase: str = "train",
    seed: int | None = None,
) -> Operation:
    """Create a dropout operation."""
    return Dropout(input_op, rate, phase=phase, seed=seed)


def relu(input_op: Operation) -> Operation:
    """Create a ReLU activation."""
    return Activation(input_op, ActivationType.RELU)


def sigmoid(input_op: Operation) -> Operation:
    """Create a sigmoid activation."""
    return Activation(input_op, ActivationType.SIGMOID)


def tanh(input_op: Operation) -> Operation:
    """Create a tanh activation."""
    return Activation(input_op, ActivationType.TANH)


def elu(input_op: Operation, alpha: float = 1.0) -> Operation:
    """Create an ELU activation."""
    return Activation(input_op, ActivationType.ELU, alpha=alpha)


def softmax(input_op: Operation, axis: int = -1) -> Operation:
    """Create a softmax activation."""
    return Activation(input_op, ActivationType.SOFTMAX, axis=axis)
