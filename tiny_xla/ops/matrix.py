from __future__ import annotations

import dataclasses

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Tensor
from ..utils.errors import ShapeError, ValidationError


@dataclasses.dataclass(frozen=True, slots=True)
class MatMulAttributes(OpAttributes):
    """Attributes for matrix multiplication."""

    transpose_a: bool = False
    transpose_b: bool = False

    def validate(self) -> bool:
        return True


@Operation.register("matmul")
class MatMul(Operation):
    """Matrix multiplication operation."""

    def __init__(
        self,
        a: Operation,
        b: Operation,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            "matmul",
            [a, b],
            MatMulAttributes(transpose_a, transpose_b),
        )

    def compute_output_type(self) -> Tensor:
        a_type = self.inputs[0].output_type
        b_type = self.inputs[1].output_type

        if len(a_type.shape) != 2 or len(b_type.shape) != 2:
            raise ShapeError(
                a_type.shape,
                b_type.shape,
                self.id,
                "MatMul input must be 2D [M, K]",
            )

        a_shape = (
            a_type.shape[::-1] if self.attributes.transpose_a else a_type.shape
        )
        b_shape = (
            b_type.shape[::-1] if self.attributes.transpose_b else b_type.shape
        )

        if a_shape[-1] != b_shape[0]:
            raise ShapeError(
                a_shape,
                b_shape,
                self.id,
                "Incompatible inner dimensions for matmul",
            )

        return Tensor(a_type.dtype, (a_shape[0], b_shape[1]))

    def validate(self) -> bool:
        if len(self.inputs) != 2:
            raise ValidationError(
                "MatMul requires exactly two inputs", self.id
            )

        _ = self.compute_output_type()  # Validate shapes
        return True

    def lower(self) -> Operation:
        """Matmul is already a basic operation."""
        return self


def matmul(
    a: Operation,
    b: Operation,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> Operation:
    """Create a matmul operation."""
    return MatMul(a, b, transpose_a, transpose_b)
