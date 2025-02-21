from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias, TypeVar

# Generic operations
T = TypeVar("T")
P = TypeVar("P", bound="Operation")  # noqa:F821

# Core tpes
NodeId: TypeAlias = int
Shape: TypeAlias = tuple[int, ...]
DType: TypeAlias = str


class XLAType(StrEnum):
    """Support XLA data types."""

    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "bool"


@dataclass(frozen=True, slots=True)
class Tensor:
    """Represents the type of a tensor."""

    dtype: XLAType
    shape: Shape

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, XLAType):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if not validate_shape(self.shape):
            raise ValueError(f"Invalid shape: {self.shape}")

    def __str__(self) -> str:
        return f"{self.dtype}{self.shape}"


class XLAPhase(StrEnum):
    """Compiler phases in order of execution."""

    PARSING = "parsing"
    LOWERING = "lowering"
    OPTIMIZATION = "optimization"
    CODE_GEN = "code_gen"


# Constants
MAX_TENSOR_RANK: Final[int] = 8
MAX_TENSOR_SIZE: Final[int] = 2**31 - 1  # Maximum size of a tensor dimension


# Common shapes
SCALAR_SHAPE: Final[Shape] = ()
VECTOR_1D: Final[Shape] = (1,)


def validate_shape(shape: Shape) -> bool:
    if len(shape) > MAX_TENSOR_RANK:
        return False
    for dim in shape:
        if dim < -1 or dim > MAX_TENSOR_SIZE:
            return False
    return True


def get_shape_size(shape: Shape) -> int:
    """Calculate the total number of elements in a shape."""
    if not shape:
        return 1
    size = 1
    for dim in shape:
        size *= dim
    return size
