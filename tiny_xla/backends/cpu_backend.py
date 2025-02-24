from __future__ import annotations

import dataclasses

from ..core.types import XLAType
from ..utils.logger import XLALogger

logger = XLALogger.get_logger(__name__)


@dataclasses.dataclass
class CPUKernelConfig:
    """Configuration for CPU kernel compilation."""

    vector_size: int = 4
    unroll_factor: int = 4
    cache_line_size: int = 64
    memory_alignment: int = 32
    use_openmp: bool = True


class MemoryLayout:
    """Represents memory layout information."""

    def __init__(self, shape: tuple[int, ...], dtype: XLAType) -> None:
        self.shape = shape
        self.dtype = dtype
        self.strides = self._compute_strides(shape)

    def _compute_strides(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute strides for contiguous layout."""
        strides = [1]

        for dim in reversed(shape[:-1]):
            strides.append(dim * strides[-1])
        return tuple(reversed(strides))

    def get_offset(self, indices: tuple[int, ...]) -> int:
        """Get linear memory offset for given indices."""
        if len(indices) != len(self.shape):
            raise ValueError("Incorrect number of indices")
        return sum(i * s for i, s in zip(indices, self.strides))


class CPUKernel:
    """Represents a generated CPU kernel."""

    def __init__(
        self,
        name: str,
        source: str,
        input_layouts: list[MemoryLayout],
        output_layout: MemoryLayout,
    ) -> None:
        self.name = name
        self.source = source
        self.input_layouts = input_layouts
        self.output_layout = output_layout
