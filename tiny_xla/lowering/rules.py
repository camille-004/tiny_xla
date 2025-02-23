from __future__ import annotations

from ..core.operation import Operation
from ..ops import (
    Conv2D,
    Pool2D,
    PoolMode,
    im2col,
    matmul,
    reduce_max,
    reshape,
    slice_array,
)
from ..utils.errors import LoweringError
from .lowering_pass import LoweringContext, LoweringRule


class Conv2DLoweringRule(LoweringRule):
    """Lower Conv2D to simpler operations.

    Strategy:
    1. Convert input to patches using im2col
    2. Reshape kernel to 2D matrix
    3. Perform matrix multiplication
    4. Reshape result
    """

    def matches(self, op: Operation) -> bool:
        return isinstance(op, Conv2D)

    def lower(
        self, op: Operation, ctx: LoweringContext
    ) -> list[Operation] | None:
        conv_op = op
        if not isinstance(conv_op, Conv2D):
            return None

        input_op = conv_op.inputs[0]
        kernel_op = conv_op.inputs[1]

        n, h, w, c_in = input_op.output_type.shape
        k_h, k_w, c_in_kernel, c_out = kernel_op.output_type.shape

        if c_in != c_in_kernel:
            raise LoweringError(
                f"Conv2D: Input channels ({c_in}) must match "
                f"output channels ({c_out})",
                conv_op.name,
                conv_op.id,
            )

        padding = conv_op.attributes.padding
        strides = conv_op.attributes.strides
        dilation = conv_op.attributes.dilation_rate

        try:
            if padding == "same":
                out_h = (h + strides[0] - 1) // strides[0]
                out_w = (w + strides[1] - 1) // strides[1]
            else:
                out_h = (h - (k_h - 1) * dilation[0] - 1) // strides[0] + 1
                out_w = (w - (k_w - 1) * dilation[1] - 1) // strides[1] + 1
            # Convert input to patches
            patches = im2col(
                input_op,
                kernel_size=(k_h, k_w),
                strides=strides,
                padding=padding,
                dilation=dilation,
            )

            patches_reshaped = reshape(
                patches, (n * out_h * out_w, k_h * k_w * c_in)
            )
            kernel_reshaped = reshape(kernel_op, (k_h * k_w * c_in, c_out))
            matmul_result = matmul(patches_reshaped, kernel_reshaped)
            result = reshape(matmul_result, (n, out_h, out_w, c_out))

            return [patches, kernel_reshaped, matmul_result, result]

        except Exception as e:
            raise LoweringError(
                f"Failed to lower Conv2D: {e}", conv_op.name, conv_op.id
            )


class MaxPool2DLoweringRule(LoweringRule):
    """Lower MaxPool2D to simpler operations.

    Strategy:
    1. Extract patches using im2col
    2. Compute maximum over patch dimension
    3. Reshape result
    """

    def matches(self, op: Operation) -> bool:
        return (
            isinstance(op, Pool2D) and op.attributes.pool_mode == PoolMode.MAX
        )

    def lower(
        self, op: Operation, ctx: LoweringContext
    ) -> list[Operation] | None:
        pool_op = op
        if not isinstance(pool_op, Pool2D):
            return None

        input_op = pool_op.inputs[0]

        n, h, w, c = input_op.output_type.shape
        p_h, p_w = pool_op.attributes.pool_size
        strides = pool_op.attributes.strides
        padding = pool_op.attributes.padding

        try:
            if padding == "same":
                out_h = (h + strides[0] - 1) // strides[0]
                out_w = (w + strides[1] - 1) // strides[1]
            else:
                out_h = (h - p_h + strides[0]) // strides[0]
                out_w = (w - p_w + strides[1]) // strides[1]

            # Extract patches
            patches = im2col(
                input_op,
                kernel_size=(p_h, p_w),
                strides=strides,
                padding=padding,
            )

            num_patches = out_h * out_w
            patches_reshaped = reshape(patches, (n, num_patches, p_h * p_w, c))
            maxes = reduce_max(patches_reshaped, axes=[2], keep_dims=False)

            result = reshape(maxes, (n, out_h, out_w, c))

            return [patches, reshape, maxes, result]

        except Exception as e:
            raise LoweringError(
                f"Failed to lower MaxPool2D: {e}", pool_op.name, pool_op.id
            )


class MatMulLoweringRule(LoweringRule):
    """Lower matrix multiplication to tiled operations.

    Strategy:
    1. Split matrices into tiles
    2. Perform multiplication on tiles
    3. Accumulate results
    """

    def __init__(self, tile_size: int = 32) -> None:
        self.tile_size = tile_size

    def matches(self, op: Operation) -> bool:
        if op.name != "matmul":
            return False

        shapes = [inp.output_type.shape for inp in op.inputs]
        return (
            len(shapes[0]) == 2
            and len(shapes[1]) == 2
            and shapes[0][1] == shapes[1][0]
        )

    def _split_tiles(
        self,
        matrix: Operation,
        tile_size: tuple[int, int],
        ctx: LoweringContext,
    ) -> list[Operation]:
        """Split matrix into tiles."""
        m, n = matrix.output_type.shape
        tile_m, tile_n = tile_size

        tiles = []
        for i in range(0, m, tile_m):
            for j in range(0, n, tile_n):
                tile = slice_array(
                    matrix,
                    start_indices=[i, j],
                    end_indices=[min(i + tile_m, m), min(j + tile_n, n)],
                )
                tiles.append(tile)

        return tiles

    def lower(
        self, op: Operation, ctx: LoweringContext
    ) -> list[Operation] | None:
        try:
            # TODO: Implement tiled multiplication when we have tile support
            a, b = op.inputs

            if ctx.target == "cpu":
                m, k = a.output_type.shape
                _, n = b.output_type.shape

                if max(m, n, k) <= 32:
                    return [matmul(a, b)]

                return [matmul(a, b)]

            return [matmul(a, b)]

        except Exception as e:
            raise LoweringError(
                f"Failed to lower matrix multiplication: {e}", op.name, op.id
            )


def create_default_lowering_rules() -> list[LoweringRule]:
    return [
        Conv2DLoweringRule(),
        MaxPool2DLoweringRule(),
        MatMulLoweringRule(),
    ]
