import pytest

from tiny_xla.core.graph import Graph
from tiny_xla.core.types import XLAType
from tiny_xla.lowering.lowering_pass import LoweringContext
from tiny_xla.lowering.rules import (
    Conv2DLoweringRule,
    MatMulLoweringRule,
    MaxPool2DLoweringRule,
    create_default_lowering_rules,
)
from tiny_xla.ops import (
    Im2Col,
    MatMul,
    ReduceMax,
    conv2d,
    matmul,
    max_pool2d,
    parameter,
    placeholder,
)


@pytest.fixture
def conv_inputs():
    """Create sample inputs for convolution."""
    images = placeholder((1, 32, 32, 3), XLAType.FLOAT32, "images")
    kernel = parameter((3, 3, 3, 16), XLAType.FLOAT32, "kernel")
    return images, kernel


@pytest.fixture
def pool_input():
    """Create sample input for pooling."""
    return placeholder((1, 32, 32, 64), XLAType.FLOAT32, "input")


@pytest.fixture
def matmul_inputs():
    """Create sample inputs for matrix multiplication."""
    a = placeholder((128, 64), XLAType.FLOAT32, "a")
    b = placeholder((64, 32), XLAType.FLOAT32, "b")
    return a, b


@pytest.fixture
def lowering_context():
    """Create lowering context."""
    return LoweringContext(
        target="cpu", graph=Graph(), replacements={}, new_ops=[]
    )


def test_conv2d_lowering_rule(conv_inputs, lowering_context):
    images, kernel = conv_inputs

    conv = conv2d(
        images,
        kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        channels_out=16,
    )

    rule = Conv2DLoweringRule()
    assert rule.matches(conv)

    ops_lowered = rule.lower(conv, lowering_context)
    assert ops_lowered is not None
    assert len(ops_lowered) > 0

    op_types = [type(op) for op in ops_lowered]  # noqa:F841
    # im2col, reshape, and multiply operations
    assert any(isinstance(op, Im2Col) for op in ops_lowered)
    assert any(isinstance(op, MatMul) for op in ops_lowered)


def test_conv2_different_padding(conv_inputs, lowering_context):
    """Test Conv2D lowering with different padding modes."""
    images, kernel = conv_inputs

    conv = conv2d(
        images,
        kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        channels_out=16,
    )

    rule = Conv2DLoweringRule()
    ops_lowered = rule.lower(conv, lowering_context)

    final_op = ops_lowered[-1]
    expected_shape = (1, 30, 30, 16)
    assert final_op.output_type.shape == expected_shape


def test_maxpool_lowering_rule(pool_input, lowering_context):
    """Test MaxPool2D lowering rule."""
    pool = max_pool2d(
        pool_input,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
    )

    rule = MaxPool2DLoweringRule()
    assert rule.matches(pool)

    ops_lowered = rule.lower(pool, lowering_context)
    assert ops_lowered is not None
    assert len(ops_lowered) > 0

    assert any(isinstance(op, Im2Col) for op in ops_lowered)
    assert any(isinstance(op, ReduceMax) for op in ops_lowered)

    final_op = ops_lowered[-1]
    assert final_op.output_type.shape == pool.output_type.shape
    assert final_op.output_type.dtype == pool.output_type.dtype


def test_matmul_lowering_rule(matmul_inputs, lowering_context):
    """Test MatMul lowering rule."""
    a, b = matmul_inputs

    mat_mul = matmul(a, b)

    rule = MatMulLoweringRule()
    assert rule.matches(mat_mul)

    ops_lowered = rule.lower(mat_mul, lowering_context)
    assert ops_lowered is not None

    final_op = ops_lowered[-1]
    expected_shape = (128, 32)
    assert final_op.output_type.shape == expected_shape


def test_lowering_shape_validation(conv_inputs, lowering_context):
    """Test shape validation during lowering."""
    images, _ = conv_inputs
    bad_kernel = parameter((3, 3, 4, 16), XLAType.FLOAT32, "bad_kernel")

    with pytest.raises(Exception):
        conv2d(
            images,
            bad_kernel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            channels_out=16,
        )


def test_lowering_rule_registration():
    """Test creation of default lowering rules."""
    rules = create_default_lowering_rules()

    rule_types = {type(rule) for rule in rules}
    assert Conv2DLoweringRule in rule_types
    assert MaxPool2DLoweringRule in rule_types
    assert MatMulLoweringRule in rule_types


def test_conv2d_strided(conv_inputs, lowering_context):
    """Test Conv2D lowering with strided convolution."""
    images, kernel = conv_inputs

    conv = conv2d(
        images,
        kernel,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        channels_out=16,
    )

    rule = Conv2DLoweringRule()
    ops_lowered = rule.lower(conv, lowering_context)

    final_op = ops_lowered[-1]
    expected_shape = (1, 16, 16, 16)
    assert final_op.output_type.shape == expected_shape


def test_maxpool_different_sizes(pool_input, lowering_context):
    """Test MaxPool2D lowering with different pool sizes."""
    pool = max_pool2d(
        pool_input,
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
    )

    rule = MaxPool2DLoweringRule()
    ops_lowered = rule.lower(pool, lowering_context)

    assert any(
        isinstance(op, Im2Col) and op.attributes.kernel_size == (3, 3)
        for op in ops_lowered
    )


def test_matmul_small_matrices(lowering_context):
    """Test matrix multiplication with small matrices."""
    a = placeholder((16, 16), XLAType.FLOAT32, "a")
    b = placeholder((16, 16), XLAType.FLOAT32, "b")

    mat_mul = matmul(a, b)

    rule = MatMulLoweringRule()
    ops_lowered = rule.lower(mat_mul, lowering_context)

    assert len(ops_lowered) == 1
    assert isinstance(ops_lowered[0], MatMul)
