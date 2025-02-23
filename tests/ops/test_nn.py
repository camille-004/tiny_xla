import pytest

from tiny_xla.core.types import XLAType
from tiny_xla.ops import (
    Conv2D,
    PadMode,
    PoolMode,
    avg_pool2d,
    constant,
    conv2d,
    max_pool2d,
    placeholder,
)
from tiny_xla.utils.errors import ValidationError


def create_constant_value(shape: tuple[int, ...], value: float = 1.0) -> list:
    if len(shape) == 0:
        return value
    return [create_constant_value(shape[1:], value) for _ in range(shape[0])]


@pytest.fixture
def sample_input():
    """Create a sample 4D input tensor."""
    return placeholder(
        shape=(1, 32, 32, 3), dtype=XLAType.FLOAT32, name="input"
    )


@pytest.fixture
def sample_kernel():
    """Create a sample convolution kernel."""
    kernel_shape = (3, 3, 3, 16)
    kernel_value = create_constant_value(kernel_shape, 1.0)
    return constant(
        value=kernel_value, shape=kernel_shape, dtype=XLAType.FLOAT32
    )


def test_conv2d_basic(sample_input, sample_kernel):
    """Test basic convolution operation."""
    conv = conv2d(
        input_op=sample_input,
        kernel=sample_kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        channels_out=16,
    )

    assert isinstance(conv, Conv2D)
    assert conv.validate()

    out_type = conv.output_type
    assert out_type.dtype == XLAType.FLOAT32
    assert out_type.shape == (1, 32, 32, 16)


def test_conv2d_valid_padding(sample_input, sample_kernel):
    """Test convolution with 'valid' padding."""
    conv = conv2d(
        input_op=sample_input,
        kernel=sample_kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        channels_out=16,
    )

    assert conv.output_type.shape == (1, 30, 30, 16)


def test_conv2d_strided(sample_input, sample_kernel):
    """Test strided convolution."""
    conv = conv2d(
        input_op=sample_input,
        kernel=sample_kernel,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        channels_out=16,
    )

    assert conv.output_type.shape == (1, 16, 16, 16)


def test_conv2d_dilation(sample_input, sample_kernel):
    """Test dilated convolution."""
    conv = conv2d(
        input_op=sample_input,
        kernel=sample_kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        channels_out=16,
        dilation_rate=(2, 2),
    )

    assert conv.validate()
    assert conv.output_type.shape == (1, 32, 32, 16)


def test_conv2d_groups(sample_input):
    """Test grouped convolution."""
    input_op = placeholder((1, 32, 32, 32), XLAType.FLOAT32, "input")
    kernel_shape = (3, 3, 4, 64)
    kernel_value = create_constant_value(kernel_shape, 1.0)

    kernel = constant(
        value=kernel_value, shape=kernel_shape, dtype=XLAType.FLOAT32
    )

    conv = conv2d(
        input_op=input_op,
        kernel=kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        channels_out=64,
        groups=8,
    )

    assert conv.validate()
    assert conv.output_type.shape == (1, 32, 32, 64)


def test_max_pool_basic(sample_input):
    """Test basic max pooling."""
    pool = max_pool2d(
        input_op=sample_input,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
    )

    assert pool.validate()
    assert pool.output_type.shape == (1, 16, 16, 3)
    assert pool.attributes.pool_mode == PoolMode.MAX


def test_avg_pool_basic(sample_input):
    """Test basic average pooling."""
    pool = avg_pool2d(
        input_op=sample_input,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
    )

    assert pool.validate()
    assert pool.output_type.shape == (1, 16, 16, 3)
    assert pool.attributes.pool_mode == PoolMode.AVG


def test_pool_same_padding(sample_input):
    """Test pooling with 'same' padding."""
    pool = max_pool2d(
        input_op=sample_input, pool_size=(3, 3), strides=(2, 2), padding="same"
    )

    assert pool.output_type.shape == (1, 16, 16, 3)


def test_pool_explicit_padding(sample_input):
    """Test pooling with explicit padding."""
    pool = max_pool2d(
        input_op=sample_input,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="explicit",
        explicit_padding=(1, 1, 1, 1),
    )

    assert pool.validate()
    assert pool.output_type.shape == (1, 17, 17, 3)


def test_invalid_conv_shapes():
    input_op = placeholder((1, 32, 32, 3), XLAType.FLOAT32, "input")
    kernel_shape = (3, 3, 4, 16)
    kernel_value = create_constant_value(kernel_shape, 1.0)
    invalid_kernel = constant(
        value=kernel_value, shape=kernel_shape, dtype=XLAType.FLOAT32
    )

    with pytest.raises(ValueError) as exc_info:
        conv2d(
            input_op=input_op,
            kernel=invalid_kernel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            channels_out=16,
        )

    assert (
        "Expected kernel shape (3, 3, 3, 16) but got "
        "kernel shape (3, 3, 4, 16)" in str(exc_info.value)
    )


def test_invalid_pool_parameters(sample_input):
    pool1 = max_pool2d(
        input_op=sample_input,
        pool_size=(0, 2),
        strides=(2, 2),
        padding="valid",
    )
    with pytest.raises(ValidationError):
        pool1.validate()

    pool2 = max_pool2d(
        input_op=sample_input,
        pool_size=(2, 2),
        strides=(0, 2),
        padding="valid",
    )
    with pytest.raises(ValidationError):
        pool2.validate()


def test_type_validation():
    """Test type validation for conv and pool ops."""
    input_op = placeholder((1, 32, 32, 3), XLAType.INT32, "input")
    kernel_shape = (3, 3, 3, 16)
    kernel_value = create_constant_value(kernel_shape, 1)
    kernel = constant(
        value=kernel_value, shape=kernel_shape, dtype=XLAType.INT32
    )

    conv = conv2d(
        input_op=input_op,
        kernel=kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        channels_out=16,
    )
    assert conv.validate()


def test_conv_attributes():
    """Test convolution attributes."""
    conv = conv2d(
        input_op=placeholder((1, 32, 32, 3), XLAType.FLOAT32, "input"),
        kernel=constant(
            value=create_constant_value((3, 3, 3, 16), 1.0),
            shape=(3, 3, 3, 16),
            dtype=XLAType.FLOAT32,
        ),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        channels_out=16,
        dilation_rate=(2, 2),
        groups=1,
    )

    attrs = conv.attributes
    assert attrs.kernel_size == (3, 3)
    assert attrs.strides == (2, 2)
    assert attrs.padding == PadMode.SAME
    assert attrs.channels_out == 16
    assert attrs.dilation_rate == (2, 2)
    assert attrs.groups == 1
