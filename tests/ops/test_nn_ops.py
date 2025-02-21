import pytest

from tiny_xla.core.types import XLAType
from tiny_xla.ops import (
    ActivationType,
    Phase,
    batch_norm,
    constant,
    dropout,
    elu,
    placeholder,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from tiny_xla.utils.errors import ValidationError


@pytest.fixture
def batch_input():
    """Create sample batch input."""
    return placeholder(
        shape=(32, 224, 224, 64), dtype=XLAType.FLOAT32, name="batch_input"
    )


@pytest.fixture
def bn_parameters():
    """Create sample batch norm parameters."""
    channels = 64
    scale = constant([1.0] * channels, (channels,), XLAType.FLOAT32)
    offset = constant([0.0] * channels, (channels,), XLAType.FLOAT32)
    mean = constant([0.0] * channels, (channels,), XLAType.FLOAT32)
    variance = constant([1.0] * channels, (channels,), XLAType.FLOAT32)
    return scale, offset, mean, variance


def test_batch_norm_training(batch_input, bn_parameters):
    """Test batch normalization n training mode."""
    scale, offset, mean, variance = bn_parameters

    bn = batch_norm(
        input_op=batch_input,
        scale=scale,
        offset=offset,
        mean=mean,
        variance=variance,
        phase=Phase.TRAIN,
    )

    assert bn.validate()
    assert bn.output_type == batch_input.output_type
    assert bn.attributes.phase == Phase.TRAIN


def test_dropout_training(batch_input):
    """Test dropout in training mode."""
    drop = dropout(input_op=batch_input, rate=0.5, phase=Phase.TRAIN)

    assert drop.validate()
    assert drop.output_type == batch_input.output_type
    assert drop.attributes.phase == Phase.TRAIN
    assert drop.attributes.rate == 0.5


def test_dropout_inference(batch_input):
    """Test dropout in inference mode."""
    drop = dropout(input_op=batch_input, rate=0.5, phase=Phase.INFERENCE)

    assert drop.validate()
    assert drop.output_type == batch_input.output_type
    assert drop.attributes.phase == Phase.INFERENCE


def test_dropout_invalid_rate(batch_input):
    """Test dropout with invalid rate values."""
    with pytest.raises(ValidationError):
        dropout(batch_input, rate=-0.1)

    with pytest.raises(ValidationError):
        dropout(batch_input, rate=1.0)


def test_activation_relu(batch_input):
    """Test ReLU activation."""
    act = relu(batch_input)

    assert act.validate()
    assert act.output_type == batch_input.output_type
    assert act.attributes.activation_type == ActivationType.RELU


def test_activation_sigmoid(batch_input):
    """Test sigmoid activation."""
    act = sigmoid(batch_input)

    assert act.validate()
    assert act.output_type == batch_input.output_type
    assert act.attributes.activation_type == ActivationType.SIGMOID


def test_activation_tanh(batch_input):
    """Test sigmoid activation."""
    act = tanh(batch_input)

    assert act.validate()
    assert act.output_type == batch_input.output_type
    assert act.attributes.activation_type == ActivationType.TANH


def test_activation_elu(batch_input):
    """Test ELU activation."""
    act = elu(batch_input)

    assert act.validate()
    assert act.output_type == batch_input.output_type
    assert act.attributes.activation_type == ActivationType.ELU
    assert act.attributes.alpha == 1.0


def test_activation_softmax(batch_input):
    """Test softmax activation."""
    act = softmax(batch_input)

    assert act.validate()
    assert act.output_type == batch_input.output_type
    assert act.attributes.activation_type == ActivationType.SOFTMAX
    assert act.attributes.axis == -1


def test_invalid_softmax_axis():
    """Test softmax with invalid axis."""
    input_op = placeholder((32, 10), XLAType.FLOAT32, "input")
    with pytest.raises(ValidationError):
        softmax(input_op, axis=2)


def test_batch_norm_momentum():
    """Test batch norm momentum parameter."""
    input_op = placeholder((32, 10), XLAType.FLOAT32, "input")
    scale = constant([1.0] * 10, (10,), XLAType.FLOAT32)
    offset = constant([0.0] * 10, (10,), XLAType.FLOAT32)
    mean = constant([0.0] * 10, (10,), XLAType.FLOAT32)
    variance = constant([1.0] * 10, (10,), XLAType.FLOAT32)

    # Valid momentum
    bn = batch_norm(
        input_op=input_op,
        scale=scale,
        offset=offset,
        mean=mean,
        variance=variance,
        momentum=0.99,
    )
    assert bn.validate()

    # Invalid momentum
    with pytest.raises(ValidationError):
        batch_norm(
            input_op=input_op,
            scale=scale,
            offset=offset,
            mean=mean,
            variance=variance,
            momentum=1.5,
        )


def test_dropout_deterministic():
    """Test dropout with fixed seed."""
    input_op = placeholder((32, 10), XLAType.FLOAT32, "input")

    drop = dropout(input_op=input_op, rate=0.5, phase=Phase.TRAIN, seed=42)

    assert drop.validate()
    assert drop.attributes.seed == 42
