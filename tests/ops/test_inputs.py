import pytest

from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.ops import (
    InputType,
    Parameter,
    Placeholder,
    Resource,
    parameter,
    placeholder,
    resource,
)
from tiny_xla.utils.errors import ValidationError


def test_placeholder_creation():
    """Test basic placeholder creation."""
    p = placeholder((-1, 224, 224, 3), XLAType.FLOAT32, "input_images")

    assert isinstance(p, Placeholder)
    assert p.output_type == Tensor(XLAType.FLOAT32, (-1, 224, 224, 3))
    assert p.validate()
    assert p.attributes.input_type == InputType.PLACEHOLDER
    assert not p.attributes.is_trainable


def test_placeholder_with_constraints():
    """Test placeholder with value constraints."""
    p = placeholder(
        shape=(224, 224),
        dtype=XLAType.FLOAT32,
        name="normalized_image",
        min_value=0.0,
        max_value=1.0,
    )

    assert p.validate()
    assert p.attributes.min_value == 0.0
    assert p.attributes.max_value == 1.0


def test_parameter_creation():
    """Test parameter creation."""
    p = parameter((256, 128), XLAType.FLOAT32, "weights")

    assert isinstance(p, Parameter)
    assert p.output_type == Tensor(XLAType.FLOAT32, (256, 128))
    assert p.validate()
    assert p.attributes.input_type == InputType.PARAMETER
    assert p.attributes.is_trainable


def test_resource_creation():
    """Test resource creation."""
    r = resource((), XLAType.INT32, "counter")

    assert isinstance(r, Resource)
    assert r.output_type == Tensor(XLAType.INT32, ())
    assert r.validate()
    assert r.attributes.input_type == InputType.RESOURCE
    assert not r.attributes.is_trainable


def test_dynamic_shapes():
    """Test placeholder with dynamic shapes."""
    # Single dynamic dimension
    p1 = placeholder((-1, 32), XLAType.FLOAT32, "batch_input")
    assert p1.validate()

    # Multiple dynamic dimensions
    p2 = placeholder((-1, -1, 3), XLAType.FLOAT32, "images")
    assert p2.validate()


def test_invalid_dynamic_shapes():
    """Test invalid dynamic shape specifications."""
    # Dynamic shapes not allowed for parameters
    with pytest.raises(ValidationError):
        parameter((-1, 128), XLAType.FLOAT32, "invalid_weights")

    # Dynamic shapes not allowed for resources
    with pytest.raises(ValidationError):
        resource((-1,), XLAType.INT32, "invalid_counter")


def test_invalid_shapes():
    """Test invalid shape specifications."""
    # Negative non-dynamic dimension
    with pytest.raises(ValidationError):
        placeholder((-2, 32), XLAType.FLOAT32, "invalid")

    # Zero dimension
    with pytest.raises(ValidationError):
        parameter((0, 32), XLAType.FLOAT32, "invalid")


def test_invalid_constraints():
    """Test invalid value constraints."""
    # Min > Max
    with pytest.raises(ValidationError):
        placeholder(
            (32,), XLAType.FLOAT32, "invalid", min_value=1.0, max_value=0.0
        )


def test_scalar_inputs():
    """Test scalar input creation."""
    # Scalar placeholder
    p = placeholder((), XLAType.FLOAT32, "scalar_input")
    assert p.output_type.shape == ()

    # Scalar parameter
    param = parameter((), XLAType.FLOAT32, "scalar_param")
    assert param.output_type.shape == ()

    # Scalar resource
    res = resource((), XLAType.INT32, "scalar_resource")
    assert res.output_type.shape == ()


def test_input_names():
    """Test input naming."""
    p = placeholder((32,), XLAType.FLOAT32, "test_input")
    assert p.attributes.name == "test_input"
    assert "test_input" in str(p)


def test_input_type_validation():
    """Test input type validation."""
    # Valid types
    placeholder((32,), XLAType.FLOAT32, "float_input")
    placeholder((32,), XLAType.INT32, "int_input")
    placeholder((32,), XLAType.BOOL, "bool_input")

    # Invalid type
    with pytest.raises(ValueError):
        placeholder((32,), "invalid_type", "invalid_input")


def test_input_metadata():
    """Test input operation metadata."""
    p = parameter((32,), XLAType.FLOAT32, "weight")

    # Parameters should be stateless
    assert p.metadata.is_stateless

    r = resource((32,), XLAType.FLOAT32, "variable")
    # Resources should not be stateless
    assert not r.metadata.is_stateless


def test_input_lowering():
    """Test input operation lowering."""
    p = placeholder((32,), XLAType.FLOAT32, "input")
    # Input operations should return themselves on lowering
    assert p.lower() is p


def test_input_validation():
    """Test comprehensive input validation."""
    # Test required name
    with pytest.raises(ValueError):
        placeholder((32,), XLAType.FLOAT32, "")

    # Test shape validation
    with pytest.raises(ValidationError):
        placeholder((10**9,), XLAType.FLOAT32, "too_large")

    # Test parameter validation
    param = parameter((32,), XLAType.FLOAT32, "param")
    assert param.attributes.is_trainable
