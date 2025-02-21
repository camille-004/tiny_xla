import pytest

from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.ops import (
    Add,
    Divide,
    Multiply,
    Subtract,
    add,
    constant,
    divide,
    multiply,
    placeholder,
    subtract,
)
from tiny_xla.utils.errors import (
    ShapeError,
    TypeCheckingError,
    ValidationError,
)


@pytest.fixture
def scalar_ops():
    """Create scalar operands."""
    x = constant([1.0], (), XLAType.FLOAT32)
    y = constant([2.0], (), XLAType.FLOAT32)
    return x, y


@pytest.fixture
def vector_ops():
    """Create vector operands."""
    x = constant([1.0, 2.0], (2,), XLAType.FLOAT32)
    y = constant([3.0, 4.0], (2,), XLAType.FLOAT32)
    return x, y


@pytest.fixture
def matrix_ops():
    """Create matrix operands."""
    x = constant([[1.0, 2.0], [3.0, 4.0]], (2, 2), XLAType.FLOAT32)
    y = constant([[5.0, 6.0], [7.0, 8.0]], (2, 2), XLAType.FLOAT32)
    return x, y


def test_add_scalar(scalar_ops):
    """Test scalar addition."""
    x, y = scalar_ops
    z = add(x, y)

    assert isinstance(z, Add)
    assert z.output_type == Tensor(XLAType.FLOAT32, ())
    assert z.validate()


def test_add_vector(vector_ops):
    """Test vector addition."""
    x, y = vector_ops
    z = add(x, y)

    assert z.output_type == Tensor(XLAType.FLOAT32, (2,))
    assert z.validate()


def test_add_matrix(matrix_ops):
    """Test matrix addition."""
    x, y = matrix_ops
    z = add(x, y)

    assert z.output_type == Tensor(XLAType.FLOAT32, (2, 2))
    assert z.validate()


def test_subtract(scalar_ops):
    """Test subtraction."""
    x, y = scalar_ops
    z = subtract(x, y)

    assert isinstance(z, Subtract)
    assert z.output_type == Tensor(XLAType.FLOAT32, ())
    assert z.validate()


def test_multiply(scalar_ops):
    """Test multiplication."""
    x, y = scalar_ops
    z = multiply(x, y)

    assert isinstance(z, Multiply)
    assert z.output_type == Tensor(XLAType.FLOAT32, ())
    assert z.validate()


def test_divide(scalar_ops):
    """Test division."""
    x, y = scalar_ops
    z = divide(x, y)

    assert isinstance(z, Divide)
    assert z.output_type == Tensor(XLAType.FLOAT32, ())
    assert z.validate()


def test_type_promotion():
    """Test type promotion rules."""
    x = constant([1], (), XLAType.INT32)
    y = constant([2.0], (), XLAType.FLOAT32)
    z = add(x, y)

    # Should promote to float32
    assert z.output_type.dtype == XLAType.FLOAT32


def test_broadcasting():
    """Test shape broadcasting."""
    x = constant([2.0], (), XLAType.FLOAT32)
    y = constant([1.0, 2.0, 3.0], (3,), XLAType.FLOAT32)
    z = add(x, y)

    assert z.output_type.shape == (3,)

    x = constant([1.0, 2.0], (2,), XLAType.FLOAT32)
    y = constant([[1.0, 2.0], [3.0, 4.0]], (2, 2), XLAType.FLOAT32)
    z = add(x, y)

    assert z.output_type.shape == (2, 2)


def test_invalid_shapes():
    """Test invalid shape combinations."""
    x = constant([1.0, 2.0], (2,), XLAType.FLOAT32)
    y = constant([1.0, 2.0, 3.0], (3,), XLAType.FLOAT32)

    with pytest.raises(ShapeError):
        add(x, y)


def test_invalid_types():
    """Test invalid type combinations."""
    x = constant([True], (), XLAType.BOOL)
    y = constant([1.0], (), XLAType.FLOAT32)

    with pytest.raises(TypeCheckingError):
        add(x, y)


def test_divide_validation():
    """Test division-specific validation."""
    x = constant([1], (), XLAType.INT32)
    y = constant([2], (), XLAType.INT32)

    # Integer division not allowed
    with pytest.raises(ValidationError):
        divide(x, y)


def test_dynamic_shapes():
    """Test operations with dynamic shapes."""
    x = placeholder((-1, 2), XLAType.FLOAT32, "x")
    y = constant([[1.0, 2.0]], (1, 2), XLAType.FLOAT32)
    z = add(x, y)

    assert z.output_type.shape[0] == -1
    assert z.output_type.shape[1] == 2


def test_operation_attributes(scalar_ops):
    """Test operation attributes."""
    x, y = scalar_ops
    z = add(x, y)

    assert z.attributes.validate()
    assert not z.attributes.broadcast_dimensions


def test_zero_size_tensors():
    """Test operations with empty tensors."""
    x = constant([], (0,), XLAType.FLOAT32)
    y = constant([], (0,), XLAType.FLOAT32)

    with pytest.raises(ValidationError):
        add(x, y)
