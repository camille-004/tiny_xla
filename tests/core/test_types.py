import pytest

from tiny_xla.core.types import Tensor, XLAType, get_shape_size, validate_shape


def test_xla_shape_values():
    """Test XLA type enumeration."""
    assert XLAType.INT32.value == "int32"
    assert XLAType.FLOAT32.value == "float32"
    assert XLAType.BOOL.value == "bool"


def test_tensor_type_creation():
    """Test tensor type creation and properties."""
    t = Tensor(XLAType.FLOAT32, (2, 3))
    assert t.dtype == XLAType.FLOAT32
    assert t.shape == (2, 3)


def test_tensor_type_equality():
    """Test tensor type equality comparison."""
    t1 = Tensor(XLAType.FLOAT32, (2, 3))
    t2 = Tensor(XLAType.FLOAT32, (2, 3))
    t3 = Tensor(XLAType.INT32, (2, 3))
    t4 = Tensor(XLAType.FLOAT32, (3, 2))

    assert t1 == t2
    assert t1 != t3
    assert t1 != t4


def test_tensor_type_string():
    """Test tensor type string representation."""
    t = Tensor(XLAType.FLOAT32, (2, 3))
    assert str(t) == "float32(2, 3)"


def test_validate_shape():
    """Test shape validation."""
    assert validate_shape(())  # Scalar
    assert validate_shape((1,))  # Vector
    assert validate_shape((2, 3))  # Matrix
    assert validate_shape((-1, 3))

    assert not validate_shape((2**31, 3))


def test_get_shape_size():
    """Test shape size calculation."""
    assert get_shape_size(()) == 1  # Scalar
    assert get_shape_size((5,)) == 5  # Vector
    assert get_shape_size((2, 3)) == 6  # Matrix
    assert get_shape_size((2, 3, 4)) == 24  # 3D tensor


def test_tensor_type_validation():
    """Test tensor type validation cases."""
    Tensor(XLAType.FLOAT32, ())
    Tensor(XLAType.INT32, (1,))
    Tensor(XLAType.BOOL, (2, 3))

    with pytest.raises(ValueError):
        Tensor("invalid_type", (2, 3))


def test_shape_edge_cases():
    """Test edge cases for shape handling."""
    assert validate_shape(())
    assert get_shape_size(()) == 1

    assert validate_shape((1,))
    assert get_shape_size((1,)) == 1
