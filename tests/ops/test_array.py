import pytest

from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.ops import (
    Constant,
    Reshape,
    Slice,
    Transpose,
    constant,
    reshape,
    slice_array,
    transpose,
)
from tiny_xla.utils.errors import ShapeError, ValidationError


@pytest.fixture
def sample_matrix():
    """Create a sample 2x3 matrix."""
    return constant(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3), XLAType.FLOAT32
    )


@pytest.fixture
def sample_tensor():
    """Create a sample 2x2x2 tensor."""
    return constant(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        (2, 2, 2),
        XLAType.FLOAT32,
    )


def test_constant_creation():
    """Test constant operation creation."""
    c = constant([1.0, 2.0], (2,), XLAType.FLOAT32)

    assert isinstance(c, Constant)
    assert c.output_type == Tensor(XLAType.FLOAT32, (2,))
    assert c.validate()


def test_reshape_basic(sample_matrix):
    """Test basic reshape operation."""
    # Reshape 2x3 to 6
    r = reshape(sample_matrix, (6,))

    assert isinstance(r, Reshape)
    assert r.output_type == Tensor(XLAType.FLOAT32, (6,))
    assert r.validate()


def test_reshape_multiple_dims(sample_matrix):
    """Test reshaping to multiple dimensions."""
    # Reshape 2x3 to 3x2
    r = reshape(sample_matrix, (3, 2))

    assert r.output_type == Tensor(XLAType.FLOAT32, (3, 2))
    assert r.validate()


def test_reshape_invalid_size(sample_matrix):
    """Test invalid reshape operations."""
    # Total size mismatch
    with pytest.raises(ShapeError):
        reshape(sample_matrix, (5,))


def test_transpose_basic(sample_matrix):
    """Test basic transpose operation."""
    t = transpose(sample_matrix)  # Default: reverse dimensions

    assert isinstance(t, Transpose)
    assert t.output_type == Tensor(XLAType.FLOAT32, (3, 2))
    assert t.validate()


def test_transpose_with_perm(sample_tensor):
    """Test transpose with custom permutation."""
    # Permute dimensions 0->2, 1->0, 2->1
    t = transpose(sample_tensor, [1, 2, 0])

    assert t.output_type == Tensor(XLAType.FLOAT32, (2, 2, 2))
    assert t.validate()


def test_transpose_invalid_perm(sample_matrix):
    """Test invalid transpose permutations."""
    # Invalid permutation length
    with pytest.raises(ShapeError):
        transpose(sample_matrix, [0, 1, 2])


def test_slice_basic(sample_matrix):
    """Test basic slice operation."""
    # Slice first row
    s = slice_array(sample_matrix, [0, 0], [1, 3])

    assert isinstance(s, Slice)
    assert s.output_type == Tensor(XLAType.FLOAT32, (1, 3))
    assert s.validate()


def test_slice_with_strides(sample_matrix):
    """Test slice operation with strides."""
    # Slice with stride 2
    s = slice_array(sample_matrix, [0, 0], [2, 3], [1, 2])

    assert s.output_type == Tensor(XLAType.FLOAT32, (2, 2))
    assert s.validate()


def test_slice_invalid_indices(sample_matrix):
    """Test invalid slice operations."""
    # Out of bounds
    with pytest.raises(ValidationError):
        slice_array(sample_matrix, [0, 0], [3, 3])


def test_shape_inference():
    """Test shape inference for array operations."""
    x = constant([[1.0]], (1, 1), XLAType.FLOAT32)

    # Test various transformations
    r = reshape(x, (1,))
    t = transpose(r)
    s = slice_array(t, [0], [1])

    assert r.output_type.shape == (1,)
    assert t.output_type.shape == (1,)
    assert s.output_type.shape == (1,)


def test_constant_validation():
    """Test constant operation validation."""
    # Invalid shape
    with pytest.raises(ValidationError):
        constant([1.0], (2,), XLAType.FLOAT32)


def test_reshape_zero_dims():
    """Test reshape with zero dimensions."""
    x = constant([1.0], (1,), XLAType.FLOAT32)

    # Reshape to scalar
    r = reshape(x, ())
    assert r.output_type.shape == ()


def test_transpose_single_dim():
    """Test transpose of single dimension tensor."""
    x = constant([1.0, 2.0], (2,), XLAType.FLOAT32)
    t = transpose(x)

    assert t.output_type.shape == (2,)


def test_slice_full_tensor(sample_matrix):
    """Test slicing entire tensor."""
    s = slice_array(sample_matrix, [0, 0], [2, 3], [1, 1])

    assert s.output_type == sample_matrix.output_type


def test_array_op_attributes():
    """Test array operation attributes."""
    x = constant([1.0], (1,), XLAType.FLOAT32)

    r = reshape(x, (1,))
    assert r.attributes.new_shape == (1,)

    t = transpose(x)
    assert not t.attributes.permutation  # Default permutation

    s = slice_array(x, [0], [1])
    assert s.attributes.strides == (1,)
