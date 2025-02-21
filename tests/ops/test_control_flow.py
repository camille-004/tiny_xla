import pytest

from tiny_xla.core.types import XLAType
from tiny_xla.ops import (
    add,
    conditional,
    constant,
    placeholder,
    while_loop,
)
from tiny_xla.utils.errors import TypeCheckingError, ValidationError


def test_conditional_type_mismatch():
    """Test conditional with mismatched branch types."""
    cond = placeholder((), XLAType.BOOL, "condition")
    x = constant(1.0, (), XLAType.FLOAT32)
    y = constant(1, (), XLAType.INT32)
    with pytest.raises(TypeCheckingError):
        op = conditional(
            predicate=cond, then_graph=[x], else_graph=[y], inputs=[]
        )
        op.validate()


def test_conditional_invalid_predicate():
    """Test conditional with invalid predicate type."""
    pred = constant(1.0, (), XLAType.FLOAT32)
    x = constant(1.0, (), XLAType.FLOAT32)
    with pytest.raises(ValidationError):
        op = conditional(
            predicate=pred, then_graph=[x], else_graph=[x], inputs=[]
        )
        op.validate()


def test_while_loop_basic():
    """Test basic while loop operation."""
    i = placeholder((), XLAType.INT32, "counter")

    def condition_graph():
        # Return scalar boolean
        return [constant(True, (), XLAType.BOOL)]

    def body_graph():
        one = constant(1, (), XLAType.INT32)
        return [add(i, one)]

    loop = while_loop(
        condition_graph=condition_graph(),
        body_graph=body_graph(),
        initial_values=[i],
    )
    assert loop.validate()


def test_while_loop_multiple_vars():
    """Test while loop with multiple loop variables."""
    # Loop variables (both int32)
    i = placeholder((), XLAType.INT32, "counter")
    acc = placeholder(
        (), XLAType.INT32, "accumulator"
    )  # changed to int32 so types match

    def condition_graph():
        return [constant(True, (), XLAType.BOOL)]

    def body_graph():
        one = constant(1, (), XLAType.INT32)
        new_i = add(i, one)
        new_acc = add(acc, one)  # now both are int32
        return [new_i, new_acc]

    loop = while_loop(
        condition_graph=condition_graph(),
        body_graph=body_graph(),
        initial_values=[i, acc],
    )
    assert loop.validate()
    assert len(loop.inputs) == 2


def test_while_loop_invalid_condition():
    """Test while loop with invalid condition type."""
    i = placeholder((), XLAType.INT32, "counter")

    def condition_graph():
        return [i]  # Returns INT32 instead of BOOL

    def body_graph():
        return [add(i, constant(1, (), XLAType.INT32))]

    with pytest.raises(ValidationError):
        while_loop(
            condition_graph=condition_graph(),
            body_graph=body_graph(),
            initial_values=[i],
        )


def test_while_loop_type_mismatch():
    """Test while loop with type mismatch between iterations."""
    i = placeholder((), XLAType.INT32, "counter")

    def condition_graph():
        return [constant(True, (), XLAType.BOOL)]

    # Body returns float instead of int32
    def body_graph():
        return [constant(1.0, (), XLAType.FLOAT32)]

    with pytest.raises(TypeCheckingError):
        while_loop(
            condition_graph=condition_graph(),
            body_graph=body_graph(),
            initial_values=[i],
        )


def test_max_iter():
    """Test while loop max iterations parameter."""
    i = placeholder((), XLAType.INT32, "counter")

    def condition_graph():
        return [constant(True, (), XLAType.BOOL)]

    def body_graph():
        return [add(i, constant(1, (), XLAType.INT32))]

    loop = while_loop(
        condition_graph=condition_graph(),
        body_graph=body_graph(),
        initial_values=[i],
        max_iter=1000,
    )
    assert loop.validate()
    with pytest.raises(ValidationError):
        while_loop(
            condition_graph=condition_graph(),
            body_graph=body_graph(),
            initial_values=[i],
            max_iter=0,
        )
