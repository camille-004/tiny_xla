from __future__ import annotations

import dataclasses

from ..core.operation import OpAttributes, Operation, OpType
from ..core.types import Tensor, XLAType
from ..utils.errors import TypeCheckingError, ValidationError


@dataclasses.dataclass(frozen=True, slots=True)
class ConditionalAttributes(OpAttributes):
    """Attributes for conditional operations."""

    then_graph: list[Operation]  # Operations for true graph
    else_graph: list[Operation]  # Operations for false branch

    def validate(self) -> bool:
        return bool(self.then_graph and self.else_graph)


@dataclasses.dataclass(frozen=True, slots=True)
class WhileAttributes(OpAttributes):
    """Attributes for while loop operations."""

    condition_graph: list[Operation]  # Operations for condition
    body_graph: list[Operation]  # Operations for loop body
    max_iter: int = 1000

    def validate(self) -> bool:
        if not (self.condition_graph and self.body_graph):
            return False
        return self.max_iter > 0


@Operation.register("conditional")
class Conditional(Operation):
    """Conditional operation (if/else).

    Takes a boolean predicate and executes either the
    thengraph or else_graph based on the predicted value.

    All inputs must be available to both branches.
    Both branches must produce outputs of the same type.
    """

    def __init__(
        self,
        predicate: Operation,
        then_graph: list[Operation],
        else_graph: list[Operation],
        inputs: list[Operation],
    ) -> None:
        # Predicate must be first input
        all_inputs = [predicate] + inputs
        super().__init__(
            OpType.ELEMENT_WISE,
            "conditional",
            all_inputs,
            ConditionalAttributes(then_graph, else_graph),
        )

    def compute_output_type(self) -> Tensor:
        """Both branches must produce same output type."""
        if not self.inputs:
            raise ValidationError("Conditional requires inputs", self.id)

        pred_type = self.inputs[0].output_type
        if pred_type.dtype != XLAType.BOOL or pred_type.shape != ():
            raise TypeCheckingError(
                Tensor(XLAType.BOOL, ()),
                pred_type,
                self.id,
                "Conditional predicate must be scalar boolean",
            )

        then_type = self.attributes.then_graph[-1].output_type
        else_type = self.attributes.else_graph[-1].output_type

        if then_type != else_type:
            raise TypeCheckingError(
                then_type,
                else_type,
                self.id,
                "Conditional branches must produce same type",
            )

        return then_type

    def validate(self) -> bool:
        """Validate the conditional operation."""
        if len(self.inputs) < 1:
            raise ValidationError(
                "Conditional requires predicate input", self.id
            )

        pred_type = self.inputs[0].output_type
        if pred_type.dtype != XLAType.BOOL or pred_type.shape != ():
            raise ValidationError("Predicate must be scalar boolean", self.id)

        attrs = self.attributes
        if not attrs.then_graph or not attrs.else_graph:
            raise ValidationError(
                "Both branches must contain operations", self.id
            )

        then_type = attrs.then_graph[-1].output_type
        else_type = attrs.else_graph[-1].output_type
        if then_type != else_type:
            raise TypeCheckingError(
                f"Branch output types don't match: {then_type} vs {else_type}",
                self.id,
            )

        return True

    def lower(self) -> Operation:
        """Control flow operations don't need lowering."""
        return self


@Operation.register("while")
class While(Operation):
    """While loop operation.

    Repeatedly executes body_graph while condition_graph returns true.
    Loop carries forward a set of loop variables that are updated each
    iteration.
    """

    def __init__(
        self,
        condition_graph: list[Operation],
        body_graph: list[Operation],
        initial_values: list[Operation],
        max_iter: int = 1000,
    ) -> None:
        super().__init__(
            OpType.ELEMENT_WISE,
            "while",
            initial_values,
            WhileAttributes(condition_graph, body_graph, max_iter),
        )
        self.validate()

    def compute_output_type(self) -> Tensor:
        """Output type matches the body graph output type."""
        body_graph = self.attributes.body_graph
        if not body_graph:
            raise ValidationError("While body cannot be empty", self.id)
        return body_graph[-1].output_type

    def validate(self) -> bool:
        attrs = self.attributes
        if not attrs.condition_graph:
            raise ValidationError("While condition cannot be empty", self.id)
        cond_type = attrs.condition_graph[-1].output_type
        if cond_type.dtype != XLAType.BOOL or cond_type.shape != ():
            raise ValidationError(
                "While condition must produce scalar boolean", self.id
            )
        if not attrs.body_graph:
            raise ValidationError("While body cannot be empty", self.id)
        if attrs.max_iter <= 0:
            raise ValidationError(
                f"Invalid max iterations: {attrs.max_iter}", self.id
            )
        init_types = [var.output_type for var in self.inputs]
        body_types = [op.output_type for op in attrs.body_graph]
        if len(init_types) != len(body_types):
            raise ValidationError(
                "Mismatch in number of loop variables", self.id
            )
        for t1, t2 in zip(init_types, body_types):
            if t1 != t2:
                raise TypeCheckingError(
                    t1, t2, self.id, "Loop variable type mismatch"
                )
        return True

    def lower(self) -> Operation:
        """Control flow operations don't need lowering."""
        return self


def conditional(
    predicate: Operation,
    then_graph: list[Operation],
    else_graph: list[Operation],
    inputs: list[Operation],
) -> Operation:
    return Conditional(predicate, then_graph, else_graph, inputs)


def while_loop(
    condition_graph: list[Operation],
    body_graph: list[Operation],
    initial_values: list[Operation],
    max_iter: int = 1000,
) -> Operation:
    """Create a while loop iteration."""
    return While(condition_graph, body_graph, initial_values, max_iter)
