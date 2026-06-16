# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Boolean combinators for discrete Hilbert constraints.

The classes in this module compose existing constraints without changing their
evaluation semantics. They are used by the operator overloads on
``AbstractConstraint`` and can also be constructed directly when explicit
constraint trees are useful.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from neuralqx.utils.struct import field

from .abstract import AbstractDiscreteConstraint


class AndConstraint(AbstractDiscreteConstraint):
    """Logical conjunction of one or more discrete constraints.

    ``AndConstraint`` is the concrete object produced by the ``&`` operator on
    constraints. It keeps the child constraints immutable and evaluates them on
    the same state batch, accepting a state only when every child accepts it.
    This is useful for composing simple predicates without defining a custom
    constraint class.

    Args:
        constraints: Child constraints evaluated from left to right.

    Attributes:
        constraints: Child constraints evaluated from left to right.
    """

    constraints: (
        tuple[AbstractDiscreteConstraint, ...] | Sequence[AbstractDiscreteConstraint]
    ) = field(static=True)
    """Child constraints evaluated from left to right."""

    def __post_init__(self) -> None:
        """Normalizes and validates the child constraint sequence."""
        constraints = tuple(self.constraints)
        if not constraints:
            raise ValueError("AndConstraint needs at least one child constraint.")
        object.__setattr__(self, "constraints", constraints)

    def __hash__(self) -> int:
        """Returns a structural hash for the conjunction."""
        return hash((type(self), self.constraints))

    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates all child constraints and combines them with logical and.

        Args:
            states: State array whose trailing dimension is the Hilbert-space
                site dimension.

        Returns:
            Boolean mask accepted by every child constraint.
        """
        out = self.constraints[0](states)
        for constraint in self.constraints[1:]:
            out = out & constraint(states)
        return out


class OrConstraint(AbstractDiscreteConstraint):
    """Logical disjunction of one or more discrete constraints.

    ``OrConstraint`` is the concrete object produced by the ``|`` operator on
    constraints. It evaluates each child on the same state batch and accepts a
    state when at least one child accepts it. The object does not short-circuit
    at Python runtime because each child returns array masks that must be
    combined with JAX operations.

    Args:
        constraints: Child constraints evaluated from left to right.

    Attributes:
        constraints: Child constraints evaluated from left to right.
    """

    constraints: (
        tuple[AbstractDiscreteConstraint, ...] | Sequence[AbstractDiscreteConstraint]
    ) = field(static=True)
    """Child constraints evaluated from left to right."""

    def __post_init__(self) -> None:
        """Normalizes and validates the child constraint sequence."""
        constraints = tuple(self.constraints)
        if not constraints:
            raise ValueError("OrConstraint needs at least one child constraint.")
        object.__setattr__(self, "constraints", constraints)

    def __hash__(self) -> int:
        """Returns a structural hash for the disjunction."""
        return hash((type(self), self.constraints))

    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates all child constraints and combines them with logical or.

        Args:
            states: State array whose trailing dimension is the Hilbert-space
                site dimension.

        Returns:
            Boolean mask accepted by at least one child constraint.
        """
        out = self.constraints[0](states)
        for constraint in self.constraints[1:]:
            out = out | constraint(states)
        return out


class NotConstraint(AbstractDiscreteConstraint):
    """Logical negation of a single discrete constraint.

    ``NotConstraint`` is the concrete object produced by the unary ``~``
    operator on a constraint. It preserves the child constraint as metadata and
    inverts the boolean mask returned by that child for the supplied state
    batch. This makes complement constraints explicit and hashable.

    Args:
        constraint: Child constraint to negate.

    Attributes:
        constraint: Child constraint to negate.
    """

    constraint: AbstractDiscreteConstraint = field(static=True)
    """Child constraint to negate."""

    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates the child constraint and inverts the boolean mask.

        Args:
            states: State array whose trailing dimension is the Hilbert-space
                site dimension.

        Returns:
            Boolean mask rejected by the child constraint.
        """
        return ~self.constraint(states)

    def __hash__(self) -> int:
        """Returns a structural hash for the negation."""
        return hash((type(self), self.constraint))


__all__ = ["AndConstraint", "NotConstraint", "OrConstraint"]
