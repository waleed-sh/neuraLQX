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


"""Abstract interfaces for Hilbert-space constraints.

Constraints are JAX-compatible predicates over flat state arrays. They are
used by discrete Hilbert spaces to validate states, restrict sampling, and
report exact constrained dimensions when a specialized implementation can do so
cheaply.
"""

from __future__ import annotations

import abc
from typing import Any

import jax.numpy as jnp

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta


class AbstractConstraint(Struct, metaclass=StructABCMeta):
    """Base protocol for predicates over Hilbert-space states.

    A constraint receives state arrays in the flat representation owned by a
    Hilbert space and returns a boolean mask over the leading batch axes. The
    trailing axis always represents sites, edges, channels, or another concrete
    layout chosen by the space.

    Subclasses must implement :meth:`__call__` with JAX-compatible array
    operations so the predicate can run inside sampling, validation, and
    proposal code. Subclasses may override :meth:`dimension` when the number of
    accepted states is available without exhaustive enumeration.
    """

    @abc.abstractmethod
    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates whether each state satisfies the constraint.

        Args:
            states: State array whose trailing dimension is the Hilbert-space
                site dimension.

        Returns:
            Boolean array with shape ``states.shape[:-1]``.
        """

    def dimension(self, hilbert: Any) -> int | None:
        """Returns the exact constrained dimension when it is cheaply known.

        Args:
            hilbert: Hilbert-space instance whose dimension is being queried.

        Returns:
            Exact number of valid states, or ``None`` when the constraint does
            not provide a cheap closed-form answer.
        """
        return None

    def __and__(self, other: AbstractConstraint) -> AbstractConstraint:
        """Builds a logical conjunction with another constraint.

        Args:
            other: Constraint to combine with this one.

        Returns:
            Constraint that accepts states accepted by both operands.
        """
        from .combinators import AndConstraint

        return AndConstraint((self, other))

    def __or__(self, other: AbstractConstraint) -> AbstractConstraint:
        """Builds a logical disjunction with another constraint.

        Args:
            other: Constraint to combine with this one.

        Returns:
            Constraint that accepts states accepted by either operand.
        """
        from .combinators import OrConstraint

        return OrConstraint((self, other))

    def __invert__(self) -> AbstractConstraint:
        """Builds a logical negation of this constraint.

        Returns:
            Constraint that accepts states rejected by this predicate.
        """
        from .combinators import NotConstraint

        return NotConstraint(self)


class AbstractDiscreteConstraint(AbstractConstraint):
    """Constraint specialization used by discrete Hilbert spaces.

    Discrete constraints operate after local-domain checks have established
    that each site value belongs to its finite local space. They therefore
    express global, cross-site, or physics-specific restrictions such as gauge
    fixing, linear equalities, or user-supplied predicates.

    Subclasses inherit the abstract :meth:`__call__` contract from
    :class:`AbstractConstraint`. They may also override
    :meth:`validate_hilbert` to reject incompatible spaces early, for example
    when the constraint expects a particular state size, local range, or graph
    layout.
    """

    def validate_hilbert(self, hilbert: Any) -> None:
        """Validates compatibility with a discrete Hilbert space.

        Args:
            hilbert: Discrete Hilbert-space instance about to use this
                constraint.

        Raises:
            TypeError: If the constraint cannot operate on ``hilbert``.
            ValueError: If ``hilbert`` has incompatible shape or domain data.
        """


__all__ = ["AbstractConstraint", "AbstractDiscreteConstraint"]
