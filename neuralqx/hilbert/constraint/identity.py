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


"""Identity constraints for discrete Hilbert spaces.

The identity constraint accepts every state that has already passed the local
domain checks of its Hilbert space. It is used as the default unconstrained
predicate.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .abstract import AbstractDiscreteConstraint


class IdentityConstraint(AbstractDiscreteConstraint):
    """Constraint that leaves a discrete Hilbert space unconstrained.

    The identity constraint is the default predicate for spaces whose only
    restrictions are their local domains. It assumes the owning Hilbert space
    has already checked that each site value is present in the corresponding
    local space, then returns ``True`` for every row in the batch.

    Because it accepts the full rectangular product basis, its dimension is the
    Hilbert space's unconstrained product dimension. The class carries no
    fields and all instances are interchangeable.
    """

    def __call__(self, states: Any) -> jnp.ndarray:
        """Returns a true mask over the batch dimensions of ``states``.

        Args:
            states: State array whose trailing dimension is the Hilbert-space
                site dimension.

        Returns:
            Boolean array filled with ``True`` and shaped like the state batch.
        """
        arr = jnp.asarray(states)
        return jnp.ones(arr.shape[:-1], dtype=jnp.bool_)

    def dimension(self, hilbert: Any) -> int | None:
        """Returns the unconstrained dimension of the Hilbert space.

        Args:
            hilbert: Discrete Hilbert-space instance exposing
                ``unconstrained_dimension``.

        Returns:
            Exact product dimension before additional constraints. Because this
            constraint accepts every locally valid state, that product is also
            the constrained dimension.
        """
        return hilbert.unconstrained_dimension

    def __hash__(self) -> int:
        """Returns a type-level hash for the stateless identity constraint."""
        return hash(type(self))


NoConstraint = IdentityConstraint


__all__ = ["IdentityConstraint", "NoConstraint"]
