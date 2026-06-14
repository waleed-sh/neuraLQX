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


"""Abstract discrete Hilbert-space interface.

Discrete Hilbert spaces use flat state arrays whose trailing dimension indexes
sites. The class centralizes local-index conversion, validity checks, sampling,
proposal dispatch, and unconstrained basis ranking for concrete graph-aware
implementations.
"""

from __future__ import annotations

import abc
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.constraint import IdentityConstraint
from neuralqx.hilbert.proposal.moves import AbstractProposalMove
from neuralqx.hilbert.utils import INDEX_LIMIT
from neuralqx.hilbert.utils import product

from .abstract import AbstractHilbertSpace


class DiscreteHilbertSpace(AbstractHilbertSpace):
    """Abstract base class for finite flat-array Hilbert spaces.

    Discrete spaces represent states as arrays whose final axis has length
    :attr:`size`. Each site has a finite local domain and can therefore be
    converted between physical values and compact local indices. The base class
    provides common validity checks, random sampling dispatch, proposal
    dispatch, basis ranking, unranking, and full-basis materialization for
    spaces that are small enough and unconstrained enough to support it.

    Subclasses must provide local-domain metadata through :attr:`local_sizes`
    and :attr:`dtype`, implement conversion in
    :meth:`states_to_local_indices` and :meth:`local_indices_to_states`, and
    implement :meth:`local_states_valid`. Concrete spaces are also expected to
    expose :attr:`size` and :attr:`constraint`, either directly or through
    mixins, so the inherited methods can validate and route operations.
    """

    @property
    @abc.abstractmethod
    def local_sizes(self) -> tuple[int, ...]:
        """Local dimension at every flat site."""

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Common dtype used for state values."""

    @abc.abstractmethod
    def states_to_local_indices(self, states: Any) -> jax.Array:
        """Maps quantum-number values to site-local integer indices.

        Args:
            states: State array whose trailing dimension equals ``size``.

        Returns:
            Integer array with the same shape as ``states``.
        """

    @abc.abstractmethod
    def local_indices_to_states(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array:
        """Maps site-local integer indices to quantum-number values.

        Args:
            indices: Integer index array whose trailing dimension equals
                ``size``.
            dtype: Optional dtype for the returned state values.

        Returns:
            State-value array with the same shape as ``indices``.
        """

    @abc.abstractmethod
    def local_states_valid(self, states: Any) -> jax.Array:
        """Checks only local-domain validity.

        Args:
            states: State array whose trailing dimension equals ``size``.

        Returns:
            Boolean mask over the batch dimensions. The result should ignore
            global constraints and report only whether each flat site value is
            present in that site's finite local space.
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of local dimensions at every flat site."""
        return self.local_sizes

    @property
    def constrained(self) -> bool:
        """Whether this space has a non-identity constraint."""
        return not isinstance(self.constraint, IdentityConstraint)

    @property
    def unconstrained_dimension(self) -> int:
        """Product dimension before applying non-local constraints."""
        return product(self.local_sizes)

    @property
    def dimension(self) -> int | None:
        """Exact constrained dimension when available from the constraint."""
        return self.constraint.dimension(self)

    @property
    def n_states(self) -> int:
        """Exact number of states.

        Raises:
            RuntimeError: If the constrained dimension is not cheaply known.
        """
        dimension = self.dimension
        if dimension is None:
            raise RuntimeError(
                "The constrained Hilbert-space dimension is not cheaply known."
            )
        return dimension

    @property
    def is_finite(self) -> bool:
        """Whether the represented basis is finite."""
        return True

    @property
    def is_indexable(self) -> bool:
        """Whether generic JAX int32 basis ranking is supported."""
        return not self.constrained and self.unconstrained_dimension <= INDEX_LIMIT

    def is_valid(self, states: Any) -> jax.Array:
        """Checks local-domain validity and global constraints.

        Args:
            states: State array whose trailing dimension equals ``size``.

        Returns:
            Boolean mask over the batch dimensions. A state is valid only when
            every site value belongs to its local domain and the space's
            constraint accepts the full row.
        """
        return self.local_states_valid(states) & self.constraint(states)

    def random_state(
        self,
        key: jax.Array,
        size: int | tuple[int, ...] | None = None,
        *,
        dtype: Any | None = None,
        max_trials: int = 1024,
    ) -> jax.Array:
        """Samples random states from this Hilbert space.

        Args:
            key: JAX pseudo-random key.
            size: Optional leading batch shape for generated states.
            dtype: Optional dtype for generated state values.
            max_trials: Maximum rejection-sampling attempts for generic
                constrained spaces.

        Returns:
            Random state array with trailing dimension ``size``.
        """
        from neuralqx.hilbert.random import random_state

        return random_state(self, key, size, dtype=dtype, max_trials=max_trials)

    def propose(
        self, move: AbstractProposalMove, key: jax.Array, states: Any
    ) -> jax.Array:
        """Applies a proposal move to existing states.

        Args:
            move: Proposal move descriptor.
            key: JAX pseudo-random key.
            states: State array to update.

        Returns:
            Proposed state array with the same shape as ``states``.
        """
        from neuralqx.hilbert.proposal import propose

        return propose(self, move, key, states)

    def states_to_numbers(self, states: Any, *, validate: bool = False) -> jax.Array:
        """Ranks unconstrained states using JAX int32 arithmetic.

        Args:
            states: State array to rank.
            validate: Whether to check membership before ranking.

        Returns:
            Integer basis labels in rightmost-fast order.
        """
        from neuralqx.hilbert.index import states_to_numbers

        return states_to_numbers(self, states, validate=validate)

    def numbers_to_states(self, numbers: Any) -> jax.Array:
        """Unranks JAX int32 basis labels into flat states.

        Args:
            numbers: Integer basis labels.

        Returns:
            State array with trailing dimension ``size``. The generic path is
            intended for unconstrained spaces whose product dimension fits the
            supported JAX indexing range.
        """
        from neuralqx.hilbert.index import numbers_to_states

        return numbers_to_states(self, numbers)

    def states_to_numbers_python(self, states: Any, *, validate: bool = False) -> Any:
        """Ranks unconstrained states using arbitrary-size Python integers.

        Args:
            states: State array to rank.
            validate: Whether to check membership before ranking.

        Returns:
            Python integer label for one state or an object array of labels.
        """
        from neuralqx.hilbert.index import states_to_numbers_python

        return states_to_numbers_python(self, states, validate=validate)

    def numbers_to_states_python(self, numbers: Any) -> jax.Array:
        """Unranks arbitrary-size Python integer labels into flat states.

        Args:
            numbers: Python integer labels or an array of labels.

        Returns:
            State array with trailing dimension ``size``. This path supports
            larger unconstrained product spaces by doing the ranking arithmetic
            with Python integers.
        """
        from neuralqx.hilbert.index import numbers_to_states_python

        return numbers_to_states_python(self, numbers)

    def all_states(self, *, max_states: int = 1_000_000) -> jax.Array:
        """Materializes every unconstrained state up to a safety limit.

        Args:
            max_states: Maximum number of states to materialize.

        Returns:
            Array containing all unconstrained basis states.

        Raises:
            RuntimeError: If the space is constrained or exceeds
                ``max_states``.
        """
        if self.constrained:
            raise RuntimeError(
                "Generic constrained all_states would require filtering."
            )
        if self.unconstrained_dimension > max_states:
            raise RuntimeError(
                f"Refusing to materialize {self.unconstrained_dimension} states; "
                f"increase max_states={max_states} explicitly if this is intentional."
            )
        return self.numbers_to_states_python(
            np.arange(self.unconstrained_dimension, dtype=object)
        )


AbstractDiscreteHilbert = DiscreteHilbertSpace


__all__ = ["AbstractDiscreteHilbert", "DiscreteHilbertSpace"]
