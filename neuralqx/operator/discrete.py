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


"""Discrete operator base functionality."""

from __future__ import annotations

import abc
from typing import Any

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import jax
from neuralqx.hilbert import DiscreteHilbertSpace

from ._batching import flatten_states
from .abstract import AbstractOperator


class DiscreteOperator(AbstractOperator):
    """Base class for operators acting on discrete Hilbert spaces."""

    hilbert: DiscreteHilbertSpace

    def __post_init__(self) -> None:
        if not isinstance(self.hilbert, DiscreteHilbertSpace):
            raise TypeError(
                "Discrete operators must act on a DiscreteHilbertSpace; "
                f"got {type(self.hilbert).__name__}."
            )

    @property
    @abc.abstractmethod
    def max_conn_size(self) -> int:
        """Maximum padded number of connected states per input state."""

    @property
    def is_diagonal(self) -> bool:
        """Whether this operator only connects a basis state to itself."""
        return False

    @abc.abstractmethod
    def get_conn_padded(
        self, states: Any, **kwargs: Any
    ) -> tuple[jax.Array, jax.Array]:
        """Return padded connected states and matrix elements."""

    def get_conn_flattened(
        self,
        states: Any,
        sections: Any | None = None,
        *,
        pad: bool = False,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Return connected states flattened over the batch and connection axes."""
        flat, _single, _batch_shape = flatten_states(states, self.hilbert.size)
        x_primes, mels = self.get_conn_padded(flat, **kwargs)
        xp_flat = x_primes.reshape((-1, self.hilbert.size))
        mel_flat = mels.reshape((-1,))

        if pad:
            counts = jnp.full((flat.shape[0],), int(mels.shape[-1]), dtype=jnp.int32)
        else:
            mask = mel_flat != 0
            xp_flat = xp_flat[mask]
            mel_flat = mel_flat[mask]
            counts = jnp.sum(mels != 0, axis=-1).astype(jnp.int32)

        cumulative = jnp.cumsum(counts)
        if sections is not None:
            sections[...] = jax.device_get(cumulative)
        return xp_flat, mel_flat

    def get_conn(self, state: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array]:
        """Return nonzero connected states for one input state."""
        flat, single, _batch_shape = flatten_states(state, self.hilbert.size)
        if not single:
            raise ValueError(
                "get_conn expects one state; use get_conn_padded for batches."
            )
        x_primes, mels = self.get_conn_padded(flat, **kwargs)
        x_primes = x_primes[0]
        mels = mels[0]
        mask = mels != 0
        return x_primes[mask], mels[mask]

    def n_conn(self, states: Any, **kwargs: Any) -> jax.Array:
        """Return the number of nonzero padded connections per input state."""
        _x_primes, mels = self.get_conn_padded(states, **kwargs)
        return jnp.sum(mels != 0, axis=-1).astype(jnp.int32)

    def to_sparse(self, *, max_states: int = 1_000_000) -> BCOO:
        """Return a sparse matrix representation for small indexable spaces.

        Rows are the input basis states and columns are the connected states
        returned by :meth:`get_conn_padded`, matching the local-estimator
        convention used throughout neuraLQX.
        """
        if not self.hilbert.is_indexable:
            raise RuntimeError("to_sparse requires an indexable Hilbert space.")

        states = self.hilbert.all_states(max_states=max_states)
        x_primes, mels = self.get_conn_padded(states, use_streaming=False)
        n_states = int(states.shape[0])
        cols = self.hilbert.states_to_numbers(x_primes).reshape((-1,))
        rows = jnp.repeat(jnp.arange(n_states, dtype=cols.dtype), int(mels.shape[-1]))
        data = mels.reshape((-1,))
        mask = data != 0
        indices = jnp.stack((rows[mask], cols[mask]), axis=-1)
        return BCOO((data[mask], indices), shape=(n_states, n_states)).sum_duplicates()

    def to_dense(self, *, max_states: int = 1_000_000) -> jax.Array:
        """Return a dense matrix representation for small indexable spaces."""
        return self.to_sparse(max_states=max_states).todense()


__all__ = ["DiscreteOperator"]
