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


"""Algorithmic JAX-native computational operators."""

from __future__ import annotations

import abc
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.sharding import place_connected_outputs
from neuralqx.jax.sharding.connected import place_operator_batch
from neuralqx.jax.vectorization import vmap_connected_kernel
from neuralqx.operator._algebra import is_scalar_like
from neuralqx.operator._algebra import is_zero_scalar_like

from ._batching import flatten_states
from ._batching import restore_connected
from ._connect import direct_connected_components
from ._connect import streamed_connected_components
from ._policy import resolve_streaming_policy
from .discrete import DiscreteOperator


class ComputationalOperator(DiscreteOperator):
    """Discrete operator defined by a JAX connected-component kernel.

    Subclasses implement :meth:`_get_conn_padded_kernel` for one basis state.
    The kernel receives a rank-1 state with shape ``(hilbert.size,)`` and must
    return connected states with shape ``(max_conn_size, hilbert.size)`` and
    matrix elements with shape ``(max_conn_size,)``. neuraLQX supplies the
    batched implementation by vectorizing this single-state kernel.
    """

    def __add__(self, other: Any) -> ComputationalOperator:
        if isinstance(other, ComputationalOperator):
            from .wrappers import SumOperator

            return SumOperator((self, other))
        return NotImplemented

    def __radd__(self, other: Any) -> ComputationalOperator:
        if is_zero_scalar_like(other):
            return self
        if isinstance(other, ComputationalOperator):
            from .wrappers import SumOperator

            return SumOperator((other, self))
        return NotImplemented

    def __sub__(self, other: Any) -> ComputationalOperator:
        if isinstance(other, ComputationalOperator):
            from .wrappers import SumOperator

            return SumOperator((self, other), coefficients=jnp.asarray([1.0, -1.0]))
        return NotImplemented

    def __rsub__(self, other: Any) -> ComputationalOperator:
        if is_zero_scalar_like(other):
            return -self
        if isinstance(other, ComputationalOperator):
            from .wrappers import SumOperator

            return SumOperator((other, self), coefficients=jnp.asarray([1.0, -1.0]))
        return NotImplemented

    def __neg__(self) -> ComputationalOperator:
        from .wrappers import ScaledOperator

        return ScaledOperator(self, -1.0)

    def __mul__(self, other: Any) -> ComputationalOperator:
        if is_scalar_like(other):
            from .wrappers import ScaledOperator

            return ScaledOperator(self, other)
        return NotImplemented

    def __rmul__(self, other: Any) -> ComputationalOperator:
        if is_scalar_like(other):
            from .wrappers import ScaledOperator

            return ScaledOperator(self, other)
        return NotImplemented

    def __matmul__(self, other: Any) -> ComputationalOperator:
        if isinstance(other, ComputationalOperator):
            from .wrappers import ProductOperator

            return ProductOperator((self, other))
        return NotImplemented

    def __rmatmul__(self, other: Any) -> ComputationalOperator:
        if isinstance(other, ComputationalOperator):
            from .wrappers import ProductOperator

            return ProductOperator((other, self))
        return NotImplemented

    def __pow__(self, power: int) -> ComputationalOperator:
        if not isinstance(power, int) or power < 1:
            return NotImplemented
        result: ComputationalOperator = self
        for _ in range(power - 1):
            result = result @ self
        return result

    def squared(self) -> ComputationalOperator:
        """Return the positive wrapper ``self.adjoint @ self``."""
        from .wrappers import Squared

        return Squared(self)

    def get_conn_padded(
        self,
        states: Any,
        *,
        use_streaming: bool | None = None,
        use_sharding: bool | None = None,
        chunk_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return padded connected states and matrix elements.

        ``use_streaming`` controls connected-component streaming/chunking inside
        this one operator. ``use_sharding`` is kept as a compatibility alias for
        older code that used that name for streaming.
        """
        if use_streaming is not None and use_sharding is not None:
            raise ValueError("Pass either use_streaming or use_sharding, not both.")
        if use_streaming is None:
            use_streaming = use_sharding
        flat, single, batch_shape = flatten_states(states, self.hilbert.size)
        stream, resolved_chunk_size = resolve_streaming_policy(
            use_streaming=use_streaming,
            chunk_size=chunk_size,
        )
        if stream:
            flat = place_operator_batch(flat)

        if stream:
            x_primes, mels = streamed_connected_components(
                self,
                flat,
                chunk_size=resolved_chunk_size,
            )
        else:
            x_primes, mels = direct_connected_components(self, flat)

        _validate_connected_shapes(
            self,
            x_primes,
            mels,
            n_states=int(flat.shape[0]),
            hilbert_size=int(flat.shape[1]),
        )
        restored = restore_connected(
            x_primes,
            mels,
            single=single,
            batch_shape=batch_shape,
        )
        return place_connected_outputs(jnp.asarray(states), *restored)

    def _get_conn_padded_batch_kernel(
        self,
        states_2d: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Vectorized padded connected-component action for rank-2 batches."""
        return vmap_connected_kernel(self._get_conn_padded_kernel, states_2d)

    @abc.abstractmethod
    def _get_conn_padded_kernel(
        self,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Kernel implementing the padded action on one basis state."""


def _validate_connected_shapes(
    operator: ComputationalOperator,
    x_primes: Any,
    mels: Any,
    *,
    n_states: int,
    hilbert_size: int,
) -> None:
    xp = jnp.asarray(x_primes)
    mel = jnp.asarray(mels)
    expected_xp = (n_states, int(operator.max_conn_size), hilbert_size)
    expected_mel = (n_states, int(operator.max_conn_size))
    if xp.shape != expected_xp:
        raise ValueError(
            f"{type(operator).__name__} kernel returned connected states with "
            f"shape {xp.shape}, expected {expected_xp}."
        )
    if mel.shape != expected_mel:
        raise ValueError(
            f"{type(operator).__name__} kernel returned matrix elements with "
            f"shape {mel.shape}, expected {expected_mel}."
        )


__all__ = ["ComputationalOperator"]
