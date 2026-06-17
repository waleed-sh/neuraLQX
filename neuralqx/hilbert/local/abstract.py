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


"""Abstract local quantum-number domains.

Local spaces describe the finite values that may appear at a single flat site
of a discrete Hilbert state. Concrete implementations provide conversion
between compact integer indices and the physical quantum-number values stored
in state arrays.
"""

from __future__ import annotations

import abc
from typing import Any

import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta


class AbstractLocalSpace(Struct, metaclass=StructABCMeta):
    """Abstract finite domain for one scalar Hilbert-space site.

    A local space defines the allowed quantum-number values at a single flat
    site and the compact integer indices used for basis ranking, random
    sampling, and generic proposal moves. Concrete subclasses decide whether
    the values come from an arithmetic range, an explicit table, or another
    finite representation.

    Subclasses must implement :attr:`local_size`, :meth:`all_values`,
    :meth:`indices_to_values`, :meth:`values_to_indices`, and
    :meth:`contains`. The inherited sampling helpers use that contract to draw
    uniform local indices and convert them back to physical values.
    """

    @property
    @abc.abstractmethod
    def local_size(self) -> int:
        """Number of allowed local quantum-number values."""

    @abc.abstractmethod
    def all_values(self, *, dtype: Any | None = None) -> jax.Array:
        """Returns all values in this local domain in basis order.

        Args:
            dtype: Optional dtype for the returned value table.

        Returns:
            One-dimensional array of allowed local quantum numbers. Entry
            ``i`` must be the value produced by ``indices_to_values(i)``.
        """

    @abc.abstractmethod
    def indices_to_values(self, indices: Any, *, dtype: Any | None = None) -> jax.Array:
        """Maps local integer indices to quantum-number values.

        Args:
            indices: Integer local indices in the range ``[0, local_size)``.
            dtype: Optional dtype for the returned values.

        Returns:
            Array of local quantum-number values with the same shape as
            ``indices``.
        """

    @abc.abstractmethod
    def values_to_indices(self, values: Any) -> jax.Array:
        """Maps local quantum-number values to integer indices.

        Args:
            values: Candidate local quantum-number values.

        Returns:
            Integer indices into this local domain. Implementations should use
            the same basis order returned by :meth:`all_values`.
        """

    @abc.abstractmethod
    def contains(self, values: Any) -> jax.Array:
        """Returns a boolean mask for values in this local domain.

        Args:
            values: Candidate local quantum-number values.

        Returns:
            Boolean array with the same shape as ``values``.
        """

    def random_indices(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """Uniformly samples local integer indices.

        Args:
            key: JAX pseudo-random key.
            shape: Output shape for sampled indices.

        Returns:
            Integer array with values in ``[0, local_size)``. Sampling is
            uniform over the finite local basis.
        """
        return jax.random.randint(
            key,
            shape=shape,
            minval=0,
            maxval=self.local_size,
            dtype=jnp.int32,
        )

    def random_values(
        self,
        key: jax.Array,
        shape: tuple[int, ...],
        *,
        dtype: Any | None = None,
    ) -> jax.Array:
        """Uniformly samples local quantum-number values.

        Args:
            key: JAX pseudo-random key.
            shape: Output shape for sampled values.
            dtype: Optional dtype for returned values.

        Returns:
            Random values drawn uniformly from this local domain.
        """
        return self.indices_to_values(self.random_indices(key, shape), dtype=dtype)


__all__ = ["AbstractLocalSpace"]
