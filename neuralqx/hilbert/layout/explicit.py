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


"""Explicit finite local domains.

An explicit local space stores the complete value table for a site whose
allowed values are finite but not necessarily arithmetic. It is the fallback
adapter for user-provided sequences of local values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralqx.hilbert.utils import canonical_dtype
from neuralqx.utils.struct import field

from .abstract import AbstractLocalSpace


class ExplicitLocalSpace(AbstractLocalSpace):
    """Finite local domain defined by an explicit value table.

    ``ExplicitLocalSpace`` is used when the allowed values at a site are not a
    simple arithmetic range. The values are stored in the order used for local
    basis indices, so index ``0`` maps to ``values[0]``, index ``1`` maps to
    ``values[1]``, and so on.

    Args:
        values: Allowed local quantum-number values.
        dtype: Optional dtype used when materializing values as JAX arrays.

    Attributes:
        values: Allowed local quantum-number values.
        dtype: Optional dtype used when materializing values as JAX arrays.
    """

    values: tuple[Any, ...] | Sequence[Any] = field(static=True)
    """Allowed local quantum-number values."""

    dtype: np.dtype | str | Any | None = field(static=True, default=None)
    """Optional dtype used when materializing values as JAX arrays."""

    def __post_init__(self) -> None:
        """Normalizes the value table and resolves its dtype."""
        values = tuple(self.values)
        if not values:
            raise ValueError("values must contain at least one local state.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype, *values))

    def __hash__(self) -> int:
        """Returns a structural hash for the explicit local space."""
        return hash((type(self), self.values, str(self.dtype)))

    @property
    def local_size(self) -> int:
        """Number of allowed local quantum-number values."""
        return len(self.values)

    def all_values(self, *, dtype: Any | None = None) -> jax.Array:
        """Returns the explicit value table in local-index order.

        Args:
            dtype: Optional dtype overriding the space dtype.

        Returns:
            One-dimensional array of allowed local values. The returned order is
            the basis order used by :meth:`indices_to_values` and
            :meth:`values_to_indices`.
        """
        return jnp.asarray(self.values, dtype=self.dtype if dtype is None else dtype)

    def indices_to_values(self, indices: Any, *, dtype: Any | None = None) -> jax.Array:
        """Maps local indices into the explicit value table.

        Args:
            indices: Integer local indices.
            dtype: Optional dtype for the returned values.

        Returns:
            Values selected from :meth:`all_values`.
        """
        return jnp.take(
            self.all_values(dtype=dtype), jnp.asarray(indices, dtype=jnp.int32), axis=0
        )

    def values_to_indices(self, values: Any) -> jax.Array:
        """Maps local values to their first matching table indices.

        Args:
            values: Candidate local values.

        Returns:
            Integer index of the matching explicit value for each input.
        """
        vals = jnp.asarray(values, dtype=self.dtype)
        table = self.all_values(dtype=self.dtype)
        return jnp.argmax(vals[..., None] == table, axis=-1).astype(jnp.int32)

    def contains(self, values: Any) -> jax.Array:
        """Checks whether candidate values are present in the explicit table.

        Args:
            values: Candidate local values.

        Returns:
            Boolean mask with the same shape as ``values``. A value is accepted
            when it equals at least one entry in the stored table after dtype
            normalization.
        """
        vals = jnp.asarray(values, dtype=self.dtype)
        table = self.all_values(dtype=self.dtype)
        return jnp.any(vals[..., None] == table, axis=-1)


__all__ = ["ExplicitLocalSpace"]
