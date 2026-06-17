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


"""Arithmetic local quantum-number ranges.

``LocalRange`` describes finite evenly spaced value sets such as spin labels,
charge labels, or bounded integer intervals. The representation stores the
first value, step, and number of entries rather than materializing the table at
construction time.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralqx.hilbert.utils import canonical_dtype
from neuralqx.utils.struct import field

from .abstract import AbstractLocalSpace


class LocalRange(AbstractLocalSpace):
    """Arithmetic local domain ``start + step * n`` for ``0 <= n < size``.

    Args:
        start: First allowed local value.
        step: Difference between adjacent allowed values.
        size: Number of allowed values.
        dtype: Optional dtype used when materializing values.

    Attributes:
        start: First allowed local value.
        step: Difference between adjacent allowed values.
        size: Number of allowed values.
        dtype: Optional dtype used when materializing values.
    """

    start: int | float = field(static=True)
    """First allowed local value."""

    step: int | float = field(static=True)
    """Difference between adjacent allowed values."""

    size: int = field(static=True)
    """Number of allowed values."""

    dtype: np.dtype | str | Any | None = field(static=True, default=None)
    """Optional dtype used when materializing values."""

    def __post_init__(self) -> None:
        """Validates the arithmetic range and resolves its dtype."""
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise TypeError(f"size must be an int, got {type(self.size).__name__}.")
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}.")
        if self.step == 0:
            raise ValueError("step must be non-zero.")
        object.__setattr__(
            self, "dtype", canonical_dtype(self.dtype, self.start, self.step)
        )

    def __hash__(self) -> int:
        """Returns a structural hash for the arithmetic local range."""
        return hash((type(self), self.start, self.step, self.size, str(self.dtype)))

    @property
    def local_size(self) -> int:
        """Number of allowed local quantum-number values."""
        return self.size

    def all_values(self, *, dtype: Any | None = None) -> jax.Array:
        """Returns all values in the arithmetic range in basis order.

        Args:
            dtype: Optional dtype overriding the space dtype.

        Returns:
            One-dimensional array containing ``start + step * n`` for every
            valid local index ``n`` from zero through ``size - 1``.
        """
        return self.indices_to_values(
            jnp.arange(self.size, dtype=jnp.int32), dtype=dtype
        )

    def indices_to_values(self, indices: Any, *, dtype: Any | None = None) -> jax.Array:
        """Maps local integer indices to arithmetic range values.

        Args:
            indices: Integer local indices.
            dtype: Optional dtype for the returned values.

        Returns:
            ``start + step * indices`` with the requested dtype.
        """
        out_dtype = self.dtype if dtype is None else np.dtype(dtype)
        idx = jnp.asarray(indices)
        start = jnp.asarray(self.start, dtype=out_dtype)
        step = jnp.asarray(self.step, dtype=out_dtype)
        return (start + idx.astype(out_dtype) * step).astype(out_dtype)

    def values_to_indices(self, values: Any) -> jax.Array:
        """Maps arithmetic range values back to local integer indices.

        Args:
            values: Candidate local values.

        Returns:
            Integer indices obtained by subtracting ``start`` and dividing by
            ``step``. Callers that need membership validation should pair this
            conversion with :meth:`contains`.
        """
        vals = jnp.asarray(values, dtype=self.dtype)
        raw = (vals - jnp.asarray(self.start, dtype=self.dtype)) / jnp.asarray(
            self.step,
            dtype=self.dtype,
        )
        if np.issubdtype(self.dtype, np.floating):
            return jnp.rint(raw).astype(jnp.int32)
        return raw.astype(jnp.int32)

    def contains(self, values: Any) -> jax.Array:
        """Checks whether values lie exactly on this finite range.

        Args:
            values: Candidate local values.

        Returns:
            Boolean mask indicating values within bounds and on the arithmetic
            grid.
        """
        vals = jnp.asarray(values, dtype=self.dtype)
        idx = self.values_to_indices(vals)
        in_range = (idx >= 0) & (idx < self.size)
        reconstructed = self.indices_to_values(idx, dtype=self.dtype)
        if np.issubdtype(self.dtype, np.floating):
            return in_range & jnp.isclose(vals, reconstructed)
        return in_range & (vals == reconstructed)


__all__ = ["LocalRange"]
