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


from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def batch_shape(size: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Normalizes a random-state batch-size declaration.

    Args:
        size: ``None`` for a single state, an integer batch size, or an
            explicit tuple of batch dimensions.

    Returns:
        Tuple of leading batch dimensions.

    Raises:
        ValueError: If an integer size is negative.
    """
    if size is None:
        return ()
    if isinstance(size, int):
        if size < 0:
            raise ValueError("size must be non-negative.")
        return (size,)
    return tuple(int(dim) for dim in size)


def ensure_2d_trailing(
    values: Any, trailing: int
) -> tuple[jax.Array, bool, tuple[int, ...]]:
    """Flattens leading batch axes while preserving a trailing state size.

    Args:
        values: Single state or batched state array.
        trailing: Expected trailing dimension.

    Returns:
        A two-dimensional array, a flag indicating whether the input was a
        single state, and the original batch shape.

    Raises:
        ValueError: If the trailing dimension does not match ``trailing``.
    """
    arr = jnp.asarray(values)
    if arr.ndim == 1:
        if arr.shape[0] != trailing:
            raise ValueError(
                f"Trailing dimension {arr.shape[0]} does not match {trailing}."
            )
        return arr[None, :], True, ()
    if arr.shape[-1] != trailing:
        raise ValueError(
            f"Trailing dimension {arr.shape[-1]} does not match {trailing}."
        )
    return arr.reshape((-1, trailing)), False, arr.shape[:-1]


def restore_trailing(
    values: jax.Array, single: bool, batch: tuple[int, ...]
) -> jax.Array:
    """Restores the shape captured by :func:`ensure_2d_trailing`.

    Args:
        values: Two-dimensional array with a trailing state dimension.
        single: Whether the original input was a single state.
        batch: Original batch shape.

    Returns:
        Single state when ``single`` is ``True``. Otherwise a batched array with
        the recorded leading shape.
    """
    if single:
        return values[0]
    return values.reshape((*batch, values.shape[-1]))


__all__ = ["batch_shape", "ensure_2d_trailing", "restore_trailing"]
