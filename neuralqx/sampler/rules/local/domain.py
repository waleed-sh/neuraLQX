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


"""Local-space domain helpers for local transition rules."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax


def local_range_parameters(
    hilbert, dtype
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
    """Return arithmetic local-space metadata when every site is range-like."""
    if not all(
        all(hasattr(local_space, name) for name in ("start", "step", "size"))
        for local_space in hilbert.local_spaces
    ):
        return None

    out_dtype = np.dtype(dtype)
    starts = np.asarray(
        [local_space.start for local_space in hilbert.local_spaces],
        dtype=out_dtype,
    )
    steps = np.asarray(
        [local_space.step for local_space in hilbert.local_spaces],
        dtype=out_dtype,
    )
    sizes = np.asarray(
        [int(local_space.size) for local_space in hilbert.local_spaces],
        dtype=np.int32,
    )
    return (
        jnp.asarray(starts, dtype=dtype),
        jnp.asarray(steps, dtype=dtype),
        jnp.asarray(sizes, dtype=jnp.int32),
    )


def all_local_sizes_equal(hilbert, size: int) -> bool:
    """Return whether every flat site has the same local cardinality."""
    return all(int(local_size) == int(size) for local_size in hilbert.local_sizes)


def homogeneous_binary_flip_parameters(
    hilbert, dtype
) -> tuple[jax.Array, jax.Array] | None:
    """Return scalar ``(start, step)`` for a homogeneous binary range space."""
    spaces = hilbert.local_spaces
    first = spaces[0]
    start = first.start
    step = first.step
    if not all(space.start == start and space.step == step for space in spaces):
        return None
    return (
        jnp.asarray(start, dtype=dtype),
        jnp.asarray(step, dtype=dtype),
    )


def local_value_table(hilbert, dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build padded local-value and validity tables for heterogeneous spaces."""
    local_sizes = tuple(int(size) for size in hilbert.local_sizes)
    max_size = max(local_sizes)
    rows = []
    masks = []
    out_dtype = np.dtype(dtype)
    for local_space, size in zip(hilbert.local_spaces, local_sizes, strict=True):
        values = local_space_values_numpy(local_space, out_dtype).reshape((size,))
        if size < max_size:
            pad = np.broadcast_to(values[:1], (max_size - size,))
            values = np.concatenate((values, pad), axis=0)
        rows.append(values)
        masks.append(np.arange(max_size) < size)
    return (
        jnp.asarray(np.stack(rows, axis=0), dtype=dtype),
        jnp.asarray(np.stack(masks, axis=0), dtype=jnp.bool_),
        jnp.asarray(local_sizes, dtype=jnp.int32),
    )


def local_space_values_numpy(local_space, dtype: np.dtype) -> np.ndarray:
    """Materialize local-space values on host for static transition metadata."""
    if hasattr(local_space, "values"):
        return np.asarray(local_space.values, dtype=dtype)
    if all(hasattr(local_space, name) for name in ("start", "step", "size")):
        indices = np.arange(int(local_space.size), dtype=np.int64)
        return (
            np.asarray(local_space.start, dtype=dtype)
            + indices.astype(dtype) * np.asarray(local_space.step, dtype=dtype)
        ).astype(dtype)
    values = jax.device_get(local_space.all_values(dtype=dtype))
    return np.asarray(values, dtype=dtype)


def values_equal(values: jax.Array, candidates: jax.Array) -> jax.Array:
    """Compare selected values against candidate local values by dtype."""
    if jnp.issubdtype(values.dtype, jnp.integer) or jnp.issubdtype(
        values.dtype, jnp.bool_
    ):
        return candidates == values[:, None]
    return jnp.isclose(candidates, values[:, None])


__all__ = [
    "all_local_sizes_equal",
    "homogeneous_binary_flip_parameters",
    "local_range_parameters",
    "local_space_values_numpy",
    "local_value_table",
    "values_equal",
]
