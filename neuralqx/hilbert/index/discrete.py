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
import numpy as np

from neuralqx.hilbert.space.discrete import DiscreteHilbertSpace


def states_to_numbers(
    space: DiscreteHilbertSpace, states: Any, *, validate: bool = False
) -> jax.Array:
    """Ranks unconstrained states in rightmost-fast lexicographic order.

    Args:
        space: Indexable unconstrained discrete Hilbert space.
        states: State array whose trailing dimension equals ``space.size``.
        validate: Whether to check membership before ranking.

    Returns:
        JAX integer labels matching the input batch shape.
    """
    _ensure_jax_indexable(space)
    if validate and not bool(jnp.all(space.is_valid(states))):
        raise ValueError("states contain values outside the Hilbert space.")

    local = space.states_to_local_indices(states).astype(jnp.int32)
    single = local.ndim == 1
    local2 = local[None, :] if single else local.reshape((-1, space.size))
    basis = jnp.asarray(_basis(space.local_sizes), dtype=jnp.int32)
    out = local2 @ basis
    return out[0] if single else out.reshape(local.shape[:-1])


def numbers_to_states(space: DiscreteHilbertSpace, numbers: Any) -> jax.Array:
    """Unranks integer labels for indexable unconstrained spaces.

    Args:
        space: Indexable unconstrained discrete Hilbert space.
        numbers: Integer labels to decode.

    Returns:
        State array with trailing dimension ``space.size``.
    """
    _ensure_jax_indexable(space)
    nums = jnp.asarray(numbers, dtype=jnp.int32)
    single = nums.ndim == 0
    flat = nums.reshape((-1,))
    digits = []
    rem = flat
    for local_size in reversed(space.local_sizes):
        rem, digit = jnp.divmod(rem, local_size)
        digits.append(digit)
    local = jnp.stack(tuple(reversed(digits)), axis=-1)
    states = space.local_indices_to_states(local)
    return states[0] if single else states.reshape((*nums.shape, space.size))


def states_to_numbers_python(
    space: DiscreteHilbertSpace, states: Any, *, validate: bool = False
) -> Any:
    """Ranks states using arbitrary-size Python integers.

    Args:
        space: Unconstrained discrete Hilbert space.
        states: State array whose trailing dimension equals ``space.size``.
        validate: Whether to check membership before ranking.

    Returns:
        Python integer for one state or an object array for batched states.
    """
    if space.constrained:
        raise RuntimeError("Generic constrained ranking would require enumeration.")
    if validate and not bool(jnp.all(space.is_valid(states))):
        raise ValueError("states contain values outside the Hilbert space.")

    local = np.asarray(jax.device_get(space.states_to_local_indices(states)))
    single = local.ndim == 1
    local2 = local[None, :] if single else local.reshape((-1, space.size))
    out = [_rank_row(row, space.local_sizes) for row in local2]
    if single:
        return out[0]
    return np.asarray(out, dtype=object).reshape(local.shape[:-1])


def numbers_to_states_python(space: DiscreteHilbertSpace, numbers: Any) -> jax.Array:
    """Unranks arbitrary-size Python integers without product materialization.

    Args:
        space: Unconstrained discrete Hilbert space.
        numbers: Python integer labels or an array of labels.

    Returns:
        State array with trailing dimension ``space.size``.
    """
    if space.constrained:
        raise RuntimeError("Generic constrained unranking would require enumeration.")
    nums = np.asarray(numbers, dtype=object)
    single = nums.ndim == 0
    rows = [
        _unrank_number(int(number), space.local_sizes) for number in nums.reshape((-1,))
    ]
    local = jnp.asarray(rows, dtype=jnp.int32)
    states = space.local_indices_to_states(local)
    return states[0] if single else states.reshape((*nums.shape, space.size))


def _basis(local_sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Computes mixed-radix strides for rightmost-fast ranking.

    Args:
        local_sizes: Local dimension at every flat site.

    Returns:
        Tuple of integer strides with the same length as ``local_sizes``.
    """
    stride = 1
    values = []
    for local_size in reversed(local_sizes):
        values.append(stride)
        stride *= local_size
    return tuple(reversed(values))


def _rank_row(row: np.ndarray, local_sizes: tuple[int, ...]) -> int:
    """Ranks one local-index row using Python integers.

    Args:
        row: One-dimensional local-index row.
        local_sizes: Local dimension at every flat site.

    Returns:
        Mixed-radix basis label for ``row``.
    """
    number = 0
    for digit, base in zip(row.tolist(), local_sizes, strict=True):
        number = number * base + int(digit)
    return number


def _unrank_number(number: int, local_sizes: tuple[int, ...]) -> list[int]:
    """Unranks one Python integer into mixed-radix digits.

    Args:
        number: Non-negative basis label.
        local_sizes: Local dimension at every flat site.

    Returns:
        Local-index row corresponding to ``number``.
    """
    if number < 0:
        raise ValueError("number must be non-negative.")
    row = [0] * len(local_sizes)
    rem = number
    for idx in range(len(local_sizes) - 1, -1, -1):
        rem, digit = divmod(rem, local_sizes[idx])
        row[idx] = digit
    if rem != 0:
        raise ValueError("number outside the Hilbert-space range.")
    return row


def _ensure_jax_indexable(space: DiscreteHilbertSpace) -> None:
    """Validates that a space supports generic JAX int32 ranking.

    Args:
        space: Discrete Hilbert space to validate.

    Raises:
        RuntimeError: If the space is constrained or too large for int32
            ranking.
    """
    if space.constrained:
        raise RuntimeError("Generic constrained ranking would require enumeration.")
    if not space.is_indexable:
        raise RuntimeError("This Hilbert space is too large for JAX int32 indexing.")


__all__ = [
    "numbers_to_states",
    "numbers_to_states_python",
    "states_to_numbers",
    "states_to_numbers_python",
]
