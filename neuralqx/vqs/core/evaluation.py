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


"""Exact and batched variational-state evaluation helpers."""

from __future__ import annotations

from functools import partial
from math import prod
from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.jax.streaming import apply_batched


def flatten_state_batch(
    states: Any, hilbert_size: int
) -> tuple[jax.Array, tuple[int, ...]]:
    """Flatten state leading axes while preserving one trailing Hilbert axis."""
    arr = jnp.asarray(states)
    if arr.ndim == 0:
        raise ValueError("states must have a trailing Hilbert dimension.")
    if arr.shape[-1] != int(hilbert_size):
        raise ValueError(
            f"states must have trailing Hilbert size {hilbert_size}; got shape {arr.shape}."
        )
    return arr.reshape((-1, int(hilbert_size))), arr.shape[:-1]


@partial(jax.jit, static_argnames=("apply_variables", "chunk_size"))
def evaluate_log_values(
    apply_variables: Any,
    variables: Any,
    states_2d: jax.Array,
    *,
    chunk_size: int | None,
) -> jax.Array:
    """Evaluate model log-amplitudes over a rank-2 state batch."""
    return apply_batched(
        apply_variables,
        variables,
        states_2d,
        chunk_size=chunk_size,
    )


def restore_log_values(values: Any, state_shape: tuple[int, ...]) -> jax.Array:
    """Restore log values to the input state's leading shape."""
    arr = jnp.asarray(values)
    if arr.shape[0] != prod(state_shape or (1,)):
        raise ValueError("Model output leading dimension does not match input states.")
    return arr.reshape((*state_shape, *arr.shape[1:]))


def amplitudes_from_log_values(log_values: Any, *, normalize: bool) -> jax.Array:
    """Convert log-amplitudes to amplitudes with stable optional normalization."""
    logs = jnp.asarray(log_values)
    if not normalize:
        return jnp.exp(logs)
    if logs.size == 0:
        return jnp.exp(logs)
    shift = jnp.max(jnp.real(logs))
    amplitudes = jnp.exp(logs - shift)
    norm = jnp.sqrt(jnp.sum(jnp.abs(amplitudes) ** 2))
    return amplitudes / norm


__all__ = [
    "amplitudes_from_log_values",
    "evaluate_log_values",
    "flatten_state_batch",
    "restore_log_values",
]
