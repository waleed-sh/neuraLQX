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


"""Chunked evaluation helpers for batched model calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

import jax


def apply_batched(
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    states: Any,
    *,
    chunk_size: int | None = None,
) -> jax.Array:
    """Evaluate ``apply_fn(parameters, states)`` with optional leading-axis chunks."""
    arr = jnp.asarray(states)
    if chunk_size is None:
        return jnp.asarray(apply_fn(parameters, arr))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if arr.shape[0] <= int(chunk_size):
        return jnp.asarray(apply_fn(parameters, arr))
    return _apply_batched_chunked(apply_fn, parameters, arr, int(chunk_size))


def _apply_batched_chunked(
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    states: jax.Array,
    chunk_size: int,
) -> jax.Array:
    n_states = int(states.shape[0])
    n_chunks = (n_states + chunk_size - 1) // chunk_size
    padded_size = n_chunks * chunk_size
    pad = padded_size - n_states
    if pad:
        fill = jnp.broadcast_to(states[-1:, :], (pad, states.shape[1]))
        states = jnp.concatenate((states, fill), axis=0)

    chunks = states.reshape((n_chunks, chunk_size, states.shape[1]))
    values = jax.lax.map(lambda chunk: apply_fn(parameters, chunk), chunks)
    return values.reshape((padded_size, *values.shape[2:]))[:n_states]


__all__ = ["apply_batched"]
