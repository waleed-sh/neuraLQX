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


"""Chunked JAX execution over a flattened leading batch axis."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.utils.typing import ConnectedKernel


def stream_flattened_kernel(
    kernel: ConnectedKernel,
    states: Any,
    *,
    chunk_size: int,
    max_conn_size: int,
    matrix_element_dtype: Any,
) -> tuple[jax.Array, jax.Array]:
    """Run a connected-component kernel over fixed-size chunks.

    ``states`` must have shape ``(n_states, hilbert_size)``. ``kernel`` is
    called with chunks of shape ``(chunk_size, hilbert_size)`` and must return
    ``(chunk_size, max_conn_size, hilbert_size)`` and
    ``(chunk_size, max_conn_size)`` arrays. Padding rows are copied from the
    final valid row and trimmed away after the scan, so kernels never need a
    special invalid-state branch.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if max_conn_size <= 0:
        raise ValueError("max_conn_size must be positive.")

    flat = jnp.asarray(states)
    if flat.ndim != 2:
        raise ValueError("stream_flattened_kernel expects a rank-2 state batch.")

    n_states = int(flat.shape[0])
    hilbert_size = int(flat.shape[1])
    if n_states == 0:
        xp = jnp.empty(
            (0, int(max_conn_size), hilbert_size),
            dtype=flat.dtype,
        )
        mels = jnp.empty((0, int(max_conn_size)), dtype=matrix_element_dtype)
        return xp, mels

    n_chunks = (n_states + int(chunk_size) - 1) // int(chunk_size)
    padded_size = n_chunks * int(chunk_size)
    pad = padded_size - n_states
    if pad:
        fill = jnp.broadcast_to(flat[-1:, :], (pad, hilbert_size))
        flat = jnp.concatenate((flat, fill), axis=0)

    chunks = flat.reshape((n_chunks, int(chunk_size), hilbert_size))
    xp_chunks, mel_chunks = jax.lax.map(kernel, chunks)

    xp = xp_chunks.reshape((padded_size, *xp_chunks.shape[2:]))[:n_states]
    mels = mel_chunks.reshape((padded_size, *mel_chunks.shape[2:]))[:n_states]
    return xp, mels


__all__ = ["stream_flattened_kernel"]
