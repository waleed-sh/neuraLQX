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


"""Shape helpers for operator connected-component batches."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax


def flatten_states(
    states: Any, hilbert_size: int
) -> tuple[jax.Array, bool, tuple[int, ...]]:
    """Flatten leading state axes into a rank-2 batch."""
    arr = jnp.asarray(states)
    if arr.ndim == 0:
        raise ValueError("Operator states must have a trailing Hilbert axis.")
    if arr.shape[-1] != hilbert_size:
        raise ValueError(
            f"State trailing dimension {arr.shape[-1]} != Hilbert size {hilbert_size}."
        )
    single = arr.ndim == 1
    if single:
        return arr.reshape((1, hilbert_size)), True, ()
    return arr.reshape((-1, hilbert_size)), False, arr.shape[:-1]


def restore_connected(
    x_primes: Any,
    mels: Any,
    *,
    single: bool,
    batch_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Restore flat connected outputs to the input leading batch shape."""
    xp = jnp.asarray(x_primes)
    mel = jnp.asarray(mels)
    if xp.ndim != 3:
        raise ValueError(
            "Connected states must have shape (batch, max_conn, hilbert_size)."
        )
    if mel.ndim != 2:
        raise ValueError("Matrix elements must have shape (batch, max_conn).")
    if xp.shape[:2] != mel.shape:
        raise ValueError(
            "Connected state and matrix-element leading shapes disagree: "
            f"{xp.shape[:2]} != {mel.shape}."
        )
    if single:
        return xp[0], mel[0]
    return (
        xp.reshape((*batch_shape, xp.shape[-2], xp.shape[-1])),
        mel.reshape((*batch_shape, mel.shape[-1])),
    )


__all__ = ["flatten_states", "restore_connected"]
