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


"""Shape normalization for chain-aware statistics."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax


def as_chain_matrix(values: Any, *, chain_axis: int | None = 0) -> jax.Array:
    """Return values as ``(n_chains, samples_per_chain)``.

    Scalar inputs become one chain with one sample. Rank-1 inputs become one
    chain. Higher-rank inputs use ``chain_axis`` as the independent-chain axis
    and flatten all remaining axes into per-chain draws.
    """
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    if arr.ndim == 1 or chain_axis is None:
        return arr.reshape((1, -1))
    axis = int(chain_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"chain_axis={chain_axis} is invalid for shape {arr.shape}.")
    arr = jnp.moveaxis(arr, axis, 0)
    return arr.reshape((arr.shape[0], -1))


def flattened_sample_count(values: Any) -> int:
    """Return total number of scalar samples in ``values``."""
    return int(jnp.asarray(values).size)


__all__ = ["as_chain_matrix", "flattened_sample_count"]
