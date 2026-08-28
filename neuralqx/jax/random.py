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


"""PRNG key helpers used across neuraLQX JAX code."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.utils.typing import Key


def as_prng_key(key: Key | None = None) -> jax.Array:
    """Return a JAX-compatible PRNG key from a seed, typed key, or legacy key."""
    if key is None:
        return jax.random.key(0)
    if isinstance(key, jax.Array):
        return key
    if isinstance(key, np.ndarray):
        if key.shape == ():
            return jax.random.key(int(key))
        return jnp.asarray(key, dtype=jnp.uint32)
    if isinstance(key, np.integer | int):
        return jax.random.key(int(key))

    arr = np.asarray(key)
    if arr.shape == ():
        return jax.random.key(int(arr))
    return jnp.asarray(arr, dtype=jnp.uint32)


def split_key(key: Key | None, num: int = 2) -> jax.Array:
    """Normalize and split a PRNG key."""
    return jax.random.split(as_prng_key(key), num)


__all__ = ["as_prng_key", "split_key"]
