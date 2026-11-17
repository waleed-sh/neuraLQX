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


"""Small predicates for modern JAX array sharding."""

from __future__ import annotations

from typing import Any

import jax


def with_weak_type(value: jax.Array, weak_type: bool) -> jax.Array:
    """Restore scalar-promotion semantics without moving an array to host memory.

    The public JAX dtype conversion clears this flag. The conversion primitive
    exposes it explicitly, preserving scan carry types after array restoration.
    """
    if bool(getattr(value, "weak_type", False)) == bool(weak_type):
        return value
    from jax._src.lax.lax import _convert_element_type

    return _convert_element_type(value, value.dtype, weak_type=bool(weak_type))


def place_global_array(value: jax.Array, sharding: jax.sharding.Sharding) -> jax.Array:
    """Place a complete local array or an existing global array on ``sharding``.

    For a local array and a mesh spanning multiple processes, each process
    supplies an array with the global shape. Only addressable slices are copied
    to its devices, and replicated slices must agree across processes. Existing
    global arrays use normal JAX resharding. Local and traced execution retain
    the ordinary ``device_put`` path.
    """
    if (
        isinstance(value, jax.core.Tracer)
        or sharding.is_fully_addressable
        or not value.is_fully_addressable
    ):
        return jax.device_put(value, sharding)
    result = jax.make_array_from_callback(
        value.shape, sharding, lambda index: value[index], dtype=value.dtype
    )
    return with_weak_type(result, getattr(value, "weak_type", False))


def is_jax_array(value: Any) -> bool:
    """Return whether ``value`` is a JAX array object."""
    return isinstance(value, jax.Array)


def is_sharded_array(value: Any) -> bool:
    """Return whether ``value`` is partitioned or replicated over many devices."""
    if not isinstance(value, jax.Array):
        return False
    try:
        return len(value.sharding.device_set) > 1
    except AttributeError:
        return False


def is_distributed_array(value: Any) -> bool:
    """Return whether a JAX array spans non-local processes."""
    if not isinstance(value, jax.Array):
        return False
    return bool(is_sharded_array(value) and not value.is_fully_addressable)


__all__ = [
    "is_distributed_array",
    "is_jax_array",
    "is_sharded_array",
    "place_global_array",
    "with_weak_type",
]
