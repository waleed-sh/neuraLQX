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


"""Sharding utilities for padded connected-component arrays."""

from __future__ import annotations

from typing import Any

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from neuralqx.jax.mesh import operator_batch_sharding


def connected_output_shardings(
    input_sharding: Any,
    input_ndim: int,
) -> tuple[NamedSharding | None, NamedSharding | None]:
    """Derive output shardings for ``get_conn_padded`` outputs.

    The input has shape ``(*batch, hilbert_size)``. Connected states have
    shape ``(*batch, max_conn_size, hilbert_size)`` and matrix elements have
    shape ``(*batch, max_conn_size)``. We preserve batch-axis sharding and keep
    both the connection and Hilbert-site axes unpartitioned. If the input site
    axis itself is sharded, ``None`` is returned because most operator kernels
    require a complete local basis state.
    """
    if not isinstance(input_sharding, NamedSharding):
        return None, None

    spec = tuple(input_sharding.spec)
    if len(spec) < input_ndim:
        spec = (None,) * (input_ndim - len(spec)) + spec
    if len(spec) != input_ndim:
        return None, None
    if spec[-1] is not None:
        return None, None

    batch_spec = spec[:-1]
    xp_spec = PartitionSpec(*batch_spec, None, None)
    mel_spec = PartitionSpec(*batch_spec, None)
    mesh = input_sharding.mesh
    return NamedSharding(mesh, xp_spec), NamedSharding(mesh, mel_spec)


def place_connected_outputs(
    states: Any,
    x_primes: jax.Array,
    mels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Place connected outputs on shardings derived from ``states`` when possible."""
    if not isinstance(states, jax.Array):
        return x_primes, mels
    try:
        state_sharding = states.sharding
    except AttributeError:
        return x_primes, mels
    xp_sharding, mel_sharding = connected_output_shardings(state_sharding, states.ndim)
    if xp_sharding is None or mel_sharding is None:
        return x_primes, mels
    return jax.device_put(x_primes, xp_sharding), jax.device_put(mels, mel_sharding)


def place_operator_batch(states: Any) -> jax.Array:
    """Place rank-2 operator input batches on the configured operator mesh."""
    if not isinstance(states, jax.Array):
        return states
    try:
        state_sharding = states.sharding
    except AttributeError:
        return states
    if isinstance(state_sharding, NamedSharding):
        return states
    sharding = operator_batch_sharding()
    if sharding is None:
        return states
    partitions = _partition_count(sharding.spec, sharding.mesh.shape)
    if partitions <= 1 or states.shape[0] % partitions != 0:
        return states
    return jax.device_put(states, sharding)


def _partition_count(spec: PartitionSpec, mesh_shape: dict[str, int]) -> int:
    axis = spec[0]
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        product = 1
        for name in axis:
            product *= int(mesh_shape[name])
        return product
    return int(mesh_shape[axis])


__all__ = [
    "connected_output_shardings",
    "place_connected_outputs",
    "place_operator_batch",
]
