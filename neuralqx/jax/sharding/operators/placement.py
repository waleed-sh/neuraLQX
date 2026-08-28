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


"""Placement helpers for independent operator-term batches."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from neuralqx.config import config
from neuralqx.jax.mesh import create_device_mesh
from neuralqx.jax.mesh import logical_axis_name


def operator_terms_shardings(
    *,
    x_primes_ndim: int,
    mels_ndim: int,
) -> tuple[NamedSharding | None, NamedSharding | None]:
    """Return shardings that partition only the leading operator-term axis."""
    if not bool(config.get("operator_sharding_enabled")):
        return None, None
    mesh = create_device_mesh(
        active_axes=("operators",),
        axis_sizes={"operators": int(jax.device_count())},
    )
    if mesh is None:
        return None, None
    axis_name = logical_axis_name("operators")
    if axis_name not in mesh.axis_names:
        return None, None
    xp_spec = PartitionSpec(axis_name, *((None,) * (x_primes_ndim - 1)))
    mel_spec = PartitionSpec(axis_name, *((None,) * (mels_ndim - 1)))
    return NamedSharding(mesh, xp_spec), NamedSharding(mesh, mel_spec)


def place_operator_terms(
    x_primes: Any,
    mels: Any,
    *,
    enabled: bool | None,
) -> tuple[jax.Array, jax.Array]:
    """Place stacked operator-term connections on the operator mesh when enabled."""
    xp = jnp.asarray(x_primes)
    mel = jnp.asarray(mels)
    if enabled is False:
        return xp, mel
    if enabled is None and not bool(config.get("operator_sharding_enabled")):
        return xp, mel

    xp_sharding, mel_sharding = operator_terms_shardings(
        x_primes_ndim=xp.ndim,
        mels_ndim=mel.ndim,
    )
    if xp_sharding is None or mel_sharding is None:
        return xp, mel

    xp, mel = _pad_terms_for_sharding(xp, mel, xp_sharding)
    return jax.device_put(xp, xp_sharding), jax.device_put(mel, mel_sharding)


def _pad_terms_for_sharding(
    x_primes: jax.Array,
    mels: jax.Array,
    sharding: NamedSharding,
) -> tuple[jax.Array, jax.Array]:
    partitions = operator_axis_partition_count(sharding.spec, sharding.mesh.shape)
    if partitions <= 1:
        return x_primes, mels
    n_terms = int(x_primes.shape[0])
    remainder = n_terms % partitions
    if remainder == 0:
        return x_primes, mels

    pad = partitions - remainder
    xp_pad_width = [(0, pad)] + [(0, 0)] * (x_primes.ndim - 1)
    mel_pad_width = [(0, pad)] + [(0, 0)] * (mels.ndim - 1)
    x_primes = jnp.pad(x_primes, xp_pad_width, mode="edge")
    mels = jnp.pad(mels, mel_pad_width, mode="constant", constant_values=0)
    return x_primes, mels


def operator_axis_partition_count(
    spec: PartitionSpec,
    mesh_shape: dict[str, int],
) -> int:
    """Return the device partition count for the leading operator axis."""
    axis = spec[0]
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        product = 1
        for name in axis:
            product *= int(mesh_shape[name])
        return product
    return int(mesh_shape[axis])


_partition_count = operator_axis_partition_count


__all__ = [
    "operator_axis_partition_count",
    "operator_terms_shardings",
    "place_operator_terms",
]
