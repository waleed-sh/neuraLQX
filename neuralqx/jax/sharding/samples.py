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


"""Sample-axis sharding helpers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from neuralqx.config import config
from neuralqx.jax.mesh import create_device_mesh
from neuralqx.jax.mesh import logical_axis_name
from neuralqx.jax.mesh import named_sharding

from .connected import _partition_count


def sample_sharding_enabled() -> bool:
    """Return whether sample/chains should be sharded across devices."""
    return bool(config.get("sample_sharding_enabled"))


def sample_axis_name() -> str:
    """Return the configured mesh axis name for sample sharding."""
    return logical_axis_name("samples")


def sample_mesh():
    """Return the sample-only mesh used by sampler execution."""
    if not sample_sharding_enabled():
        return None
    return create_device_mesh(active_axes=("samples",))


def sample_partition_spec(ndim: int = 2) -> PartitionSpec:
    """Return the leading-sample-axis partition spec for an array rank."""
    if ndim <= 0:
        raise ValueError("sample sharding requires an array with at least one axis.")
    return PartitionSpec(sample_axis_name(), *((None,) * (int(ndim) - 1)))


def sample_batch_sharding(ndim: int = 2) -> NamedSharding | None:
    """Return sharding for arrays whose leading axis is samples/chains."""
    if not sample_sharding_enabled():
        return None
    return named_sharding(
        sample_partition_spec(ndim),
        active_axes=("samples",),
    )


def sample_partition_count(ndim: int = 1) -> int:
    """Return the number of partitions used along the leading sample axis."""
    sharding = sample_batch_sharding(ndim)
    if sharding is None:
        return 1
    return _partition_count(sharding.spec, sharding.mesh.shape)


def can_shard_samples(leading_size: int, *, ndim: int = 1) -> bool:
    """Return whether a leading-axis batch can be explicitly sample-sharded."""
    partitions = sample_partition_count(ndim)
    return bool(partitions > 1 and int(leading_size) % partitions == 0)


def place_sample_batch(samples: Any) -> Any:
    """Place a leading-sample-axis array on the configured sample mesh when useful."""
    if not isinstance(samples, jax.Array):
        return samples
    sharding = sample_batch_sharding(samples.ndim)
    if sharding is None:
        return samples
    partitions = _partition_count(sharding.spec, sharding.mesh.shape)
    if partitions <= 1 or samples.shape[0] % partitions != 0:
        return samples
    return jax.device_put(samples, sharding)


def make_sample_zeros(shape: tuple[int, ...], dtype: Any) -> jax.Array:
    """Create a zero array on sample sharding without materializing all shards."""
    out_shape = tuple(int(dim) for dim in shape)
    sharding = sample_batch_sharding(len(out_shape))
    if sharding is None:
        return jnp.zeros(out_shape, dtype=dtype)
    partitions = _partition_count(sharding.spec, sharding.mesh.shape)
    if partitions <= 1 or out_shape[0] % partitions != 0:
        return jnp.zeros(out_shape, dtype=dtype)
    if hasattr(jax, "make_array_from_callback"):
        np_dtype = np.dtype(dtype)
        return jax.make_array_from_callback(
            out_shape,
            sharding,
            lambda index: np.zeros(_index_shape(index, out_shape), dtype=np_dtype),
            dtype=dtype,
        )
    return jax.device_put(jnp.zeros(out_shape, dtype=dtype), sharding)


def shard_sample_keys(key: jax.Array, count: int) -> jax.Array:
    """Split ``key`` into a leading sample-sharded vector of keys."""
    return place_sample_batch(jax.random.split(key, int(count)))


def _index_shape(index: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    if index is None:
        return shape
    out = []
    for entry, dim in zip(index, shape, strict=True):
        if isinstance(entry, slice):
            start, stop, step = entry.indices(dim)
            if step == 1:
                out.append(max(stop - start, 0))
            else:
                out.append(len(range(start, stop, step)))
        elif entry is Ellipsis:
            out.append(dim)
    return tuple(out)


__all__ = [
    "can_shard_samples",
    "make_sample_zeros",
    "place_sample_batch",
    "sample_axis_name",
    "sample_batch_sharding",
    "sample_mesh",
    "sample_partition_count",
    "sample_partition_spec",
    "sample_sharding_enabled",
    "shard_sample_keys",
]
