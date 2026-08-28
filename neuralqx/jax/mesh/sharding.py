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


"""NamedSharding helpers backed by configured neuraLQX meshes."""

from __future__ import annotations

from collections.abc import Sequence

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from .config import active_logical_axes
from .config import logical_axis_name
from .create import create_device_mesh


def named_sharding(
    spec: PartitionSpec,
    *,
    active_axes: Sequence[str] | None = None,
) -> NamedSharding | None:
    """Build a ``NamedSharding`` on the configured neuraLQX mesh."""
    mesh = create_device_mesh(active_axes=active_axes)
    if mesh is None:
        return None
    return NamedSharding(mesh, spec)


def operator_partition_axis_names() -> tuple[str, ...]:
    """Return configured mesh axis names used for operator batch work."""
    axes = active_logical_axes()
    logical = tuple(axis for axis in ("samples", "operators") if axis in axes)
    return tuple(logical_axis_name(axis) for axis in logical)


def operator_batch_sharding() -> NamedSharding | None:
    """Return sharding for rank-2 operator batches ``(batch, hilbert_size)``."""
    axis_names = operator_partition_axis_names()
    if not axis_names:
        return None
    return named_sharding(PartitionSpec(axis_names, None))


__all__ = [
    "named_sharding",
    "operator_batch_sharding",
    "operator_partition_axis_names",
]
