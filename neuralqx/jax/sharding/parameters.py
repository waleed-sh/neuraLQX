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


"""Parameter-tree sharding helpers.

The current neuraLQX neural-network layer does not yet attach fine-grained
partition metadata to individual parameters. Until `neuralqx.nn` grows sharded
layers, the safe default is to replicate parameter trees over the configured
mesh when parameter sharding is enabled.
"""

from __future__ import annotations

from typing import Any

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from neuralqx.config import config
from neuralqx.jax.mesh import create_device_mesh


def parameter_sharding_enabled() -> bool:
    """Return whether parameter sharding is enabled in the current config."""
    return bool(config.get("parameter_sharding_enabled"))


def parameter_mesh():
    """Return the active mesh for parameter trees, or ``None`` when disabled."""
    if not parameter_sharding_enabled():
        return None
    return create_device_mesh()


def replicated_parameter_sharding() -> NamedSharding | None:
    """Return a replicated sharding over the active mesh for parameter leaves."""
    mesh = parameter_mesh()
    if mesh is None:
        return None
    return NamedSharding(mesh, PartitionSpec())


def place_parameters(tree: Any) -> Any:
    """Place a parameter pytree on the configured replicated parameter sharding."""
    sharding = replicated_parameter_sharding()
    if sharding is None:
        return tree

    def place_leaf(value: Any) -> Any:
        if isinstance(value, jax.Array):
            return jax.device_put(value, sharding)
        return value

    return jax.tree_util.tree_map(place_leaf, tree)


def constrain_parameters(tree: Any) -> Any:
    """Apply a replicated sharding constraint to parameter-array leaves."""
    sharding = replicated_parameter_sharding()
    if sharding is None:
        return tree

    def constrain_leaf(value: Any) -> Any:
        if isinstance(value, jax.Array):
            return jax.lax.with_sharding_constraint(value, sharding)
        return value

    return jax.tree_util.tree_map(constrain_leaf, tree)


__all__ = [
    "constrain_parameters",
    "parameter_mesh",
    "parameter_sharding_enabled",
    "place_parameters",
    "replicated_parameter_sharding",
]
