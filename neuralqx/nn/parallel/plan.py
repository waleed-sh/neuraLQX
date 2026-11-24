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


"""Validated immutable parameter layouts shared by framework adapters."""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import lru_cache
from math import prod

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from neuralqx.jax.sharding.arrays import place_global_array
from neuralqx.jax.tensor_parallel import tensor_axis_name
from neuralqx.jax.tensor_parallel import tensor_mesh
from neuralqx.jax.tensor_parallel import tensor_partition_count
from neuralqx.utils.errors import NeuralqxValueError


def parameter_path(path):
    """Normalize JAX key paths, excluding Flax's variable value wrapper."""
    return "/".join(
        str(
            k.key
            if isinstance(k, jax.tree_util.DictKey)
            else k.idx if isinstance(k, jax.tree_util.SequenceKey) else k.name
        )
        for k in path
        if not isinstance(k, jax.tree_util.GetAttrKey)
        or k.name not in {"value", "raw_value"}
    )


@dataclass(frozen=True)
class ParameterPartition:
    """One parameter's global shape and tensor partition specification.

    ``path`` is a slash-separated framework parameter path. ``spec`` names
    only the tensor axis. Independent work axes replicate this parameter.
    """

    path: str
    shape: tuple[int, ...]
    spec: P


@dataclass(frozen=True)
class TensorParallelPlan:
    """Validated tensor layouts for one parameter pytree.

    Plans preserve global array shapes. Parameters are replicated over sample
    and operator groups and partitioned within each cooperating tensor group.
    ``parameters`` exposes every chosen layout, including replicated leaves.
    No parameter padding, truncation, or numerical approximation is performed.
    """

    parameters: tuple[ParameterPartition, ...]
    """Per-leaf decisions in the parameter tree's flattening order."""
    treedef: object
    """JAX tree definition needed to reconstruct parameter-layout pytrees."""
    partitions: int
    """Number of cooperating tensor workers used when validating the plan."""

    @property
    def specs(self):
        """A partition-specification pytree matching the model parameters."""
        return self.treedef.unflatten([entry.spec for entry in self.parameters])

    @property
    def sharded_parameter_count(self):
        """Number of scalar parameters partitioned over the tensor axis."""
        if self.partitions <= 1:
            return 0
        return sum(prod(entry.shape) for entry in self.parameters if any(entry.spec))

    def place(self, parameters):
        """Place global parameter arrays without changing their values."""
        mesh = tensor_mesh()
        return jax.tree.map(
            lambda x, spec: place_parameter(x, NamedSharding(mesh, spec)),
            parameters,
            self.specs,
        )

    def constrain(self, parameters):
        """Apply layout constraints during tracing, retaining manual work axes."""
        active = jax.sharding.get_abstract_mesh()
        use_context = tensor_axis_name() in active.axis_names
        mesh = None if use_context else tensor_mesh()
        return jax.tree.map(
            lambda x, spec: jax.lax.with_sharding_constraint(
                x, spec if use_context else NamedSharding(mesh, spec)
            ),
            parameters,
            self.specs,
        )


def plan_parameters(parameters, rules=(), *, generic=False, required_patterns=()):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(parameters)
    shapes = tuple((parameter_path(path), tuple(value.shape)) for path, value in leaves)
    return _plan(
        treedef,
        shapes,
        tuple(rules),
        generic,
        tuple(required_patterns),
        tensor_axis_name(),
        tensor_partition_count(),
    )


def place_parameter(value, sharding):
    """Retain abstract restore templates without allocating placeholder arrays."""
    if isinstance(value, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=sharding, weak_type=value.weak_type
        )
    if isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer):
        if value.sharding == sharding:
            return value
    return place_global_array(value, sharding)


@lru_cache(maxsize=128)
def _plan(treedef, shapes, rules, generic, required_patterns, axis, count):
    for pattern, candidate in rules:
        if not isinstance(pattern, str) or not isinstance(candidate, P):
            raise NeuralqxValueError(
                "Tensor rules must map string paths to PartitionSpec objects."
            )
    for pattern in required_patterns:
        if not any(fnmatchcase(path, pattern) for path, _ in shapes):
            raise NeuralqxValueError(
                f"Tensor parameter rule {pattern!r} matches no parameter."
            )
    entries = []
    for path, shape in shapes:
        spec = None
        for pattern, candidate in rules:
            if fnmatchcase(path, pattern):
                spec = candidate
        if spec is None and generic and len(shape) >= 2:
            candidates = (
                (len(shape) - 1, *range(len(shape) - 1))
                if shape[-1] > 2
                else tuple(range(len(shape)))
            )
            dim = next(
                (i for i in candidates if shape[i] >= count and shape[i] % count == 0),
                None,
            )
            if dim is not None:
                spec = P(*(axis if i == dim else None for i in range(len(shape))))
        spec = P() if spec is None else P(*(axis if x == "tensor" else x for x in spec))
        if len(spec) > len(shape) or sum(x is not None for x in spec) > 1:
            raise NeuralqxValueError(
                f"Invalid tensor partition specification {spec} for {path}: {shape}."
            )
        for dim, name in enumerate(spec):
            if name is not None and (name != axis or shape[dim] % count):
                raise NeuralqxValueError(
                    f"Tensor partition {spec} does not divide parameter {path}: {shape} over {count} devices."
                )
        entries.append(ParameterPartition(path, shape, spec))
    plan = TensorParallelPlan(tuple(entries), treedef, count)
    if count > 1 and plan.sharded_parameter_count == 0:
        raise NeuralqxValueError(
            "No parameter can be tensor-partitioned with this plan. Reduce mesh_tensor_axis_size "
            "or supply TensorParallel parameter_rules for the custom model."
        )
    return plan


__all__ = ["ParameterPartition", "TensorParallelPlan"]
