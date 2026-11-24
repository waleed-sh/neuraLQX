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


"""Topology-aware meshes with a persistent tensor group across VMC phases."""

from functools import lru_cache
from math import isqrt
from math import prod

import jax
from jax.sharding import PartitionSpec as P
from neuralqx.config import config
from neuralqx.utils.errors import NeuralqxValueError

from .config import tensor_axis_name


def _factor(value, bound):
    return next(n for n in range(max(1, int(bound)), 0, -1) if value % n == 0)


@lru_cache(maxsize=32)
def _mesh(devices, names, samples, operators, sizes):
    count = len(devices)
    s, o, t = sizes
    if len(set(names)) != 3:
        raise NeuralqxValueError(
            "Sample, operator, and tensor mesh names must be distinct."
        )
    for name, enabled, size in zip(
        names, (samples, operators, True), sizes, strict=True
    ):
        if size is not None and (size <= 0 or (not enabled and size != 1)):
            raise NeuralqxValueError(
                f"Mesh axis {name!r} has invalid size {size}. Disabled work axes must have size 1."
            )
    explicit = prod(size for size in sizes if size is not None)
    if not count or count % explicit:
        raise NeuralqxValueError(
            "Explicit tensor mesh sizes must divide the global device count."
        )
    if t is None:
        remaining = count // ((s or 1) * (o or 1))
        dimensions = 1 + int(samples and s is None) + int(operators and o is None)
        bound = remaining if dimensions == 1 else round(remaining ** (1 / dimensions))
        t = _factor(remaining, bound)
        if t == 1 and remaining > 1:
            t = next(n for n in range(2, remaining + 1) if remaining % n == 0)
    if count % t:
        raise NeuralqxValueError(
            "mesh_tensor_axis_size must divide the global device count."
        )
    remaining = count // t
    if not samples:
        s = 1
    if not operators:
        o = 1
    if s is None and o is None:
        o = _factor(remaining, isqrt(remaining))
        s = remaining // o
    elif s is None:
        s = remaining // o
    elif o is None:
        o = remaining // s
    if s * o * t != count:
        raise NeuralqxValueError(
            f"Sample/operator/tensor mesh {(s, o, t)} must use all {count} devices. "
            "Enable an independent work axis or increase the tensor partition count."
        )
    return jax.make_mesh(
        (s, o, t), names, devices=devices, axis_types=(jax.sharding.AxisType.Auto,) * 3
    )


def tensor_mesh():
    """Return the shared sample/operator/tensor mesh.

    The innermost tensor group remains fixed throughout a VMC step. Sampling
    can use both outer axes for independent chains. Operator evaluation uses
    them separately. JAX chooses a topology-aware ordering of the devices.
    """
    return _mesh(
        tuple(jax.devices()),
        (
            str(config.get("mesh_sample_axis_name")),
            str(config.get("mesh_operator_axis_name")),
            tensor_axis_name(),
        ),
        bool(config.get("sample_sharding_enabled")),
        bool(config.get("operator_sharding_enabled")),
        (
            config.get("mesh_sample_axis_size"),
            config.get("mesh_operator_axis_size"),
            config.get("mesh_tensor_axis_size"),
        ),
    )


def tensor_partition_count() -> int:
    """Number of devices cooperating on one model evaluation."""
    return int(tensor_mesh().shape[tensor_axis_name()])


def sample_spec(ndim=1):
    """Partition distinct samples, replicating each across its tensor group."""
    if not config.get("sample_sharding_enabled"):
        return P()
    mesh = tensor_mesh()
    axes = tuple(name for name in mesh.axis_names if name != tensor_axis_name())
    return P(axes, *((None,) * (ndim - 1)))


__all__ = ["tensor_mesh", "tensor_partition_count"]
