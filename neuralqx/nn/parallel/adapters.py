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


"""Public customization for networks using Flax or pure JAX callables."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jax.sharding import PartitionSpec


@dataclass(frozen=True)
class TensorParallel:
    """Supply tensor partition rules for a custom NNX, Linen, or JAX model.

    Pass this adapter as ``model`` to ``VariationalState``. Enable network
    parallelism before importing neuraLQX, for example with
    ``NEURALQX_SHARDING=tensor`` or ``samples,operators,tensor``. Set
    ``NEURALQX_MESH_TENSOR_AXIS_SIZE`` to choose the cooperating group size.

    ``parameter_rules`` maps slash-separated paths within the parameter tree,
    without an outer ``params/`` prefix, or glob patterns
    to JAX ``PartitionSpec`` objects. For example, ``{"dense/kernel":
    PartitionSpec(None, "tensor"), "head/kernel": PartitionSpec("tensor", None)}``
    partitions a hidden projection by output features and its head by input
    features. Rules override architecture defaults. Unmatched parameters keep
    their defaults or remain replicated. Dimensions must divide the tensor
    group size exactly. Scalars and small output biases can remain replicated.

    The adapter uses ordinary JAX differentiation and does not change the
    wavefunction. Custom models must remain deterministic and independent of
    other batch entries. Normalization must use global feature dimensions.

    Args:
        model: An initialized NNX model, Linen module, or callable accepting
            ``(parameters, configurations)``.
        parameter_rules: Explicit parameter paths and their tensor layouts.
        init_fn: Initializer for a callable model, if variables are not supplied.
        expects_variables: A callable accepts the full variable mapping when true.
            ``None`` inherits the framework or variational-state setting.
    """

    model: Any
    parameter_rules: Mapping[str, PartitionSpec]
    init_fn: Any = None
    expects_variables: bool | None = None


__all__ = ["TensorParallel"]
