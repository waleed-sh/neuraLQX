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


"""Weight extraction and transfer without replacing a model's definition."""

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.jax.sharding.arrays import place_global_array
from neuralqx.jax.sharding.arrays import with_weak_type

from .factory import as_framework


def model_parameters(source, *, variables=None, parameters=None):
    """Extract existing weights without initializing or evaluating a network.

    NNX modules own their variables. Unbound Linen modules, functions and
    custom framework adapters instead need an explicit parameter or variable
    tree. Bound Linen modules expose the variables held by their scope.
    """
    if parameters is not None and variables is not None:
        raise ValueError("Pass either parameters or variables, not both.")
    if parameters is not None:
        return parameters

    from flax import linen
    from flax import nnx

    if isinstance(source, nnx.Module):
        if variables is None:
            return nnx.state(source, nnx.Param)
        # Extraction must not apply the current global placement policy to a
        # donor network before the destination has validated its parameters.
        from .flax import FlaxNNXFramework

        return FlaxNNXFramework(None).parameters_from_variables(variables)
    if (
        variables is None
        and isinstance(source, linen.Module)
        and source.scope is not None
    ):
        variables = source.variables
    framework, _ = as_framework(source)
    if variables is None:
        raise ValueError(
            "This model does not own initialized weights. Pass variables=... or "
            "parameters=..., or pass a variational state containing its trained weights."
        )
    return framework.parameters_from_variables(variables)


def _parameter_structure(definition):
    """Compare parameter containers while keeping NNX metadata destination-owned."""
    data = definition.node_data()
    if data is None:
        return None
    node_type, metadata = data
    from flax import nnx
    from flax.core import FrozenDict

    if node_type in (dict, FrozenDict):
        node_type, metadata = dict, tuple(metadata)
    elif issubclass(node_type, nnx.Variable):
        # Axis annotations and variable hooks belong to the receiving model.
        # The variable class and the structure of its value must still match.
        metadata = None
    return (
        node_type,
        metadata,
        tuple(_parameter_structure(c) for c in definition.children()),
    )


def _place_like(value, template):
    """Retain target placement, including a distributed-to-local transfer.

    A local destination needs the complete parameter, not just the donor's
    local shard. JAX can first replicate it across the donor mesh and expose
    the local device buffers without copying the tensor into Python memory.
    """
    sharding = getattr(template, "sharding", None)
    if sharding is not None:
        if sharding.is_fully_addressable and not value.is_fully_addressable:
            from jax.experimental import multihost_utils

            from jax.sharding import Mesh
            from jax.sharding import PartitionSpec

            mesh = getattr(value.sharding, "mesh", None)
            if mesh is None:
                devices = sorted(
                    value.sharding.device_set,
                    key=lambda device: (device.process_index, device.id),
                )
                mesh = Mesh(np.asarray(devices), ("weight_transfer",))
            value = multihost_utils.global_array_to_host_local_array(
                value, mesh, PartitionSpec()
            )
        value = place_global_array(value, sharding)
    weak_type = getattr(template, "weak_type", None)
    if weak_type is None:
        weak_type = getattr(jax.typeof(template), "weak_type", False)
    return with_weak_type(value, weak_type)


def transfer_parameter_tree(source, target):
    """Rebuild a target parameter tree using compatible source array values.

    Parameter paths, container types, shapes and dtypes must match. Ordinary
    dictionaries and Flax frozen dictionaries are interchangeable. All leaves
    are checked before any device transfers. NNX variable metadata, scalar
    promotion flags and array placements come from the target. Static model
    code and configuration are outside this parameter-layout check.

    Device arrays stay on devices. Resharding uses each target leaf's existing
    placement, including global arrays spanning JAX processes. In distributed
    execution every process must participate with the same global parameter
    layout. Fully addressable donor arrays must contain the same complete
    values on every participating process.
    """
    source_pairs, source_tree = jax.tree_util.tree_flatten_with_path(source)
    target_pairs, target_tree = jax.tree_util.tree_flatten_with_path(target)
    source_paths = [jax.tree_util.keystr(path) for path, _ in source_pairs]
    target_paths = [jax.tree_util.keystr(path) for path, _ in target_pairs]
    if source_paths != target_paths or _parameter_structure(
        source_tree
    ) != _parameter_structure(target_tree):
        missing = sorted(set(target_paths) - set(source_paths))
        extra = sorted(set(source_paths) - set(target_paths))
        detail = (
            f" Missing: {missing[:5]}. Unexpected: {extra[:5]}."
            if missing or extra
            else ""
        )
        raise ValueError(
            "Source and destination parameter structures do not match." + detail
        )

    for path, (_, value), (_, template) in zip(
        target_paths, source_pairs, target_pairs, strict=True
    ):
        if isinstance(value, (jax.ShapeDtypeStruct, jax.core.Tracer)):
            raise TypeError(
                f"Source parameter {path} must contain concrete array values."
            )
        value_spec = value if hasattr(value, "dtype") else np.asarray(value)
        target_spec = template if hasattr(template, "dtype") else np.asarray(template)
        if value_spec.shape != target_spec.shape:
            raise ValueError(
                f"Parameter {path} has source shape {value_spec.shape}, "
                f"expected {target_spec.shape}."
            )
        if value_spec.dtype != target_spec.dtype:
            raise ValueError(
                f"Parameter {path} has source dtype {value_spec.dtype}, "
                f"expected {target_spec.dtype}. No implicit casting is performed."
            )
        if jax.dtypes.canonicalize_dtype(value_spec.dtype) != value_spec.dtype:
            raise ValueError(
                f"Transferring parameter {path} with dtype {value_spec.dtype} would lose "
                "precision. Enable JAX 64-bit support before transferring these weights."
            )

    leaves = []
    for (_, value), (_, template) in zip(source_pairs, target_pairs, strict=True):
        if isinstance(value, np.ndarray):
            # A subsequent mutation of a user's NumPy array must not change
            # the transferred weights through a CPU buffer alias.
            value = value.copy()
        leaves.append(_place_like(jnp.asarray(value), template))
    return target_tree.unflatten(leaves)
