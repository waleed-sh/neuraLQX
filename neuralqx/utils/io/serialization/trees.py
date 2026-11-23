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


"""Portable array leaves and template-based reconstruction of arbitrary pytrees."""

import hashlib

import numpy as np

import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from neuralqx.jax.sharding.arrays import place_global_array
from neuralqx.jax.sharding.arrays import with_weak_type

from ._diagnostics import _MISSING
from ._diagnostics import differences
from ._diagnostics import mismatch_report
from .collective import agree
from .collective import collective_call
from .errors import CheckpointError


def replicated_sharding():
    """Return a replicated placement spanning the currently initialized devices."""
    return NamedSharding(Mesh(np.asarray(jax.devices()), ("nqx_io",)), PartitionSpec())


def restore_weak_type(value, weak_type):
    """Restore JAX's scalar-promotion flag without copying a device array to host.

    JAX's public dtype conversion clears weak typing. Its conversion primitive
    exposes the required flag, which is part of an array's abstract value and
    can affect both numerical precision and scan carry compatibility.
    """
    return with_weak_type(value, weak_type)


def _structure(definition):
    data = definition.node_data()
    if data is None:
        return None
    cls, auxiliary = data
    return [
        f"{cls.__module__}.{cls.__qualname__}",
        _auxiliary(auxiliary),
        [_structure(c) for c in definition.children()],
    ]


def _auxiliary(value):
    if value is None or type(value) in (bool, str, int, float):
        return value
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_auxiliary(item) for item in value]]
    if isinstance(value, dict):
        return [
            "dict",
            [[_auxiliary(key), _auxiliary(item)] for key, item in value.items()],
        ]
    if isinstance(value, jax.sharding.Sharding):
        # Placement is selected by the restoration template, not old devices.
        return {"placement": True}
    if isinstance(value, type) or callable(value):
        return {
            "definition": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    from .registry import encode_config

    return encode_config(value)


def _describe(value, path):
    is_jax = isinstance(value, (jax.Array, jax.ShapeDtypeStruct))
    if not is_jax and not isinstance(
        value, (np.ndarray, np.generic, bool, int, float, complex)
    ):
        raise CheckpointError(
            f"Unsupported state leaf at {path}: {type(value).__name__}."
        )
    kind = (
        "jax"
        if is_jax
        else (
            "numpy"
            if isinstance(value, (np.ndarray, np.generic))
            else type(value).__name__
        )
    )
    if isinstance(value, np.generic):
        kind = "numpy_scalar"
    array = value if is_jax else np.asarray(value)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        if isinstance(value, jax.ShapeDtypeStruct):
            raise CheckpointError(
                "Use an actual typed RNG key as the restoration template."
            )
        raw = jax.random.key_data(value)
        return {
            "path": path,
            "kind": kind,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "key_impl": str(jax.random.key_impl(value)),
            "storage_shape": list(raw.shape),
            "storage_dtype": str(raw.dtype),
        }
    dtype = np.dtype(array.dtype)
    if dtype.hasobject or dtype.kind in "SUV":
        # JAX's extended floating types may use NumPy kind V without being opaque.
        if dtype.hasobject or not jax.dtypes.issubdtype(dtype, np.number):
            raise CheckpointError(f"Unsupported state dtype at {path}: {dtype}.")
    return {
        "path": path,
        "kind": kind,
        "weak_type": bool(getattr(array, "weak_type", False)),
        "shape": list(array.shape),
        "dtype": str(dtype),
        "storage_shape": list(array.shape),
        "storage_dtype": str(dtype),
    }


def describe_tree(tree):
    """Describe leaf paths, container types, exact dtypes and RNG implementations."""
    pairs, definition = jax.tree_util.tree_flatten_with_path(tree)
    descriptions = [
        _describe(value, jax.tree_util.keystr(path)) for path, value in pairs
    ]
    for item in descriptions:
        if 0 in item["storage_shape"]:
            # Orbax rejects zero-sized tensors. Their shape and dtype contain
            # all information, so the stored value is a one-byte placeholder.
            item.update(empty=True, storage_shape=[], storage_dtype="uint8")
    return {"structure": _structure(definition), "leaves": descriptions}


def prepare_arrays(tree):
    """Extract leaves without gathering globally sharded arrays onto a host.

    Typed RNG keys are stored as uint32 data plus their implementation name.
    Process-local replicated leaves are checked for agreement before being
    promoted to global arrays. Divergent independent per-process states must
    be represented as global arrays, or saved by independent JAX jobs, instead
    of silently discarding any rank.
    """
    schema = collective_call(lambda: describe_tree(tree), "state description")
    agree(schema, "state structure, shapes and dtypes")
    leaves = jax.tree.leaves(tree)
    arrays = {}
    local_hashes = {}

    def prepare():
        for index, (value, item) in enumerate(
            zip(leaves, schema["leaves"], strict=True)
        ):
            name = f"leaf_{index:08d}"
            if isinstance(value, jax.ShapeDtypeStruct):
                raise CheckpointError(
                    f"Cannot save a shape-only leaf at {item['path']}. Supply its array values."
                )
            if item.get("empty"):
                arrays[name] = np.zeros((), dtype=np.uint8)
                continue
            if item.get("key_impl"):
                value = jax.random.key_data(value)
            if not isinstance(value, jax.Array):
                value = np.asarray(value)
            if jax.process_count() > 1 and (
                not isinstance(value, jax.Array) or value.is_fully_addressable
            ):
                host = np.asarray(jax.device_get(value))
                local_hashes[name] = hashlib.sha256(host.tobytes(order="C")).hexdigest()
                if isinstance(value, jax.Array):
                    value = place_global_array(value, replicated_sharding())
            arrays[name] = value
        if not arrays:
            arrays["empty_tree"] = np.zeros((), dtype=np.uint8)

    collective_call(prepare, "array preparation")
    agree(local_hashes, "process-local replicated values")
    return schema, arrays


def restore_targets(template, schema):
    """Validate a template and specify each leaf's placement on the current mesh."""
    actual = describe_tree(template)
    if actual != schema:
        old = {v["path"]: v for v in schema["leaves"]}
        new = {v["path"]: v for v in actual["leaves"]}

        def details():
            for path in sorted(old.keys() | new.keys()):
                # Storage shapes and dtypes are derived from the logical leaf
                # metadata. Avoid repeating them unless they alone differ.
                fields = ("kind", "shape", "dtype", "weak_type", "key_impl")
                a = (
                    {k: old[path][k] for k in fields if k in old[path]}
                    if path in old
                    else _MISSING
                )
                b = (
                    {k: new[path][k] for k in fields if k in new[path]}
                    if path in new
                    else _MISSING
                )
                if a != b:
                    yield from differences(a, b, f"state{path}")
                elif old[path] != new[path]:
                    yield from differences(old[path], new[path], f"state{path}")
            if old == new:
                yield from differences(
                    schema["structure"], actual["structure"], "state.structure"
                )

        raise CheckpointError(
            mismatch_report(
                "Checkpoint and target state differ. Shapes, dtypes and structure must match exactly.",
                details(),
            )
        )
    targets = {}
    for index, (value, item) in enumerate(
        zip(jax.tree.leaves(template), schema["leaves"], strict=True)
    ):
        dtype = np.dtype(item["storage_dtype"])
        if item["kind"] == "jax":
            original_dtype = dtype if item.get("key_impl") else np.dtype(item["dtype"])
            if jax.dtypes.canonicalize_dtype(original_dtype) != original_dtype:
                raise CheckpointError(
                    f"Restoring {original_dtype} would lose precision. Enable jax_enable_x64 before loading."
                )
            raw = jax.random.key_data(value) if item.get("key_impl") else value
            sharding = getattr(raw, "sharding", None)
            if (
                sharding is None
                or (
                    jax.process_count() > 1
                    and isinstance(sharding, jax.sharding.SingleDeviceSharding)
                )
                or (
                    isinstance(raw, jax.Array)
                    and not raw.committed
                    and jax.device_count() > 1
                )
            ):
                sharding = replicated_sharding()
        else:
            sharding = None
        if item.get("empty"):
            sharding = None
        targets[f"leaf_{index:08d}"] = jax.ShapeDtypeStruct(
            tuple(item["storage_shape"]), dtype, sharding=sharding
        )
    if not targets:
        targets["empty_tree"] = jax.ShapeDtypeStruct((), np.uint8)
    return targets


def rebuild_tree(template, schema, arrays):
    """Restore original container types and typed RNG keys without mutating a target."""
    leaves = []
    template_leaves = jax.tree.leaves(template)
    for index, item in enumerate(schema["leaves"]):
        value = arrays[f"leaf_{index:08d}"]
        if (
            list(value.shape) != item["storage_shape"]
            or str(value.dtype) != item["storage_dtype"]
        ):
            raise CheckpointError(
                f"Restored array does not match its manifest at {item['path']}."
            )
        if item.get("empty"):
            if item.get("key_impl"):
                value = template_leaves[index]
            else:
                value = np.empty(tuple(item["shape"]), dtype=np.dtype(item["dtype"]))
                if item["kind"] == "jax":
                    sharding = (
                        getattr(template_leaves[index], "sharding", None)
                        or replicated_sharding()
                    )
                    value = jax.make_array_from_callback(
                        value.shape,
                        sharding,
                        lambda position, host=value: host[position],
                    )
                    value = restore_weak_type(value, item.get("weak_type", False))
        elif item.get("key_impl"):
            value = jax.random.wrap_key_data(value, impl=item["key_impl"])
        elif item["kind"] in ("bool", "int", "float", "complex"):
            value = np.asarray(value).item()
        elif item["kind"] == "numpy_scalar":
            value = np.asarray(value)[()]
        elif item["kind"] == "jax":
            value = restore_weak_type(value, item.get("weak_type", False))
        leaves.append(value)
    return jax.tree.structure(template).unflatten(leaves)
