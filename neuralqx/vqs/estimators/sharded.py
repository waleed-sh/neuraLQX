from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import RLock

import jax.numpy as jnp
import numpy as np

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from neuralqx.utils.errors import NeuralqxValueError

_operators = OrderedDict()
_operator_lock = RLock()
_executors = {}


def _device_executor(size):
    with _operator_lock:
        if size not in _executors:
            _executors[size] = ThreadPoolExecutor(
                max_workers=size, thread_name_prefix="nqx-connections"
            )
        return _executors[size]


def _parallel_preparation(shards):
    return len(shards) > 1 and all(s.device.platform == "gpu" for s in shards)


def sample_layout(samples):
    if isinstance(samples, jax.core.Tracer) or not isinstance(samples, jax.Array):
        return None
    layout = samples.sharding
    if not isinstance(layout, NamedSharding) or layout.is_fully_replicated:
        return None
    if not layout.spec or layout.spec[0] is None or any(layout.spec[1:]):
        return None
    axes = layout.spec[0] if isinstance(layout.spec[0], tuple) else (layout.spec[0],)
    if np.prod([layout.mesh.shape[name] for name in axes]) != layout.mesh.size:
        return None
    return layout


def local_model_layout(samples, variables):
    layout = sample_layout(samples)
    if layout is None:
        return None
    for leaf in jax.tree_util.tree_leaves(variables):
        if isinstance(leaf, jax.core.Tracer):
            return None
        if isinstance(leaf, jax.Array) and len(leaf.devices()) > 1:
            if not leaf.sharding.is_fully_replicated:
                return None
    return layout


def _on_device(operator, device):
    key = (id(operator), device)
    with _operator_lock:
        found = _operators.get(key)
        if found is not None and found[0] is operator:
            _operators.move_to_end(key)
            return found[1]

        def local_leaf(value):
            if isinstance(value, jax.Array) and not value.is_fully_addressable:
                if not value.sharding.is_fully_replicated:
                    raise NeuralqxValueError(
                        "Sample-local connection preparation requires replicated operator arrays."
                    )
                # Restored coefficients may be globally replicated. Use their
                # local replica without requesting data from another process.
                for shard in value.addressable_shards:
                    if shard.device == device:
                        return shard.data
                value = value.addressable_shards[0].data
            return jax.device_put(value, device)

        placed = jax.tree.map(local_leaf, operator)
        _operators[key] = (operator, placed)
        while len(_operators) > 32:
            _operators.popitem(last=False)
        return placed


@partial(jax.jit, static_argnames=("capacity",))
def _pad_connections(targets, amplitudes, owners, offset, *, capacity):
    padding = capacity - len(amplitudes)
    return (
        jnp.pad(targets, ((0, padding), (0, 0))),
        jnp.pad(amplitudes, (0, padding)),
        jnp.pad(owners, (0, padding)) + offset,
    )


def prepare_sharded_connections(operator, samples, prepare, **kwargs):
    layout = sample_layout(samples)
    if layout is None:
        return None
    shards = samples.addressable_shards
    if not shards:
        return None

    def local(shard):
        # Each call sees a genuine single-device array, retaining eager tensor
        # reuse and compaction without gathering samples from other devices.
        with jax.default_device(shard.device):
            parent = _on_device(operator, shard.device)
            targets, amplitudes, owners = prepare(parent, shard.data, **kwargs)
            if owners is None:
                width = amplitudes.shape[-1]
                owners = jnp.repeat(jnp.arange(len(shard.data)), width)
                targets = targets.reshape((-1, samples.shape[-1]))
                amplitudes = amplitudes.reshape(-1)
            return shard, targets, amplitudes, owners

    if _parallel_preparation(shards):
        rows = list(_device_executor(len(shards)).map(local, shards))
    else:
        rows = [local(s) for s in shards]
    local_capacity = max(len(row[2]) for row in rows)
    local_size = shards[0].data.shape[0]
    partitions = samples.shape[0] // local_size
    output_layout = NamedSharding(layout.mesh, P(layout.spec[0]))
    capacity = local_capacity
    if not samples.is_fully_addressable:
        # Only a capacity is reduced across hosts. Connections never cross the
        # process boundary, even when sparsity differs between sample shards.
        counts = jax.make_array_from_single_device_arrays(
            (partitions,),
            output_layout,
            [
                jax.device_put(np.array([local_capacity], np.int64), s.device)
                for s in shards
            ],
        )
        capacity = int(jnp.max(counts))
    buffers = [[], [], []]
    for shard, targets, amplitudes, owners in rows:
        with jax.default_device(shard.device):
            padded = _pad_connections(
                targets,
                amplitudes,
                owners,
                shard.index[0].start or 0,
                capacity=capacity,
            )
        for result, array in zip(buffers, padded, strict=True):
            result.append(array)
    return tuple(
        jax.make_array_from_single_device_arrays(
            (capacity * partitions, *arrays[0].shape[1:]), output_layout, arrays
        )
        for arrays in buffers
    )
