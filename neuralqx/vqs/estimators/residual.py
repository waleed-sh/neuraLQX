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

r"""Positive residual estimators on output configurations reached by an action.

Let x be a source sample, D(x) its distinct reachable outputs in the selected
region, and y uniform in D(x). Define

    S(y) = sum_{z: A[y, E(z)] != 0} |psi(z)|^2.

Then |D(x)| |(A E psi)(y)|^2 / S(y), averaged over Born-distributed x and
the conditional output draw, equals the selected action norm divided by
||psi||^2. Indeed the sum of |psi(x)|^2 / S(y) over predecessors of y is
one. No marginal output density, second independent action estimate, or
expanded Gram matrix is needed. A source proposal other than Born adds the
usual |psi(x)|^2 / q(x) weight to the numerator and normalization.

The proposal is fixed by operator connectivity, so gradients differentiate
the numerator, S(y), and the Born weights while holding the sampled labels
fixed. Coalescing precedes every support decision. Duplicate paths therefore
interfere before either the action is squared or a degree is counted.
"""

from dataclasses import dataclass
from functools import partial
from numbers import Integral

import jax.numpy as jnp

import jax
from neuralqx.hilbert.local import LocalRange
from neuralqx.jax.streaming import apply_batched
from neuralqx.jax.streaming import apply_connected
from neuralqx.operator.connectivity import coalesce_connections

from .connectivity import _capacity
from .connectivity import prepare_connections
from .sharded import sample_layout


@dataclass(frozen=True)
class ResidualPlan:
    """Parameter-independent sparse rows at sampled output configurations.

    ``base_ids`` identifies the source amplitude used for each row's extracted
    diagonal. ``output_ids`` maps the original source/probe events onto those
    rows. Several events can share a row without changing their proposal
    factors or their positions in the sample batch. The other sparse arrays
    index unique network inputs and retain coherent matrix-element sums.

    Plans belong to one evaluation. They do not add a persistent operator
    cache or retain wavefunction values across parameter updates.
    """

    arrays: tuple
    n_outputs: int
    layout: object = None


def _sort_keys(states, ranges):
    """Pack arithmetic local labels into lossless lexicographic sort keys.

    Earlier sites occupy more significant bits. Integer ranges are converted
    to ascending local indices first. Floating labels, custom local domains,
    and sites requiring more than 32 bits keep the original label keys.
    This is an exact encoding, not a hash or a basis rank.
    """
    if (
        ranges is None
        or not jnp.issubdtype(states.dtype, jnp.integer)
        or any(
            not isinstance(start, Integral) or not isinstance(step, Integral)
            for start, step, _ in ranges
        )
        or max(((size - 1).bit_length() for _, _, size in ranges), default=0) > 32
    ):
        return states
    unsigned = jnp.dtype(f"uint{states.dtype.itemsize * 8}")
    words, remaining = [], 32
    word = jnp.zeros(len(states), jnp.uint32)
    for site, (start, step, size) in enumerate(ranges):
        width = (size - 1).bit_length()
        if remaining < width:
            words.append(word)
            word, remaining = jnp.zeros_like(word), 32
        if width:
            first = min(start, start + step * (size - 1))
            # Unsigned subtraction also handles domains spanning the signed
            # dtype boundary without overflowing the nonnegative distance.
            delta = states[:, site].astype(unsigned) - jnp.asarray(
                first, states.dtype
            ).astype(unsigned)
            stride = jnp.asarray(abs(step), unsigned)
            index = (delta // stride).astype(jnp.uint32)
            remaining -= width
            word = word | (index << remaining)
    return jnp.stack((*words, word), axis=1)


@partial(jax.jit, static_argnames=("ranges",))
def _coalesce_flat(states, mels, owners, origins, *, ranges):
    """Coalesce compact rows without a host count or a padded row expansion."""
    if not len(mels):
        return states, mels, owners
    # Operator padding need not contain valid labels. Give zero entries a
    # valid representative before encoding local indices into bounded words.
    states = jnp.where((mels != 0)[:, None], states, origins[owners])
    keys = _sort_keys(states, ranges)
    order = jnp.lexsort(
        (*(keys[:, i] for i in range(keys.shape[1] - 1, -1, -1)), owners)
    )
    x, m, o = states[order], mels[order], owners[order]
    keys = keys[order]
    start = jnp.concatenate(
        (jnp.ones(1, bool), (o[1:] != o[:-1]) | jnp.any(keys[1:] != keys[:-1], axis=1))
    )
    ids = jax.lax.associative_scan(jnp.maximum, jnp.where(start, jnp.arange(len(m)), 0))
    return x, jnp.zeros_like(m).at[ids].add(m), o


def _rows(operator, states):
    """Use compact operator caches when available and native padded rows otherwise."""
    compact = getattr(operator, "get_conn_compact", None)
    packed = None if compact is None else compact(states)
    if packed is not None:
        targets, mels, owners = packed
        spaces = getattr(operator.hilbert, "local_spaces", ())
        ranges = (
            tuple((space.start, space.step, space.local_size) for space in spaces)
            if len(spaces) == states.shape[1]
            and all(isinstance(s, LocalRange) for s in spaces)
            else None
        )
        return (
            *_coalesce_flat(
                targets, mels, owners.astype(jnp.int32), states, ranges=ranges
            ),
            True,
        )
    targets, mels = coalesce_connections(
        *operator.get_conn_padded(states, use_streaming=False)
    )
    owners = jnp.repeat(jnp.arange(len(states), dtype=jnp.int32), mels.shape[1])
    return targets.reshape((-1, states.shape[1])), mels.reshape(-1), owners, False


@partial(jax.jit, static_argnames=("n_outputs", "exhaustive"))
def _draw(
    rows, targets, mels, owners, key, offset, source_ids=None, *, n_outputs, exhaustive
):
    """Choose unique nonzero destinations uniformly within each source column."""
    active = mels != 0
    dtype = jnp.result_type(mels.real, jnp.float32)
    degree = jax.ops.segment_sum(active.astype(jnp.int32), owners, len(rows))
    if source_ids is None:
        source_ids = jnp.arange(len(rows))
    starts = (jnp.cumsum(degree) - degree)[source_ids]
    degree = degree[source_ids]
    rows = rows[source_ids]
    if not len(mels):
        return jnp.repeat(rows, n_outputs, axis=0), jnp.zeros(
            len(rows) * n_outputs, dtype
        )
    # _rows returns coalesced entries in owner order. Stable compaction
    # therefore supplies CSR entry positions without sorting them again.
    order = jnp.nonzero(active, size=len(mels), fill_value=0)[0]
    if exhaustive:
        rank = jnp.broadcast_to(jnp.arange(n_outputs), (len(rows), n_outputs))
    else:
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i + offset))(
            jnp.arange(len(rows))
        )
        rank = jax.vmap(
            lambda k, d: jax.random.randint(k, (n_outputs,), 0, jnp.maximum(d, 1))
        )(keys, degree)
    slots = order[jnp.minimum(starts[:, None] + rank, len(mels) - 1)]
    valid = rank < degree[:, None]
    selected = jnp.where(valid[..., None], targets[slots], rows[:, None])
    factor = (
        jnp.ones_like(rank, dtype=dtype)
        if exhaustive
        else degree[:, None].astype(dtype) / n_outputs
    )
    return selected.reshape((-1, rows.shape[1])), jnp.where(valid, factor, 0).reshape(
        -1
    )


@partial(jax.jit, static_argnames=("part", "n_outputs", "exhaustive"))
def _select_outputs(
    embedding,
    rows,
    targets,
    mels,
    owners,
    key,
    offset,
    source_ids=None,
    *,
    part,
    n_outputs,
    exhaustive,
):
    """Fuse image membership and output selection on the owning device."""
    if part != "all":
        _, inside = embedding.project(targets)
        mels = jnp.where(inside if part == "retained" else ~inside, mels, 0)
    return _draw(
        rows,
        targets,
        mels,
        owners,
        key,
        offset,
        source_ids,
        n_outputs=n_outputs,
        exhaustive=exhaustive,
    )


@partial(jax.jit, static_argnames=("count",))
def _project_rows(
    embedding, original, targets, mels, owners, factors, size, start, *, count
):
    """Project incoming rows and mask padding in one compiled operation.

    A constrained source can have a substantial validity predicate. Keeping
    it inside this kernel avoids tracing and dispatching its individual
    operations for every connection block.
    """
    projected, inside = embedding.project(targets)
    valid = inside & (factors[owners] != 0) & (owners < size * count)
    mels = jnp.where(valid, mels, 0)
    projected = jnp.where((mels != 0)[:, None], projected, original[owners // count])
    owners = jnp.minimum(owners, size * count - 1) + start * count
    return projected, mels, owners


@jax.jit
def _unique_indices(states):
    """Index equal configurations without moving their labels to the host."""
    if states.shape[1] == 0:
        zeros = jnp.zeros(len(states), jnp.int32)
        return zeros, zeros, jnp.asarray(1, jnp.int32)
    order = jnp.lexsort(
        tuple(states[:, i] for i in range(states.shape[1] - 1, -1, -1))
    ).astype(jnp.int32)
    sorted_states = states[order]
    starts = jnp.concatenate(
        (jnp.ones(1, bool), jnp.any(sorted_states[1:] != sorted_states[:-1], axis=1))
    )
    ids = jnp.cumsum(starts, dtype=jnp.int32) - 1
    inverse = jnp.zeros(len(states), jnp.int32).at[order].set(ids)
    first = order[jnp.nonzero(starts, size=len(states), fill_value=0)[0]]
    return first, inverse, jnp.count_nonzero(starts)


def _unique_rows(states):
    """Deduplicate eager rows into a bounded, reusable capacity bucket."""
    first, inverse, count = _unique_indices(states)
    capacity = min(len(states), _capacity(int(count)))
    if capacity == len(states):
        # Reordering a nearly unique batch saves no operator rows in its
        # capacity bucket and can make otherwise balanced blocks uneven.
        indices = jnp.arange(len(states), dtype=jnp.int32)
        return states, indices, indices
    first = first[:capacity]
    return states[first], inverse, first


@partial(jax.jit, static_argnames=("capacity",))
def _pad_rows(targets, mels, owners, size, offset, *, capacity):
    """Give compact blocks a common capacity before concatenation."""
    padding = capacity - len(mels)
    return (
        jnp.pad(targets, ((0, padding), (0, 0)), mode="edge"),
        jnp.pad(jnp.where(owners < size, mels, 0), (0, padding)),
        jnp.pad(jnp.minimum(owners, size - 1), (0, padding)) + offset,
    )


def _blocked_rows(operator, states, chunk):
    """Assemble coalesced compact rows with stable block capacities."""
    chunk = min(chunk, len(states))
    blocks = []
    for start in range(0, len(states), chunk):
        size = min(chunk, len(states) - start)
        batch = jnp.pad(
            states[start : start + size], ((0, chunk - size), (0, 0)), mode="edge"
        )
        targets, mels, owners, _ = _rows(operator, batch)
        if not len(mels):
            targets, mels, owners = (
                batch[:1],
                jnp.zeros(1, mels.dtype),
                jnp.zeros(1, jnp.int32),
            )
        blocks.append((targets, mels, owners, size, start))
    capacity = max(1, max(len(block[1]) for block in blocks))
    padded = [_pad_rows(*block, capacity=capacity) for block in blocks]
    return tuple(jnp.concatenate(group) for group in zip(*padded, strict=True))


def _compact_chunks(spec, samples, count):
    """Amortize CPU dispatch while bounding each block's connection payload.

    The estimate uses the operator's declared maximum row width, including
    labels, amplitudes and row indices. It targets at most 64 MiB of declared
    connection data per block, with one row as the minimum. Capacity padding,
    compiler temporaries and the complete estimator plan are additional.
    No tensors are retained between calls. Explicit limits and accelerator
    execution keep their existing block sizes.
    """
    if spec.connection_chunk_size is not None:
        return spec.connection_chunk_size, spec.connection_chunk_size * count
    if next(iter(samples.devices())).platform != "cpu":
        return 128, 128 * count

    def bounded(operator, requested):
        width = operator.hilbert.size * samples.dtype.itemsize
        width += jnp.dtype(operator.dtype).itemsize + jnp.dtype(jnp.int32).itemsize
        row_bytes = max(1, operator.max_conn_size * width)
        limit = max(1, int((64 * 1024**2) // row_bytes))
        # Stable powers of two avoid an extra family of executable shapes.
        return min(requested, 1 << (limit.bit_length() - 1))

    return bounded(spec.operator.adjoint, 512), bounded(spec.operator, 512 * count)


def _compact_plan(spec, samples, key, offset):
    """Reuse source columns and sampled output rows within a sample shard.

    Output draws still use the original global sample index. Repeated sources
    keep independent draws, while equal outputs share the same coherent
    action and predecessor mass. The inverse maps restore every contribution
    to its original chain and sample before statistics or differentiation.
    """
    action, embedding = spec.operator, spec.embedding
    reverse = action.adjoint
    count = max(1, reverse.max_conn_size) if spec.n_probes is None else spec.n_probes
    sources, source_ids, _ = _unique_rows(samples)
    encoded = embedding.embed(sources)
    source_chunk, output_chunk = _compact_chunks(spec, encoded, count)
    targets, mels, owners = _blocked_rows(reverse, encoded, source_chunk)
    outputs, factors = _select_outputs(
        embedding,
        encoded,
        targets,
        mels,
        owners,
        key,
        offset,
        source_ids,
        part=spec.part,
        n_outputs=count,
        exhaustive=spec.n_probes is None,
    )
    outputs, output_ids, first = _unique_rows(outputs)
    base_ids = first // count
    bases = samples[base_ids]
    targets, mels, owners = _blocked_rows(action, outputs, output_chunk)
    targets, mels, owners = _project_rows(
        embedding,
        bases,
        targets,
        mels,
        owners,
        jnp.ones(len(bases)),
        len(bases),
        0,
        count=1,
    )
    plan = prepare_connections(embedding.source, bases, targets, mels, owners)
    return (
        plan.states,
        plan.mels,
        plan.owners,
        plan.target_ids,
        plan.diagonal,
        factors,
        base_ids,
        output_ids,
    ), count


@jax.jit
def _native_plan(spec, samples, key, offset):
    """Fuse bounded native preparation without sparsity counts on the host."""
    action, embedding = spec.operator, spec.embedding
    reverse = action.adjoint
    count = max(1, reverse.max_conn_size) if spec.n_probes is None else spec.n_probes
    requested = (
        128 if spec.connection_chunk_size is None else spec.connection_chunk_size
    )
    chunk = min(requested, len(samples))
    blocks = (len(samples) + chunk - 1) // chunk
    padded = jnp.pad(samples, ((0, blocks * chunk - len(samples)), (0, 0)), mode="edge")

    def block(args):
        index, original = args
        encoded = embedding.embed(original)
        targets, mels = coalesce_connections(
            *reverse.get_conn_padded(encoded, use_streaming=False)
        )
        owners = jnp.repeat(jnp.arange(chunk, dtype=jnp.int32), mels.shape[-1])
        targets, mels = targets.reshape((-1, encoded.shape[-1])), mels.reshape(-1)
        if spec.part != "all":
            _, inside = embedding.project(targets)
            mels = jnp.where(inside if spec.part == "retained" else ~inside, mels, 0)
        outputs, factors = _draw(
            encoded,
            targets,
            mels,
            owners,
            key,
            offset + index * chunk,
            n_outputs=count,
            exhaustive=spec.n_probes is None,
        )
        targets, mels = coalesce_connections(
            *action.get_conn_padded(outputs, use_streaming=False)
        )
        projected, inside = embedding.project(targets)
        mels = jnp.where(inside & (factors[:, None] != 0), mels, 0)
        bases = jnp.repeat(original, count, axis=0)
        projected = jnp.where((mels != 0)[..., None], projected, bases[:, None])
        owners = jnp.repeat(
            jnp.arange(chunk * count, dtype=jnp.int32) + index * chunk * count,
            mels.shape[-1],
        )
        return (
            projected.reshape((-1, samples.shape[-1])),
            mels.reshape(-1),
            owners,
            factors,
        )

    targets, mels, owners, factors = jax.lax.map(
        block,
        (
            jnp.arange(blocks, dtype=jnp.int32),
            padded.reshape((blocks, chunk, samples.shape[-1])),
        ),
    )
    rows = len(samples) * count
    targets, mels, owners = (
        targets.reshape((-1, samples.shape[-1])),
        mels.reshape(-1),
        owners.reshape(-1),
    )
    mels = jnp.where(owners < rows, mels, 0)
    owners = jnp.minimum(owners, rows - 1)
    return (
        targets,
        mels,
        owners,
        jnp.arange(len(mels), dtype=jnp.int32),
        jnp.zeros(rows, mels.dtype),
        factors.reshape(-1)[:rows],
        jnp.arange(rows, dtype=jnp.int32) // count,
        jnp.arange(rows, dtype=jnp.int32),
    )


def _local_plan(spec, samples, key, offset):
    """Prepare bounded source batches and reuse compact recoupling paths."""
    action, embedding = spec.operator, spec.embedding
    identity = embedding.encode is None and embedding.source == embedding.target
    if spec.part == "outgoing" and (identity or action.is_diagonal):
        # No output can leave the image in these cases. Avoid even preparing
        # the parent connections, rather than discovering zeros after a draw.
        return (
            jnp.empty((0, samples.shape[-1]), samples.dtype),
            jnp.empty(0, action.dtype),
            jnp.empty(0, jnp.int32),
            jnp.empty(0, jnp.int32),
            jnp.zeros(len(samples), action.dtype),
            jnp.zeros(len(samples), spec.dtype),
            jnp.arange(len(samples), dtype=jnp.int32),
            jnp.arange(len(samples), dtype=jnp.int32),
        ), 1
    reverse = action.adjoint
    count = max(1, reverse.max_conn_size) if spec.n_probes is None else spec.n_probes
    if (
        isinstance(samples, jax.core.Tracer)
        or len(samples.devices()) != 1
        or (
            not hasattr(action, "get_conn_compact")
            and not hasattr(reverse, "get_conn_compact")
        )
    ):
        return _native_plan(spec, samples, key, offset), count
    return _compact_plan(spec, samples, key, offset)


def prepare_residual(vstate, spec, samples):
    """Freeze auxiliary draws and prepare each addressable sample shard locally."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from .sharded import _device_executor
    from .sharded import _on_device
    from .sharded import _parallel_preparation

    flat = samples.reshape((-1, spec.hilbert.size))
    if not len(flat):
        raise ValueError("Residual estimation requires a nonempty sample batch.")
    key = vstate._estimator_key
    if spec.n_probes is not None:
        vstate._estimator_key, key = jax.random.split(key)
    layout = sample_layout(flat)
    if layout is None:
        arrays, count = _local_plan(spec, flat, key, 0)
        return ResidualPlan(arrays, count)
    shards = flat.addressable_shards

    def local(shard):
        with jax.default_device(shard.device):
            local_key = key
            if isinstance(key, jax.Array) and not key.is_fully_addressable:
                local_key = key.addressable_shards[0].data
            return _local_plan(
                _on_device(spec, shard.device),
                shard.data,
                jax.device_put(local_key, shard.device),
                shard.index[0].start or 0,
            )

    plans = (
        list(_device_executor(len(shards)).map(local, shards))
        if _parallel_preparation(shards)
        else [local(s) for s in shards]
    )
    count = plans[0][1]
    partitions = len(flat) // len(shards[0].data)
    placement = NamedSharding(layout.mesh, P(layout.spec[0]))
    capacities = [max(len(p[0][i]) for p in plans) for i in (0, 1, 4)]
    if not flat.is_fully_addressable:
        sizes = jax.make_array_from_single_device_arrays(
            (partitions, 3),
            placement,
            [jax.device_put(jnp.asarray([capacities]), s.device) for s in shards],
        )
        capacities = list(map(int, jnp.max(sizes, axis=0)))
    buffers = [[] for _ in range(8)]
    for shard, (arrays, _) in zip(shards, plans, strict=True):
        with jax.default_device(shard.device):
            states, mels, owners, ids, diagonal, factors, base_ids, output_ids = arrays
            source_offset = shard.index[0].start or 0
            partition = source_offset // len(shard.data)
            # Empty plans still need a valid filler configuration for model calls.
            padding = capacities[0] - len(states)
            states = jnp.concatenate(
                (states, jnp.broadcast_to(shard.data[0], (padding, flat.shape[1])))
            )
            padding = capacities[1] - len(mels)
            mels = jnp.pad(mels, (0, padding))
            owners = jnp.pad(owners, (0, padding)) + partition * capacities[2]
            ids = jnp.pad(ids, (0, padding)) + partition * capacities[0]
            padding = capacities[2] - len(diagonal)
            diagonal = jnp.pad(diagonal, (0, padding))
            base_ids = jnp.pad(base_ids, (0, padding)) + source_offset
            output_ids = output_ids + partition * capacities[2]
            for result, value in zip(
                buffers,
                (states, mels, owners, ids, diagonal, factors, base_ids, output_ids),
                strict=True,
            ):
                result.append(value)
    arrays = tuple(
        jax.make_array_from_single_device_arrays(
            (partitions * len(group[0]), *group[0].shape[1:]), placement, group
        )
        for group in buffers
    )
    return ResidualPlan(arrays, count, layout)


@partial(jax.jit, static_argnames=("apply", "n_outputs", "chunk_size", "axis_name"))
def _values(
    apply, variables, samples, arrays, density, *, n_outputs, chunk_size, axis_name=None
):
    """Evaluate coherent action norms and their positive importance weights."""
    states, mels, owners, ids, diagonal, factors, base_ids, output_ids = arrays
    logs = apply_batched(apply, variables, samples, chunk_size=chunk_size)
    base = logs[base_ids]
    rows = len(base)
    active = mels != 0
    if len(states):
        # Each unique input can reuse any source that reaches its output row.
        # Choose one on device. Full-evaluation fallbacks ignore this map, so
        # XLA can eliminate it when a model's reuse policy declines the work.
        target_owners = jax.ops.segment_min(
            jnp.where(active, base_ids[owners], len(samples)), ids, len(states)
        )
        target_owners = jnp.minimum(target_owners, len(samples) - 1)
        target_logs = apply_connected(
            apply, variables, samples, states, target_owners, chunk_size=chunk_size
        )
    else:
        target_logs = jnp.empty(0, logs.dtype)
    scale = jnp.where(diagonal != 0, base.real, -jnp.inf)
    if len(mels):
        scale = jnp.maximum(
            scale,
            jax.ops.segment_max(
                jnp.where(active, target_logs[ids].real, -jnp.inf), owners, rows
            ),
        )
    scale = jax.lax.stop_gradient(jnp.where(jnp.isfinite(scale), scale, 0))
    a = jnp.exp(jnp.where(diagonal != 0, base - scale, 0))
    acted = diagonal * a
    mass = jnp.where(diagonal != 0, abs(a) ** 2, 0)
    if len(mels):
        a = jnp.exp(jnp.where(active, target_logs[ids] - scale[owners], 0))
        acted = acted + jax.ops.segment_sum(mels * a, owners, rows)
        mass = mass + jax.ops.segment_sum(
            jnp.where(active, abs(a) ** 2, 0), owners, rows
        )
    norms = abs(acted) ** 2 / jnp.where(mass > 0, mass, 1)
    selected = factors * norms[output_ids]
    local = jnp.sum(selected.reshape((len(samples), n_outputs)), axis=1)
    logw = 2 * logs.real - density
    shift = jax.lax.stop_gradient(jnp.max(logw))
    if axis_name is not None:
        shift = jax.lax.pmax(shift, axis_name)
    weight = jnp.exp(logw - jnp.where(jnp.isfinite(shift), shift, 0))
    return weight * local, weight


@partial(jax.jit, static_argnames=("apply", "n_outputs", "chunk_size", "layout"))
def _sharded_values(
    apply, variables, samples, arrays, density, *, n_outputs, chunk_size, layout
):
    """Keep action rows and network evaluations on their owning sample devices."""
    from jax.sharding import PartitionSpec as P
    from neuralqx.jax.shard_map import shard_map

    axis = layout.spec[0]
    axes = axis if isinstance(axis, tuple) else (axis,)

    def local(variables, samples, arrays, density):
        index = 0
        for name in axes:
            index = index * layout.mesh.shape[name] + jax.lax.axis_index(name)
        states, mels, owners, ids, diagonal, factors, base_ids, output_ids = arrays
        arrays = (
            states,
            mels,
            owners - index * len(diagonal),
            ids - index * len(states),
            diagonal,
            factors,
            base_ids - index * len(samples),
            output_ids - index * len(diagonal),
        )
        return _values(
            apply,
            variables,
            samples,
            arrays,
            density,
            n_outputs=n_outputs,
            chunk_size=chunk_size,
            axis_name=axes,
        )

    return shard_map(
        local,
        mesh=layout.mesh,
        in_specs=(P(), P(axis), tuple(P(axis) for _ in arrays), P(axis)),
        out_specs=(P(axis), P(axis)),
    )(variables, samples, arrays, density)


def evaluate_residual(vstate, variables, samples, plan, density, options):
    """Preserve parameter sharding instead of replicating a model-parallel tree."""
    layout = plan.layout
    if any(
        isinstance(a, jax.Array) and not a.sharding.is_fully_replicated
        for a in jax.tree.leaves(vstate.variables)
    ):
        layout = None
    evaluate = _values if layout is None else _sharded_values
    extra = {} if layout is None else {"layout": layout}
    return evaluate(
        vstate.apply_variables_kernel,
        variables,
        samples,
        plan.arrays,
        density,
        n_outputs=plan.n_outputs,
        chunk_size=options.chunk_size,
        **extra,
    )
