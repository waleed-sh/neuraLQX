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

"""Shared network evaluation for eager, single-device discrete estimators.

Operator execution remains in the operator subsystem. Here a batch's computed
connections become a reusable, parameter-independent sparse evaluation plan.
Only counts cross to Python, sorting, summation and gathers run in JAX.
"""

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp

import jax
from neuralqx.jax.streaming import apply_batched
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.connectivity import pack_local_indices

from .operators import OperatorTerm
from .operators import linear_operator_terms


def _capacity(count):
    return 1 << (max(1, count) - 1).bit_length()


@partial(jax.jit, static_argnames=("capacity",))
def _pack(states, mels, owners, *, capacity):
    active = mels != 0
    count = jnp.count_nonzero(active)
    idx = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    return (
        states[idx],
        jnp.where(jnp.arange(capacity) < count, mels[idx], 0),
        owners[idx],
    )


@jax.jit
def _coalesce(states, mels, owners, keys):
    # Targets are the primary key, owners the secondary key. Equal targets
    # stay adjacent even when they belong to different sources or squares.
    order = jnp.lexsort(
        (owners, *(keys[:, i] for i in range(keys.shape[1] - 1, -1, -1)))
    )
    keys, owners, mels = keys[order], owners[order], mels[order]
    starts = jnp.concatenate(
        (
            jnp.ones(1, bool),
            (owners[1:] != owners[:-1]) | jnp.any(keys[1:] != keys[:-1], axis=-1),
        )
    )
    first = jax.lax.associative_scan(
        jnp.maximum, jnp.where(starts, jnp.arange(len(mels)), 0)
    )
    sums = jnp.zeros_like(mels).at[first].add(mels)
    return states[order], sums, owners, keys


@partial(jax.jit, static_argnames=("capacity",))
def _survivors(states, mels, owners, keys, *, capacity):
    count = jnp.count_nonzero(mels)
    idx = jnp.nonzero(mels != 0, size=capacity, fill_value=0)[0]
    keys = keys[idx]
    active = jnp.arange(capacity) < count
    starts = active & jnp.concatenate(
        (jnp.ones(1, bool), jnp.any(keys[1:] != keys[:-1], axis=-1))
    )
    ids = jnp.maximum(jnp.cumsum(starts, dtype=jnp.int32) - 1, 0)
    return states[idx], jnp.where(active, mels[idx], 0), owners[idx], ids, starts


@partial(jax.jit, static_argnames=("capacity",))
def _unique_states(states, starts, *, capacity):
    return states[jnp.nonzero(starts, size=capacity, fill_value=0)[0]]


@partial(jax.jit, static_argnames=("n_rows",))
def _extract_diagonal(samples, states, mels, owners, *, n_rows):
    diagonal = jnp.all(states == samples[owners % len(samples)], axis=-1)
    values = jnp.zeros(n_rows, mels.dtype).at[owners].add(jnp.where(diagonal, mels, 0))
    return values, jnp.where(diagonal, 0, mels)


@dataclass(frozen=True)
class ConnectionPlan:
    """A batch-local plan, never cache it across changing samples/operators."""

    states: jax.Array
    mels: jax.Array
    owners: jax.Array
    target_ids: jax.Array
    diagonal: jax.Array
    coefficients: jax.Array
    squared: tuple[bool, ...]
    n_unique: int
    n_contributions: int
    n_input_nonzero: int

    def evaluate(self, apply_fn, variables, samples, *, chunk_size):
        return _evaluate(
            apply_fn,
            variables,
            samples,
            self.states,
            self.mels,
            self.owners,
            self.target_ids,
            self.diagonal,
            self.coefficients,
            jnp.asarray(self.n_unique),
            squared=self.squared,
            chunk_size=chunk_size,
        )


def prepare_connections(
    hilbert, samples, states, mels, owners, *, squared=(False,), coefficients=None
):
    """Build an exact coalesced plan from flat connections and source owners.

    An owner is ``group * batch_size + sample``. Groups keep separate squared
    actions independent while allowing them to share network evaluations.
    Fixed operator amplitudes are required: differentiating through support
    selection with respect to amplitudes is intentionally not supported.
    """
    if any(isinstance(a, jax.core.Tracer) for a in (samples, states, mels, owners)):
        return None
    n_rows = len(samples) * len(squared)
    coefficients = (
        jnp.ones(len(squared)) if coefficients is None else jnp.asarray(coefficients)
    )
    diagonal, mels = _extract_diagonal(samples, states, mels, owners, n_rows=n_rows)
    n_input = int(jnp.count_nonzero(mels))
    if n_input:
        states, mels, owners = _pack(
            states, mels, owners, capacity=min(len(mels), _capacity(n_input))
        )
        # Zero padding can contain arbitrary labels. Replace it with a valid
        # sampled state before converting to local indices.
        states = jnp.where((mels != 0)[:, None], states, samples[0])
        keys = pack_local_indices(
            hilbert.states_to_local_indices(states),
            local_sizes=tuple(hilbert.local_sizes),
        )
        states, mels, owners, keys = _coalesce(states, mels, owners, keys)
        count = int(jnp.count_nonzero(mels))
    else:
        count = 0
    if count:
        states, mels, owners, ids, starts = _survivors(
            states, mels, owners, keys, capacity=min(len(mels), _capacity(count))
        )
        n_unique = int(jnp.count_nonzero(starts))
        states = _unique_states(states, starts, capacity=_capacity(n_unique))
    else:
        states = jnp.empty((0, samples.shape[-1]), samples.dtype)
        mels, owners, ids = (
            jnp.empty(0, mels.dtype),
            jnp.empty(0, jnp.int32),
            jnp.empty(0, jnp.int32),
        )
        n_unique = 0
    return ConnectionPlan(
        states,
        mels,
        owners,
        ids,
        diagonal,
        coefficients,
        tuple(squared),
        n_unique,
        count,
        n_input,
    )


def prepare_estimator(
    operator,
    samples,
    *,
    operator_sharding=None,
    operator_streaming=None,
    operator_streaming_chunk_size=None,
    enabled=True,
):
    """Prepare supported eager estimators, or return None for the existing path.

    Explicit sharding, multiple devices and enclosing JIT keep their existing
    execution policy. Nonlinear/custom wrappers are dispatched as before.
    """
    if (
        not enabled
        or operator_sharding
        or jax.device_count() != 1
        or isinstance(samples, jax.core.Tracer)
    ):
        return None
    terms = _combine_identical_terms(linear_operator_terms(operator))
    if not terms or any(
        not isinstance(t.operator, (DiscreteOperator, Squared)) for t in terms
    ):
        return None
    if any(
        isinstance(t.operator, Squared)
        and (
            type(t.operator) is not Squared
            or not isinstance(t.operator.operator, DiscreteOperator)
        )
        for t in terms
    ):
        return None
    # Avoid sorting tiny rows, where a single ordinary model batch is cheaper.
    # A single square still benefits when its parent has many branches.
    parents = tuple(
        t.operator.operator if isinstance(t.operator, Squared) else t.operator
        for t in terms
    )
    flat = jnp.asarray(samples).reshape((-1, parents[0].hilbert.size))
    if len(flat) * sum(p.max_conn_size for p in parents) < 512:
        return None
    states, mels, owners = [], [], []
    computed = {}
    has_linear = any(isinstance(t.operator, DiscreteOperator) for t in terms)
    squared, coefficients = ([False], [1.0]) if has_linear else ([], [])
    for term, parent in zip(terms, parents, strict=True):
        if isinstance(term.operator, Squared):
            group = len(squared)
            squared.append(True)
            coefficients.append(term.coefficient)
            children = _combine_identical_terms(linear_operator_terms(parent))
        else:
            group = 0
            children = (term,)
        for child in children:
            op = child.operator
            if isinstance(child.coefficient, jax.core.Tracer):
                return None
            if bool(jnp.asarray(child.coefficient) == 0):
                continue
            compact = getattr(op, "get_conn_compact", None)
            packed = computed.get(id(op))
            if packed is None:
                packed = (
                    compact(flat)
                    if compact is not None
                    and not operator_streaming
                    and operator_streaming_chunk_size is None
                    else None
                )
            if packed is None:
                xp, mel = op.get_conn_padded(
                    flat,
                    use_streaming=operator_streaming,
                    chunk_size=operator_streaming_chunk_size,
                )
                xp, mel = xp.reshape((-1, flat.shape[-1])), mel.reshape(-1)
                source = jnp.repeat(
                    jnp.arange(len(flat), dtype=jnp.int32), op.max_conn_size
                )
            else:
                xp, mel, source = packed
            computed[id(op)] = (xp, mel, source)
            states.append(xp)
            mels.append(mel * child.coefficient)
            owners.append(source + group * len(flat))
    if not states:
        states = [jnp.empty((0, flat.shape[-1]), flat.dtype)]
        mels = [jnp.empty(0, jnp.result_type(*(p.dtype for p in parents)))]
        owners = [jnp.empty(0, jnp.int32)]
    return prepare_connections(
        parents[0].hilbert,
        flat,
        jnp.concatenate(states),
        jnp.concatenate(mels),
        jnp.concatenate(owners),
        squared=tuple(squared),
        coefficients=coefficients,
    )


def _combine_identical_terms(terms):
    """Combine references to the same operator, without comparing JAX arrays."""
    result = {}
    for term in terms:
        key = id(term.operator)
        coefficient = term.coefficient
        if key in result:
            coefficient = coefficient + result[key].coefficient
        result[key] = OperatorTerm(term.operator, coefficient)
    return tuple(result.values())


def _apply_active(apply_fn, variables, states, count, chunk_size):
    # Static buckets reuse executables, but completely empty chunks execute no
    # network. The last partially filled chunk has at most chunk_size-1 pads.
    chunk = min(len(states), 256 if chunk_size is None else chunk_size)
    n_chunks = (len(states) + chunk - 1) // chunk
    padded = jnp.pad(states, ((0, n_chunks * chunk - len(states)), (0, 0)), mode="edge")
    batches = padded.reshape((n_chunks, chunk, states.shape[-1]))
    dtype = jax.eval_shape(
        apply_fn,
        variables,
        jax.ShapeDtypeStruct((chunk, states.shape[-1]), states.dtype),
    ).dtype

    def evaluate(args):
        i, x = args
        return jax.lax.cond(
            i * chunk < count,
            lambda: apply_fn(variables, x),
            lambda: jnp.zeros(chunk, dtype),
        )

    values = jax.lax.map(jax.checkpoint(evaluate), (jnp.arange(n_chunks), batches))
    return values.reshape(-1)[: len(states)]


@partial(jax.jit, static_argnames=("apply_fn", "squared", "chunk_size"))
def _evaluate(
    apply_fn,
    variables,
    samples,
    states,
    mels,
    owners,
    target_ids,
    diagonal,
    coefficients,
    count,
    *,
    squared,
    chunk_size,
):
    values = diagonal
    if len(states):
        base = apply_batched(apply_fn, variables, samples, chunk_size=chunk_size)
        targets = _apply_active(apply_fn, variables, states, count, chunk_size)
        delta = jnp.where(
            mels != 0, targets[target_ids] - base[owners % len(samples)], 0
        )
        contributions = mels * jnp.exp(delta)
        values = values.astype(contributions.dtype).at[owners].add(contributions)
    values = values.reshape((len(squared), len(samples)))
    total = None
    for group in range(len(squared)):
        v = abs(values[group]) ** 2 if squared[group] else values[group]
        v = coefficients[group] * v
        total = v if total is None else total + v
    return total
