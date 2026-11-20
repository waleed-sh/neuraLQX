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


"""Exact sparse transition densities for operator-connected postprocessing."""

from functools import partial

import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax
from neuralqx.jax.streaming import apply_batched
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.connectivity import coalesce_connections
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.vqs.estimators.operators import linear_operator_terms


def blur_operators(spec, hilbert):
    """Resolve factor columns and reject missing or incompatible adjoints."""

    def factors(operator):
        if isinstance(operator, PenaltyCost):
            yield from factors(operator.cost_operator)
        elif isinstance(operator, Squared):
            yield operator.operator
        else:
            pieces = linear_operator_terms(operator)
            if len(pieces) == 1 and pieces[0].operator is operator:
                yield operator
            else:
                for piece in pieces:
                    yield from factors(piece.operator)

    terms = []
    for op in factors(spec.operator):
        if not isinstance(op, DiscreteOperator):
            raise NeuralqxTypeError(
                "OperatorBlur requires discrete matrix-row operators."
            )
        if op.hilbert != hilbert:
            raise NeuralqxValueError(
                "The blur operator and sampler have different Hilbert spaces."
            )
        adjoint = op.adjoint
        if not isinstance(adjoint, DiscreteOperator):
            raise NeuralqxTypeError(
                "OperatorBlur requires an explicit discrete adjoint."
            )
        terms.append((op, adjoint))
    return tuple(terms)


def _rows(operator, states):
    xp, mel = operator.get_conn_padded(states)
    xp, mel = coalesce_connections(xp, mel)
    mask = (mel != 0) & jnp.any(xp != states[:, None, :], axis=-1)
    xp = jnp.where(mask[..., None], xp, states[:, None, :])
    return xp, mask


@partial(jax.jit, static_argnames=("chunk_size",))
def _degrees(operator, states, *, chunk_size):
    def one(state):
        _, mask = _rows(operator, state[None, :])
        return jnp.sum(mask)

    return jax.lax.map(one, states, batch_size=chunk_size)


@jax.jit
def _choose(operator, states, keys, probability):
    xp, mask = _rows(operator, states)
    if xp.shape[1] == 0:
        return states
    degree = jnp.sum(mask, axis=1)
    uniforms = jax.vmap(lambda key: jax.random.uniform(key, (2,)))(keys)
    rank = jnp.floor(uniforms[:, 0] * jnp.maximum(degree, 1)).astype(jnp.int32)
    slots = jnp.argmax(mask & (jnp.cumsum(mask, axis=1) == rank[:, None] + 1), axis=1)
    moved = xp[jnp.arange(len(states)), slots]
    move = (degree > 0) & (uniforms[:, 1] < probability)
    return jnp.where(move[:, None], moved, states)


def _density(
    operators,
    apply,
    variables,
    states,
    exponent,
    probability,
    chunk_size,
    model_chunk_size,
):
    """Evaluate the full mixture, including proposals from other components."""
    logs = apply_batched(apply, variables, states, chunk_size=model_chunk_size).real
    log_terms, scores = [], []
    for incoming, outgoing in operators:
        out_degree = _degrees(outgoing, states, chunk_size=chunk_size)
        self_probability = jnp.where(out_degree > 0, 1 - probability, 1.0)
        xp, mask = _rows(incoming, states)
        if xp.shape[1] == 0:
            incoming_log = jnp.full((len(states),), -jnp.inf)
            incoming_score = jnp.zeros((len(states),))
        else:
            targets = xp.reshape((-1, states.shape[-1]))
            degree = _degrees(outgoing, targets, chunk_size=chunk_size).reshape(
                mask.shape
            )
            neighbour_log = apply_batched(
                apply, variables, targets, chunk_size=model_chunk_size
            ).real.reshape(mask.shape)
            log_mass = (
                exponent * neighbour_log
                - jnp.log(jnp.maximum(degree, 1))
                + jnp.log(probability)
            )
            log_mass = jnp.where(mask, log_mass, -jnp.inf)
            incoming_log = logsumexp(log_mass, axis=1)
            normalized = jnp.exp(
                log_mass
                - jnp.where(jnp.isfinite(incoming_log), incoming_log, 0)[:, None]
            )
            incoming_score = jnp.sum(
                normalized
                * jnp.where(mask & jnp.isfinite(neighbour_log), neighbour_log, 0),
                axis=1,
            )
        self_log = exponent * logs + jnp.log(self_probability)
        combined = jnp.logaddexp(self_log, incoming_log)
        safe = jnp.where(jnp.isfinite(combined), combined, 0)
        score = (
            jnp.exp(self_log - safe) * jnp.where(jnp.isfinite(logs), logs, 0)
            + jnp.exp(incoming_log - safe) * incoming_score
        )
        log_terms.append(combined)
        scores.append(score)
    stacked = jnp.stack(log_terms)
    total = logsumexp(stacked, axis=0)
    score = jnp.sum(
        jnp.exp(stacked - jnp.where(jnp.isfinite(total), total, 0)) * jnp.stack(scores),
        axis=0,
    )
    return total - jnp.log(len(operators)), score


@partial(
    jax.jit, static_argnames=("apply", "probability", "chunk_size", "model_chunk_size")
)
def _blur_local(
    operators,
    apply,
    variables,
    flat,
    exponent,
    key,
    offset,
    *,
    probability,
    chunk_size,
    model_chunk_size,
):
    """Process fixed-size local blocks without a host loop or a two-hop tensor."""
    n = len(flat)
    width = min(n, chunk_size)
    padding = (-n) % width
    blocks = jnp.pad(flat, ((0, padding), (0, 0)), mode="edge").reshape(
        (-1, width, flat.shape[-1])
    )
    indices = (jnp.arange(n + padding, dtype=jnp.uint32) + offset).reshape((-1, width))

    def block(inputs):
        states, ids = inputs
        keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(ids)
        selected = jax.vmap(
            lambda k: jax.random.randint(
                jax.random.fold_in(k, 0), (), 0, len(operators)
            )
        )(keys)
        result = states
        for i, (_, outgoing) in enumerate(operators):
            move_keys = jax.vmap(lambda k, i=i: jax.random.fold_in(k, i + 1))(keys)
            moved = _choose(outgoing, states, move_keys, probability)
            result = jnp.where((selected == i)[:, None], moved, result)
        density, score = _density(
            operators,
            apply,
            variables,
            result,
            exponent,
            probability,
            chunk_size,
            model_chunk_size,
        )
        return result, density, score

    results, density, score = jax.lax.map(block, (blocks, indices))
    return (
        results.reshape((-1, flat.shape[-1]))[:n],
        density.reshape(-1)[:n],
        score.reshape(-1)[:n],
    )


def blur_samples(
    spec, operators, apply, variables, samples, exponent, key, model_chunk_size
):
    """Draw a sparse blur and return its exact unnormalized mixture density.

    Sample-parallel execution visits only the local device's configurations.
    Fixed-size blocks bound reverse-degree work without allocating a two-hop
    connection tensor. The only global reduction is the exponent-score mean.
    Model-partitioned variables retain their ordinary JAX sharding semantics.
    """
    from jax.sharding import PartitionSpec as P
    from neuralqx.jax.shard_map import shard_map
    from neuralqx.vqs.estimators.sharded import local_model_layout

    shape = samples.shape
    flat = samples.reshape((-1, shape[-1]))
    layout = local_model_layout(flat, variables)
    kwargs = dict(
        probability=spec.probability,
        chunk_size=spec.chunk_size,
        model_chunk_size=model_chunk_size,
    )
    compact = any(
        callable(getattr(op, "get_conn_compact", None))
        for pair in operators
        for op in pair
    )
    model_partitioned = any(
        isinstance(v, jax.Array) and not v.sharding.is_fully_replicated
        for v in jax.tree.leaves(variables)
    )
    if compact and not model_partitioned:
        from .prepared_blur import prepared_blur

        result, density, score = prepared_blur(
            spec, operators, apply, variables, flat, exponent, key, model_chunk_size
        )
    elif layout is None:
        result, density, score = _blur_local(
            operators, apply, variables, flat, exponent, key, 0, **kwargs
        )
    else:
        axis = layout.spec[0]
        axes = axis if isinstance(axis, tuple) else (axis,)

        def local(operators, variables, flat, exponent, key):
            index = 0
            for name in axes:
                index = index * layout.mesh.shape[name] + jax.lax.axis_index(name)
            return _blur_local(
                operators,
                apply,
                variables,
                flat,
                exponent,
                key,
                index * len(flat),
                **kwargs,
            )

        mapped = shard_map(
            local,
            mesh=layout.mesh,
            in_specs=(P(), P(), P(axis), P(), P()),
            out_specs=(P(axis), P(axis), P(axis)),
        )
        result, density, score = mapped(operators, variables, flat, exponent, key)
    base_logs = apply_batched(apply, variables, flat, chunk_size=model_chunk_size).real
    score = score - jnp.mean(base_logs)
    return result.reshape(shape), density.reshape(shape[:-1]), score.reshape(shape[:-1])
