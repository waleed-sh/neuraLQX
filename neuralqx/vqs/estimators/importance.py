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


"""Amplitude-based importance estimators with a frozen proposal measure.

Numerators are evaluated before dividing by a wavefunction amplitude. This
retains residual contributions at configurations where the source amplitude
vanishes and avoids multiplying an infinite local value by a zero weight.
"""

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp

import jax
from neuralqx.jax.streaming import apply_batched
from neuralqx.jax.streaming import apply_connected
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.operator.wrappers import penalty_local_value_coefficients
from neuralqx.sampler.importance import WeightedLocalEstimators
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

from .connectivity import ConnectionPlan
from .connectivity import _apply_active
from .connectivity import prepare_estimator
from .kernels import prepare_discrete_connections
from .operators import linear_operator_terms
from .operators import needs_nodal_residual
from .sharded import sample_layout
from .utils import validate_operator_hilbert


@dataclass(frozen=True)
class _Combination:
    children: tuple
    penalties: tuple


def prepare_importance(vstate, operator, samples, options):
    """Prepare sparse actions once, outside differentiation of network variables."""
    validate_operator_hilbert(vstate, operator)
    measure = vstate.sampling_measure_for(samples)
    if measure is None:
        from neuralqx.sampler.importance import SamplingMeasure

        if vstate.machine_pow != 2:
            raise NeuralqxValueError(
                "Embedded residuals require Born sampling or an explicit importance sampling measure."
            )
        logs = apply_batched(
            vstate.apply_variables_kernel,
            vstate.variables,
            samples.reshape((-1, vstate.hilbert.size)),
            chunk_size=options.chunk_size,
        )
        density = jax.lax.stop_gradient(2 * logs.real).reshape(samples.shape[:-1])
        measure = SamplingMeasure(density, jnp.zeros_like(density), 2.0, -1)
    flat = samples.reshape((-1, vstate.hilbert.size))
    from neuralqx.operator.embedding import FullResidual
    from neuralqx.operator.embedding import contains_residual

    from .residual import prepare_residual

    if isinstance(operator, Squared) and needs_nodal_residual(vstate, operator):
        from neuralqx.hilbert import BasisEmbedding

        # This is an identity embedding, not a projection of the operator.
        # Output draws retain action on nodes excluded by the ansatz.
        embedding = BasisEmbedding(vstate.hilbert, vstate.hilbert)
        operator = FullResidual(operator, embedding)
    if isinstance(operator, FullResidual):
        plan = prepare_residual(vstate, operator, samples)
        return flat, plan, (), jnp.ones(1), measure
    terms = linear_operator_terms(operator)
    if any(
        isinstance(t.operator, PenaltyCost)
        or contains_residual(t.operator)
        or needs_nodal_residual(vstate, t.operator)
        for t in terms
    ):
        children, penalties = [], []
        for term in terms:
            penalty = term.operator if isinstance(term.operator, PenaltyCost) else None
            parent = term.operator if penalty is None else penalty.cost_operator
            children.append(prepare_importance(vstate, parent, samples, options))
            penalties.append(penalty)
        return (
            flat,
            _Combination(tuple(children), tuple(penalties)),
            (),
            jnp.asarray([t.coefficient for t in terms]),
            measure,
        )
    plan = prepare_estimator(
        operator,
        samples,
        operator_sharding=options.operator_sharding,
        operator_streaming=options.operator_streaming,
        operator_streaming_chunk_size=options.operator_streaming_chunk_size,
        enabled=options.connection_deduplication,
    )
    if plan is not None:
        return flat, plan, plan.squared, plan.coefficients, measure
    connections = []
    squared = []
    coefficients = []
    for term in linear_operator_terms(operator):
        square = isinstance(term.operator, Squared)
        op = term.operator.operator if square else term.operator
        if not isinstance(op, DiscreteOperator):
            raise NeuralqxTypeError(
                "Importance estimators require discrete operators or their squared markers."
            )
        connections.append(
            prepare_discrete_connections(
                op,
                flat,
                operator_streaming=options.operator_streaming,
                operator_streaming_chunk_size=options.operator_streaming_chunk_size,
            )
        )
        squared.append(square)
        coefficients.append(term.coefficient)
    return flat, tuple(connections), tuple(squared), jnp.asarray(coefficients), measure


@partial(jax.jit, static_argnames=("apply", "squared", "chunk_size"))
def planned_amplitude_values(
    apply, variables, samples, arrays, log_density, *, squared, chunk_size
):
    """Reuse unique target evaluations without dividing by a source amplitude."""
    states, mels, owners, ids, diagonal, coefficients, count = arrays
    logs = apply_batched(apply, variables, samples, chunk_size=chunk_size)
    shift = jax.lax.stop_gradient(jnp.max(2 * logs.real - log_density) / 2)
    reference = log_density / 2 + jnp.where(jnp.isfinite(shift), shift, 0)
    amplitude = jnp.exp(logs - reference)
    values = diagonal * jnp.tile(amplitude, len(squared))
    if len(states):
        targets = _apply_active(apply, variables, states, count, chunk_size)
        delta = jnp.where(mels != 0, targets[ids] - reference[owners % len(samples)], 0)
        contributions = mels * jnp.exp(delta)
        values = values.astype(contributions.dtype).at[owners].add(contributions)
    values = values.reshape((len(squared), len(samples)))
    total = sum(
        coefficients[i]
        * (abs(values[i]) ** 2 if square else jnp.conj(amplitude) * values[i])
        for i, square in enumerate(squared)
    )
    return total, abs(amplitude) ** 2


@partial(jax.jit, static_argnames=("apply", "squared", "chunk_size", "axis_name"))
def amplitude_values(
    apply,
    variables,
    samples,
    connections,
    coefficients,
    log_density,
    *,
    squared,
    chunk_size,
    axis_name=None,
):
    """Evaluate a numerator and a Born normalizer with a common stable scale."""
    logs = apply_batched(apply, variables, samples, chunk_size=chunk_size)
    logw = 2 * logs.real - log_density
    shift = jax.lax.stop_gradient(jnp.max(logw) / 2)
    if axis_name is not None:
        shift = jax.lax.pmax(shift, axis_name)
    shift = jnp.where(jnp.isfinite(shift), shift, 0)
    reference = 0.5 * log_density + shift
    amplitude = jnp.exp(logs - reference)
    denominator = abs(amplitude) ** 2
    dtype = jnp.result_type(logs if not all(squared) else logs.real, coefficients)
    total = jnp.zeros(len(samples), dtype=dtype)
    for i, (targets, mels, owners) in enumerate(connections):
        if owners is None:
            owners = jnp.repeat(jnp.arange(len(samples)), targets.shape[1])
        targets = targets.reshape((-1, samples.shape[-1]))
        mels = mels.reshape((-1,))
        targets = jnp.where((mels != 0)[:, None], targets, samples[owners])

        # Rematerialize connected activations during the pullback instead of
        # retaining the forward activations of every operator term.
        def action(variables, targets, mels, owners):
            connected = apply_connected(
                apply, variables, samples, targets, owners, chunk_size=chunk_size
            )
            values = mels * jnp.exp(
                jnp.where(mels != 0, connected - reference[owners], 0)
            )
            return jnp.zeros(len(samples), values.dtype).at[owners].add(values)

        acted = jax.checkpoint(action)(variables, targets, mels, owners)
        value = abs(acted) ** 2 if squared[i] else jnp.conj(amplitude) * acted
        total = total + coefficients[i] * value
    return total, denominator


@partial(jax.jit, static_argnames=("apply", "squared", "chunk_size", "layout"))
def sharded_amplitude_values(
    apply,
    variables,
    samples,
    connections,
    coefficients,
    log_density,
    *,
    squared,
    chunk_size,
    layout,
):
    """Evaluate local connections without gathering configurations between devices."""
    from jax.sharding import PartitionSpec as P
    from neuralqx.jax.shard_map import shard_map

    axis = layout.spec[0]
    axes = axis if isinstance(axis, tuple) else (axis,)

    def local(variables, samples, connections, coefficients, density):
        index = 0
        for name in axes:
            index = index * layout.mesh.shape[name] + jax.lax.axis_index(name)
        local_connections = tuple(
            (x, m, None if owner is None else owner - index * len(samples))
            for x, m, owner in connections
        )
        return amplitude_values(
            apply,
            variables,
            samples,
            local_connections,
            coefficients,
            density,
            squared=squared,
            chunk_size=chunk_size,
            axis_name=axes,
        )

    specs = tuple(
        (P(axis), P(axis), None if o is None else P(axis)) for _, _, o in connections
    )
    mapped = shard_map(
        local,
        mesh=layout.mesh,
        in_specs=(P(), P(axis), specs, P(), P(axis)),
        out_specs=(P(axis), P(axis)),
    )
    return mapped(variables, samples, connections, coefficients, log_density)


def evaluate_importance(vstate, variables, prepared, options):
    """Select sample-local execution while respecting model-parallel variables."""
    flat, connections, squared, coefficients, measure = prepared
    from .residual import ResidualPlan
    from .residual import evaluate_residual

    if isinstance(connections, ResidualPlan):
        return evaluate_residual(
            vstate,
            variables,
            flat,
            connections,
            measure.log_density.reshape(-1),
            options,
        )
    if isinstance(connections, _Combination):
        total = 0
        for i, (child, penalty) in enumerate(
            zip(connections.children, connections.penalties, strict=True)
        ):
            numerator, denominator = evaluate_importance(
                vstate, variables, child, options
            )
            if penalty is not None:
                mean = jnp.sum(numerator) / jnp.sum(denominator)
                scale, shift = penalty_local_value_coefficients(penalty, mean)
                numerator = scale * numerator + shift * denominator
            total = total + coefficients[i] * numerator
        return total, denominator
    if isinstance(connections, ConnectionPlan):
        plan = connections
        arrays = (
            plan.states,
            plan.mels,
            plan.owners,
            plan.target_ids,
            plan.diagonal,
            plan.coefficients,
            jnp.asarray(plan.n_unique),
        )
        return planned_amplitude_values(
            vstate.apply_variables_kernel,
            variables,
            flat,
            arrays,
            measure.log_density.reshape(-1),
            squared=squared,
            chunk_size=options.chunk_size,
        )
    layout = sample_layout(flat)
    if layout is not None and any(
        isinstance(a, jax.Array) and not a.sharding.is_fully_replicated
        for a in jax.tree.leaves(vstate.variables)
    ):
        layout = None
    function = amplitude_values if layout is None else sharded_amplitude_values
    extra = {} if layout is None else {"layout": layout}
    return function(
        vstate.apply_variables_kernel,
        variables,
        flat,
        connections,
        coefficients,
        measure.log_density.reshape(-1),
        squared=squared,
        chunk_size=options.chunk_size,
        **extra,
    )


def importance_local_estimators(vstate, operator, samples, options):
    """Compute density-corrected local contributions and ratio statistics."""
    prepared = prepare_importance(vstate, operator, samples, options)
    numerator, denominator = evaluate_importance(
        vstate, vstate.variables, prepared, options
    )
    shape = samples.shape[:-1]
    estimates = WeightedLocalEstimators(
        numerator.reshape(shape), denominator.reshape(shape)
    )
    vstate._cache_importance_weights(samples, estimates.weights)
    return estimates
