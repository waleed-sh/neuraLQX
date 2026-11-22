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


"""Physical gradients and adaptive sampling under a frozen proposal density."""

from functools import partial

import jax.numpy as jnp

import jax
from neuralqx.jax.differentiation import vjp
from neuralqx.jax.differentiation.real import split_tree_to_real
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.profile._instrumentation import instrument
from neuralqx.profile._instrumentation import region
from neuralqx.sampler.importance import WeightedLocalEstimators
from neuralqx.vqs.estimators.importance import _amplitude_reference
from neuralqx.vqs.estimators.importance import _amplitude_term
from neuralqx.vqs.estimators.importance import evaluate_importance
from neuralqx.vqs.estimators.importance import prepare_importance
from neuralqx.vqs.estimators.operators import linear_operator_terms
from neuralqx.vqs.estimators.operators import sequence_is_hermitian
from neuralqx.vqs.estimators.options import LocalEstimatorOptions
from neuralqx.vqs.estimators.sharded import local_model_layout


@instrument("gradient.importance", detail="detailed", category="neuralqx.gradient")
def importance_expect_and_grad(vstate, operator, kwargs, *, expectation_transform=None):
    """Differentiate the physical ratio while keeping the sampling law fixed.

    Both the numerator and normalization derivative are included. Splitting
    parameter coordinates preserves derivatives of non-holomorphic networks
    and complex objectives. The estimator is self-normalized, with its usual
    finite-batch bias, and requires no derivative through MCMC transitions.
    """
    kwargs = dict(kwargs)
    samples = kwargs.pop("samples", None)
    samples = vstate.samples if samples is None else samples
    options = LocalEstimatorOptions.resolve(vstate, **kwargs)
    prepared = prepare_importance(vstate, operator, samples, options)
    coordinates, reassemble = split_tree_to_real(vstate.parameters)
    transform = expectation_transform
    if transform is None:
        transform = jnp.real if sequence_is_hermitian(operator) else lambda value: value

    def values(real_parameters):
        variables = vstate.merge_variables_kernel(
            reassemble(real_parameters), vstate.model_state
        )
        return evaluate_importance(vstate, variables, prepared, options)

    def loss(real_parameters):
        numerator, denominator = values(real_parameters)
        mean = jnp.sum(numerator) / jnp.sum(denominator)
        objective = transform(mean)
        return objective, (numerator, denominator)

    if (
        expectation_transform is None
        and isinstance(prepared[1], tuple)
        and prepared[2]
        and all(prepared[2])
        and not jnp.iscomplexobj(prepared[3])
    ):
        numerator, denominator, gradient = _squared_importance(
            vstate, prepared, options
        )
    else:
        with region(
            "gradient.forward_linearize",
            detail="developer",
            category="neuralqx.gradient",
        ):
            objective, pullback, (numerator, denominator) = vjp(
                loss, coordinates, has_aux=True
            )
        with region(
            "gradient.backward_pullback",
            detail="developer",
            category="neuralqx.gradient",
        ):
            gradient = reassemble(pullback(jnp.ones_like(objective))[0])
    shape = samples.shape[:-1]
    estimates = WeightedLocalEstimators(
        numerator.reshape(shape), denominator.reshape(shape)
    )
    vstate._cache_importance_weights(samples, estimates.weights)
    if not any(
        isinstance(term.operator, PenaltyCost)
        for term in linear_operator_terms(operator)
    ):
        _adapt(vstate, values, coordinates, prepared[-1], estimates, transform)
    return estimates.to_stats(), gradient


@partial(jax.jit, static_argnames=("apply", "merge", "chunk_size"))
def _squared_numerator_grad(
    apply,
    merge,
    parameters,
    state,
    samples,
    connection,
    reference,
    coefficient,
    norm,
    *,
    chunk_size,
):
    """Differentiate one numerator while its network activations are still live."""
    coordinates, reassemble = split_tree_to_real(parameters)

    def loss(p):
        numerator = _amplitude_term(
            apply,
            merge(reassemble(p), state),
            samples,
            connection,
            jnp.zeros(len(samples)),
            reference,
            coefficient,
            square=True,
            chunk_size=chunk_size,
            rematerialize=False,
        )
        return jnp.sum(numerator) / norm, numerator

    (_, numerator), gradient = jax.value_and_grad(loss, has_aux=True)(coordinates)
    return numerator, reassemble(gradient)


@partial(jax.jit, static_argnames=("apply", "merge", "chunk_size"))
def _normalization_grad(
    apply, merge, parameters, state, samples, reference, mean, norm, *, chunk_size
):
    from neuralqx.jax.streaming import apply_batched

    coordinates, reassemble = split_tree_to_real(parameters)

    def loss(p):
        logs = apply_batched(
            apply, merge(reassemble(p), state), samples, chunk_size=chunk_size
        )
        return -mean * jnp.sum(abs(jnp.exp(logs - reference)) ** 2) / norm

    return reassemble(jax.grad(loss)(coordinates))


@jax.jit
def _add_gradient(left, right):
    return jax.tree.map(jnp.add, left, right)


def _squared_importance(vstate, prepared, options):
    samples, connections, _, coefficients, measure = prepared
    layout = local_model_layout(samples, vstate.variables)
    if layout is not None:
        return _sharded_squared_importance(
            vstate.apply_variables_kernel,
            vstate.merge_variables_kernel,
            vstate.parameters,
            vstate.model_state,
            samples,
            connections,
            coefficients,
            measure.log_density.reshape(-1),
            chunk_size=options.chunk_size,
            layout=layout,
        )
    return _squared_importance_values_grad(
        vstate.apply_variables_kernel,
        vstate.merge_variables_kernel,
        vstate.parameters,
        vstate.model_state,
        samples,
        connections,
        coefficients,
        measure.log_density.reshape(-1),
        chunk_size=options.chunk_size,
    )


def _squared_importance_values_grad(
    apply,
    merge,
    parameters,
    state,
    samples,
    connections,
    coefficients,
    density,
    *,
    chunk_size,
    axis_name=None,
):
    """Use d(N/D) = dN/D - (N/D) dD/D with a frozen density and scale.

    Separate term kernels reuse executables as compact capacities change.
    Each term finishes its pullback before releasing its activations. The
    normalization derivative uses only the source batch.
    """
    if axis_name is not None:
        from .gradients import _vary

        parameters = _vary(parameters, axis_name)
    amplitude, reference = _amplitude_reference(
        apply,
        merge(parameters, state),
        samples,
        density,
        chunk_size=chunk_size,
        axis_name=axis_name,
    )
    denominator = abs(amplitude) ** 2
    norm = jnp.sum(denominator)
    if axis_name is not None:
        norm = jax.lax.psum(norm, axis_name)
    # Connected amplitudes and coefficients may have greater precision than
    # the model output. A scan carry must start with the promoted term dtype.
    dtype = jnp.result_type(
        denominator, coefficients, *(m.real for _, m, _ in connections)
    )
    numerator = jnp.zeros(denominator.shape, dtype)
    gradient = jax.tree.map(lambda p: jnp.zeros(p.shape, p.dtype), parameters)

    def add_term(carry, inputs):
        numerator, gradient = carry
        connection, coefficient = inputs
        values, grad = _squared_numerator_grad(
            apply,
            merge,
            parameters,
            state,
            samples,
            connection,
            reference,
            coefficient,
            norm,
            chunk_size=chunk_size,
        )
        return (numerator + values, _add_gradient(gradient, grad)), None

    if axis_name is None:
        for connection, coefficient in zip(connections, coefficients, strict=True):
            (numerator, gradient), _ = add_term(
                (numerator, gradient), (connection, coefficient)
            )
    else:
        # The enclosing shard_map is one executable. Scan equal-shaped terms
        # so it cannot retain every vertex's large network activations at once.
        # Samples remain parallel within each shard, with one gradient reduction.
        buckets = {}
        for connection, coefficient in zip(connections, coefficients, strict=True):
            key = tuple(
                (a.shape, a.dtype) if a is not None else None for a in connection
            )
            buckets.setdefault(key, []).append((connection, coefficient))
        carry = _vary((numerator, gradient), axis_name)
        for bucket in buckets.values():
            stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *bucket)
            carry, _ = jax.lax.scan(add_term, carry, stacked)
        numerator, gradient = carry
    mean = jnp.sum(numerator) / norm
    if axis_name is not None:
        mean = jax.lax.psum(mean, axis_name)
    source_gradient = _normalization_grad(
        apply,
        merge,
        parameters,
        state,
        samples,
        reference,
        mean,
        norm,
        chunk_size=chunk_size,
    )
    gradient = _add_gradient(gradient, source_gradient)
    if axis_name is not None:
        gradient = jax.lax.psum(gradient, axis_name)
    return numerator, denominator, gradient


@partial(jax.jit, static_argnames=("apply", "merge", "chunk_size", "layout"))
def _sharded_squared_importance(
    apply,
    merge,
    parameters,
    state,
    samples,
    connections,
    coefficients,
    density,
    *,
    chunk_size,
    layout,
):
    from jax.sharding import PartitionSpec as P
    from neuralqx.jax.shard_map import shard_map

    axis = layout.spec[0]
    axes = axis if isinstance(axis, tuple) else (axis,)

    def local(parameters, state, samples, connections, coefficients, density):
        index = 0
        for name in axes:
            index = index * layout.mesh.shape[name] + jax.lax.axis_index(name)
        local_connections = tuple(
            (x, m, None if o is None else o - index * len(samples))
            for x, m, o in connections
        )
        return _squared_importance_values_grad(
            apply,
            merge,
            parameters,
            state,
            samples,
            local_connections,
            coefficients,
            density,
            chunk_size=chunk_size,
            axis_name=axes,
        )

    specs = tuple(
        (P(axis), P(axis), None if o is None else P(axis)) for _, _, o in connections
    )
    mapped = shard_map(
        local,
        mesh=layout.mesh,
        in_specs=(P(), P(), P(axis), specs, P(), P(axis)),
        out_specs=(P(axis), P(axis), P()),
    )
    return mapped(parameters, state, samples, connections, coefficients, density)


@instrument("sampling.adapt_exponent", detail="developer", category="neuralqx.sampling")
def _adapt(vstate, values, coordinates, measure, estimates, transform):
    config = getattr(vstate.sampler, "importance_sampling", None)
    if config is None or not config.adaptive or measure.batch_id < 0:
        return
    if (
        measure.batch_id == vstate._importance_last_adapted
        or measure.batch_id % config.update_every
    ):
        return
    vstate._importance_last_adapted = measure.batch_id
    mean = jax.lax.stop_gradient(estimates.mean)
    normalization = jax.lax.stop_gradient(jnp.mean(estimates.denominator))

    score = measure.exponent_score.reshape(-1)
    score = score - jnp.mean(score)
    variance = 0.0
    slope = 0.0
    vstate._importance_key, key = jax.random.split(vstate._importance_key)
    # Rademacher probes estimate the trace of the gradient covariance without
    # an N_samples by N_parameters Jacobian or a per-sample gradient tree.
    for probe in range(config.probes):
        keys = jax.random.split(jax.random.fold_in(key, probe), len(coordinates))
        tangent = tuple(
            jax.random.rademacher(k, x.shape, dtype=x.dtype)
            for k, x in zip(keys, coordinates, strict=True)
        )
        _, (dn, dd) = jax.jvp(values, (coordinates,), (tangent,))
        direction = (jnp.mean(dn) - mean * jnp.mean(dd)) / normalization
        # The ratio gradient has two estimated normalizers, its Born norm
        # and its energy mean. Include both in its influence function.
        n = estimates.numerator.reshape(-1)
        d = estimates.denominator.reshape(-1)
        influence = (
            dn
            - mean * dd
            - direction * d
            - (jnp.mean(dd) / normalization) * (n - mean * d)
        ) / normalization
        energy_influence = (n - mean * d) / normalization

        def transformed(e, g, direction=direction):
            first = jax.jvp(transform, (mean,), (g,))[1]

            def directional(m):
                return jax.jvp(transform, (m,), (direction,))[1]

            second = jax.jvp(directional, (mean,), (e,))[1]
            return first + second

        influence = jax.vmap(transformed)(energy_influence, influence)
        fluctuation = abs(influence) ** 2
        variance = variance + jnp.mean(fluctuation)
        slope = slope - jnp.mean(score * fluctuation)
    valid = jnp.isfinite(variance) & jnp.isfinite(slope) & (variance > 0)
    step = jnp.where(
        valid,
        -config.learning_rate
        * slope
        / jnp.maximum(variance, jnp.finfo(jnp.asarray(variance).dtype).tiny),
        0.0,
    )
    step = jnp.clip(step, -config.max_change, config.max_change)
    ess_fraction = estimates.ess / estimates.denominator.size
    step = jnp.where(ess_fraction < config.min_ess_fraction, config.max_change, step)
    vstate._importance_exponent = jax.lax.stop_gradient(
        jnp.clip(measure.exponent + step, config.min_exponent, config.max_exponent)
    )
    vstate._sampler_state_stale = True
