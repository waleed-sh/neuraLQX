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

import jax.numpy as jnp

import jax
from neuralqx.jax.differentiation import vjp
from neuralqx.jax.differentiation.real import split_tree_to_real
from neuralqx.sampler.importance import WeightedLocalEstimators
from neuralqx.vqs.estimators.importance import _Combination
from neuralqx.vqs.estimators.importance import evaluate_importance
from neuralqx.vqs.estimators.importance import prepare_importance
from neuralqx.vqs.estimators.operators import sequence_is_hermitian
from neuralqx.vqs.estimators.options import LocalEstimatorOptions


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

    objective, pullback, (numerator, denominator) = vjp(loss, coordinates, has_aux=True)
    gradient = reassemble(pullback(jnp.ones_like(objective))[0])
    shape = samples.shape[:-1]
    estimates = WeightedLocalEstimators(
        numerator.reshape(shape), denominator.reshape(shape)
    )
    vstate._cache_importance_weights(samples, estimates.weights)
    if not isinstance(prepared[1], _Combination):
        _adapt(vstate, values, coordinates, prepared[-1], estimates, transform)
    return estimates.to_stats(), gradient


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
