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


"""Expectation-gradient and force estimators."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.differentiation import vjp as neuralqx_vjp
from neuralqx.jax.differentiation.real import split_tree_to_real
from neuralqx.jax.streaming import apply_batched
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.vqs.estimators import Stats
from neuralqx.vqs.estimators import local_estimators
from neuralqx.vqs.estimators import statistics
from neuralqx.vqs.estimators.connectivity import prepare_estimator
from neuralqx.vqs.estimators.kernels import discrete_operator_local_values
from neuralqx.vqs.estimators.kernels import discrete_operator_terms_local_values
from neuralqx.vqs.estimators.kernels import prepare_discrete_connections
from neuralqx.vqs.estimators.kernels import squared_operator_local_values
from neuralqx.vqs.estimators.operators import discrete_linear_terms
from neuralqx.vqs.estimators.operators import is_operator_sequence
from neuralqx.vqs.estimators.operators import linear_operator_terms
from neuralqx.vqs.estimators.options import LocalEstimatorOptions
from neuralqx.vqs.estimators.utils import flatten_samples
from neuralqx.vqs.estimators.utils import validate_operator_hilbert

from .forces import can_use_covariance
from .forces import expect_and_forces
from .forces import force_to_grad
from .forces import machine_pow_is_two
from .penalties import penalty_expect_and_grad


def expect_and_grad(
    vstate: Any,
    operator: Any,
    *,
    preconditioner: Any | None = None,
    step: Any | None = None,
    use_covariance: bool | None = None,
    **kwargs: Any,
) -> tuple[Stats, Any]:
    """Estimate expectation statistics and optimizer-compatible gradients."""
    if isinstance(operator, PenaltyCost):
        stats, grad = penalty_expect_and_grad(
            vstate,
            operator,
            dict(kwargs),
            use_covariance,
        )
    else:
        covariance_valid = can_use_covariance(operator, machine_pow=vstate.machine_pow)
        if use_covariance is None:
            use_covariance = covariance_valid
        elif use_covariance and not covariance_valid:
            raise ValueError(
                "Covariance gradients require a Hermitian linear estimator and either "
                "machine_pow=2 or a diagonal operator. Use use_covariance=False "
                "to include the local estimator's parameter derivative."
            )
        if use_covariance:
            stats, forces = expect_and_forces(vstate, operator, **kwargs)
            grad = force_to_grad(
                forces,
                vstate.parameters,
                machine_pow=vstate.machine_pow,
            )
        else:
            stats, grad = _expect_and_grad_noncovariance(vstate, operator, dict(kwargs))
    if preconditioner is not None:
        grad = preconditioner(vstate, grad, step=step, samples=kwargs.get("samples"))
    return stats, grad


def _expect_and_grad_noncovariance(
    vstate: Any,
    operator: Any,
    kwargs: dict[str, Any],
    *,
    expectation_transform: Any = None,
) -> tuple[Stats, Any]:
    if isinstance(operator, PenaltyCost):
        return penalty_expect_and_grad(vstate, operator, kwargs, False)

    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = vstate.samples
    flat_samples, sample_shape = flatten_samples(samples, vstate.hilbert.size)
    options = LocalEstimatorOptions.resolve(
        vstate,
        options=kwargs.pop("options", None),
        chunk_size=kwargs.pop("chunk_size", None),
        operator_sharding=kwargs.pop("operator_sharding", None),
        operator_streaming=kwargs.pop("operator_streaming", None),
        operator_streaming_chunk_size=kwargs.pop("operator_streaming_chunk_size", None),
        operator_chunk_size=kwargs.pop("operator_chunk_size", None),
        connection_deduplication=kwargs.pop("connection_deduplication", None),
    )
    if kwargs:
        raise TypeError(
            f"Unexpected expect_and_grad keyword argument(s): {', '.join(sorted(kwargs))}."
        )
    validate_operator_hilbert(vstate, operator)
    if expectation_transform is None:
        terms = linear_operator_terms(operator)
        if any(isinstance(term.operator, PenaltyCost) for term in terms):
            gradients = [
                jax.tree_util.tree_map(
                    lambda leaf, coefficient=term.coefficient: coefficient * leaf,
                    expect_and_grad(
                        vstate,
                        term.operator,
                        samples=samples,
                        options=options,
                        use_covariance=False,
                    )[1],
                )
                for term in terms
            ]
            total = gradients[0]
            for gradient in gradients[1:]:
                total = _tree_add(total, gradient)
            return (
                local_estimators(
                    vstate, operator, samples=samples, options=options
                ).to_stats(),
                total,
            )
        biadjoint = _expect_and_grad_biadjoint(vstate, operator, samples, options)
        if biadjoint is not None:
            return biadjoint
    model_chunk_size = options.chunk_size
    plan = prepare_estimator(
        operator,
        samples,
        operator_sharding=options.operator_sharding,
        operator_streaming=options.operator_streaming,
        operator_streaming_chunk_size=options.operator_streaming_chunk_size,
        enabled=options.connection_deduplication,
    )

    squared_terms = tuple(operator) if is_operator_sequence(operator) else (operator,)
    if (
        plan is None
        and expectation_transform is None
        and squared_terms
        and all(
            isinstance(term, Squared) and isinstance(term.operator, DiscreteOperator)
            for term in squared_terms
        )
    ):
        # Connections are independent of the network parameters. Prepare them
        # before AD so sparsity counts cannot synchronize between model passes.
        connections = tuple(
            prepare_discrete_connections(
                term.operator,
                flat_samples,
                operator_streaming=options.operator_streaming,
                operator_streaming_chunk_size=options.operator_streaming_chunk_size,
            )
            for term in squared_terms
        )
        return _squared_expect_and_grad(
            vstate.apply_variables_kernel,
            vstate.merge_variables_kernel,
            vstate.parameters,
            vstate.model_state,
            flat_samples,
            connections,
            vstate.machine_pow,
            sample_shape=sample_shape,
            chunk_size=model_chunk_size,
        )

    coordinates, reassemble = split_tree_to_real(vstate.parameters)

    def values_for_params(real_parameters):
        parameters = reassemble(real_parameters)
        if plan is not None:
            variables = vstate.merge_variables_kernel(parameters, vstate.model_state)
            return plan.evaluate(
                vstate.apply_variables_kernel,
                variables,
                flat_samples,
                chunk_size=model_chunk_size,
            )
        return _local_values_for_params(
            vstate,
            operator,
            samples,
            parameters,
            chunk_size=model_chunk_size,
            operator_sharding=options.operator_sharding,
            operator_streaming=options.operator_streaming,
            operator_streaming_chunk_size=options.operator_streaming_chunk_size,
        ).reshape((-1,))

    def log_pdf_for_params(real_parameters):
        parameters = reassemble(real_parameters)
        variables = vstate.merge_variables_kernel(parameters, vstate.model_state)
        logpsi = apply_batched(
            vstate.apply_variables_kernel,
            variables,
            flat_samples,
            chunk_size=model_chunk_size,
        )
        return jnp.asarray(vstate.machine_pow) * jnp.real(logpsi)

    def objective_for_params(real_parameters):
        values = values_for_params(real_parameters)
        log_pdf = log_pdf_for_params(real_parameters)
        # This term is zero in value and contributes the normalized sampling
        # score to the derivative. Keep the direct estimator derivative in the
        # same graph so both contributions share one pullback.
        centered = jax.lax.stop_gradient(values - jnp.mean(values))
        score = log_pdf - jax.lax.stop_gradient(log_pdf)
        objective = jnp.mean(values + centered * score)
        if expectation_transform is not None:
            objective = expectation_transform(objective)
        return objective, values

    # Real coordinates preserve both channels of a complex objective.
    # Reassembly gives dE/dx + i dE/dy for z=x+iy, also for mixed trees and
    # non-holomorphic models. Conjugating this pullback would change dE.
    objective, pullback, values = neuralqx_vjp(
        objective_for_params, coordinates, has_aux=True
    )
    stats = statistics(values.reshape(sample_shape))
    gradient = reassemble(pullback(jnp.ones_like(objective))[0])
    return stats, gradient


@partial(
    jax.jit,
    static_argnames=(
        "apply_variables",
        "merge_variables",
        "sample_shape",
        "chunk_size",
    ),
)
def _squared_expect_and_grad(
    apply_variables,
    merge_variables,
    parameters,
    model_state,
    samples,
    connections,
    machine_pow,
    *,
    sample_shape,
    chunk_size,
):
    def log_value(params):
        variables = merge_variables(params, model_state)
        return apply_batched(apply_variables, variables, samples, chunk_size=chunk_size)

    logpsi, source_pullback = jax.vjp(log_value, parameters)
    value_dtype = jnp.result_type(
        logpsi.real.dtype,
        *(jnp.empty((), mels.dtype).real.dtype for _, mels, _ in connections),
    )
    carry = (
        jnp.zeros(len(samples), value_dtype),
        jax.tree_util.tree_map(jnp.zeros_like, parameters),
    )

    # Scan equal-sized terms so only one vertex's network activations are live
    # at a time. An unrolled VJP retains all vertices and can require gigabytes.
    buckets = {}
    for targets, mels, owners in connections:
        key = (targets.shape, targets.dtype, mels.shape, mels.dtype, owners is None)
        buckets.setdefault(key, []).append((targets, mels, owners))

    def term_gradient(carry, connection):
        total, gradient = carry
        targets, mels, owners = connection
        sources = samples[:, None, :] if owners is None else samples[owners]
        targets = jnp.where((mels != 0)[..., None], targets, sources)

        def target_loss(params):
            variables = merge_variables(params, model_state)
            target_values = apply_batched(
                apply_variables,
                variables,
                targets.reshape((-1, samples.shape[-1])),
                chunk_size=chunk_size,
            )
            if owners is None:
                delta = target_values.reshape(mels.shape) - logpsi[:, None]
            else:
                delta = target_values - logpsi[owners]
            contributions = mels * jnp.exp(jnp.where(mels != 0, delta, 0))
            if owners is None:
                local = jnp.sum(contributions, axis=-1)
            else:
                local = (
                    jnp.zeros(len(samples), contributions.dtype)
                    .at[owners]
                    .add(contributions)
                )
            values = jnp.abs(local) ** 2
            return jnp.mean(values), values

        (_, values), target_gradient = jax.value_and_grad(target_loss, has_aux=True)(
            parameters
        )
        return (total + values, _tree_add(gradient, target_gradient)), None

    for bucket in buckets.values():
        stacked = jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays), *bucket)
        carry, _ = jax.lax.scan(term_gradient, carry, stacked)

    values, gradient = carry
    stats = statistics(values.reshape(sample_shape))
    # The denominator contributes -2 f d Re(logpsi). Add the sampling score
    # p (f - mean(f)) d Re(logpsi) before the shared source pullback. This keeps
    # the full estimator derivative, including for machine_pow other than two.
    source_cotangent = (
        (machine_pow - 2) * values - machine_pow * stats.mean
    ) / values.size
    source_gradient = source_pullback(source_cotangent.astype(logpsi.dtype))[0]
    gradient = _tree_add(gradient, source_gradient)
    return stats, jax.tree_util.tree_map(jnp.conjugate, gradient)


def _expect_and_grad_biadjoint(
    vstate: Any,
    operator: Any,
    samples: Any,
    options: LocalEstimatorOptions,
) -> tuple[Stats, Any] | None:
    r"""Use both matrix actions to avoid differentiating connected-state models.

    For each real parameter coordinate, Born sampling gives
    :math:`\partial E_A = \operatorname{Cov}(O^*, L_A)
    + \operatorname{Cov}(O, L_{A^\dagger}^*)`.
    This identity applies to linear local estimators. Positive norm estimators
    and nonlinear penalties must retain their explicit parameter derivatives.
    """
    if not machine_pow_is_two(vstate.machine_pow):
        return None
    if discrete_linear_terms(operator) is None:
        return None
    if can_use_covariance(operator):
        return None
    adjoint = _operator_adjoint_or_none(operator)
    if adjoint is None:
        return None

    estimators = local_estimators(vstate, operator, samples=samples, options=options)
    stats = estimators.to_stats()
    adjoint_estimators = (
        estimators
        if adjoint is operator
        else local_estimators(vstate, adjoint, samples=samples, options=options)
    )
    # Only the mean is needed here, not another set of chain diagnostics.
    values = jnp.asarray(estimators.data).reshape((-1,))
    adjoint_values = jnp.asarray(adjoint_estimators.data).reshape((-1,))
    flat_samples, _ = flatten_samples(samples, vstate.hilbert.size)
    gradient = _biadjoint_gradient(
        vstate.apply_variables_kernel,
        vstate.merge_variables_kernel,
        vstate.parameters,
        vstate.model_state,
        flat_samples,
        values - stats.mean,
        adjoint_values - jnp.mean(adjoint_values),
        chunk_size=options.chunk_size,
    )
    return stats, gradient


@partial(jax.jit, static_argnames=("apply_variables", "merge_variables", "chunk_size"))
def _biadjoint_gradient(
    apply_variables,
    merge_variables,
    parameters,
    model_state,
    samples,
    centered,
    adjoint_centered,
    *,
    chunk_size,
):
    """Pull back amplitude and phase contributions on the sampled states once."""
    coordinates, reassemble = split_tree_to_real(parameters)

    def log_channels(real_parameters):
        variables = merge_variables(reassemble(real_parameters), model_state)
        logpsi = apply_batched(
            apply_variables, variables, samples, chunk_size=chunk_size
        )
        return logpsi.real, logpsi.imag

    (real_logpsi, imag_logpsi), pullback = jax.vjp(log_channels, coordinates)
    # Write O = du + i dv. The full complex derivative is
    # du (L_A + conj(L_Adagger)) + i dv (conj(L_Adagger) - L_A).
    # Pull back its real and imaginary parts without projecting either away.
    amplitude = (centered + jnp.conjugate(adjoint_centered)) / centered.size
    phase = 1j * (jnp.conjugate(adjoint_centered) - centered) / centered.size
    real_gradient = pullback(
        (amplitude.real.astype(real_logpsi.dtype), phase.real.astype(imag_logpsi.dtype))
    )[0]
    if not jnp.iscomplexobj(centered) and not jnp.iscomplexobj(adjoint_centered):
        return reassemble(real_gradient)
    imag_gradient = pullback(
        (amplitude.imag.astype(real_logpsi.dtype), phase.imag.astype(imag_logpsi.dtype))
    )[0]
    return reassemble(
        jax.tree_util.tree_map(
            lambda real, imag: real + 1j * imag, real_gradient, imag_gradient
        )
    )


def _operator_adjoint_or_none(operator: Any) -> Any | None:
    if is_operator_sequence(operator):
        adjoints = []
        for item in operator:
            adjoint = _operator_adjoint_or_none(item)
            if adjoint is None:
                return None
            adjoints.append(adjoint)
        return tuple(adjoints)
    try:
        return operator.adjoint
    except (AttributeError, NotImplementedError):
        return None


def _local_values_for_params(
    vstate: Any,
    operator: Any,
    samples: Any,
    parameters: Any,
    *,
    chunk_size: int | None,
    operator_sharding: bool | None,
    operator_streaming: bool | None,
    operator_streaming_chunk_size: int | None,
) -> jax.Array:
    variables = vstate.merge_variables_kernel(parameters, vstate.model_state)
    terms = discrete_linear_terms(operator)
    if terms is not None and (is_operator_sequence(operator) or len(terms) > 1):
        return discrete_operator_terms_local_values(
            vstate.apply_variables_kernel,
            variables,
            samples,
            terms,
            chunk_size=chunk_size,
            operator_sharding=operator_sharding,
            operator_streaming=operator_streaming,
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        )
    if is_operator_sequence(operator):
        total = None
        for item in operator:
            values = _local_values_for_params(
                vstate,
                item,
                samples,
                parameters,
                chunk_size=chunk_size,
                operator_sharding=operator_sharding,
                operator_streaming=operator_streaming,
                operator_streaming_chunk_size=operator_streaming_chunk_size,
            )
            total = values if total is None else total + values
        assert total is not None
        return total
    if isinstance(operator, Squared):
        return squared_operator_local_values(
            vstate.apply_variables_kernel,
            variables,
            samples,
            operator,
            chunk_size=chunk_size,
            operator_streaming=operator_streaming,
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        )
    if isinstance(operator, DiscreteOperator):
        return discrete_operator_local_values(
            vstate.apply_variables_kernel,
            variables,
            samples,
            operator,
            chunk_size=chunk_size,
            operator_streaming=operator_streaming,
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        )
    raise NotImplementedError(
        f"Non-covariance gradients are not implemented for operator type {type(operator).__name__}."
    )


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


__all__ = [
    "expect_and_forces",
    "expect_and_grad",
    "force_to_grad",
]
