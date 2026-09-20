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

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.differentiation import vjp as neuralqx_vjp
from neuralqx.jax.streaming import apply_batched
from neuralqx.jax.tree import tree_any_complex
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.vqs.estimators import Stats
from neuralqx.vqs.estimators import local_estimators
from neuralqx.vqs.estimators import statistics
from neuralqx.vqs.estimators.kernels import discrete_operator_local_values
from neuralqx.vqs.estimators.kernels import discrete_operator_terms_local_values
from neuralqx.vqs.estimators.kernels import squared_operator_local_values
from neuralqx.vqs.estimators.operators import discrete_linear_terms
from neuralqx.vqs.estimators.operators import is_operator_sequence
from neuralqx.vqs.estimators.utils import flatten_samples

from .forces import can_use_covariance
from .forces import expect_and_forces
from .forces import force_to_grad
from .forces import forces_from_cotangent
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
        if use_covariance is None:
            use_covariance = can_use_covariance(operator)
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
) -> tuple[Stats, Any]:
    if isinstance(operator, PenaltyCost):
        return penalty_expect_and_grad(vstate, operator, kwargs, False)

    biadjoint = _expect_and_grad_biadjoint(vstate, operator, dict(kwargs))
    if biadjoint is not None:
        return biadjoint

    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = vstate.samples
    flat_samples, sample_shape = flatten_samples(samples, vstate.hilbert.size)
    n_samples = jnp.asarray(flat_samples.shape[0], dtype=jnp.float32)
    model_chunk_size = kwargs.pop("chunk_size", vstate.chunk_size)
    operator_streaming_chunk_size = kwargs.get(
        "operator_streaming_chunk_size",
        kwargs.get("operator_chunk_size"),
    )

    def values_for_params(parameters):
        return _local_values_for_params(
            vstate,
            operator,
            samples,
            parameters,
            chunk_size=model_chunk_size,
            operator_sharding=kwargs.get("operator_sharding"),
            operator_streaming=kwargs.get("operator_streaming"),
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        ).reshape((-1,))

    def log_pdf_for_params(parameters):
        variables = vstate.merge_variables_kernel(parameters, vstate.model_state)
        logpsi = apply_batched(
            vstate.apply_variables_kernel,
            variables,
            flat_samples,
            chunk_size=model_chunk_size,
        )
        return jnp.asarray(vstate.machine_pow) * jnp.real(logpsi)

    values, values_pullback = neuralqx_vjp(
        values_for_params,
        vstate.parameters,
        conjugate=True,
    )
    stats = statistics(values.reshape(sample_shape))
    direct_grad = values_pullback(jnp.ones_like(values) / values.size)[0]

    log_pdf, score_pullback = neuralqx_vjp(log_pdf_for_params, vstate.parameters)
    centered = values - stats.mean
    if jnp.iscomplexobj(centered):
        score_real = score_pullback(
            (jnp.real(centered) / n_samples).astype(log_pdf.dtype)
        )[0]
        score_imag = score_pullback(
            (jnp.imag(centered) / n_samples).astype(log_pdf.dtype)
        )[0]
        score_grad = jax.tree_util.tree_map(
            lambda real, imag: real + 1j * imag,
            score_real,
            score_imag,
        )
    else:
        score_grad = score_pullback((centered / n_samples).astype(log_pdf.dtype))[0]
    return stats, _tree_add(direct_grad, score_grad)


def _expect_and_grad_biadjoint(
    vstate: Any,
    operator: Any,
    kwargs: dict[str, Any],
) -> tuple[Stats, Any] | None:
    """Try the adjoint-assisted covariance gradient for non-Hermitian operators."""
    if isinstance(operator, Squared):
        return None
    if bool(getattr(operator, "is_hermitian", False)):
        return None
    if tree_any_complex(vstate.parameters):
        return None
    if not _machine_pow_is_two(vstate.machine_pow):
        return None

    adjoint = _operator_adjoint_or_none(operator)
    if adjoint is None:
        return None

    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = vstate.samples

    options_kwargs = {
        "options": kwargs.pop("options", None),
        "chunk_size": kwargs.pop("chunk_size", None),
        "operator_sharding": kwargs.pop("operator_sharding", None),
        "operator_streaming": kwargs.pop("operator_streaming", None),
        "operator_streaming_chunk_size": kwargs.pop(
            "operator_streaming_chunk_size", None
        ),
        "operator_chunk_size": kwargs.pop("operator_chunk_size", None),
    }
    options_kwargs = {
        key: value for key, value in options_kwargs.items() if value is not None
    }

    estimators = local_estimators(
        vstate,
        operator,
        samples=samples,
        **options_kwargs,
    )
    stats = estimators.to_stats()

    if adjoint is operator:
        centered = jnp.asarray(estimators.data).reshape((-1,)) - stats.mean
        n_samples = jnp.asarray(centered.shape[0], dtype=jnp.result_type(centered))
        cotangent = (jnp.conjugate(centered) + centered) / n_samples
        forces = forces_from_cotangent(vstate, samples, cotangent)
        return stats, _combined_biadjoint_force_to_grad(forces, vstate.parameters)
    else:
        adjoint_estimators = local_estimators(
            vstate,
            adjoint,
            samples=samples,
            **options_kwargs,
        )
        adjoint_stats = adjoint_estimators.to_stats()
        centered = jnp.asarray(estimators.data).reshape((-1,)) - stats.mean
        adjoint_centered = (
            jnp.asarray(adjoint_estimators.data).reshape((-1,)) - adjoint_stats.mean
        )
        n_samples = jnp.asarray(centered.shape[0], dtype=jnp.result_type(centered))
        cotangent = (jnp.conjugate(centered) + adjoint_centered) / n_samples
        forces = forces_from_cotangent(vstate, samples, cotangent)

    grad = _combined_biadjoint_force_to_grad(forces, vstate.parameters)
    return stats, grad


@jax.jit
def _combined_biadjoint_force_to_grad(forces: Any, parameters: Any) -> Any:
    """Cast a combined biadjoint pullback to optimizer-compatible dtypes."""

    def convert(force_leaf: Any, param_leaf: Any) -> Any:
        force_arr = jnp.asarray(force_leaf)
        param_arr = jnp.asarray(param_leaf)
        if jnp.iscomplexobj(param_arr):
            return force_arr.astype(param_arr.dtype)
        return jnp.real(force_arr).astype(param_arr.dtype)

    return jax.tree_util.tree_map(convert, forces, parameters)


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


def _machine_pow_is_two(machine_pow: Any) -> bool:
    try:
        return bool(jnp.isclose(jnp.asarray(machine_pow), 2.0))
    except TypeError:
        return False


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
        "Non-covariance gradients are not implemented for operator type "
        f"{type(operator).__name__}."
    )


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


__all__ = [
    "expect_and_forces",
    "expect_and_grad",
    "force_to_grad",
]
