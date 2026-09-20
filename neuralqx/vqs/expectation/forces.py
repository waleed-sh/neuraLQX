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


"""Covariance-force estimators for variational states."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from plum import dispatch

import jax
from neuralqx.jax.differentiation import vjp as neuralqx_vjp
from neuralqx.jax.sharding import place_sample_batch
from neuralqx.jax.streaming import apply_batched
from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.vqs.estimators import LocalEstimatorOptions
from neuralqx.vqs.estimators import Stats
from neuralqx.vqs.estimators import local_estimators
from neuralqx.vqs.estimators import statistics
from neuralqx.vqs.estimators.kernels import local_values_from_connections
from neuralqx.vqs.estimators.operators import discrete_linear_terms
from neuralqx.vqs.estimators.operators import is_operator_sequence
from neuralqx.vqs.estimators.operators import sequence_is_hermitian
from neuralqx.vqs.estimators.utils import flatten_samples
from neuralqx.vqs.estimators.utils import validate_operator_hilbert

from .penalties import penalty_expect_and_forces


def expect_and_forces(
    vstate: Any,
    operator: Any,
    **kwargs: Any,
) -> tuple[Stats, Any]:
    """Estimate expectation statistics and covariance forces."""
    if is_operator_sequence(operator):
        return _expect_and_forces_sequence(vstate, operator, dict(kwargs))
    return _expect_and_forces(vstate, operator, dict(kwargs))


def _expect_and_forces_sequence(
    vstate: Any,
    operator: Any,
    kwargs: dict[str, Any],
) -> tuple[Stats, Any]:
    if not sequence_is_hermitian(operator):
        raise ValueError("Forces are only defined for Hermitian operators.")
    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = vstate.samples
    estimators = local_estimators(vstate, operator, samples=samples, **kwargs)
    stats = estimators.to_stats()
    forces = forces_from_local_values(vstate, samples, estimators.data, stats=stats)
    return stats, forces


@dispatch
def _expect_and_forces(vstate: Any, operator: PenaltyCost, kwargs: dict[str, Any]):
    return penalty_expect_and_forces(vstate, operator, kwargs)


@_expect_and_forces.dispatch
def _expect_and_forces_discrete(
    vstate: Any, operator: DiscreteOperator, kwargs: dict[str, Any]
):
    if not getattr(operator, "is_hermitian", False):
        raise ValueError("Forces are only defined for Hermitian operators.")
    terms = discrete_linear_terms(operator)
    if terms is not None and len(terms) > 1:
        return _expect_and_forces_sequence(vstate, (operator,), kwargs)
    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = vstate.samples
    operator_streaming_chunk_size = kwargs.pop("operator_streaming_chunk_size", None)
    operator_chunk_size = kwargs.pop("operator_chunk_size", None)
    options = LocalEstimatorOptions.resolve(
        vstate,
        options=kwargs.pop("options", None),
        chunk_size=kwargs.pop("chunk_size", None),
        operator_sharding=kwargs.pop("operator_sharding", None),
        operator_streaming=kwargs.pop("operator_streaming", None),
        operator_streaming_chunk_size=operator_streaming_chunk_size,
        operator_chunk_size=operator_chunk_size,
    )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected expect_and_forces keyword argument(s): {unknown}.")
    validate_operator_hilbert(vstate, operator)
    flat_samples, sample_shape = flatten_samples(samples, operator.hilbert.size)
    x_primes, mels = operator.get_conn_padded(
        flat_samples,
        use_streaming=options.operator_streaming,
        chunk_size=options.operator_streaming_chunk_size,
    )
    return _expect_and_forces_from_connections_kernel(
        vstate.apply_variables_kernel,
        vstate.merge_variables_kernel,
        vstate.variables,
        vstate.parameters,
        vstate.model_state,
        flat_samples,
        x_primes,
        mels,
        sample_shape,
        options.chunk_size,
    )


@dispatch
def _expect_and_forces(vstate: Any, operator: Any, kwargs: dict[str, Any]):
    if isinstance(operator, Squared):
        raise ValueError(
            "Covariance forces are not valid for Squared operators because "
            "their positive local estimator depends on the variational "
            "parameters. Use expect_and_grad instead."
        )
    if not getattr(operator, "is_hermitian", False):
        raise ValueError("Forces are only defined for Hermitian operators.")
    samples = kwargs.pop("samples", None)
    if samples is None:
        samples = vstate.samples
    estimators = local_estimators(vstate, operator, samples=samples, **kwargs)
    stats = estimators.to_stats()
    forces = forces_from_local_values(vstate, samples, estimators.data, stats=stats)
    return stats, forces


@partial(
    jax.jit,
    static_argnames=(
        "apply_variables",
        "merge_variables",
        "sample_shape",
        "chunk_size",
    ),
)
def _expect_and_forces_from_connections_kernel(
    apply_variables: Any,
    merge_variables: Any,
    variables: Any,
    parameters: Any,
    model_state: Any,
    flat_samples: jax.Array,
    x_primes: jax.Array,
    mels: jax.Array,
    sample_shape: tuple[int, ...],
    chunk_size: int | None,
) -> tuple[Stats, Any]:
    flat_values = local_values_from_connections(
        apply_variables,
        variables,
        flat_samples,
        x_primes,
        mels,
        chunk_size=chunk_size,
    )
    stats = statistics(flat_values.reshape(sample_shape))
    forces = _forces_from_local_values_kernel(
        apply_variables,
        merge_variables,
        parameters,
        model_state,
        flat_samples,
        flat_values,
        stats.mean,
        chunk_size,
    )
    return stats, forces


def forces_from_local_values(
    vstate: Any,
    samples: Any,
    local_values: Any,
    *,
    stats: Stats | None = None,
) -> Any:
    """Compute covariance forces using one VJP through ``logpsi``."""
    flat_samples, _sample_shape = flatten_samples(samples, vstate.hilbert.size)
    flat_values = jnp.asarray(local_values).reshape((-1,))
    if stats is None:
        stats = statistics(local_values)
    flat_samples = place_sample_batch(flat_samples)
    flat_values = place_sample_batch(flat_values)
    return _forces_from_local_values_kernel(
        vstate.apply_variables_kernel,
        vstate.merge_variables_kernel,
        vstate.parameters,
        vstate.model_state,
        flat_samples,
        flat_values,
        stats.mean,
        vstate.chunk_size,
    )


def forces_from_cotangent(
    vstate: Any,
    samples: Any,
    cotangent: Any,
) -> Any:
    """Pull back a precomputed log-amplitude cotangent over ``samples``."""
    flat_samples, _sample_shape = flatten_samples(samples, vstate.hilbert.size)
    flat_cotangent = jnp.asarray(cotangent).reshape((-1,))
    flat_samples = place_sample_batch(flat_samples)
    flat_cotangent = place_sample_batch(flat_cotangent)
    return _forces_from_cotangent_kernel(
        vstate.apply_variables_kernel,
        vstate.merge_variables_kernel,
        vstate.parameters,
        vstate.model_state,
        flat_samples,
        flat_cotangent,
        vstate.chunk_size,
    )


@partial(
    jax.jit,
    static_argnames=("apply_variables", "merge_variables", "chunk_size"),
)
def _forces_from_local_values_kernel(
    apply_variables: Any,
    merge_variables: Any,
    parameters: Any,
    model_state: Any,
    flat_samples: jax.Array,
    flat_values: jax.Array,
    mean: Any,
    chunk_size: int | None,
) -> Any:
    centered = flat_values - mean
    n_samples = jnp.asarray(flat_values.shape[0], dtype=jnp.result_type(centered))
    cotangent = jnp.conjugate(centered) / n_samples
    return _forces_from_cotangent_kernel(
        apply_variables,
        merge_variables,
        parameters,
        model_state,
        flat_samples,
        cotangent,
        chunk_size,
    )


@partial(
    jax.jit,
    static_argnames=("apply_variables", "merge_variables", "chunk_size"),
)
def _forces_from_cotangent_kernel(
    apply_variables: Any,
    merge_variables: Any,
    parameters: Any,
    model_state: Any,
    flat_samples: jax.Array,
    cotangent: jax.Array,
    chunk_size: int | None,
) -> Any:

    def logpsi_params(params):
        variables = merge_variables(params, model_state)
        return apply_batched(
            apply_variables,
            variables,
            flat_samples,
            chunk_size=chunk_size,
        )

    _logpsi, pullback = neuralqx_vjp(logpsi_params, parameters, conjugate=True)
    return pullback(cotangent)[0]


@jax.jit
def force_to_grad(forces: Any, parameters: Any, *, machine_pow: Any = 2.0) -> Any:
    """Convert covariance forces to JAX optimizer-compatible gradients."""

    def convert(force_leaf: Any, param_leaf: Any) -> Any:
        force_arr = jnp.asarray(force_leaf)
        param_arr = jnp.asarray(param_leaf)
        scale = jnp.asarray(machine_pow, dtype=force_arr.dtype)
        if jnp.iscomplexobj(param_arr):
            return (scale * force_arr).astype(param_arr.dtype)
        return (scale * jnp.real(force_arr)).astype(param_arr.dtype)

    return jax.tree_util.tree_map(convert, forces, parameters)


def can_use_covariance(operator: Any) -> bool:
    """Return whether the fast covariance force path is valid."""
    if is_operator_sequence(operator):
        return sequence_is_hermitian(operator)
    if isinstance(operator, Squared):
        return False
    if isinstance(operator, PenaltyCost):
        return bool(
            operator.is_hermitian and can_use_covariance(operator.cost_operator)
        )
    return bool(getattr(operator, "is_hermitian", False))


__all__ = [
    "can_use_covariance",
    "expect_and_forces",
    "force_to_grad",
    "forces_from_cotangent",
    "forces_from_local_values",
]
