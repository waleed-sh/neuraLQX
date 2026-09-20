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


"""Plum-dispatched local estimator entry points."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from plum import dispatch

from neuralqx.operator import DiscreteOperator
from neuralqx.operator import Squared
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.operator.wrappers import penalty_local_value_coefficients
from neuralqx.utils.typing import EstimatorRule

from .kernels import discrete_operator_local_values
from .kernels import discrete_operator_terms_local_values
from .kernels import squared_operator_local_values
from .operators import discrete_linear_terms
from .operators import is_operator_sequence
from .options import LocalEstimatorOptions
from .statistics import LocalEstimators
from .utils import validate_operator_hilbert


def local_estimators(
    vstate: Any,
    operator: Any,
    *,
    samples: Any | None = None,
    options: LocalEstimatorOptions | None = None,
    chunk_size: int | None = None,
    operator_sharding: bool | None = None,
    operator_streaming: bool | None = None,
    operator_streaming_chunk_size: int | None = None,
    operator_chunk_size: int | None = None,
) -> LocalEstimators:
    """Compute per-sample local estimators for ``operator`` on ``vstate``."""
    if samples is None:
        samples = vstate.samples
    options = LocalEstimatorOptions.resolve(
        vstate,
        options=options,
        chunk_size=chunk_size,
        operator_sharding=operator_sharding,
        operator_streaming=operator_streaming,
        operator_streaming_chunk_size=operator_streaming_chunk_size,
        operator_chunk_size=operator_chunk_size,
    )
    validate_operator_hilbert(vstate, operator)
    terms = discrete_linear_terms(operator)
    if terms is not None and (is_operator_sequence(operator) or len(terms) > 1):
        values = discrete_operator_terms_local_values(
            vstate.apply_variables_kernel,
            vstate.variables,
            samples,
            terms,
            chunk_size=options.chunk_size,
            operator_sharding=options.operator_sharding,
            operator_streaming=options.operator_streaming,
            operator_streaming_chunk_size=options.operator_streaming_chunk_size,
        )
        return LocalEstimators(values)
    if is_operator_sequence(operator):
        total = None
        for item in operator:
            values = local_estimators(
                vstate, item, samples=samples, options=options
            ).data
            total = values if total is None else total + values
        assert total is not None
        return LocalEstimators(total)
    return local_estimator_kernel(
        vstate,
        operator,
        samples,
        options,
    )


@dispatch
def local_estimator_kernel(
    vstate: Any,
    operator: DiscreteOperator,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    values = discrete_operator_local_values(
        vstate.apply_variables_kernel,
        vstate.variables,
        samples,
        operator,
        chunk_size=options.chunk_size,
        operator_streaming=options.operator_streaming,
        operator_streaming_chunk_size=options.operator_streaming_chunk_size,
    )
    return LocalEstimators(values)


@local_estimator_kernel.dispatch
def _squared_local_estimator_kernel(
    vstate: Any,
    operator: Squared,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    values = squared_operator_local_values(
        vstate.apply_variables_kernel,
        vstate.variables,
        samples,
        operator,
        chunk_size=options.chunk_size,
        operator_streaming=options.operator_streaming,
        operator_streaming_chunk_size=options.operator_streaming_chunk_size,
    )
    return LocalEstimators(values)


@local_estimator_kernel.dispatch
def _penalty_local_estimator_kernel(
    vstate: Any,
    penalty: PenaltyCost,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    parent = local_estimators(
        vstate,
        penalty.cost_operator,
        samples=samples,
        options=options,
    )
    parent_stats = parent.to_stats()
    scale, shift = penalty_local_value_coefficients(penalty, parent_stats.mean)
    values = jnp.asarray(scale) * parent.data + jnp.asarray(shift)
    return LocalEstimators(values)


@local_estimator_kernel.dispatch(precedence=-100)
def local_estimator_kernel_fallback(
    vstate: Any,
    operator: Any,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    del vstate, samples, options
    raise NotImplementedError(
        "local_estimators is not implemented for operator type "
        f"{type(operator).__name__}."
    )


def register_local_estimator(
    fn: EstimatorRule | None = None,
    *,
    precedence: int = 0,
):
    """Register a custom local-estimator kernel.

    Custom kernels should have signature
    ``(vstate, operator, samples, options) -> LocalEstimators`` and type-annotate
    the operator argument. The returned values must preserve ``samples.shape[:-1]``.
    """
    decorator = local_estimator_kernel.dispatch(precedence=precedence)
    if fn is None:
        return decorator
    return decorator(fn)


__all__ = [
    "EstimatorRule",
    "LocalEstimatorOptions",
    "LocalEstimators",
    "local_estimator_kernel",
    "local_estimators",
    "register_local_estimator",
]
