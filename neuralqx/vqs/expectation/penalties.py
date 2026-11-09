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


"""Expectation, force, and gradient rules for penalty objectives."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from plum import dispatch

import jax
from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.operator.wrappers import Squared
from neuralqx.operator.wrappers import SumOfSquares
from neuralqx.operator.wrappers import penalty_local_value_coefficients
from neuralqx.utils.typing import PenaltyExpectationRule
from neuralqx.utils.typing import PenaltyForcesRule
from neuralqx.utils.typing import PenaltyGradientRule
from neuralqx.vqs.estimators import Stats
from neuralqx.vqs.estimators import local_estimators
from neuralqx.vqs.estimators.penalties import transform_stats


@dispatch
def penalty_expect(
    vstate: Any,
    operator: PenaltyCost,
    kwargs: dict[str, Any],
) -> Stats:
    """Estimate a penalty objective from transformed local estimators."""
    return local_estimators(vstate, operator, **kwargs).to_stats()


@dispatch
def penalty_expect_and_forces(
    vstate: Any,
    operator: PenaltyCost,
    kwargs: dict[str, Any],
) -> tuple[Stats, Any]:
    """Estimate a penalty objective and covariance forces.

    The default rule applies the local-estimator chain rule
    ``F'(mean(parent)) * parent_loc + constant``. Custom nonlinear penalties can
    register a more specialised implementation for their subclass.
    """
    from .forces import expect_and_forces

    parent_stats, parent_forces = expect_and_forces(
        vstate,
        operator.cost_operator,
        **kwargs,
    )
    scale, shift = penalty_local_value_coefficients(operator, parent_stats.mean)
    stats = transform_stats(parent_stats, scale, shift)
    forces = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(scale) * leaf,
        parent_forces,
    )
    return stats, forces


@dispatch
def penalty_expect_and_grad(
    vstate: Any,
    operator: PenaltyCost,
    kwargs: dict[str, Any],
    use_covariance: Any,
) -> tuple[Stats, Any]:
    """Estimate a penalty objective and optimiser-compatible gradients."""
    from .forces import can_use_covariance
    from .forces import force_to_grad
    from .gradients import _expect_and_grad_noncovariance
    from .gradients import expect_and_grad

    if use_covariance is None:
        use_covariance = can_use_covariance(operator, machine_pow=vstate.machine_pow)
    if use_covariance:
        if not can_use_covariance(operator, machine_pow=vstate.machine_pow):
            raise ValueError(
                "Covariance gradients are not valid for this penalty and sampling "
                "distribution. Use use_covariance=False for the full derivative."
            )
        stats, forces = penalty_expect_and_forces(vstate, operator, kwargs)
        grad = force_to_grad(
            forces,
            vstate.parameters,
            machine_pow=vstate.machine_pow,
        )
        return stats, grad

    if not operator.is_linear_penalty() and not isinstance(
        operator.cost_operator, (Squared, SumOfSquares)
    ):
        # Apply the scalar objective before combining complex parameter
        # coordinates. A real-part penalty cannot in general be differentiated
        # by multiplying a complex parent's already assembled gradient.
        parent = operator.cost_operator
        transform = operator.expectation_value
        penalties = [operator]
        while isinstance(parent, PenaltyCost):
            outer, inner = transform, parent.expectation_value

            def transform(mean, outer=outer, inner=inner):
                return outer(inner(mean))

            penalties.append(parent)
            parent = parent.cost_operator
        stats, grad = _expect_and_grad_noncovariance(
            vstate, parent, dict(kwargs), expectation_transform=transform
        )
        for penalty in reversed(penalties):
            scale, shift = penalty_local_value_coefficients(penalty, stats.mean)
            stats = transform_stats(stats, scale, shift)
        return stats, grad

    parent_stats, parent_grad = expect_and_grad(
        vstate, operator.cost_operator, use_covariance=False, **kwargs
    )
    scale, shift = penalty_local_value_coefficients(operator, parent_stats.mean)
    stats = transform_stats(parent_stats, scale, shift)
    grad = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(scale) * leaf,
        parent_grad,
    )
    return stats, grad


def register_penalty_expect(
    fn: PenaltyExpectationRule | None = None,
    *,
    precedence: int = 0,
):
    """Register a custom expectation rule for a penalty subclass."""
    decorator = penalty_expect.dispatch(precedence=precedence)
    if fn is None:
        return decorator
    return decorator(fn)


def register_penalty_expect_and_forces(
    fn: PenaltyForcesRule | None = None,
    *,
    precedence: int = 0,
):
    """Register a custom force rule for a penalty subclass."""
    decorator = penalty_expect_and_forces.dispatch(precedence=precedence)
    if fn is None:
        return decorator
    return decorator(fn)


def register_penalty_expect_and_grad(
    fn: PenaltyGradientRule | None = None,
    *,
    precedence: int = 0,
):
    """Register a custom gradient rule for a penalty subclass."""
    decorator = penalty_expect_and_grad.dispatch(precedence=precedence)
    if fn is None:
        return decorator
    return decorator(fn)


__all__ = [
    "PenaltyExpectationRule",
    "PenaltyForcesRule",
    "PenaltyGradientRule",
    "penalty_expect",
    "penalty_expect_and_forces",
    "penalty_expect_and_grad",
    "register_penalty_expect",
    "register_penalty_expect_and_forces",
    "register_penalty_expect_and_grad",
]
