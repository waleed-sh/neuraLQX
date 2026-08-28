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


"""Expectation-level cost wrappers for operators."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from plum import dispatch

from neuralqx.operator._algebra import is_known_real_scalar
from neuralqx.operator.abstract import AbstractOperator
from neuralqx.utils.struct import field

from ._utils import canonical_scalar
from .base import WrappedOperator


class PenaltyCost(WrappedOperator):
    """A penalty objective driven by the expectation of one operator.

    The default penalty is linear: ``factor * <operator>``. Nonlinear subclasses
    define the same small protocol used by the expectation engine:
    ``expectation_value``, ``expectation_gradient``, and
    ``local_value_coefficients``.
    """

    factor: Any = field(default=1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "factor", canonical_scalar(self.factor))

    @property
    def cost_operator(self) -> AbstractOperator:
        return self.operator

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(jnp.result_type(self.operator.dtype, self.factor))

    @property
    def is_hermitian(self) -> bool:
        return bool(self.operator.is_hermitian and is_known_real_scalar(self.factor))

    @property
    def adjoint(self) -> PenaltyCost:
        return self

    def is_linear_penalty(self) -> bool:
        """Return whether the penalty is a pure rescaling of the parent."""
        return type(self) is PenaltyCost

    def linear_scale(self) -> jnp.ndarray:
        """Return the fast-path scale for linear penalties."""
        return jnp.asarray(self.factor, dtype=self.dtype)

    def expectation_value(self, parent_expectation: Any) -> jnp.ndarray:
        """Map ``<operator>`` to the penalty objective contribution."""
        return (
            jnp.asarray(self.factor, dtype=jnp.result_type(parent_expectation))
            * parent_expectation
        )

    def expectation_gradient(self, parent_expectation: Any) -> jnp.ndarray:
        """Return derivative with respect to the parent expectation."""
        return jnp.asarray(self.factor, dtype=jnp.result_type(parent_expectation))

    def local_value_coefficients(
        self,
        parent_expectation: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``scale, shift`` for an affine transformed local estimator."""
        scale = self.expectation_gradient(parent_expectation)
        value = self.expectation_value(parent_expectation)
        return scale, value - scale * parent_expectation


class InverseExpectationCost(PenaltyCost):
    """Penalty objective ``factor / (alpha + real(<operator>) + eps)^2``."""

    alpha: Any = field(default=0.0)
    eps: float = field(static=True, default=1e-24)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.operator.is_hermitian:
            raise ValueError("InverseExpectationCost requires a Hermitian operator.")
        object.__setattr__(self, "alpha", canonical_scalar(self.alpha))

    def is_linear_penalty(self) -> bool:
        return False

    def expectation_value(self, parent_expectation: Any) -> jnp.ndarray:
        parent = jnp.real(parent_expectation)
        denom = self.alpha + parent + self.eps
        return self.factor / (denom * denom)

    def expectation_gradient(self, parent_expectation: Any) -> jnp.ndarray:
        parent = jnp.real(parent_expectation)
        denom = self.alpha + parent + self.eps
        return (-2.0 * self.factor) / (denom * denom * denom)

    def local_value_coefficients(
        self,
        parent_expectation: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        parent = jnp.real(parent_expectation)
        scale = self.expectation_gradient(parent)
        value = self.expectation_value(parent)
        return scale, value - scale * parent


@dispatch
def penalty_is_linear(operator: PenaltyCost) -> bool:
    """Return whether a penalty wrapper is a constant local-estimator scaling."""
    return bool(operator.is_linear_penalty())


@dispatch
def penalty_linear_scale(operator: PenaltyCost) -> jnp.ndarray:
    """Return the constant local-estimator scale for a linear penalty."""
    return operator.linear_scale()


@dispatch
def penalty_expectation_value(
    operator: PenaltyCost,
    parent_expectation,
) -> jnp.ndarray:
    """Evaluate the penalty objective from the parent local-estimator mean."""
    return operator.expectation_value(parent_expectation)


@dispatch
def penalty_expectation_gradient(
    operator: PenaltyCost,
    parent_expectation,
) -> jnp.ndarray:
    """Evaluate derivative with respect to the parent local-estimator mean."""
    return operator.expectation_gradient(parent_expectation)


@dispatch
def penalty_local_value_coefficients(
    operator: PenaltyCost,
    parent_expectation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return affine coefficients for transforming parent local estimators."""
    return operator.local_value_coefficients(parent_expectation)


__all__ = [
    "InverseExpectationCost",
    "PenaltyCost",
    "penalty_expectation_gradient",
    "penalty_expectation_value",
    "penalty_is_linear",
    "penalty_linear_scale",
    "penalty_local_value_coefficients",
]
