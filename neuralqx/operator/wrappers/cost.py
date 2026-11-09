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
    r"""An expectation-level objective built from another operator.

    The base penalty maps the parent expectation $m=\langle A\rangle$ to
    $f(m)=\lambda m$. Subclasses can supply a nonlinear objective by overriding
    `expectation_value`, `expectation_gradient`, and `local_value_coefficients`.

    This wrapper describes an objective for the expectation engine. It does
    not represent a new matrix or provide connected configurations. For a
    nonlinear penalty, applying $f$ to a sampled local value would give a
    different objective from $f(\langle A\rangle)$.

    Attributes:
        operator (AbstractOperator): Parent operator from which this wrapper is constructed.
        factor (Any): Overall penalty weight $\lambda$.
    """

    factor: Any = field(default=1.0)
    r"""Overall penalty weight $\lambda$."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        super().__post_init__()
        object.__setattr__(self, "factor", canonical_scalar(self.factor))

    @property
    def cost_operator(self) -> AbstractOperator:
        """Parent operator whose expectation determines this penalty."""
        return self.operator

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.dtype(jnp.result_type(self.operator.dtype, self.factor))

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return bool(self.operator.is_hermitian and is_known_real_scalar(self.factor))

    @property
    def adjoint(self) -> PenaltyCost:
        """Return this expectation-objective marker without constructing a matrix adjoint."""
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
    r"""An inverse-square penalty on the expectation of a Hermitian operator.

    For $m=\operatorname{Re}\langle A\rangle$, the objective and its derivative are

    $$
    f(m)=\frac{\lambda}{(\alpha+m+\varepsilon)^2},\qquad
    f'(m)=-\frac{2\lambda}{(\alpha+m+\varepsilon)^3}.
    $$

    The offset `eps` is added to the denominator. It is not an absolute-value
    floor, so the objective remains singular when the full denominator is
    zero. The parent operator must declare itself Hermitian.

    Attributes:
        operator (AbstractOperator): Parent operator from which this wrapper is constructed.
        factor (Any): Overall penalty weight $\lambda$.
        alpha (Any): Offset added to the real parent expectation in the denominator.
        eps (float): Small additive denominator offset, without clipping or absolute values.
    """

    alpha: Any = field(default=0.0)
    """Offset added to the real parent expectation in the denominator."""

    eps: float = field(static=True, default=1e-24)
    """Small additive denominator offset, without clipping or absolute values."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        super().__post_init__()
        if not self.operator.is_hermitian:
            raise ValueError("InverseExpectationCost requires a Hermitian operator.")
        object.__setattr__(self, "alpha", canonical_scalar(self.alpha))

    def is_linear_penalty(self) -> bool:
        """Return false because the inverse-square objective depends nonlinearly on the mean."""
        return False

    def expectation_value(self, parent_expectation: Any) -> jnp.ndarray:
        r"""Evaluate $\lambda/(\alpha+\operatorname{Re}(m)+\varepsilon)^2$ at the parent mean."""
        parent = jnp.real(parent_expectation)
        denom = self.alpha + parent + self.eps
        return self.factor / (denom * denom)

    def expectation_gradient(self, parent_expectation: Any) -> jnp.ndarray:
        r"""Evaluate the derivative $-2\lambda/(\alpha+\operatorname{Re}(m)+\varepsilon)^3$."""
        parent = jnp.real(parent_expectation)
        denom = self.alpha + parent + self.eps
        return (-2.0 * self.factor) / (denom * denom * denom)

    def local_value_coefficients(
        self,
        parent_expectation: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the scale and shift for the affine local estimator.

        The scale is the derivative at the current real parent mean. The shift
        makes the transformed estimator's mean equal the penalty objective.
        """
        parent = jnp.real(parent_expectation)
        scale = self.expectation_gradient(parent)
        value = self.expectation_value(parent)
        return scale, value - scale * parent_expectation


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
