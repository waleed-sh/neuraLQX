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

# Copyright 2021 The NetKet Authors - All rights reserved.
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

import jax.numpy as jnp

from netket.operator import AbstractOperator
from netket.operator._lazy import WrappedOperator
from netket.operator._lazy import Transpose
from netket.utils.dispatch import dispatch

from plum import parametric

from neuralqx.utils.errors import NonHermitianInverseCostError
from neuralqx.utils.parsing import strict_type


@parametric
class PenaltyCost(WrappedOperator):
    """
    A wrapper lazily representing a penalty term to be added to the cost function such that:
        cost = <C> + factor * <P>

    The penalty term operator term can be any discrete operator, but should always have an attribute
    `factor`, which will then be used to weight the penalty being applied to the cost function
    """

    parent: AbstractOperator

    def __init__(self, op: AbstractOperator, factor: float):
        super().__init__(op)
        self._factor = factor

    @property
    def cost_operator(self):
        return self.parent

    @property
    def dtype(self):
        return self.parent.dtype

    def collect(self):
        return self.parent.H.collect()

    @property
    def T(self):
        self.parent.c @ self.parent.T
        self.parent
        return self.parent.conjugate()

    def conjugate(self):
        return Transpose(self.parent)

    @property
    def H(self):
        return self

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def factor(self):
        return self._factor

    def is_linear_penalty(self) -> bool:
        """
        Return whether this wrapper is a pure constant rescaling of its parent.

        The concrete :class:`PenaltyCost` keeps the historical fast path:
        its local estimator is the parent local estimator multiplied by
        :attr:`factor` outside the kernel. Subclasses are routed through the
        generic affine objective protocol by default, which lets users only
        override the objective value and derivative without touching MCState
        internals. A subclass that is still strictly linear can override this
        method or the dispatched :func:`penalty_is_linear` implementation.
        """
        return strict_type(self) is PenaltyCost

    def linear_scale(self):
        """
        Return the constant scale for the linear fast path.

        This is used only when :meth:`is_linear_penalty` is true.
        """
        return self.factor

    def expectation_value(self, parent_expectation):
        """
        Map ``<parent>`` to this penalty contribution.

        Subclasses may override this method, or register a Plum dispatch
        implementation for :func:`penalty_expectation_value`.
        """
        return self.factor * parent_expectation

    def expectation_gradient(self, parent_expectation):
        """
        Return ``d expectation_value(<parent>) / d <parent>``.

        The returned scalar multiplies the wrapped operator's forces/gradient.
        """
        return jnp.asarray(self.factor, dtype=jnp.result_type(parent_expectation))

    def local_value_coefficients(self, parent_expectation):
        """
        Return ``(scale, shift)`` for the affine local estimator.

        For any differentiable scalar objective ``F(<O>)`` the estimator

            ``F'(<O>) * O_loc(sigma) + F(<O>) - F'(<O>) * <O>``

        has the correct Monte Carlo mean and yields the chain-rule gradient.
        """
        scale = penalty_expectation_gradient(self, parent_expectation)
        value = penalty_expectation_value(self, parent_expectation)
        return scale, value - scale * parent_expectation

    def __repr__(self):
        return (
            f"PenaltyCost(CostOperator = {type(self.cost_operator).__name__}, "
            f"factor = {self.factor})"
        )

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def _op__matmul__(self, other):
        return self.collect() @ other


@parametric
class InverseExpectationCost(PenaltyCost):
    """
    A wrapper lazily representing a penalty term to be added to the cost function such that:
        cost = <C> + factor / (alpha + <V>)^2

    The penalty term operator term can be any discrete operator, but should always have an attribute
    `factor`, which will then be used to weight the penalty being applied to the cost function and
    an attribute `alpha`, which will be in the denominator of the inverse volume computation
    """

    parent: AbstractOperator

    eps: float = 1e-24

    def __init__(self, op: AbstractOperator, factor: float, alpha: float):
        super().__init__(op, factor=factor)
        self._factor = factor
        self._alpha = alpha

        # check for hermiticity as we use the covariance method to compute the expectation
        # value of the wrapped operator
        if not op.is_hermitian:
            raise NonHermitianInverseCostError()

    @property
    def cost_operator(self):
        return self.parent

    @property
    def dtype(self):
        return self.parent.dtype

    def collect(self):
        raise NotImplementedError(
            f".collect() is not implemented for {type(self).__name__} type."
        )
        # return self.parent.H.collect()

    @property
    def T(self):
        # self.parent.c @ self.parent.T
        # self.parent
        # return self.parent.conjugate()
        raise NotImplementedError(
            f".T is not implemented for {type(self).__name__} type."
        )

    def conjugate(self):
        return Transpose(self.parent)

    @property
    def H(self):
        return self

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def factor(self):
        return self._factor

    @property
    def alpha(self):
        return self._alpha

    def is_linear_penalty(self) -> bool:
        return False

    def expectation_value(self, parent_expectation):
        parent_expectation = jnp.real(parent_expectation)
        denom = self.alpha + parent_expectation + self.eps
        return self.factor / (denom**2)

    def expectation_gradient(self, parent_expectation):
        parent_expectation = jnp.real(parent_expectation)
        denom = self.alpha + parent_expectation + self.eps
        return (-2.0 * self.factor) / (denom**3)

    def local_value_coefficients(self, parent_expectation):
        parent_expectation = jnp.real(parent_expectation)
        scale = penalty_expectation_gradient(self, parent_expectation)
        value = penalty_expectation_value(self, parent_expectation)
        return scale, value - scale * parent_expectation

    def __repr__(self):
        return (
            f"InverseExpectationCost(CostOperator = {type(self.cost_operator).__name__}, "
            f"factor = {self.factor}, alpha = {self.alpha})"
        )

    def __mul__(self, other):
        # return self.collect() * other
        raise NotImplementedError(
            f"multiplication is not implemented for {type(self).__name__} type."
        )

    def __rmul__(self, other):
        # return other * self.collect()
        raise NotImplementedError(
            f"multiplication is not implemented for {type(self).__name__} type."
        )

    def _op__matmul__(self, other):
        # return self.collect() @ other
        raise NotImplementedError(
            f"multiplication is not implemented for {type(self).__name__} type."
        )


@dispatch
def penalty_is_linear(operator: PenaltyCost) -> bool:
    """
    Return whether a penalty wrapper can use the constant-scale fast path.

    This is a Plum-dispatched extension point. Register a more specific
    implementation for custom wrappers that need different routing.
    """
    return bool(operator.is_linear_penalty())


@dispatch
def penalty_linear_scale(operator: PenaltyCost):
    """
    Return the constant local-estimator scale for a linear penalty wrapper.
    """
    return operator.linear_scale()


@dispatch
def penalty_expectation_value(operator: PenaltyCost, parent_expectation):
    """
    Evaluate the penalty objective from the wrapped operator expectation.

    Custom subclasses can override :meth:`PenaltyCost.expectation_value` or
    register a more specific Plum implementation of this function.
    """
    return operator.expectation_value(parent_expectation)


@dispatch
def penalty_expectation_gradient(operator: PenaltyCost, parent_expectation):
    """
    Evaluate the derivative of the penalty objective with respect to ``<O>``.
    """
    return operator.expectation_gradient(parent_expectation)


@dispatch
def penalty_local_value_coefficients(operator: PenaltyCost, parent_expectation):
    """
    Return affine local-estimator coefficients ``(scale, shift)``.

    The local estimator is ``scale * O_loc + shift``. This is the only
    information the expectation, forces and gradient paths need for nonlinear
    expectation-level penalty objectives.
    """
    return operator.local_value_coefficients(parent_expectation)
