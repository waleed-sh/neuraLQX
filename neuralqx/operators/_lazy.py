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

from netket.operator import AbstractOperator
from netket.operator._lazy import WrappedOperator
from netket.operator._lazy import Transpose

from plum import parametric

from neuralqx.utils.errors import NonHermitianInverseCostError


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
