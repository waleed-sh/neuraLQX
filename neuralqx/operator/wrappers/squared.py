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


"""Semantic squared-operator marker."""

from __future__ import annotations

import jax.numpy as jnp

from ._utils import result_dtype
from .base import WrappedOperator


class Squared(WrappedOperator):
    r"""A lazy marker for the positive operator $A^\dagger A$.

    For a normalized state, its expectation is $\|A\psi\|^2$. The VQS layer
    recognizes this object and selects an estimator without expanding the
    product's connectivity. Consequently this wrapper does not provide
    `get_conn_padded` or matrix conversion. Use `A.adjoint @ A` when an explicit
    computational product is needed.

    This operation differs from `A @ A` when $A$ is not Hermitian.
    With matrix-row connectivity, the estimator samples

    $$
    \left|\frac{(A\psi)(q)}{\psi(q)}\right|^2,
    $$

    whose Born-weighted mean is $\langle A^\dagger A\rangle$.
    Thus `Squared(H.adjoint)` represents $H H^\dagger$, while `Squared(H)`
    represents $H^\dagger H$. Taking a quadratic constraint does not change
    the meaning of the parent's `adjoint` property.

    Attributes:
        operator (AbstractOperator): Parent operator from which this wrapper is constructed.
    """

    def __add__(self, other):
        """Add independent squared terms without expanding product connections."""
        from .sum_of_squares import SumOfSquares

        if isinstance(other, Squared):
            return SumOfSquares((self.operator, other.operator))
        if isinstance(other, SumOfSquares):
            return SumOfSquares((self.operator, *other.operators))
        return NotImplemented

    def __radd__(self, other):
        """Accept the zero starting value used by Python's sum."""
        from neuralqx.operator._algebra import is_zero_scalar_like

        return self if is_zero_scalar_like(other) else NotImplemented

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return result_dtype(self.operator)

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return True

    @property
    def is_diagonal(self) -> bool:
        """Whether every nonzero matrix element connects a configuration to itself."""
        return bool(getattr(self.operator, "is_diagonal", False))

    @property
    def adjoint(self) -> Squared:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self


__all__ = ["Squared"]
