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

import abc

from typing import Union

import numpy as np
import jax
import jax.numpy as jnp


from netket.operator import DiscreteJaxOperator

from neuralqx.debug import timeit
from neuralqx.utils.numbers import hermitian_flag_for_sum_with_scalar
from neuralqx.utils.numbers import is_scalar_like
from neuralqx.utils.numbers import promote_constant_for_op_dtype

ArrayLike = Union[np.ndarray, jax.numpy.ndarray]


class ComputationalJaxOperator(DiscreteJaxOperator):
    """A renamed abstract base class identical to DiscreteJaxOperator from NetKet.

    This class exists primarily for semantic clarity, e.g. when distinguishing
    between computational and other operator types.
    """

    _is_ket_action: bool = True
    """If True, the local estimator of the operator becomes one compatible with a ket action rather than bra action."""

    @property
    def is_ket_action(self) -> bool:
        """
        If True, when computing the expectation value for the operator, a local estimator taking into account that the
        operator acts on a ket, instead of a bra, is used.
        :return:
        """
        return self._is_ket_action

    @is_ket_action.setter
    def is_ket_action(self, val: bool) -> None:
        self._is_ket_action = val

    @timeit(tag="GCP_COMPOP_JAX", warn_ms=1000)
    def get_conn_padded(self, x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        r"""Finds the connected elements of the Operator. This method
        can be executed inside of a JAX function transformation.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.

        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """
        return self._get_conn_padded(x)

    @abc.abstractmethod
    def _get_conn_padded(self, x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Core function that will call a kernel that computes the connected configurations of the operator.

        This function implements the low-level computational routine for determining all states
        :math:`x'` connected to each configuration :math:`x` in a given batch, as well as their
        corresponding matrix elements :math:`O(x, x')`.

        It is typically invoked by :meth:`get_conn_padded`, and must be written in a way that is
        compatible with JAX transformations such as :func:`jax.jit`, :func:`jax.vmap`, and
        :func:`jax.grad`. In particular, all array operations should be JAX-compatible and avoid
        Python-side control flow or shape-dependent logic.
        """

    def __mul__(self, other):
        if isinstance(other, ComputationalJaxOperator):
            from ...computational.wrappers import ProductJax

            return ProductJax(self, other)

        if is_scalar_like(other):
            from ...computational.wrappers import ScaledJax

            return ScaledJax(self, other)

        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, ComputationalJaxOperator):
            return self.__mul__(other)

        if is_scalar_like(other):
            from ...computational.wrappers import ScaledJax

            return ScaledJax(self, other)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ComputationalJaxOperator):
            return self.__mul__(other)

        if is_scalar_like(other):
            from ...computational.wrappers import ScaledJax

            return ScaledJax(self, other)

        return NotImplemented

    def __add__(self, other):
        if isinstance(other, ComputationalJaxOperator):
            from ...computational.wrappers import SumJax

            is_hermitian = self.is_hermitian and other.is_hermitian
            return SumJax(self, other, is_hermitian=is_hermitian)

        if is_scalar_like(other):
            from ...computational.misc import IdentityOperator
            from ...computational.wrappers import SumJax

            target_dtype, const_cast = promote_constant_for_op_dtype(self.dtype, other)
            # c_py = to_Python_scalar(const_cast)

            I = IdentityOperator(self.hilbert, const_cast, jax=True)
            is_hermitian = hermitian_flag_for_sum_with_scalar(
                self.is_hermitian, const_cast
            )

            return SumJax(self, I, is_hermitian=is_hermitian)

        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, ComputationalJaxOperator) or is_scalar_like(other):
            return self.__add__(other)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, ComputationalJaxOperator) or is_scalar_like(other):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ComputationalJaxOperator):
            from ...computational.wrappers import SumJax

            is_hermitian = self.is_hermitian and other.is_hermitian
            return SumJax(self, other, subtract=True, is_hermitian=is_hermitian)

        if is_scalar_like(other):
            from ...computational.misc import IdentityOperator
            from ...computational.wrappers import SumJax

            target_dtype, const_cast = promote_constant_for_op_dtype(self.dtype, other)
            I = IdentityOperator(self.hilbert, const_cast, jax=True)

            # OP - c·I
            is_hermitian = hermitian_flag_for_sum_with_scalar(
                self.is_hermitian, const_cast
            )
            return SumJax(self, I, subtract=True, is_hermitian=is_hermitian)

        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, ComputationalJaxOperator):
            return self.__sub__(other)

        if is_scalar_like(other):
            from ...computational.misc import IdentityOperator
            from ...computational.wrappers import SumJax

            target_dtype, const_cast = promote_constant_for_op_dtype(self.dtype, other)
            I = IdentityOperator(self.hilbert, const_cast, jax=True)

            # (c·I) - OP
            is_hermitian = hermitian_flag_for_sum_with_scalar(
                self.is_hermitian, const_cast
            )
            return SumJax(I, self, subtract=True, is_hermitian=is_hermitian)

        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, ComputationalJaxOperator) or is_scalar_like(other):
            return self.__sub__(other)
        return NotImplemented

    def __neg__(self):
        from ...computational.wrappers import ScaledJax

        return ScaledJax(self, -1.0)

    def __truediv__(self, other):
        if is_scalar_like(other):
            from ...computational.wrappers import ScaledJax

            target_dtype, c_cast = promote_constant_for_op_dtype(self.dtype, other)

            # guard zero division with friendly error
            c = jnp.asarray(c_cast, dtype=target_dtype)
            if (jnp.iscomplexobj(c) and jnp.all(jnp.isclose(c, 0))) or (
                not jnp.iscomplexobj(c) and float(c) == 0.0
            ):
                raise ZeroDivisionError("Operator division by zero scalar.")
            return ScaledJax(self, (1.0 / c))

        return NotImplemented
