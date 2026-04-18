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

from abc import abstractmethod

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import BCSR
from jax.experimental.sparse import JAXSparse

from netket.hilbert import DiscreteHilbert

from netket.utils.types import DType

from neuralqx.utils.errors import SuboptimalOperatorForGPUWarning
from neuralqx.utils.numbers import is_scalar_like
from neuralqx.utils.numbers import hermitian_flag_for_sum_with_scalar
from neuralqx.utils.numbers import promote_constant_for_op_dtype

from .._discrete_operator import DiscreteOperator


class ComputationalOperator(DiscreteOperator):

    def __init__(self, hilbert: DiscreteHilbert):
        if jax.default_backend() == "gpu":
            SuboptimalOperatorForGPUWarning()
        super().__init__(hilbert)

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

    @property
    @abstractmethod
    def is_hermitian(self) -> bool:
        """
        This function must return either ``True`` or ``False`` based on whether your operator is
        Hermitian or not. Note that unlike for the case of ``LocalOperator`` types, you must specify
        by-hand whether this operator is Hermitian or not, there is no implementation to deduce
        that information for you as there is no matrices stored in this operator type.

        This property plays a role in determining the computational path to be taken when computing
        gradients. If you specify that the operator is Hermitian while in reality it is not, the
        computed gradients will be incorrect.

        :return: ``True`` if this operator is Hermitian, ``False`` otherwise
        """

    @property
    @abstractmethod
    def dtype(self) -> DType:
        """
        Specify a JAX NumPy or NumPy dtype for this operator (that is, what is the dtype of the
        matrix elements returned by this operator)
        """

    @property
    def adjoint(self) -> "ComputationalOperator":
        """
        Return the adjoint (Hermitian conjugate) of this operator as a fully constructed
        operator of the same family type.

        Subclasses must override this property. For Hermitian operators the natural
        implementation is ``return self``. For non-Hermitian operators the implementation
        should construct and return the conjugate-transpose operator.

        Raises:
            NotImplementedError: if this operator family has not implemented an adjoint.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the `adjoint` property. "
            "Override it in your subclass to enable the bi-covariance gradient path."
        )

    @abstractmethod
    def _get_conn_padded_kernel(self, x_local: np.ndarray | jnp.ndarray):
        r"""
        This function should implement an algorithm that finds the connected elements of the
        operator.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of size :math:`B \times M`
        where :math:`B` size of the batch and :math:`M` size of the hilbert space, finds all states
        :math:`y_i^1, ..., y_i^K` connected to every :math:`x_i`.

        This function should returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every :math:`y_i`.
        """

    def to_sparse(self) -> JAXSparse:
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The sparse JAX matrix representation of the operator.
        """

        # TODO: If the operator get_conn_padded uses shard_map, the
        # replication of all_states will lead to a crash when
        # the n_samples cannot be divided by the number of ranks.
        # this should be fixed.
        x = self.hilbert.all_states()
        x = jnp.asarray(x, dtype=jnp.int64)
        n = x.shape[0]
        xp, mels = self.get_conn_padded(x)
        a = mels.ravel()
        i = np.broadcast_to(np.arange(n)[..., None], mels.shape).ravel()
        j = self.hilbert.states_to_numbers(xp).ravel()
        ij = np.concatenate((i[:, None], j[:, None]), axis=1)
        return BCSR.from_bcoo(BCOO((a, ij), shape=(n, n)))

    def to_dense(self) -> np.ndarray:
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The dense matrix representation of the operator as a JAX Array.
        """
        return self.to_sparse().todense()

    def get_conn_flattened(
        self,
        x: np.ndarray,
        sections: np.ndarray,
        pad: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

        Starting from a given quantum number :math:`x`, it finds all
        other quantum numbers  :math:`x'` such that the matrix element
        :math:`O(x,x')` is different from zero. In general there will be
        several different connected states :math:`x'` satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape
        :code:`(batch_size,hilbert.size)`.

        Args:
            x: A matrix of shape `(batch_size, hilbert.size)`
                containing the batch of quantum numbers x.
            sections: An array of sections for the flattened x'.
                See numpy.split for the meaning of sections.

        Returns:
            (matrix, array): The connected states x', flattened together in
                a single matrix.
                An array containing the matrix elements :math:`O(x,x')`
                associated to each x'.

        """
        del pad

        xp, mels = self.get_conn_padded(x)
        n_conns = mels.shape[1]
        xp = xp.reshape(-1, xp.shape[-1])
        mels = mels.reshape(-1)
        sections[:] = n_conns
        return xp, mels

    def __mul__(self, other):
        if isinstance(other, ComputationalOperator):
            from ...computational.wrappers import Product

            return Product(self, other)

        if is_scalar_like(other):
            from ...computational.wrappers import Scaled

            return Scaled(self, other)

        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, ComputationalOperator):
            return self.__mul__(other)

        if is_scalar_like(other):
            from ...computational.wrappers import Scaled

            return Scaled(self, other)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ComputationalOperator):
            return self.__mul__(other)

        if is_scalar_like(other):
            from ...computational.wrappers import Scaled

            return Scaled(self, other)

        return NotImplemented

    def __add__(self, other):
        if isinstance(other, ComputationalOperator):
            from ...computational.wrappers import Sum

            is_hermitian = self.is_hermitian and other.is_hermitian
            return Sum(self, other, is_hermitian=is_hermitian)

        if is_scalar_like(other):
            from ...computational.misc import IdentityOperator
            from ...computational.wrappers import Sum

            target_dtype, const_cast = promote_constant_for_op_dtype(self.dtype, other)
            # c_py = to_Python_scalar(const_cast)

            I = IdentityOperator(self.hilbert, const_cast, jax=False)
            is_hermitian = hermitian_flag_for_sum_with_scalar(
                self.is_hermitian, const_cast
            )

            return Sum(self, I, is_hermitian=is_hermitian)

        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, ComputationalOperator) or is_scalar_like(other):
            return self.__add__(other)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, ComputationalOperator) or is_scalar_like(other):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ComputationalOperator):
            from ...computational.wrappers import Sum

            is_hermitian = self.is_hermitian and other.is_hermitian
            return Sum(self, other, subtract=True, is_hermitian=is_hermitian)

        if is_scalar_like(other):
            from ...computational.misc import IdentityOperator
            from ...computational.wrappers import Sum

            target_dtype, const_cast = promote_constant_for_op_dtype(self.dtype, other)
            I = IdentityOperator(self.hilbert, const_cast, jax=False)

            # OP - c·I
            is_hermitian = hermitian_flag_for_sum_with_scalar(
                self.is_hermitian, const_cast
            )
            return Sum(self, I, subtract=True, is_hermitian=is_hermitian)

        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, ComputationalOperator):
            return self.__sub__(other)

        if is_scalar_like(other):
            from ...computational.misc import IdentityOperator
            from ...computational.wrappers import Sum

            target_dtype, const_cast = promote_constant_for_op_dtype(self.dtype, other)
            I = IdentityOperator(self.hilbert, const_cast, jax=False)

            # (c·I) - OP
            is_hermitian = hermitian_flag_for_sum_with_scalar(
                self.is_hermitian, const_cast
            )
            return Sum(I, self, subtract=True, is_hermitian=is_hermitian)

        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, ComputationalOperator) or is_scalar_like(other):
            return self.__sub__(other)
        return NotImplemented

    def __neg__(self):
        from ...computational.wrappers import Scaled

        return Scaled(self, -1.0)

    def __truediv__(self, other):
        if is_scalar_like(other):
            from ...computational.wrappers import Scaled

            target_dtype, c_cast = promote_constant_for_op_dtype(self.dtype, other)

            # guard zero division with friendly error
            c = jnp.asarray(c_cast, dtype=target_dtype)
            if (jnp.iscomplexobj(c) and jnp.all(jnp.isclose(c, 0))) or (
                not jnp.iscomplexobj(c) and float(c) == 0.0
            ):
                raise ZeroDivisionError("Operator division by zero scalar.")
            return Scaled(self, (1.0 / c))

        return NotImplemented
