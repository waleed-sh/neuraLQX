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

"""
This file includes the implementation of raising and lowering operators as well as flux operators
which are used in quantum reduced loop gravity.
"""

from typing import Union

import numpy as np
import jax.numpy as jnp
from scipy import sparse as _sparse

from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.operators.types import FunctionalLocalOperator


def creation(
    H: AbstractHilbertInterface,
    site: int,
    n: int,
) -> FunctionalLocalOperator:
    """
    This function will yield a NetKet LocalOperator which acts on the given site and increments its
    associated quantum number by N. If that happens to be above the cutoff of the Hilbert space, it
    produces the zero state.

    Note:
        The produced operator is not cyclic in the sense that when raising the highest possible
        basis vector, it gives the lowest possible basis vector. This is different from how the
        holonomy operator acts.

    :param H: the Hilbert space this operator is to be defined on
    :param site: the site to act on
    :param n: the amount of increment to be applied to the site's label
    """

    # get the number of local states at the given vertex
    # this will determine the dimension of the matrix acting on that vertex
    dim = H.hilbert.size_at_index(site)

    # create an empty matrix of size (dim x dim)
    C = np.zeros((dim, dim))

    # populate the matrix accordingly based on how many increments n are requested
    # this eventually is the creation operator with n = 1, to the power of the requested n
    # which has the form:   Cᵢⱼ = δᵢ₊ₙ,ⱼ
    for i in range(dim - n):
        C[i, i + n] = 1

    # turn it into sparse
    C = _sparse.coo_matrix(C)

    # return the LocalOperator
    return FunctionalLocalOperator(H.hilbert, C, [site], 0, dtype=jnp.float64)


def annihilation(
    H: AbstractHilbertInterface,
    site: int,
    n: int,
) -> FunctionalLocalOperator:
    """
    This function will yield a NetKet LocalOperator which acts on the given site and decrements its
    associated quantum number by N. If that happens to be below the cutoff of the Hilbert space, it
    produces the zero state.

    Note:
        The produced operator is not cyclic in the sense that when annihilating the lowest possible
        basis vector, it gives the lowest possible basis vector back. This is different from how the
        holonomy operator acts.

    :param H: the Hilbert space this operator is to be defined on
    :param site: the site to act on
    :param n: the amount of decrement to be applied to the site's label
    """

    # get the number of local states at the given vertex
    # this will determine the dimension of the matrix acting on that vertex
    dim = H.hilbert.size_at_index(site)

    # create an empty matrix of size (dim x dim)
    A = np.zeros((dim, dim))

    # populate the matrix accordingly based on how many decrements n are requested
    # this eventually is the annihilation operator with n = 1, to the power of the requested n
    # which has the form:   Aᵢⱼ = δᵢ,ⱼ₊ₙ
    for i in range(n, dim):
        A[i, i - n] = 1

    # turn it into sparse
    A = _sparse.coo_matrix(A)

    # return the LocalOperator
    return FunctionalLocalOperator(H.hilbert, A, [site], 0, dtype=jnp.float64)


def flux(
    H: AbstractHilbertInterface,
    site: int,
    power: float = 1,
    inverse: bool = False,
) -> FunctionalLocalOperator:
    """
    The flux operator of quantum reduced loop gravity. This acts as a number operator essentially.
    You can also specify the desired power to output not just the associated spin, but the spin to
    that power as the eigenvalue of this operator.

    :param H: the Hilbert space this operator is to be defined on
    :param site: the site the operator acts on
    :param power: the power to which the eigenvalues should be raised
    :param inverse: a flag to determine if we want the eigenvalues to be 1/eigenvalue or not
    :return:
    """

    def _inv(e: Union[float, int], power: float) -> Union[float, int]:
        """
        A helper function which returns either 1/e or e based on the inverse flag
        """

        if e == 0:
            return 0

        if inverse:
            return (1 / e) ** power
        return e**power

    # create the diagonal
    # here, we will replace any 0 basis labels by a small value epsilon
    # this epsilon will not have a power applied to it, since 0**power is still 0...
    diag = np.array(
        [float(_inv(n, power)) for n in H.core.allowed_basis_states.all_states()]
    )

    # construct the matrix
    E = np.diag(diag, 0)

    E = _sparse.coo_matrix(E)

    return FunctionalLocalOperator(H.hilbert, E, [site], 0, dtype=jnp.float64)
