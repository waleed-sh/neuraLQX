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
This file contains the implementation of the U(1) holonomy operator
"""

import numpy as np
from scipy import sparse as _sparse

from netket.operator import LocalOperator

from neuralqx.hilbert import AbstractHilbertInterface


def holonomy(
    H: AbstractHilbertInterface,
    site: int,
    adjoint: int = 0,
    o_type: str = "creation",
) -> LocalOperator:
    """
    A method to create a NetKet operator that acts as the holonomy operator in U(1) theory
    on a given vertex. It can either be a creation or annihilation holonomy as well as an adjoint.

    :param H: the Hilbert space the operator acts on
    :param site: (int):  the vertex it acts on in the graph
    :param adjoint: (int):  an identifier to check if the adjoint of the operator is requested
    :param o_type: (String):  an identifier to check the operation type (annihilation or creation)
    :param is_4d: a flag to check if we are working in 4 spacetime dimensions
    :return: (netket.operator.LocalOperator): the holonomy operator on the given vertex
    """

    # get the number of local states at the given vertex
    # this will determine the dimension of the matrix acting on that vertex
    N = H.hilbert_netket.size_at_index(site)

    # create an array of size N of just 1's
    D = np.array([1 for _ in np.arange(1, N)])

    # check the requested operation type
    if o_type == "creation":
        # if the requested operation is creation, insert this array above the diagonal
        matrix = np.diag(D, 1)

        # set the lowest right element in the matrix to 1 to impose the periodic condition
        # that if a vertex has maximum state, and we apply the creation holonomy to it, it is
        # given the minimum state
        matrix[N - 1, 0] = 1
    else:
        # if the requested operation is annihilation, insert this array below the diagonal
        matrix = np.diag(D, -1)

        # set the right most top element in the matrix to 1 to impose the periodic condition
        # that if a vertex has minimum state, and we apply the annihilation holonomy to it, it
        # is given the maximum state
        matrix[0, N - 1] = 1

    # check if the adjoint of the holonomy is requested
    if adjoint == 1:
        # if so, we take the adjoint of the matrix
        matrix = np.conj(matrix.T)

    # create a Coordinate sparse matrix
    matrix = _sparse.coo_matrix(matrix)

    # return the Netket operator defined on the Hilbert space acting on the given vertex with the
    # coordinate sparse matrix representing the holonomy operator constructed
    return LocalOperator(H.hilbert_netket, matrix, [site], 0)
