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
This file contains some functions representing characterisation operators
"""

from typing import Callable

from scipy import sparse as _sparse
import netket as nk
import numpy as np

from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.utils.errors import InvalidChargeError


def charge_coloring(
    H: AbstractHilbertInterface,
    site: int,
    charge: int,
):
    """
    A local operator which acts on the given edge and is always zero unless the edge has the
    specified charge

    :param H: the Hilbert space it acts on
    :param site: the site it acts on
    :param charge: the charge it looks for
    """

    # get the allowed states
    D = H.hilbert.allowed_basis_states.all_states()

    if charge > H.hilbert.cutoff:
        raise InvalidChargeError(charge, H.cutoff)

    # create a diagonal with only 1 at the specified charge position
    diag = np.array([1 if x == charge else 0 for x in D])

    # create the operator matrix
    matrix = _sparse.diags(diag, 0, format="coo")

    return nk.operator.LocalOperator(H.hilbert_netket, matrix, [site], 0)


def coloring(
    H: AbstractHilbertInterface,
    site: int,
    charge_vector: tuple[int, ...],
):
    """
    A local operator which acts on the given edge and is always zero unless the edge has the
    specified charge vector

    :param H: the Hilbert space it acts on
    :param site: the edge it acts on (this should be on the first level graph for an N-dim gauge
        group
    :param charge_vector: the charge vector specified
    """

    shift_factor = H.tiny_size

    if site >= shift_factor:
        raise ValueError(
            f"The allowed edges are the ones in the first level graph. "
            f"These are edges labeled by integers 0 to {shift_factor - 1}. Please make sure you "
            f"correctly select the edges only"
            f"in the first graph, the edges in higher level graphs will be automatically specified."
        )

    return (
        charge_coloring(H, site, charge_vector[0])
        * charge_coloring(H, site + shift_factor, charge_vector[1])
        * charge_coloring(H, site + 2 * shift_factor, charge_vector[2])
    )


def n_point_function(
    H: AbstractHilbertInterface,
    operator: Callable,
    edges: tuple[int, ...],
    charge_vectors: tuple[tuple[int, ...], ...],
):
    """
    A function which returns the N-point function of a given operator. Here, "N" is determined by
    the number of edges passed onto the function. Hence, for 2 edges being specified you get the
    2-point function of the specified operator.

    :param H: the Hilbert space it acts on
    :param operator: the operator requested, this should be a callable, not a LocalOperator
    :param edges: the list of edges it acts on
    :param charge_vectors: the list of charge vectors, each corresponding to the edge and the same
        index in the edges param
    """

    if len(edges) != len(charge_vectors):
        raise ValueError(
            f"You have entered an unequal amount of edges (length {len(edges)}) and charge "
            f"vectors (length {len(charge_vectors)})"
        )

    if len(edges) == 1:
        return operator(H, edges[0], charge_vectors[0])

    return n_point_function(H, operator, edges[1:], charge_vectors[1:]) * operator(
        H, edges[0], charge_vectors[0]
    )
