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
This file contains the implementation of the U(1) number operator
"""

# TODO: these implementations assume integer step. See creation.py for arbitrary step

from typing import Union

from netket.operator import LocalOperator
from scipy import sparse as _sparse
import jax.numpy as jnp
import numpy as np

from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.operators.types import FunctionalLocalOperator


def _create_delta(
    is_complex: bool,
    complex_gc: bool,
    D: np.ndarray,
) -> np.ndarray:
    """
    create an appropriate delta matrix given the bool conditions
    """

    if is_complex and not complex_gc:
        return np.diag(
            np.array([1 + 1 * 1j for _ in range(-np.max(D), np.max(D) + 1, 1)])
        )

    return np.identity(D.shape[0])


def _get_quantum_number_kernel(
    H: AbstractHilbertInterface,
    is_complex: bool = False,
    complex_gc: bool = False,
) -> tuple[np.ndarray, np.dtype]:

    if is_complex and not complex_gc:
        # get the allowed spin
        D = H.hilbert.allowed_basis_states.all_states()

        # get the maximum spin
        N = np.max(D)

        # create a diagonal with -N-Ni ... N+Ni
        complex_diagonal = np.array([n + n * 1j for n in range(-N, N + 1, 1)])

        # create the diagonal matrix
        diagonal_matrix = np.diag(complex_diagonal, 0)

        dtype = jnp.complex128

    elif complex_gc and not is_complex:
        # create a real value diagonal matrix with values -N to N on the diagonal
        D = H.hilbert.allowed_basis_states.all_states()
        diagonal_matrix = np.diag(D, 0)

        dtype = jnp.complex128
    else:
        # create a real value diagonal matrix with values -N to N on the diagonal
        D = H.hilbert.allowed_basis_states.all_states()
        diagonal_matrix = np.diag(D, 0)

        dtype = jnp.float64

    return diagonal_matrix, dtype


def get_quantum_number(
    H: AbstractHilbertInterface,
    site: int,
    is_complex: bool = False,
    complex_gc: bool = False,
    is_4d: bool = False,
) -> Union[LocalOperator, FunctionalLocalOperator]:
    """
    A function to return the quantum number at the given site. Essentially, it acts like a number
    operator which is just a diagonal matrix where the diagonal is composed of the allowed
    quantum numbers

    :param H: the Hilbert space the operator acts on
    :param site: (int): the vertex on which the operator should act
    :return: (netket.operator.LocalOperator): the operator returning the quantum number on the given
        site
    """

    matrix, dtype = _get_quantum_number_kernel(H, is_complex, complex_gc)

    """
    Changes:
        Added is_4d checks
        
    Reasons:
        This should be a modified operator which handles real and imaginary parts alone otherwise it
        will cause a mismatch in signs in the 4d volume operator
    """
    if is_4d:
        return FunctionalLocalOperator(
            H.hilbert_netket,
            _sparse.coo_matrix(matrix),
            [site],
            dtype=dtype,
            specialised=True,
        )

    # return the local operator
    return LocalOperator(
        H.hilbert_netket,
        _sparse.coo_matrix(matrix),
        [site],
        dtype=dtype,
    )


def shifted_get_quantum_number(
    H: AbstractHilbertInterface,
    site: int,
    shift_dict: dict,
    graph_level: int,
    comp_idx: int,
    direction: str,
    is_complex: bool = False,
    complex_gc: bool = False,
    is_4d: bool = False,
) -> Union[LocalOperator, FunctionalLocalOperator]:
    """
    A function to return the shifted quantum number at the given site. Essentially, it acts like a
    number operator which is just a diagonal matrix where the diagonal is composed of the allowed
    quantum numbers. The shift is introduced by adding as many deltas as there are in the `h`
    sub-dict of the shift_dict and subtracting as many deltas as there are in the `ha` sub-dict.

    :param complex_gc: a flag to determine whether we want complex valued operators to be used in
     the Gauss constraint
    :param is_complex: a flag to determine whether we want complex valued operators
    :param comp_idx: the current component graph level being computed in the triple-scalar-product
    :param graph_level: the current graph level we are in
    :param shift_dict: the dict containing all the shifts
    :param is_4d: a flag to determine whether we work in 4-spacetime dimensions
    :param H: the Hilbert space the operator acts on
    :param site: the vertex on which the operator should act
    :param direction: the direction of the shift
    :return: (netket.operator.LocalOperator): the operator returning the quantum number on the given
        site
    """

    # if the comp_idx != graph_level, then we just return the normal number operator because the
    # δ_IJ will be zero
    if comp_idx != graph_level:
        return get_quantum_number(
            H,
            site,
            is_complex=is_complex,
            complex_gc=complex_gc,
            is_4d=is_4d,
            # _mtrc = True,
        )

    matrix, dtype = _get_quantum_number_kernel(H, is_complex, complex_gc)

    # now we have the normal number operator stored in `matrix` acting on the `site`-th edge
    # we loop through the shift_dict and if we find a shift for the same `site`-th edge, we insert
    # a contribution for the deltas accordingly (based on if it is in `h` or `ha`

    # create the delta to be added/subtracted, just the identity matrix
    δ = _create_delta(
        is_complex,
        complex_gc,
        H.hilbert.allowed_basis_states.all_states(),
    )

    # direction factor to reverse the + or - of deltas
    df = -1 if direction == "left" else 1

    # dev: can we get away with just the shift_dict? the edges and site will already be shifted, so
    #  graph level can be inferred from there. Having hem, though, enables us to skip these loops

    # loop through the holonomy contributions first
    for edge in shift_dict["h"]:
        if edge == site:
            # dev: here mod_add can be used
            matrix = matrix - df * δ

    # loop through the adjoint holonomy contributions
    for edge in shift_dict["ha"]:
        if edge == site:
            # dev: here mod_add can be used
            matrix = matrix + df * δ

    """
    Changes:
        Added is_4d checks

    Reasons:
        This should be a modified operator which handles real and imaginary parts alone otherwise it
        will cause a mismatch in signs in the 4d volume operator
    """
    # return the shifted matrix
    if is_4d:
        return FunctionalLocalOperator(
            H.hilbert_netket,
            _sparse.coo_matrix(matrix),
            [site],
            dtype=dtype,
            specialised=True,
        )

    return LocalOperator(
        H.hilbert_netket,
        _sparse.coo_matrix(matrix),
        [site],
        dtype=dtype,
    )
