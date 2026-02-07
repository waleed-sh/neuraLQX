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

# Copyright 2022 The NetKet Authors - All rights reserved.
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
This file contains by-hand implementation of matrix multiplication for both dense NumPy and sparse
scipy matrices such that the multiplication of complex valued matrices are altered.

NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

from typing import Union

import numpy as np
import jax
import numpy.core.numeric as _nx

from scipy.sparse import spmatrix
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import bsr_matrix
from scipy.sparse import issparse
from scipy.sparse import hstack
from scipy.sparse import csc_matrix

from numba import vectorize
from numba import float64
from numba import complex128

from netket.utils.types import Array

from .helpers import _reorder_kronecker_product
from .helpers import _eye_like

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ModuleNotFoundError:
    _HAS_NUMBA = False


def _modified_c_mult(
    a: Union[np.ndarray, jax.Array, int, float, complex],
    b: Union[np.ndarray, jax.Array, int, float, complex],
) -> Union[np.ndarray, jax.Array, int, float, complex]:
    """
    A function which will take two inputs, either arrays (JAX or NumPy) or primitive numerical
    dtypes, and multiply them in a special way. This special was is that if one of them is real
    valued (or has zero imaginary part) and the other is complex valued, it will only multiply the
    real parts. It only multiplies the complex parts of both inputs have non-zero imaginary parts.
    """

    # get the real part to be returned
    real_part = np.where(a.real * b.real == -0.0, 0.0, a.real * b.real)

    # check for imaginary parts
    if np.any(b.imag) and np.any(a.imag):
        # both a and b are complex
        # multiply the imaginary parts
        return real_part + 1j * a.imag * b.imag

    elif not np.any(b.imag) and np.any(a.imag):
        # b is real and a is complex
        # multiply the real part of a with b, and leave imag(a) alone
        return np.where(
            ~np.isclose(np.abs(b.real), 0, atol=1e-2),
            real_part + 1j * a.imag,
            real_part,
        )
    elif np.any(b.imag) and not np.any(a.imag):
        # a is real and b is complex
        # multiply the real part of b with a, and leave imag(b) alone
        return np.where(
            ~np.isclose(np.abs(a.real), 0, atol=1e-2),
            real_part + 1j * b.imag,
            real_part,
        )
    else:
        # both a and b are real
        return real_part


# below are re-implementation of NumPy/scipy kron, dot and multiply functions for ndarrays or sparse
# arrays which utilise the above complex multiplication rule


def _npkron(a, b):

    b = np.asanyarray(b)
    a = np.array(a, copy=False, subok=True, ndmin=b.ndim)
    is_any_mat = isinstance(a, np.matrix) or isinstance(b, np.matrix)
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)

    if nda == 0 or ndb == 0:
        return _modified_c_mult(a, b)

    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = np.reshape(a, as_)
    if not b.flags.contiguous:
        b = np.reshape(b, bs)

    # Equalise the shapes by prepending smaller one with 1s
    as_ = (1,) * max(0, ndb - nda) + as_
    bs = (1,) * max(0, nda - ndb) + bs

    # Insert empty dimensions
    a_arr = np.expand_dims(a, axis=tuple(range(ndb - nda)))
    b_arr = np.expand_dims(b, axis=tuple(range(nda - ndb)))

    # Compute the product
    a_arr = np.expand_dims(a_arr, axis=tuple(range(1, nd * 2, 2)))
    b_arr = np.expand_dims(b_arr, axis=tuple(range(0, nd * 2, 2)))
    # In case of `mat`, convert result to `array`
    result = _modified_c_mult(a_arr, b_arr)

    # Reshape back
    result = result.reshape(_nx.multiply(as_, bs))

    return result if not is_any_mat else np.matrix(result, copy=False)


def _spkron(A, B, format=None):

    B = coo_matrix(B)

    if (format is None or format == "bsr") and 2 * B.nnz >= B.shape[0] * B.shape[1]:
        # B is fairly dense, use BSR
        A = csr_matrix(A, copy=True)
        output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_matrix(output_shape).asformat(format)

        B = B.toarray()
        data = A.data.repeat(B.size).reshape(-1, B.shape[0], B.shape[1])
        data = _modified_c_mult(data, B)

        return bsr_matrix((data, A.indices, A.indptr), shape=output_shape)
    else:
        # use COO
        A = coo_matrix(A)
        output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_matrix(output_shape).asformat(format)

        # expand entries of a into blocks
        row = A.row.repeat(B.nnz)
        col = A.col.repeat(B.nnz)
        data = A.data.repeat(B.nnz)

        if (
            max(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
            > np.iinfo("int32").max
        ):
            row = row.astype(np.int64)
            col = col.astype(np.int64)

        row *= B.shape[0]
        col *= B.shape[1]

        # increment block indices
        row, col = row.reshape(-1, B.nnz), col.reshape(-1, B.nnz)
        row += B.row
        col += B.col
        row, col = row.reshape(-1), col.reshape(-1)

        # compute block entries
        data = _modified_c_mult(data.reshape(-1, B.nnz), B.data)
        data = data.reshape(-1)

        return coo_matrix((data, (row, col)), shape=output_shape).asformat(format)


def _fdot(A, B):
    if issparse(A) and issparse(B):

        if _HAS_NUMBA:
            return _spdot_numba(A, B)
        else:
            return _spdot(A, B)

    # _sp = False
    #
    # if issparse(A) and issparse(B):
    #     A = np.asarray(A.todense())
    #     B = np.asarray(B.todense())
    #     _sp = True

    if _HAS_NUMBA:
        # TODO: THE NUMBA VERSIONS ARE DEPRECATED UNTIL ALL TESTS INT uidTests.ipynb PASS!!!
        # res = _fdot_numba(A, B)
        res = (_modified_c_mult(A[:, :, None], B[None, :, :])).sum(axis=1)
    else:
        res = (_modified_c_mult(A[:, :, None], B[None, :, :])).sum(axis=1)

    # if _sp:
    #     res = coo_matrix(res)

    return res


@njit(parallel=True, fastmath=True, cache=True)
def _fdot_numba(A, B):
    """
    Numba supported A @ B operation with a custom multiply rule. Assumes numpy.ndarray as inputs.
    """

    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError("Shapes not aligned")

    out_r = np.zeros((m, p), dtype=A.real.dtype)
    out_i = np.zeros((m, p), dtype=A.real.dtype)

    for i in prange(m):
        for k in range(n):
            ar = A[i, k].real
            ai = A[i, k].imag
            for j in range(p):
                br = B[k, j].real
                bi = B[k, j].imag

                # real part always ar * br
                out_r[i, j] += ar * br

                # imag part, branchless with arithmetic masks
                if ai != 0.0 and bi != 0.0:
                    out_i[i, j] += ai * bi
                elif ai != 0.0:
                    out_i[i, j] += ai
                elif bi != 0.0:
                    out_i[i, j] += bi

    return out_r + 1j * out_i


def _spdot(A, B):
    A = coo_matrix(A)
    B = coo_matrix(B)

    # Check if the number of columns in a is equal to the number of rows in b
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "Number of columns in a must be equal to the number of rows in b"
        )

    # Perform element-wise multiplication for each pair of non-zero elements
    row_indices = []
    col_indices = []
    data = []

    for row_A, col_A, val_A in zip(A.row, A.col, A.data):
        for row_B, col_B, val_B in zip(B.row, B.col, B.data):
            if col_A == row_B:
                row_indices.append(row_A)
                col_indices.append(col_B)
                data.append(_modified_c_mult(val_A, val_B))

    # Construct a new COO matrix with the result
    result = coo_matrix(
        (data, (row_indices, col_indices)), shape=(A.shape[0], B.shape[1])
    ).tocsr()

    # Sum along the second axis (axis=1)
    result_sum = hstack([result.getcol(i) for i in range(result.shape[1])])

    return result_sum


# Numba ufunc for specialized complex multiplication
@vectorize([complex128(float64, float64, float64, float64)], nopython=True)
def c_mult_numba(ar, ai, br, bi):
    """
    Specialized multiplication: combines real and imaginary parts according to rules:
    - If both have imaginary parts: multiply imag parts.
    - If only one is complex: multiply its imag with real of the other, respecting tolerance.
    - Otherwise: real multiplication only.
    """
    real_part = ar * br
    # avoid -0.0
    if real_part == -0.0:
        real_part = 0.0
    # determine imaginary part
    if ai != 0.0 and bi != 0.0:
        imag_part = ai * bi
    elif ai != 0.0:
        imag_part = ai if abs(br) > 1e-2 else 0.0
    elif bi != 0.0:
        imag_part = bi if abs(ar) > 1e-2 else 0.0
    else:
        imag_part = 0.0
    return real_part + 1j * imag_part


def _spdot_numba(A, B):
    A = csc_matrix(A)
    B = csr_matrix(B)
    row_idxs, col_idxs, data = [], [], []

    # for each shared dimension k, batch outer-product of nonzeros
    for k in range(A.shape[1]):
        a_start, a_end = A.indptr[k], A.indptr[k + 1]
        b_start, b_end = B.indptr[k], B.indptr[k + 1]

        a_rows = A.indices[a_start:a_end]
        a_vals = A.data[a_start:a_end]
        b_cols = B.indices[b_start:b_end]
        b_vals = B.data[b_start:b_end]

        if a_rows.size == 0 or b_cols.size == 0:
            continue

        # build the broadcasted index arrays
        rr = np.repeat(a_rows, b_cols.size)
        cc = np.tile(b_cols, a_rows.size)
        AA = np.repeat(a_vals, b_vals.size)
        BB = np.tile(b_vals, a_vals.size)

        dd = c_mult_numba(AA.real, AA.imag, BB.real, BB.imag)

        row_idxs.append(rr)
        col_idxs.append(cc)
        data.append(dd)

    row = np.concatenate(row_idxs)
    col = np.concatenate(col_idxs)
    dat = np.concatenate(data)

    C = coo_matrix((dat, (row, col)), shape=(A.shape[0], B.shape[1])).tocsr()
    return C


def _kron(A, B):
    if isinstance(A, spmatrix) and isinstance(B, spmatrix):
        res = _spkron(A, B, format="coo")
    elif isinstance(A, spmatrix) ^ isinstance(B, spmatrix):
        res = _spkron(A, B, format="coo")
    else:
        res = _npkron(A, B)
    return res


# helper function to multiply NetKet sub-operators


def _multiply_operators(
    hilbert,
    support_A: tuple,
    A: Array,
    support_B: tuple,
    B: Array,
    *,
    dtype,
) -> tuple[tuple, Array]:
    """
    Returns the `Tuple[acting_on, Matrix]` representing the operator obtained by
    multiplying the two input operators A and B.
    """
    support_A = np.asarray(support_A)
    support_B = np.asarray(support_B)

    inters = np.intersect1d(support_A, support_B, return_indices=False)

    if support_A.size == support_B.size and np.array_equal(support_A, support_B):
        return tuple(support_A), _fdot(A, B)  # A @ B
    elif inters.size == 0:
        # disjoint supports
        support = tuple(np.concatenate([support_A, support_B]))
        operator = _kron(A, B)
        operator, support = _reorder_kronecker_product(hilbert, operator, support)
        return tuple(support), operator
    else:
        _support_A = list(support_A)
        _support_B = list(support_B)
        _A = A.copy()
        _B = B.copy()

        # expand _act to match _act_i
        supp_B_min = min(support_B)
        for site in support_A:
            if site not in support_B:
                I = _eye_like(hilbert.shape[site], dtype=dtype, like=B)
                if site < supp_B_min:
                    _support_B = [site] + _support_B
                    _B = _kron(I, _B)
                else:  # site > actmax
                    _support_B = _support_B + [site]
                    _B = _kron(_B, I)

        supp_A_min = min(support_A)
        for site in support_B:
            if site not in support_A:
                I = _eye_like(hilbert.shape[site], dtype=dtype, like=A)
                if site < supp_A_min:
                    _support_A = [site] + _support_A
                    _A = _kron(I, _A)
                else:  # site > actmax
                    _support_A = _support_A + [site]
                    _A = _kron(_A, I)

        # reorder
        _A, _support_A = _reorder_kronecker_product(hilbert, _A, _support_A)
        _B, _support_B = _reorder_kronecker_product(hilbert, _B, _support_B)

        if len(_support_A) == len(_support_B) and np.array_equal(
            _support_A, _support_B
        ):
            # back to the case of non-intersecting with same support
            return tuple(_support_A), _fdot(_A, _B)  # _A @ _B
        else:
            raise ValueError("Something failed")
