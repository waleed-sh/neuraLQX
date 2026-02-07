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


"""
NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

import humanize.filesize

import ast
import os
from datetime import datetime

# ---------------------- WARNING WARNING WARNING ----------------------------
# Do not modify any parts of this file unless you know what you are doing...
# This file handles a special type of Local Operators which is used in some
# constraints.


from typing import Union
from typing import Optional
from typing import TYPE_CHECKING

from textwrap import dedent
import numbers

import jax
import numpy as np
import jax.numpy as jnp

from scipy import sparse
from scipy.sparse import spmatrix
from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import bsr_matrix
from scipy.sparse import hstack
from scipy.sparse import csc_matrix

try:
    from numba import njit
    from numba import prange
    from numba import jit
    from numba import vectorize
    from numba import float64
    from numba import complex128

    _HAS_NUMBA = True
except ModuleNotFoundError:
    _HAS_NUMBA = False

from netket.utils.numbers import dtype as _dtype, is_scalar
from netket.utils.numbers import dtype as is_scalar
from netket.hilbert import AbstractHilbert, Fock
from netket.hilbert import Fock
from netket.jax import canonicalize_dtypes
from netket.utils.types import DType
from netket.utils.types import Array
from netket.operator import Squared
from netket.operator import Transpose
from netket.operator._local_operator.convert import local_operators_to_pauli_strings
from netket.operator._discrete_operator import DiscreteOperator as DiscreteOperatorNK
from netket.operator._local_operator.base import LocalOperatorBase as NKLOPB

from neuralqx.operators.types.functional_local_operator.base import (
    LocalOperatorBase as NQXLOPB,
)
from neuralqx import cfg
from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.utils.misc.auth import get_hash
from neuralqx.utils.io.printing import NQXPrinter
from neuralqx.utils.serialization import serialize
from neuralqx.utils.serialization import SerializerEngine, deserialize
from neuralqx.utils.experimental import experimental
from neuralqx.utils.misc import ActingOnModifier

# import the local DiscreteOperator to implement JAX friendly sharding for Numba operators
from neuralqx.operators.types._discrete_operator import DiscreteOperator

if TYPE_CHECKING:
    from netket.operator._pauli_strings import PauliStrings


def is_hermitian(a: np.ndarray, rtol=1e-05, atol=1e-08) -> bool:
    if issparse(a):
        return np.allclose(
            a.real.todense(), a.real.T.conj().todense(), rtol=rtol, atol=atol
        )
    else:
        return np.allclose(a.real, a.real.T.conj(), rtol=rtol, atol=atol)


def _is_sorted(a):
    for i in range(len(a) - 1):
        if a[i + 1] < a[i]:
            return False
    return True


@jit(nopython=True)
def _number_to_state(number, hilbert_size_per_site, local_states_per_site, out):
    out[:] = local_states_per_site[:, 0]
    size = out.shape[0]

    ip = number
    k = size - 1
    while ip > 0:
        local_size = hilbert_size_per_site[k]
        out[k] = local_states_per_site[k, ip % local_size]
        ip = ip // local_size
        k -= 1

    return out


@jit(nopython=True)
def _number_to_state_jax(number, hilbert_size_per_site, out):
    out[:] = 0
    size = out.shape[0]

    ip = number
    k = size - 1
    while ip > 0:
        local_size = hilbert_size_per_site[k]
        out[k] = ip % local_size
        ip = ip // local_size
        k -= 1

    return out


@experimental
class LocalOperatorBase(DiscreteOperator):
    """A custom local operator. This is a sum of an arbitrary number of operators
    acting locally on a limited set of k quantum numbers (i.e. k-local,
    in the quantum information sense).
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[list[Array], Array] = [],
        acting_on: Union[list[int], list[list[int]]] = [],
        constant: numbers.Number = 0,
        dtype: Optional[DType] = None,
        *,
        mel_cutoff: float = 1.0e-10,
        ident: str,
    ):
        r"""
        Constructs a new ``LocalOperator`` given a hilbert space and (if
        specified) a constant level shift.

        Args:
           hilbert: Hilbert space the operator acts on.
           operators: A list of operators, in matrix form. Supports NumPy dense or scipy
           sparse format
           acting_on: A list of list of sites, which the corresponding operators act on. This
                should be constructed such that :code:`operators[i]` acts on the sites
                :code:`acting_on[i]`.
                If operators is not a list of operators, acting_on should just be the list of
                corresponding sites.
           constant: Constant diagonal shift of the operator, equivalent to
                :math:`+\text{c}\hat{I}`. Default is 0.0.
           dtype: The datatype to use for the matrix elements. Defaults to double precision if
                available.
           mel_cutoff (float): a cutoff to remove small matrix elements (default = 1e-10)

        Examples:
           Constructs a ``LocalOperator`` without any operators.

           >>> from netket.hilbert import Spin
           >>> from netket.operator import LocalOperator
           >>> hi = Spin(0.5)**20
           >>> empty_hat = LocalOperator(hi)
           >>> print(len(empty_hat.acting_on))
           0
        """
        super().__init__(hilbert)

        """
        Changes:
        Identifier added so that every LocalOperator object has a unique user set ID. 

        Reason:
            This identifier will be used to mark acting_on tuples such that two LocalOperator 
            objects cannot add operators acting on the same support.
        """
        self.__ident__ = ident

        """
        Changes:
            The following is added to allow for modifying acting_on

        Reason:
            The LOP object now has an instance based ActingOnModifier which will
            handle modifying/purifying acting_on 
        """
        self._aom = ActingOnModifier(ident)

        self.mel_cutoff = mel_cutoff
        self._initialized = None
        self._is_hermitian = None

        if not all(
            [_is_sorted(hilbert.states_at_index(i)) for i in range(hilbert.size)]
        ):
            raise ValueError(
                dedent(
                    """LocalOperator needs an hilbert space with sorted state values at
                every site.
                """
                )
            )

        """
        Comments:
            Now, the canonicalized input is guaranteed to be marked and contain duplicate
            acting_on sites which are modified by `-1` sites
        """
        # Canonicalize input. From now on input is guaranteed to be in canonical order
        operators, acting_on, dtype = self.canonicalize_input(
            self.hilbert, operators, acting_on, constant, dtype=dtype
        )
        self._dtype = dtype
        self._constant = np.array(constant, dtype=dtype)

        self._operators_dict = {}
        for op, aon in zip(operators, acting_on):
            self._add_operator(aon, op)

    """
    Status:
        Completed

    Changes:
        Operators with the same support are now not going to add
    """

    def _add_operator(self, acting_on: tuple, operator: Array):
        """
        Adds an operator acting on a subset of sites.

        Does not modify in-place the operators themselves which are treated as
        immutables.
        """
        assert isinstance(acting_on, tuple)
        # acting_on_key = tuple(acting_on)
        if acting_on in self._operators_dict:
            """
            Changes:
                The following line is added to modify the acting_on. The line
                implementing addition of operators with the same support is removed

            Reason:
                The acting_on is modified to avoid adding operators which have
                the same support

            Comments:
                The new acting_on is now guaranteed to be non-existent in the list
                of acting_on of this instance
            """
            acting_on = self._aom.insert_highest_modification(
                list(self._operators_dict.keys()), acting_on
            )
            # operator = self._operators_dict[acting_on] + operator

        """
        Comments:
            A new operator is now added to the instance's _operators_dict with a 
            modified acting_on 
        """
        self._operators_dict[acting_on] = operator

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def operators(self) -> list[np.ndarray]:
        """List of the matrices of the operators encoded in this Local Operator.
        Returns a copy.
        """
        return list(self._operators_dict.values())

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def _operators(self) -> list[np.ndarray]:
        return self.operators

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def acting_on(self) -> list[list[int]]:
        """List containing the list of the sites on which every operator acts.

        Every operator `self.operators[i]` acts on the sites `self.acting_on[i]`
        """
        return list(self._operators_dict.keys())

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def n_operators(self) -> int:
        """The total number of operators that were summed upon to build this operator.

        This excludes an optional identity term that is tracked by
        :attr:`netket.operator.LocalOperator.constant`.
        """
        return len(self._operators_dict)

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def dtype(self) -> DType:
        """DType of the matrix elements of this operator."""
        return self._dtype

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def size(self) -> int:
        return self._size

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        # TODO: (VolodyaCO) I guess that if we have an operator with diagonal elements equal to
        #  1j*C+Y, some complex constant, and self._constant=-1j*C, then the actual diagonal would
        #  be Y. How do we check hermiticity taking into account the diagonal elements as well as
        #  the self._constant? For the moment I just check hermiticity of the added constant, which
        #  must be real.
        if self._is_hermitian is None:
            self._is_hermitian = all(map(is_hermitian, self.operators)) and np.isreal(
                self._constant.real
            )

        return self._is_hermitian

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def mel_cutoff(self) -> float:
        r"""float: The cutoff for matrix elements.
        Only matrix elements such that abs(O(i,i))>mel_cutoff
        are considered"""
        return self._mel_cutoff

    """
    Status:
        Completed

    Changes:
        None
    """

    @mel_cutoff.setter
    def mel_cutoff(self, mel_cutoff):
        self._mel_cutoff = mel_cutoff
        assert self.mel_cutoff >= 0

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def constant(self) -> numbers.Number:
        """
        A constant multiplying the identity added to the diagonal of the operator.
        """
        return self._constant

    """
    Status:
        Completed

    Changes:
        None
    """

    def to_pauli_strings(self, **kwargs) -> "PauliStrings":  # noqa: F821
        """Convert to PauliStrings object"""
        return local_operators_to_pauli_strings(
            self.hilbert,
            self.operators,
            self.acting_on,
            self.constant,
            self.dtype,
            **kwargs,
        )

    """
    Status:
        Completed

    Changes:
        None
    """

    def copy(self, *, dtype: Optional[DType] = None, _cls=None):
        """Returns a copy of the operator, while optionally changing the dtype
        of the operator.

        Args:
            dtype: optional dtype

        Internal args:
            _cls: used to specify the target class
        """

        if dtype is None:
            dtype = self.dtype

        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")

        if _cls is None:
            _cls = type(self)

        """
        Comments:
            Copying the current instance should also copy its identifier
        """
        new = _cls(
            self.hilbert, constant=self.constant, dtype=dtype, ident=self.__ident__
        )
        new.mel_cutoff = self.mel_cutoff

        if dtype == self.dtype:
            new._operators_dict = self._operators_dict.copy()
        else:
            new._operators_dict = {
                aon: self.cast_operator_matrix_dtype(op, dtype)
                for aon, op in self._operators_dict.items()
            }

        return new

    """
    Status:
        Completed

    Changes:
        None
    """

    def transpose(self, *, concrete=False):
        r"""LocalOperator: Returns the transpose of this operator."""
        if concrete:
            new = self.copy()
            for aon in new._operators_dict.keys():
                new._operators_dict[aon] = new._operators_dict[aon].transpose()
            return new
        else:
            return Transpose(self)

    """
    Status:
        Completed

    Changes:
        None
    """

    def conjugate(self, *, concrete=False):
        r"""LocalOperator: Returns the complex conjugate of this operator."""
        new = self.copy()
        for aon in new._operators_dict.keys():
            new._operators_dict[aon] = new._operators_dict[aon].copy().conjugate()
        return new

    """
    Status:
        Completed

    Changes:
        None
    """

    def __radd__(self, other):
        return self.__add__(other)

    """
    Status:
        Completed

    Changes:
        None
    """

    def __sub__(self, other):
        return self + (-other)

    """
    Status:
        Completed

    Changes:
        None
    """

    def __rsub__(self, other):
        return other + (-self)

    """
    Status:
        Completed

    Changes:
        None
    """

    def __isub__(self, other):
        return self.__iadd__(-other)

    """
    Status:
        Completed

    Changes:
        None
    """

    def __neg__(self):
        return -1 * self

    """
    Status:
        Completed

    Changes:
        None
    """

    def __add__(self, other: Union["LocalOperatorBase", numbers.Number]):
        op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
        op = op.__iadd__(other)
        return op

    """
    Status:
        Completed

    Changes:
        The NetKet logic is preserved. Addition of operators with the same
        acting_on is handled in _add_operator() hence no changes are needed
        here
    """

    def __iadd__(self, other):
        if (
            isinstance(other, LocalOperatorBase)
            or isinstance(other, NKLOPB)
            or isinstance(other, NQXLOPB)
        ):
            if self.hilbert != other.hilbert:
                return NotImplemented

            if not np.can_cast(other.dtype, self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator with dtype {other.dtype} "
                    f"to operator with dtype {self.dtype}"
                )

            self.mel_cutoff = min(other.mel_cutoff, self.mel_cutoff)
            self._constant += other.constant.item()
            for aon, op in other._operators_dict.items():
                """
                Changes:
                    The aon from the new operator is now marked

                Reason:
                    If we add two types, the acting_on of the other
                    LocalOperator are likely unmarked

                Comments:
                    This method call already handles the avoidance of addition
                    of two operators with the same support. If there is one which
                    already exists with this support, a new one will be inserted
                    with a modified support
                """
                self._add_operator(self._aom.mark_aon(aon), op)

            self._reset_caches()
            return self
        if is_scalar(other):
            if not np.can_cast(type(other), self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator with dtype {type(other)} "
                    f"to operator with dtype {self.dtype}"
                )

            self._reset_caches()
            self._constant += other
            return self
        return NotImplemented

    """
    Status:
        Completed

    Changes:
        None
    """

    def __truediv__(self, other):
        if not is_scalar(other):
            raise TypeError("Only division by a scalar number is supported.")

        if other == 0:
            raise ValueError("Dividing by 0")
        return self.__mul__(1.0 / other)

    """
    Status:
        Completed

    Changes:
        None
    """

    def __rmul__(self, other):
        return self.__mul__(other)

    """
    Status:
        Completed

    Changes:
        None
    """

    def __mul__(self, other):
        if isinstance(other, DiscreteOperator) or isinstance(other, DiscreteOperatorNK):
            op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
            return op.__imatmul__(other)
        elif is_scalar(other):
            op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
            return op.__imul__(other)
        return NotImplemented

    """
    Status:
        Completed

    Changes:
        None
    """

    def __imul__(self, other):
        if isinstance(other, DiscreteOperator) or isinstance(other, DiscreteOperatorNK):
            return self.__imatmul__(other)
        elif is_scalar(other):
            if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator of type {type(other)} and "
                    f"dtype {_dtype(other)} to operator with dtype {self.dtype}"
                )
            other = np.asarray(
                other, dtype=jnp.promote_types(self.dtype, _dtype(other))
            )

            self._constant *= other
            if np.abs(other) <= self.mel_cutoff:
                self._operators_dict = {}
            else:
                for key in self._operators_dict:
                    self._operators_dict[key] = other * self._operators_dict[key]
                    """
                    Status:
                        Completed
                        
                    Changes:
                        Here, we have to use the modified multiplication rule as we are multiplying
                        the constant of the other operator with this operator's matrices.
                    """
                    # self._operators_dict[key] = self._broadcastMultiplyReal(
                    #     other, self._operators_dict[key]
                    # )

            self._reset_caches()
            return self

        return NotImplemented

    """
    Status:
        Completed

    Changes:
        None
    """

    def __imatmul__(self, other):
        if (
            not isinstance(other, LocalOperatorBase)
            and not isinstance(other, NKLOPB)
            and not isinstance(other, NQXLOPB)
        ):
            return NotImplemented

        if not np.can_cast(other.dtype, self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )

        return self._op_imatmul_(other)

    """
    Status:
        Completed

    Changes:
        None
    """

    def _op__matmul__(self, other: "LocalOperatorBase") -> "LocalOperatorBase":
        if (
            not isinstance(other, LocalOperatorBase)
            and not isinstance(other, NKLOPB)
            and not isinstance(other, NQXLOPB)
        ):
            return NotImplemented
        op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
        return op._op_imatmul_(other)

    """
    Status:
        Completed

    Changes:
        The NetKet logic is preserved. The two possible alterations of operators exists
        in the case of
            (i)     α ∑ᵢBᵢ
            (ii)    ∑ᵢⱼAᵢBⱼ
            (iii)   αβ + β ∑ᵢAᵢ
        For (i), the _add_operator() method already handles possible duplicate acting_on
        For (ii), the _add_operator() method already handles possible duplicate acting_on,
        and the _multiply_operators() also returns supports which are ensured to not exist
        in this instance's acting_on
        For (iii), the multiplication β ∑ᵢAᵢ is now using specialised functions as we do not
        wish to multiply the imaginary parts of the current operator by anything
    """

    def _op_imatmul_(self, other: "LocalOperatorBase") -> "LocalOperatorBase":
        if (
            not isinstance(other, LocalOperatorBase)
            and not isinstance(other, NKLOPB)
            and not isinstance(other, NQXLOPB)
        ):
            return NotImplemented

        self.mel_cutoff = min(other.mel_cutoff, self.mel_cutoff)

        # (α + ∑ᵢAᵢ)(β + ∑ᵢBᵢ) =
        # = αβ + α ∑ᵢBᵢ + β ∑ᵢAᵢ + ∑ᵢⱼAᵢBⱼ
        # = β(α + ∑ᵢAᵢ) + α ∑ᵢBᵢ + ∑ᵢⱼAᵢBⱼ

        α = self.constant.item()
        β = other.constant.item()
        # copy A dict because it is modified inplace in __imul__(β) and add_operators
        A_op_dict = self._operators_dict.copy()
        B_op_dict = other._operators_dict

        """
        Comments:
            This will in-place modify self._operators_dict. If β < mel_cutoff, then
            self._operators_dict will reset to {}
        """
        # αβ + β ∑ᵢAᵢ
        self.__imul__(β)

        # α ∑ᵢBᵢ
        if np.abs(α) > self.mel_cutoff:
            for aon, op in B_op_dict.items():
                """
                Changes:
                    Added a line to mark the aon

                Reasons:
                    The aon is probably unmarked since it is coming from the other
                    LocalOperator which can be a NetKet default. To ensure correct
                    insertion into the current instance's acting_on, it has to be
                    marked with the current instance's identifier

                Comments:
                    Here, _add_operator() already handles possible duplicate acting_on
                """
                self._add_operator(self._aom.mark_aon(aon), α * op)

        # ∑ᵢⱼAᵢBⱼ
        for supp_A_i, A_i in A_op_dict.items():
            for supp_B_j, B_j in B_op_dict.items():
                """
                Comments:
                    Here, _add_operator() already handles possible duplicate acting_on.
                    _mulitply_operators() also ensures that the returned support is non-
                    existent in the instance's acting_on
                """
                self._add_operator(
                    *self._multiply_operators(
                        self.hilbert,
                        supp_A_i,
                        A_i,
                        supp_B_j,
                        B_j,
                        dtype=self.dtype,
                    )
                )

        is_hermitian_A = self._is_hermitian
        is_hermitian_B = other._is_hermitian

        self._reset_caches()

        if is_hermitian_A and is_hermitian_B:
            self._is_hermitian = is_hermitian_A

        return self

    """
    Status:
        Completed

    Changes:
        None
    """

    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None

    """
    Status:
        Completed

    Changes:
        None
    """

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        self._setup()
        return self._max_conn_size

    """
    Status:
        Completed

    Changes:
        None
    """

    def __repr__(self):
        ao = self.acting_on

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        return (
            f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str}, "
            f"constant={self.constant}, dtype={self.dtype}, ID={self.__ident__})"
        )

    """
    Status:
        Completed

    Changes:
        Supports are purified prior to multiplication as they are needed to determine
        the type of multiplication based on their overlap. The returned supports obey
        NetKet logic except they are marked and ensured to not exist in the current
        instance's acting_on list
    """

    # TODO: support sparse arrays without returning dense arrays
    def _multiply_operators(
        self,
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
        """
        Changes:
            The following 2 lines are added to purify the supports prior to casting 
            as numpy arrays

        Reason:
            The methods implemented in ActingOnModifier require tuples. The NetKet
            logic is preserved by purifying the supports prior to the matrix
            multiplication process

        Comments:
            Now, support_A and support_B are ensured to be NetKet friendly supports
            which do not contain identifiers or `-1` sites
        """
        support_A, count_minus_A = self._aom.get_count_minus(support_A)
        support_B, count_minus_B = self._aom.get_count_minus(support_B)

        support_A = np.asarray(support_A)
        support_B = np.asarray(support_B)

        inters = np.intersect1d(support_A, support_B, return_indices=False)

        if support_A.size == support_B.size and np.array_equal(support_A, support_B):
            # TODO: Note: COO @ COO -> CSR, this may insert CSR matrices in our
            # format. Should we enforce COO?
            # If yes, the logic in the compilation can be simplified to always assume
            # COO.
            """
            Changes:
                The following line is added to ensure that the new support about to be
                returned does not exist in the current instance's list of acting_on

            Reason:
                If a duplicate support is returned, it can cause that the corresponding
                operator in the instance's operator's dict be overwritten

            Comments:
                Now, support_A is ensured to be a unique support, marked with an identifier,
                that does not exist in the current instance's list of acting_on
            """
            support_A = self._aom.insert_highest_modification(
                list(self._operators_dict.keys()), tuple(support_A)
            )

            """
            Changes:
                1. Returning support_A as is, without casting
                2. Computing the multiplication using the modified rule

            Reason:
                insert_highest_modification() will return a tuple type
            """
            # return tuple(support_A), A @ B
            # return support_A, A @ B
            return support_A, self._fdot(A, B)

        elif inters.size == 0:
            # disjoint supports
            support = tuple(np.concatenate([support_A, support_B]))
            """
            Changes:
                Using the _ckron instead of _kron to implement the multiplication
                using the modified rule
            """
            operator = self._ckron(A, B)

            """
            Changes:
                None

            Comments:
                Note that here support has to be pure since _reorder_kronecker_product() requires a 
                pure acting_on
            """
            operator, support = self._reorder_kronecker_product(
                hilbert, operator, support
            )

            """
            Changes:
                The following line is added to ensure that the new support about to be 
                returned does not exist in the current instance's list of acting_on

            Reason: 
                 If a duplicate support is returned, it can cause that the corresponding
                operator in the instance's operator's dict be overwritten

            Comments:
                Now, support is ensured to be a unique support, marked with an identifier,
                that does not exist in the current instance's list of acting_on
            """
            support = self._aom.insert_highest_modification(
                list(self._operators_dict.keys()), tuple(support)
            )

            """
            Changes:
                Returning support as is, without casting

            Reason:
                insert_highest_modification() will return a tuple type
            """
            # return tuple(support), operator
            return support, operator
        else:
            _support_A = list(support_A)
            _support_B = list(support_B)
            _A = A.copy()
            _B = B.copy()

            # expand _act to match _act_i
            supp_B_min = min(support_B)
            for site in support_A:
                if site not in support_B:
                    I = self._eye_like(hilbert.shape[site], dtype=dtype, like=B)
                    if site < supp_B_min:
                        _support_B = [site, *_support_B]
                        """
                        Changes:
                            Using the _ckron instead of _kron to implement the multiplication
                            using the modified rule
                        """
                        _B = self._ckron(I, _B)
                    else:  # site > actmax
                        _support_B = [*_support_B, site]
                        """
                        Changes:
                            Using the _ckron instead of _kron to implement the multiplication
                            using the modified rule
                        """
                        _B = self._ckron(_B, I)

            supp_A_min = min(support_A)
            for site in support_B:
                if site not in support_A:
                    I = self._eye_like(hilbert.shape[site], dtype=dtype, like=A)
                    if site < supp_A_min:
                        _support_A = [site, *_support_A]
                        """
                        Changes:
                            Using the _ckron instead of _kron to implement the multiplication
                            using the modified rule
                        """
                        _A = self._ckron(I, _A)
                    else:  # site > actmax
                        _support_A = [*_support_A, site]
                        """
                        Changes:
                            Using the _ckron instead of _kron to implement the multiplication
                            using the modified rule
                        """
                        _A = self._ckron(_A, I)

            """
            Changes:
                None

            Comments:
                Note that here support has to be pure since _reorder_kronecker_product() requires a 
                pure acting_on
            """
            # reorder
            _A, _support_A = self._reorder_kronecker_product(hilbert, _A, _support_A)
            _B, _support_B = self._reorder_kronecker_product(hilbert, _B, _support_B)

            if len(_support_A) == len(_support_B) and np.array_equal(
                _support_A, _support_B
            ):
                # back to the case of non-intersecting with same support
                # TODO: Note: COO @ COO -> CSR

                """
                Changes:
                    The following line is added to ensure that the new support about to be
                    returned does not exist in the current instance's list of acting_on

                Reason:
                    If a duplicate support is returned, it can cause that the corresponding
                    operator in the instance's operator's dict be overwritten

                Comments:
                    Now, _support_A is ensured to be a unique support, marked with an identifier,
                    that does not exist in the current instance's list of acting_on
                """
                _support_A = self._aom.insert_highest_modification(
                    list(self._operators_dict.keys()), tuple(_support_A)
                )

                """
                Changes:
                    1. Returning _support_A as is, without casting
                    2. Using the _fdot to implement the product A @ B using the modified
                       multiplication rule

                Reason:
                    insert_highest_modification() will return a tuple type
                """
                # return tuple(_support_A), _A @ _B
                # return _support_A, _A @ _B
                return _support_A, self._fdot(_A, _B)
            else:
                raise ValueError("Something failed")

    """
    Status:
        Completed

    Changes:
        None
    """

    @staticmethod
    def cast_operator_matrix_dtype(matrix: Array, dtype: DType):
        """
        Changes the dtype of a matrix, without changing the structural type of the object.

        This makes sure that if you pass sparse arrays to a LocalOperator, they remain
        sparse even if you change the dtype
        """
        # must copy
        # return np.asarray(matrix, dtype=dtype)
        return matrix.astype(dtype)

    """
    Status:
        Completed

    Changes:
        None
    """

    @staticmethod
    def _standardize_matrix_input_type(op):
        """
        Standardize the structural type of operators stored in LocalOperator.

        Eventually, we could also support spmatrices (but some work will be needed.)
        """
        if isinstance(op, list):
            return np.asarray(op)
        elif sparse.issparse(op):
            return op.tocoo()
        elif isinstance(op, np.matrix):
            # np.matrix does not respect the ndarray interface
            return np.asarray(op)
        else:
            return op

    """
    Status:
        Completed

    Changes:
        None
    """

    @staticmethod
    def _kron(A, B):
        if isinstance(A, spmatrix) and isinstance(B, spmatrix):
            res = sparse.kron(A, B, format="coo")
        elif isinstance(A, spmatrix) ^ isinstance(B, spmatrix):
            # XOR, meaning only one is sparse matrix
            # Then we maintain sparse format (Is this a good idea?)
            res = sparse.kron(A, B, format="coo")
        else:
            res = np.kron(np.asarray(A), np.asarray(B))
        return res

    """
    Status:
        Completed

    Changes:
        None
    """

    @staticmethod
    def _eye_like(size, *, dtype, like):
        if sparse.issparse(like):
            return sparse.eye(size, dtype=dtype)
        else:
            return np.eye(size, dtype=dtype)

    """
    Status:
        Completed

    Changes:
        The returned canonicalized acting_on sites are now modified
        to include an identifier and any duplicates would include in them
        at least one `-1` site
    """

    def canonicalize_input(
        self,
        hilbert: AbstractHilbert,
        operators,
        acting_on,
        constant,
        *,
        dtype=None,
    ):
        """
        Takes as inputs the inputs to the constructor of LocalOperator and canonicalizes
        them by ensuring the following holds:
         - acting_on is a list of list
         - acting_on[i] are sorted
         - operators is list of matrices
         - all dtypes match

        Args:
            hilbert: The hilbert space

        Returns:
            List of operators, acting ons and dtypes.
        """
        markings = None

        # check if passing a single operator or a list of operators
        if isinstance(acting_on, numbers.Number):
            acting_on = [acting_on]

        is_nested = any(hasattr(i, "__len__") for i in acting_on)
        if not is_nested:
            operators = [operators]
            acting_on = [acting_on]

        if all(len(aon) == 0 for aon in acting_on):
            operators = []
            acting_on = []
        else:

            """
            Changes:
                Get the markings as well

            Reason:
                The markings are needed when transforming between Numba and Jax representations
                of this LOP
            """
            acting_on, markings = self._aom.get_pure_aon_iter_marked(acting_on)

            if max(map(max, acting_on)) >= hilbert.size or min(map(min, acting_on)) < 0:
                raise ValueError("An operator acts on an invalid set of sites.")

            """
            Changes:
                Removed check for duplicate sites

            Reason:
                This LOP type may have duplicate sites at init and this would raise
                an error
            """
            # for i, aon in enumerate(acting_on):
            #     if len(aon) != len(set(aon)):
            #         raise ValueError(
            #             f"The operator at index {i} acts on duplicated sites {aon}"
            #         )

        """
        Changes:
            None

        Reason:
            The current acting_on sites are still not modified
        """
        acting_on = [tuple(aon) for aon in acting_on]
        # operators = [np.asarray(operator) for operator in operators]

        """
        Changes:
            None

        Reason:
            Modified _standardize_matrix_input_type  is identical to original
        """
        operators = [self._standardize_matrix_input_type(op) for op in operators]

        dtype = canonicalize_dtypes(float, *operators, constant, dtype=dtype)

        canonicalized_operators = []
        canonicalized_acting_on = []
        for operator, acting_on in zip(operators, acting_on):

            """
            Changes:
                None

            Reason:
                The acting_on passed to check_valid-opmatrix are not modified
            """
            self.check_valid_opmatrix(hilbert, operator, acting_on)

            if operator.dtype is not dtype:
                """
                Changes:
                    None

                Reason:
                    Modified cast_operator_matrix_dtype is identical to the original
                """
                operator = self.cast_operator_matrix_dtype(operator, dtype=dtype)

            """
            Changes:
                None

            Reason:
                The acting_on passed to _reorder_kronecker_product are not modified
            """
            # re-sort the operator
            operator, acting_on = self._reorder_kronecker_product(
                hilbert, operator, acting_on
            )
            canonicalized_operators.append(operator)
            canonicalized_acting_on.append(acting_on)

        """
        Changes:
            The following line is added such that now each of the acting_on in the list of 
            canonicalized_acting_on are now modified to include an identifier and `-1` sites

        Reason:
            We handle modification of acting_on at the end of the canonicalization process. In 
            this case, we have a list of acting_on sites each of type tuple. Duplicates may be
            present. We handle duplicates by appending as many `-1` as necessary and append the
            identifier to each acting_on 
        """
        marked_canonicalized_acting_on = self._aom.mark_canonicalized_acting_on(
            canonicalized_acting_on, markings
        )

        return canonicalized_operators, marked_canonicalized_acting_on, dtype

    """
    Status:
        Completed

    Changes:
        The NetKet logic is preserved. This method assumes that the passed acting_on
        is pure. It is only used in canonicalize_input() whereby acting_on sites are
        purified prior to call
    """

    @staticmethod
    def check_valid_opmatrix(hi, mat, acting_on):

        expected_size = np.prod([hi.shape[aon] for aon in acting_on])

        if mat.shape != (expected_size, expected_size):
            raise ValueError(
                f"The matrix of the sub-operator acting on sites {acting_on} "
                f"must have shape {expected_size, expected_size}, "
                f"but it has shape {mat.shape}."
            )

    """
    Status:
        Completed

    Changes:
        The NetKet logic is preserved. This method assumes that the passed acting_on
        is pure. It is only used in canonicalize_input() and _multiply_operators() 
        whereby in both cases acting_on sites are purified prior to call
    """

    # TODO: support sparse arrays without returning dense arrays
    @staticmethod
    def _reorder_kronecker_product(hi, mat, acting_on) -> tuple[Array, tuple]:
        """
        Reorders the matrix resulting from a kronecker product of several
        operators in such a way to sort acting_on.

        A conceptual example is the following:
        if `mat = Â ⊗ B̂ ⊗ Ĉ` and `acting_on = [[2],[1],[3]`
        you will get `result = B̂ ⊗ Â ⊗ Ĉ, [[1], [2], [3]].

        However, essentially, A,B,C represent some operators acting on
        thei sub-space acting_on[1], [2] and [3] of the hilbert space.

        This function also handles any possible set of values in acting_on.

        The inner logic uses the Fock.all_states(), number_to_state and
        state_to_number to perform the re-ordering.
        """

        acting_on_sorted = np.sort(acting_on)
        if np.array_equal(acting_on_sorted, acting_on):
            return mat, acting_on

        # could write custom binary <-> int logic instead of using Fock...
        # Since i need to work with bit-strings (where instead of bits i
        # have integers, in order to support arbitrary size spaces) this
        # is exactly what hilbert.to_number() and vice versa do.

        # target ordering binary representation
        hi_subspace = Fock(hi.shape[acting_on_sorted[0]] - 1)
        for site in acting_on_sorted[1:]:
            hi_subspace = hi_subspace * Fock(hi.shape[site] - 1)

        hi_unsorted_subspace = Fock(hi.shape[acting_on[0]] - 1)
        for site in acting_on[1:]:
            hi_unsorted_subspace = hi_unsorted_subspace * Fock(hi.shape[site] - 1)

        # find how to map target ordering back to unordered
        acting_on_unsorted_ids = np.zeros(len(acting_on), dtype=np.intp)
        for i, site in enumerate(acting_on):
            acting_on_unsorted_ids[i] = np.argmax(site == acting_on_sorted)

        # now it is valid that
        # acting_on_sorted == acting_on[acting_on_unsorted_ids]

        # generate n-bit strings in the target ordering
        v = hi_subspace.all_states()

        # convert them to origin (unordered) ordering
        v_unsorted = v[:, acting_on_unsorted_ids]
        # convert the unordered bit-strings to numbers in the target space.
        n_unsorted = np.asarray(hi_unsorted_subspace.states_to_numbers(v_unsorted))

        # reorder the matrix
        if sparse.issparse(mat):
            # covert the indices from going from old -> new to new->old
            idxs = np.argsort(n_unsorted)

            # sparse matrices don't support indexing, and can be sorted more
            # quickly
            mat_sorted = mat.copy()
            mat_sorted.row[:] = idxs[mat.row]
            mat_sorted.col[:] = idxs[mat.col]
        else:
            mat_sorted = mat[n_unsorted, :][:, n_unsorted]

        return mat_sorted, tuple(acting_on_sorted)

    """
    Status:
        Completed

    Changes:
        The NetKet implementation is preserved, after a minor modification to purify the
        list of acting_on for the current instance. This function is used in _setup() prior
        to calling get_conn_flattened(), get_conn_filtered(), .... To this end, all 
        operators have been preserved, even with duplicate sites. No operator addition
        is performed in neither _setup() nor get_conn_...(). Therefore, the NetKet logic
        is left as is since this function is responsible for computing the non-zero mels
        for every operator and takes into account its acting_on to produce the correct
        local basis states (e.g. operator indices which will correspond to global indices).
        There is no reason to keep modified acting_on and return them as they are not used
        in the computation of mels.
        
    Update 24.02.2025:
        Added a `jax` flag. This is supposed to determine which indexing to use. This code had
        originally modified NetKet v3.12, and in the latest versions they have modified their 
        Hilbert space indexing. In order to use MarkedLocalOperatorJax, we have to use the new
        indexing...
    """

    def pack_internals(
        self,
        hilbert: AbstractHilbert,
        operators_dict: dict,
        constant,
        dtype: DType,
        mel_cutoff: float,
        jax: bool = False,
    ):
        """
        Take the internal lazy representation of a local operator and returns the arrays
        needed for the numba implementation.

        This takes as input a dictionary with Tuples as keys, the `acting_on` and matrices as values
        The keys represent the sites upon which the matrix acts.
        It is assumed that the integer in the tuples are sorted.

        Returns a dictionary with all the data fields
        """

        """
        Changes:
            The following line is added so that the list of acting_on saved in operator_dict.keys() 
            is purified prior to the NetKet routine

        Reason:
            The acting_on sites now need to be accurate to produce the correct local basis
            states which correspond to the (i, j) indices of the non-zero mels of the 
            corresponding operator. There is no point in having the modified acting_on

        Comments:
            Now, op_acting_on is ensured to be a list of pure acting_on sites corresponding
            to every operator
        """
        op_acting_on = self._aom.get_count_minus_iter(list(operators_dict.keys()))[0]
        # op_acting_on = list(operators_dict.keys())

        operators = list(operators_dict.values())
        n_operators = len(operators_dict)

        """Analyze the operator strings and precompute arrays for get_conn inference"""

        # how many sites each operator is acting on
        acting_size = np.array([len(aon) for aon in op_acting_on], dtype=np.intp)

        # compute the maximum number of off-diagonal nonzeros (over all rows) of each operator
        op_n_conns_offdiag = self.max_nonzero_per_row(operators, mel_cutoff)

        # Support empty types such as the identity.
        if len(acting_size) > 0:
            # maximum number of sites any operator is acting on
            max_acting_on_sz = np.max(acting_size)

            # max local hilbert size of all sites acted on by any operator
            max_local_hilbert_size = max(
                [max(map(hilbert.size_at_index, aon)) for aon in op_acting_on]
            )
            # maximum size of any operator
            # (maximum size of the matrix / prod of local hilbert spaces)
            max_op_size = max(map(lambda x: x.shape[0], operators))
            # maximum number of off-diagonal nonzeros of any operator
            max_op_size_offdiag = np.max(op_n_conns_offdiag)
        else:
            max_acting_on_sz = 0
            max_local_hilbert_size = 0
            max_op_size = 0
            max_op_size_offdiag = 0

        # matrix storing which sites each operator acts on, padded with -1
        acting_on = np.full((n_operators, max_acting_on_sz), -1, dtype=np.intp)
        for i, aon in enumerate(op_acting_on):
            acting_on[i][: len(aon)] = aon

        ###
        # allocate empty arrays which are filled below

        # array which will be storing the local states
        # of each site each operator is acting on
        local_states = np.full(
            (n_operators, max_acting_on_sz, max_local_hilbert_size), np.nan
        )

        # array storing the basis for each site each operator is acting on
        # The basis is an integer used to map (indices of) local states on sites
        # to (indices of) states in the space spanned by all sites the op is acting on
        # (it is simply the product of number of local states of the sites before)
        basis = np.full((n_operators, max_acting_on_sz), 0x7FFFFFFF, dtype=np.int64)

        diag_mels = np.full((n_operators, max_op_size), np.nan, dtype=dtype)

        mels = np.full(
            (n_operators, max_op_size, max_op_size_offdiag),
            np.nan,
            dtype=dtype,
        )
        # x_prime contains the local state after the operator has been applied
        # for the sites the operator is acting on, for each row
        # (each row also corresponds to a list of local states)
        x_prime = np.full(
            (n_operators, max_op_size, max_op_size_offdiag, max_acting_on_sz),
            -1,
            dtype=np.float64,
        )
        # store the number of off-diagonal nonzeros per row of each operator
        n_conns = np.full((n_operators, max_op_size), 0, dtype=np.intp)

        ###
        # iterate over all operators
        for i, (aon, op) in enumerate(operators_dict.items()):
            """
            Changes:
                Added the following line to ensure aon is pure

            Reason:
                Here, we are iterating over the modified acting_on directly from
                the operators_dict. Not purifying them would raise various errors
                such as hilbert.shape[] calling str valued sites

            Comments:
                The following line now ensures that the current aon is pure
            """
            aon = self._aom.get_pure_aon(aon)

            # how many sites this operator is acting on
            aon_size = len(aon)

            n_local_states_per_site = np.asarray(
                [hilbert.size_at_index(i) for i in aon]
            )

            # add an operator to local_states
            for j, site in enumerate(aon):
                local_states[i, j, : hilbert.shape[site]] = np.asarray(
                    hilbert.states_at_index(site)
                )

            # compute the basis of each site of this operator
            # i.e. the product of the number of local states of all sites before it
            ba = 1
            for s in range(aon_size):
                basis[i, s] = ba
                ba *= hilbert.shape[aon[aon_size - s - 1]]

            if sparse.issparse(op):
                if not isinstance(op, sparse.csr_matrix):
                    op = op.tocsr()
                # Extract the sparse matrix representation to NumPy arrays
                data = np.array(op.data, copy=False)
                indices = np.array(op.indices, copy=False)
                indptr = np.array(op.indptr, copy=False)

                self._append_matrix_sparse(
                    data,
                    indices,
                    indptr,
                    aon_size,
                    local_states[i],
                    n_local_states_per_site,
                    mel_cutoff,
                    diag_mels[i],
                    mels[i],
                    x_prime[i],
                    n_conns[i],
                    jax,
                )

            else:
                self._append_matrix(
                    op,
                    aon_size,
                    local_states[i],
                    n_local_states_per_site,
                    mel_cutoff,
                    diag_mels[i],
                    mels[i],
                    x_prime[i],
                    n_conns[i],
                    jax,
                )

        nonzero_diagonal = (
            np.any(np.abs(diag_mels) >= mel_cutoff) or np.abs(constant) >= mel_cutoff
        )

        max_conn_size = 1 if nonzero_diagonal else 0
        # estimate max_conn_size with the sum of the
        # maximum number of off-diagonal nonzeros of all operators
        max_conn_size = max_conn_size + np.sum(op_n_conns_offdiag)

        instance_identifiers, instance_u_identifiers = self._aom.get_identifiers(
            list(operators_dict.keys())
        )

        return {
            "acting_on": acting_on,
            "acting_size": acting_size,
            "diag_mels": diag_mels,
            "mels": mels,
            "x_prime": x_prime,
            "n_conns": n_conns,
            "local_states": local_states,
            "basis": basis,
            "nonzero_diagonal": nonzero_diagonal,
            "max_conn_size": max_conn_size,
            "identifiers": instance_identifiers,
            "uids": instance_u_identifiers,
        }

    """
    Status:
        Completed

    Changes:
        Added a `jax` flag. This is supposed to determine which indexing to use. This code had
        originally modified NetKet v3.12, and in the latest versions they have modified their 
        Hilbert space indexing. In order to use MarkedLocalOperatorJax, we have to use the new
        indexing...
    """

    @staticmethod
    @jit(nopython=True)
    def _append_matrix(
        operator,
        acting_size,
        local_states_per_site,
        hilb_size_per_site,
        epsilon,
        diag_mels,
        mels,
        x_prime,
        n_conns,
        jax,
    ):
        """
        Appends
        """
        op_size = operator.shape[0]
        assert op_size == operator.shape[1]
        # iterate over rows
        for i in range(op_size):
            # set diag mels
            diag_mels[i] = operator[i, i]
            # count number of connected elements
            n_conns[i] = 0  # k_conn = 0
            # iterate over cols
            for j in range(op_size):
                # off-diagonal, non-zero
                if i != j and np.abs(operator[i, j]) > epsilon:
                    k_conn = n_conns[i]
                    # set off-doagnoal mels
                    mels[i, k_conn] = operator[i, j]
                    # convert the col to the corresponding local states
                    # and store it in x_prime
                    if jax:
                        _number_to_state_jax(
                            j,
                            hilb_size_per_site,
                            x_prime[i, k_conn, :acting_size],
                        )
                    else:
                        _number_to_state(
                            j,
                            hilb_size_per_site,
                            local_states_per_site[:acting_size, :],
                            x_prime[i, k_conn, :acting_size],
                        )
                    n_conns[i] += 1  # k_conn=k_conn+1

    """
    Status:
        Completed

    Changes:
        Added a `jax` flag. This is supposed to determine which indexing to use. This code had
        originally modified NetKet v3.12, and in the latest versions they have modified their 
        Hilbert space indexing. In order to use MarkedLocalOperatorJax, we have to use the new
        indexing...    """

    @staticmethod
    @jit(nopython=True)
    def _append_matrix_sparse(
        data,
        indices,
        indptr,
        acting_size,
        local_states_per_site,
        hilb_size_per_site,
        epsilon,
        diag_mels,
        mels,
        x_prime,
        n_conns,
        jax,
    ):
        """
        Equivalent to _append_matrix, but takes as input the three arrays
        'data, indices, indptr' of the CSR sparse format instead of a NumPy
        dense matrix.
        """
        op_size = len(indptr) - 1

        for i in range(op_size):
            # If the diagonal element was not found in the data, set it to 0
            diag_mels[i] = 0
            for index in range(indptr[i], indptr[i + 1]):
                j = indices[index]
                val = data[index]

                if i == j:  # Diagonal elements
                    diag_mels[i] = val

                elif np.abs(val) > epsilon:  # Non-diagonal elements
                    k_conn = n_conns[i]
                    mels[i, k_conn] = val
                    if jax:
                        _number_to_state_jax(
                            j,
                            hilb_size_per_site,
                            x_prime[i, k_conn, :acting_size],
                        )
                    else:
                        _number_to_state(
                            j,
                            hilb_size_per_site,
                            local_states_per_site[:acting_size, :],
                            x_prime[i, k_conn, :acting_size],
                        )
                    n_conns[i] += 1

    """
    Status:
        Completed

    Changes:
        None
    """

    # Generated by chatting with chatgpt
    # https://chat.openai.com/share/9fb4b0d3-0306-41ca-aefc-eee780a2dc02
    @staticmethod
    def max_nonzero_per_row(operators, cutoff):
        """
        This function counts the maximum number of nonzero entries per row, excluding the
        diagonal, among all operators in a local operator.

        Prior to this function, netket defaulted to returning the maximum shape of the
        operators minus 1, but that was largely memory inefficient.

        This function has a computational overhead, as we have to iterate through all
        operators, but it allows us to be much happier in terms of memory cost.
        """
        max_counts = []
        for matrix in operators:
            # Check if the matrix is sparse
            if sparse.issparse(matrix):
                # simple implementation, raises warning
                # matrix = matrix.copy()  # Ensure we don't modify the original matrix
                # matrix.data[np.abs(matrix.data) < cutoff] = 0  # Apply cutoff
                # matrix.setdiag(0)  # Set diagonal entries to zero

                # alternative implementation. Does not raise warning but
                # I'm unsure if it's any faster...
                # Convert to dok_matrix format for efficient modification
                dok_matrix = matrix.todok()
                dok_matrix.setdiag(0)

                # Collect keys to delete
                keys_to_delete = [
                    (i, j) for (i, j), v in dok_matrix.items() if abs(v) < cutoff
                ]

                # Delete keys
                for key in keys_to_delete:
                    del dok_matrix[key]

                row_counts = dok_matrix.tocsr().getnnz(
                    axis=1
                )  # Count non-zero entries in each row
            else:
                matrix = (
                    matrix.copy()
                )  # Make a copy to avoid modifying the original matrix
                mask = (
                    np.abs(matrix) >= cutoff
                )  # Create a mask of entries above the cutoff
                np.fill_diagonal(mask, 0)  # Set diagonal entries to zero
                row_counts = np.count_nonzero(
                    mask, axis=1
                )  # Count non-zero entries in each row

            max_counts.append(np.max(row_counts))
        return np.array(max_counts, dtype=np.int32)

    def _ckron(
        self,
        A: Union[spmatrix, np.ndarray],
        B: Union[spmatrix, np.ndarray],
    ) -> Union[spmatrix, np.ndarray]:
        """
        A wrapper function which will compute np.kron for either scipy sparse matrices or NumPy
        arrays in the modified multiplication manner which multiplies only real parts of complex
        numbers. For example:
            .. math::
                (x + iy)(a + i0) = ax + iy
        """

        # dispatch dedicated kron function depending on input type
        if isinstance(A, spmatrix) and isinstance(B, spmatrix):
            res = self._spkron(A, B, format="coo")

        # if one of them (XOR) is sparse, go for _spkron
        elif isinstance(A, spmatrix) ^ isinstance(B, spmatrix):
            res = self._spkron(A, B)
        else:
            res = self._npkron(A, B)
        return res

    def _npkron(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> Union[np.ndarray, np.matrix]:
        """
        A method which implements the numpy.kron() multiplication but using the modified
        multiplication rule
        """

        b = np.asanyarray(b)
        a = np.array(a, copy=False, subok=True, ndmin=b.ndim)
        is_any_mat = isinstance(a, np.matrix) or isinstance(b, np.matrix)
        ndb, nda = b.ndim, a.ndim
        nd = max(ndb, nda)

        if nda == 0 or ndb == 0:
            return self._modified_c_mult(a, b)

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
        result = self._modified_c_mult(a_arr, b_arr)

        # Reshape back
        import numpy.core.numeric as _nx

        result = result.reshape(_nx.multiply(as_, bs))

        return result if not is_any_mat else np.matrix(result, copy=False)

    def _spkron(
        self,
        A: spmatrix,
        B: spmatrix,
        format=None,
    ) -> spmatrix:
        """
        A method which implements the scipy kron() function but using the modified multiplication
        rule
        """

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
            data = self._modified_c_mult(data, B)

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
            data = self._modified_c_mult(data.reshape(-1, B.nnz), B.data)
            data = data.reshape(-1)

            return coo_matrix((data, (row, col)), shape=output_shape).asformat(format)

    def _fdot(
        self,
        A: Union[np.ndarray, spmatrix],
        B: Union[np.ndarray, spmatrix],
    ) -> Union[np.ndarray, spmatrix]:
        """
        A method which implements the equivalent of A @ B for two arrays A and B basedd on their
        type (sparse, NumPy) using the modified multiplication rule
        """

        # dispatch dedicated functions depending on the type of A and B
        if issparse(A) and issparse(B):
            # A = A.toarray()
            # B = B.toarray()

            # if _HAS_NUMBA:
            #     return coo_matrix(self._fdot_numba(A, B))
            # else:
            #     raise ModuleNotFoundError

            if _HAS_NUMBA:
                return self._spdot_numba(A, B)
            else:
                return self._spdot(A, B)

        # if at least one of them is numpy.ndarray, go for dense
        if isinstance(A, np.ndarray) or isinstance(B, np.ndarray):
            if issparse(A):
                A = A.toarray()
            if issparse(B):
                B = B.toarray()

        if _HAS_NUMBA:
            # TODO: THE NUMBA VERSIONS ARE DEPRECATED UNTIL ALL TESTS PASS!!!
            # res = self._fdot_numba(A, B)
            res = (self._modified_c_mult(A[:, :, None], B[None, :, :])).sum(axis=1)
        else:
            res = (self._modified_c_mult(A[:, :, None], B[None, :, :])).sum(axis=1)

        return res

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _fdot_numba(A, B):
        """
        Numba supported A @ B operation with a custom multiply rule.
        Assumes numpy.ndarray as inputs.
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

    def _spdot_numba(
        self,
        A: spmatrix,
        B: spmatrix,
    ) -> spmatrix:

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

            dd = self._c_mult_numba(AA.real, AA.imag, BB.real, BB.imag)

            row_idxs.append(rr)
            col_idxs.append(cc)
            data.append(dd)

        row = np.concatenate(row_idxs)
        col = np.concatenate(col_idxs)
        dat = np.concatenate(data)

        return coo_matrix((dat, (row, col)), shape=(A.shape[0], B.shape[1])).tocsr()

    def _spdot(
        self,
        A: spmatrix,
        B: spmatrix,
    ) -> spmatrix:
        """
        A method which computes the product A @ B for two sparse matrices using the modified
        multiplication rule
        """

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
                    data.append(self._modified_c_mult(val_A, val_B))

        # Construct a new COO matrix with the result
        result = coo_matrix(
            (data, (row_indices, col_indices)), shape=(A.shape[0], B.shape[1])
        ).tocsr()

        # Sum along the second axis (axis=1)
        result_sum = hstack([result.getcol(i) for i in range(result.shape[1])])

        return result_sum

    @staticmethod
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

    @staticmethod
    @vectorize([complex128(float64, float64, float64, float64)], nopython=True)
    def _c_mult_numba(ar, ai, br, bi):
        """
        Numba ufunc for specialized complex multiplication
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

    @staticmethod
    def _get_const(c):
        if c.dtype == complex:
            return f"{c.real} + {c.imag}j"
        else:
            return f"{c.real}"

    def _serialise_to_disc(
        self, H: AbstractHilbertInterface, op_list: list = None, squared: bool = True
    ) -> None:
        """
        Save the operator(s) to disc. If a list of operators is provided, each operator is saved in
        its own info dict. All info dicts are then wrapped into a top-level dict with the key
        "operator_list" and saved
        """

        # build Hilbert and graph info
        is_constrained = H.hilbert.constrained
        hilbert_info = {
            "cutoff": H.cutoff,
            "is_constrained": is_constrained,
            "constraint_array": H.core._jax_ce if is_constrained else None,
        }
        graph_info = {
            "is_planar": H.graph.is_planar,
            "edges": H.graph.edges,
        }

        if op_list is not None:
            # process each operator in the list if a list is provided
            op_infos = []

            for op in op_list:
                op_info = {
                    "op_type": "trmc",
                    "hilbert": hilbert_info,
                    "graph": graph_info,
                    "operators_dict": serialize(op._operators_dict),
                    "dtype": "c" if op.dtype == complex else "r",
                    "constant": self._get_const(op._constant),
                    "mel_cutoff": op.mel_cutoff,
                    "ident": op.__ident__ or "",
                    "squared": squared,
                }
                op_infos.append(op_info)

            # wrap the list in a marker dictionary
            info = {"operator_list": op_infos}
        else:
            # no list is provided: single operator case (using self)
            info = {
                "op_type": "trmc",
                "hilbert": hilbert_info,
                "graph": graph_info,
                "operators_dict": serialize(self._operators_dict),
                "dtype": "c" if self.dtype == complex else "r",
                "constant": self._get_const(self._constant),
                "mel_cutoff": self.mel_cutoff,
                "ident": self.__ident__ or "",
                "squared": squared,
            }

        # save to disc in our cache directory (~/.neuralqx_trmc_cache)
        cache_dir = cfg.get_static("Cache Directory")

        # TODO: ensure rank 0 IO only here
        # create it if it doesnt exist
        os.makedirs(cache_dir, exist_ok=True)

        # save the serialised file
        filepath = os.path.join(
            cache_dir,
            f"{get_hash()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.msgpack",
        )

        SerializerEngine.save_to_file(info, filepath)

        NQXPrinter().print(
            f"Cached file {filepath} of size {humanize.filesize.naturalsize((os.path.getsize(filepath)), format = '%.2f')}"
        )

    def _load_from_disc(
        self,
        H: AbstractHilbertInterface,
        op_type: str = "trmc",
    ) -> Union["MarkedLocalOperator", None]:
        """
        Traverse the cache directory and check all cached files. If a file contains a list of
        operator info (under the key "operator_list"), deserialize every operator and return them as
        a list. Otherwise, return a single operator instance if a match is found.
        """

        # build current Hilbert and graph info
        is_constrained = H.hilbert.constrained
        hilbert_info = {
            "cutoff": H.cutoff,
            "is_constrained": is_constrained,
            "constraint_array": H.core._jax_ce if is_constrained else None,
        }
        graph_info = {
            "is_planar": H.graph.is_planar,
            "edges": H.graph.edges,
        }

        # get the cache directory path
        cache_dir = cfg.get_static("Cache Directory")

        # TODO: ensure rank 0 IO only here

        if not os.path.exists(cache_dir) or not os.path.isdir(cache_dir):
            raise NotADirectoryError(
                f"Cache directory {cache_dir} does not exist or is not a directory."
            )

        # found_ops = []

        # traverse all files ending with ".msgpack"
        for filename in os.listdir(cache_dir):
            if filename.endswith(".msgpack"):
                full_path = os.path.join(cache_dir, filename)
                try:
                    snapshot = SerializerEngine.load_from_file(full_path, raw=True)
                    data = deserialize(snapshot)
                except Exception as e:
                    print(f"Error processing file {full_path}: {e}")
                    continue

                # check if the data contains a list of operators
                if "operator_list" in data:
                    matching_ops = []
                    for op_info in data["operator_list"]:
                        if (
                            op_info.get("op_type") == op_type
                            and op_info.get("hilbert") == hilbert_info
                            and op_info.get("graph") == graph_info
                        ):
                            # TODO: move this out so that we just print once
                            NQXPrinter().print(
                                f"Loading cached file {full_path} of size "
                                f"{humanize.filesize.naturalsize(os.path.getsize(full_path), format = '%.2f')}"
                            )

                            # create a new operator instance from op_info
                            # assuming same class as self
                            _cls = type(self)

                            new_op = _cls(
                                self.hilbert,
                                constant=ast.literal_eval(op_info.get("constant")),
                                dtype=(
                                    np.complex128
                                    if op_info.get("dtype") == "c"
                                    else np.float64
                                ),
                                ident=op_info.get("ident"),
                            )

                            # set the cutoff
                            new_op.mel_cutoff = op_info.get("mel_cutoff")

                            # copy over the operators dict
                            new_op._operators_dict = deserialize(
                                op_info.get("operators_dict")
                            ).copy()

                            # square it if necessary
                            if op_info.get("squared") is True:
                                new_op = Squared(new_op)

                            # append to the list of found operators
                            matching_ops.append(new_op)

                    if matching_ops:
                        # as soon as we find a matching file, return its operators
                        return (
                            matching_ops[0] if len(matching_ops) == 1 else matching_ops
                        )

                else:
                    # single operator file
                    if (
                        data.get("op_type") == op_type
                        and data.get("hilbert") == hilbert_info
                        and data.get("graph") == graph_info
                    ):
                        NQXPrinter().print(
                            f"Loading cached file {full_path} of size "
                            f"{humanize.filesize.naturalsize((os.path.getsize(full_path)), format = '%.2f')}"
                        )

                        # assuming again the same type as self
                        _cls = type(self)

                        new_op = _cls(
                            self.hilbert,
                            constant=ast.literal_eval(data.get("constant")),
                            dtype=(
                                np.complex128
                                if data.get("dtype") == "c"
                                else np.float64
                            ),
                            ident=data.get("ident"),
                        )

                        # set the mel cutoff and copy all operators and supports
                        new_op.mel_cutoff = data.get("mel_cutoff")
                        new_op._operators_dict = deserialize(
                            data.get("operators_dict")
                        ).copy()

                        if data.get("squared") is True:
                            new_op = Squared(new_op)

                        return new_op

        # if we found nothing, we return None
        return None


# ---------------------- WARNING WARNING WARNING ----------------------------
