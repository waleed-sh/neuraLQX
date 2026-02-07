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

# Copyright 2023 The NetKet Authors - All rights reserved.
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

# ---------------------------- WARNING WARNING WARNING ----------------------------
#
#   This implementation is supposed to be used STRICTLY for simulation purposes.
#   DO NOT construct operators using this type. This implementation is not so
#   kosher. Subtle issues exist. And these issues are like in-place modifications
#   of the operator do not change the hash of the operator, meaning that JAX will
#   not recompile the operator, which will lead to wrong results.
#   See: https://github.com/orgs/netket/discussions/2113#discussioncomment-13933215
#
#   This implementation stays JAX-friendly, while using the underlying Numba
#   implementation for the .get_conn_padded() via a jax.pure_callback. This means
#   that this implementation has some overhead. Typically, this is negligible.
#   However, at every evaluation of the operator (e.g. expectation value), the
#   Numba operator may get rebuilt, which if it is a large operator, will make the
#   overhead worse.
#
#   If you want to run simulations using GPUs, we are currently working with NetKet
#   to get GPU support for Numba operators. The implementation exists, and you can
#   simply run GPU simulations using standard Numba operators, although it has not
#   yet been green flagged by NetKet. It is also SUBSTANTIALLY faster than this JAX
#   implementation due to the lack of overhead.
#   See: https://github.com/netket/netket/issues/2114
#
# ---------------------------- WARNING WARNING WARNING ----------------------------

from typing import TYPE_CHECKING

import numpy as np
import jax

from jax.tree_util import register_pytree_node_class

from netket.errors import JaxOperatorNotConvertibleToNumba
from netket.operator._discrete_operator_jax import DiscreteJaxOperator

from neuralqx.utils.experimental import experimental
from .base import LocalOperatorBase
from neuralqx.utils.jax import replicate_sharding

if TYPE_CHECKING:
    from .numba import FunctionalLocalOperator


@experimental
@register_pytree_node_class
class FunctionalLocalOperatorJax(LocalOperatorBase, DiscreteJaxOperator):
    """
    A JAX-compatible FunctionalLocalOperator which works with JAX sharding. It does not implement
    the .get_conn_padded() function in JAX, but it executes a jax.pure_callback to the Numba version
    and that is sharded across devices.
    """

    # can still go back to Numba
    _convertible: bool = True

    def _numba_operator(self) -> "FunctionalLocalOperator":
        """
        Cached Numba version of the operator
        """

        if not hasattr(self, "_cached_numba_op"):

            # if we dont have a cached version, construct the operator once and cache it
            from .numba import FunctionalLocalOperator

            if self._convertible is False:
                # check if convertible or not
                raise JaxOperatorNotConvertibleToNumba(self)

            # construct the concrete FunctionalLocalOperator and cache
            self._cached_numba_op = self.to_numba_operator()

        # return the cache
        return self._cached_numba_op

    def to_numba_operator(self) -> "FunctionalLocalOperator":
        """
        Convert the JAX-compatible operator to a Numba version
        """

        if self._convertible is False:
            raise JaxOperatorNotConvertibleToNumba(self)

        from .numba import FunctionalLocalOperator

        return FunctionalLocalOperator(
            self.hilbert,
            self.operators,
            self.acting_on,
            self.constant,
            dtype=self.dtype,
            mel_cutoff=self.mel_cutoff,
            mels_func=self._mels_func,
            specialised=self._specialised,
            force_hermitian=self._force_hermitian,
        )

    def _setup(self, force: bool = False):
        """
        Setup up the JAX-compatible FunctionalLocalOperator. This is done once-per-object and the
        _setup() of the Numba based implementation is called instead.
        """

        if force or not getattr(self, "_initialized", False):

            # get the Numba operator
            op_nb = self._numba_operator()

            # setup the Numba operator
            op_nb._setup()

            # store some things for tracing
            self._max_conn_size = int(op_nb._max_conn_size)
            self._nonzero_diagonal = bool(op_nb._nonzero_diagonal)

            # set as initialised to avoid doing this again
            self._initialized = True

    def _get_conn_padded(self, x):
        """
        Host‑side Numba lookup wrapped in a sharded jax.pure_callback.
        """

        # setup first, will call Numba's setup if not already initialised
        self._setup()

        # get the maximum number of connections and the expected dtype of the matrix elements
        max_conn = self._max_conn_size
        mel_dtype = self._mels_dtype

        def _py_kernel(x_np: np.ndarray):
            """
            The kernel function for the jax.pure_callback
            """

            # call the Numba .get_conn_padded()
            xp_np, mels_np = self._numba_operator().get_conn_padded(x_np)

            # pad to common max_conn, in case the Numba version did not do a good job
            if xp_np.shape[1] < max_conn:
                pad = max_conn - xp_np.shape[1]
                xp_pad = np.repeat(x_np[:, None, :], pad, 1)
                mels_pad = np.zeros((x_np.shape[0], pad), dtype=mels_np.dtype)
                xp_np = np.concatenate([xp_np, xp_pad], axis=1)
                mels_np = np.concatenate([mels_np, mels_pad], axis=1)

            # ensure dtype matches spec
            mels_np = mels_np.astype(mel_dtype, copy=False)

            # return the values
            return xp_np, mels_np

        # attach static attributes used by wrapper
        _py_kernel.max_conn = max_conn
        _py_kernel.mel_dtype = mel_dtype

        # wrap the Python kernel so it is called once per host shard
        xp, mels = replicate_sharding(_py_kernel)(x)

        # on‑device post‑processing
        # count the number of nonzero connections (nonzero means > the mels cutoff
        n_conn = (jax.numpy.abs(mels) > self._mel_cutoff).sum(
            axis=-1, dtype=jax.numpy.int32
        )

        # return the values
        return xp, mels, n_conn

    def get_conn_padded(self, x):
        xp, mels, _ = self._get_conn_padded(x)
        return xp, mels

    def n_conn(self, x):
        _, _, n_conn = self._get_conn_padded(x)
        return n_conn

    # We will stay object opaque for flatten/unflatten
    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return aux_data

    def to_pauli_strings(self):
        raise NotImplementedError(
            "This operator cannot be converted into Pauli strings type"
        )
