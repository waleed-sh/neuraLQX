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
NOTE:   The content of this file is entirely copied from NetKet's source code.
        Their original copyright below applies.
"""

# Copyright 2021-2023 The NetKet Authors - All rights reserved.
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

import numpy as np
import jax.numpy as jnp
import jax

from numba import jit
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import issparse

from netket import config
from netket.hilbert import DiscreteHilbert
from netket.operator import AbstractOperator
from netket.utils.optional_deps import import_optional_dependency
from netket.jax import sharding as nk_shard

from neuralqx.debug import timeit
from neuralqx.utils.jax.sharding import derive_output_shardings, is_distributed_array


class DiscreteOperator(AbstractOperator[DiscreteHilbert]):
    r"""This class is the base class for operators defined on a
    discrete Hilbert space. Users interested in implementing new
    quantum Operators for discrete Hilbert spaces should derive
    their own class from this class
    """

    def __init__(self, hilbert: DiscreteHilbert):
        if not isinstance(hilbert, DiscreteHilbert):
            raise ValueError(
                "A Discrete Operator can only act upon a discrete Hilbert space."
            )
        super().__init__(hilbert)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        raise NotImplementedError

    def _get_conn_padded_kernel(self, x_local: np.ndarray | jnp.ndarray):
        r"""
        Computes the connected components of the operator for a given batch of input quantum
        numbers `x`. This executes on a single device.

        Args:
            x: A matrix of shape `(batch_size, hilbert.size)`
                containing the batch of quantum numbers x.
        """

        x_local = np.asarray(x_local)

        # number of sites
        n_visible = x_local.shape[-1]

        # total number of samples in the (possibly batched) input
        n_samples = x_local.size // n_visible

        # scratch array that `get_conn_flattened` fills with section sizes
        sections = np.empty(n_samples, dtype=np.int32)

        # compute the connected components
        x_primes, mels = self.get_conn_flattened(
            x_local.reshape(-1, n_visible), sections, pad=True
        )

        # the number of connected states per input sample
        n_primes = int(sections[0])

        # restore the original batch shape and add n_primes before the site axis
        x_primes = x_primes.reshape(
            *x_local.shape[:-1],
            n_primes,
            n_visible,
        )

        mels = mels.reshape(*x_local.shape[:-1], n_primes)

        return x_primes, mels

    #
    #
    #   (02.08.2025)
    #   Note:
    #       The following implementation is still a WIP and has not received a green light from
    #       NetKet yet. Eventually, this will become part of their source code. When working with
    #       serial mode or MPI, you should not have any issues. When working with GPU/JAX sharding
    #       then you might observe slow/excessive communication, leading to slower than intended
    #       simulations. There should not be any difference in the numerical values computed
    #       however. If you observe any, please let us know.
    #       To stay up to date with the NetKet development regarding this feature,
    #       See: https://github.com/netket/netket/issues/2114
    @timeit(tag="GCP_COMPOP", warn_ms=1000)
    def get_conn_padded(
        self,
        x: np.ndarray | jax.Array,
        *,
        use_sharding: bool | None = None,
    ):
        r"""Finds the connected elements of the Operator.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        This now checks if the input array is a JAX Array and is sharded. If so, it extracts the
        shards, computes the connected components, then joins the shards and returns the result.

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.

            use_sharding : A flag to force sharding (Default `None`).

                            - If set to `True`, then it forces the sharded implementation if
                              possible (raises a `ValueError` if `x` is not already sharded).
                            - If set to `False`, it forces the single-device implementation by first
                              gathering all `x` (if sharded) and computes the connected components
                              on a single device.
                            - If set to `None`, it triggers an automatic branch which will choose
                              the sharded path if: (i) `config.config.netket_experimental_sharding`
                              is `True`, (ii) more than one device is visible for JAX and (iii) `x`
                              is a sharded array.

        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """

        # Non-JAX input: always single-device.
        if not isinstance(x, jax.Array):
            return self._get_conn_padded_kernel(x)

        input_is_distributed = is_distributed_array(x)

        auto_shard = (
            config.netket_experimental_sharding
            and (jax.device_count() > 1)
            and input_is_distributed
        )
        do_shard = auto_shard if use_sharding is None else use_sharding

        if do_shard and not input_is_distributed:
            raise ValueError(
                "`use_sharding=True` but the input `x` is not distributed/sharded."
            )

        # Forced single-device path:
        if not do_shard:
            # If distributed, gather to a single fully-replicated host-visible array.
            x0 = nk_shard.gather(x) if input_is_distributed else x
            return self._get_conn_padded_kernel(x0)

        # Sharded path:
        in_sharding = x.sharding
        xp_sharding, ml_sharding = derive_output_shardings(in_sharding, x.ndim)

        # If we can't safely derive output shardings (e.g. site axis sharded), fallback:
        if xp_sharding is None or ml_sharding is None:
            x0 = nk_shard.gather(x)
            return self._get_conn_padded_kernel(x0)

        K = int(self.max_conn_size)
        M = int(x.shape[-1])
        xp_shape = (*x.shape[:-1], K, M)
        ml_shape = (*x.shape[:-1], K)

        xp_by_dev = {}
        ml_by_dev = {}

        for shard in x.addressable_shards:
            if shard.device in xp_by_dev:
                # multiple shards per device: unusual; safest is to fallback
                x0 = nk_shard.gather(x)
                return self._get_conn_padded_kernel(x0)

            xp_local, ml_local = self._get_conn_padded_kernel(shard.data)

            xp_by_dev[shard.device] = jax.device_put(
                jnp.asarray(xp_local), shard.device
            )
            ml_by_dev[shard.device] = jax.device_put(
                jnp.asarray(ml_local), shard.device
            )

        xp_arrays = [xp_by_dev[d] for d in xp_sharding.addressable_devices]
        ml_arrays = [ml_by_dev[d] for d in ml_sharding.addressable_devices]

        xp = jax.make_array_from_single_device_arrays(xp_shape, xp_sharding, xp_arrays)
        ml = jax.make_array_from_single_device_arrays(ml_shape, ml_sharding, ml_arrays)

        return xp, ml

    def get_conn_flattened(
        self, x: np.ndarray, sections: np.ndarray, pad: bool = False
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
            pad: If True, the output x' is padded to the maximum
                number of connected states. If False, the output x'
                is not padded and the number of connected states
                is given by the sections array. (Defaults: ``False``)

        Returns:
            (matrix, array): The connected states x', flattened together in
                a single matrix.
                An array containing the matrix elements :math:`O(x,x')`
                associated to each x'.

        """
        raise NotImplementedError(f"""
            The method get_conn_flattened has not been implemented for the object of
            type {type(self)}.

            This may happen if you defined a custom class inheriting from DiscreteOperator
            and you have not implemented this method. In that case, you should define
            `get_conn_flattened(self, x: array, sections: array)` according to the
            docstring provided on the documentation.

            Otherwise, please open an issue on netket's github repository.
            """)

    def get_conn(self, x: np.ndarray):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        Args:
            x: An array of shape `(hilbert.size, )` containing the quantum numbers x.

        Returns:
            matrix: The connected states x' of shape (N_connected,hilbert.size)
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        Raise:
            ValueError: If the given quantum number is not compatible with the hilbert space.
        """
        if x.ndim != 1:
            raise ValueError(
                "get_conn does not support batches. Please use get_conn_flattened instead."
            )
        if x.shape[0] != self.hilbert.size:
            raise ValueError(
                "The given quantum numbers do not match the hilbert space because"
                f"it has shape {x.shape} of which[0] but expected {self.hilbert.size}."
            )

        return self.get_conn_flattened(
            x.reshape((1, -1)),
            np.ones(1),
        )

    def n_conn(self, x, out=None) -> np.ndarray:
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = np.empty(x.shape[0], dtype=np.int32)
        self.get_conn_flattened(x, out)
        out = self._n_conn_from_sections(out)

        return out

    @staticmethod
    @jit(nopython=True)
    def _n_conn_from_sections(out):
        low = 0
        for i in range(out.shape[0]):
            old_out = out[i]
            out[i] = out[i] - low
            low = old_out

        return out

    def to_sparse(self) -> _csr_matrix:
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The sparse matrix representation of the operator.
        """
        concrete_op = self.collect()
        hilb = self.hilbert

        x = hilb.all_states()

        sections = np.empty(x.shape[0], dtype=np.int32)
        x_prime, mels = concrete_op.get_conn_flattened(x, sections)

        numbers = hilb.states_to_numbers(x_prime)

        sections1 = np.empty(sections.size + 1, dtype=np.int32)
        sections1[1:] = sections
        sections1[0] = 0

        ## eliminate duplicates from numbers
        # rows_indices = compute_row_indices(hilb.states_to_numbers(x), sections1)

        return _csr_matrix(
            (mels, numbers, sections1),
            shape=(self.hilbert.n_states, self.hilbert.n_states),
        )

        # return _csr_matrix(
        #    (mels, (rows_indices, numbers)),
        #    shape=(self.hilbert.n_states, self.hilbert.n_states),
        # )

    def to_dense(self) -> np.ndarray:
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The dense matrix representation of the operator as a Numpy array.
        """
        return self.to_sparse().todense().A

    def to_qobj(self):  # -> "qutip.Qobj"
        r"""Convert the operator to a qutip's Qobj.

        Returns:
            A :class:`qutip.Qobj` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        return qutip.Qobj(
            self.to_sparse(), dims=[list(self.hilbert.shape), list(self.hilbert.shape)]
        )

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.apply(v)

    def apply(self, v: np.ndarray) -> np.ndarray:
        op = self.to_linear_operator()
        return op @ v

    def __matmul__(self, other):
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, jnp.ndarray)
            or issparse(other)
        ):
            return self.apply(other)
        elif isinstance(other, AbstractOperator):
            return self._op__matmul__(other)
        else:
            return NotImplemented

    def _op__matmul__(self, other):
        "Implementation on subclasses of __matmul__"
        return NotImplemented

    def __rmatmul__(self, other):
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, jnp.ndarray)
            or issparse(other)
        ):
            # return self.apply(other)
            return NotImplemented
        elif isinstance(other, AbstractOperator):
            return self._op__rmatmul__(other)
        else:
            return NotImplemented

    def _op__rmatmul__(self, other):
        "Implementation on subclasses of __matmul__"
        return NotImplemented

    def to_linear_operator(self):
        return self.to_sparse()
