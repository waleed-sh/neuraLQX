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

# ---------------------- WARNING WARNING WARNING ----------------------------
# Do not modify any parts of this file unless you know what you are doing...
# This file handles a special type of Local Operators which is used in some constraints.


from typing import TYPE_CHECKING

import numba
import numpy as np

from numba import jit
from numba import njit, prange

from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from .base import LocalOperatorBase
from neuralqx.utils.experimental import experimental

if TYPE_CHECKING:
    from .jax import MarkedLocalOperatorJax


@experimental
class MarkedLocalOperator(LocalOperatorBase):
    """A custom local operator. This is a sum of an arbitrary number of operators
    acting locally on a limited set of k quantum numbers (i.e. k-local,
    in the quantum information sense).
    """

    __module__ = "neuralqx.operators.types"

    """
    Status:
        Completed

    Changes:
        None
    """

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            data = self.pack_internals(
                self.hilbert,
                self._operators_dict,
                self.constant,
                self.dtype,
                self.mel_cutoff,
            )

            self._acting_on = data["acting_on"]
            self._acting_size = data["acting_size"]
            self._diag_mels = data["diag_mels"]
            self._mels = data["mels"]
            self._x_prime = data["x_prime"]
            self._n_conns = data["n_conns"]
            self._local_states = data["local_states"]
            self._basis = data["basis"]
            self._nonzero_diagonal = data["nonzero_diagonal"]
            self._max_conn_size = data["max_conn_size"]
            self._initialized = True

            """
            Changes:
                Added two lines to store the identifiers and unique identifiers in
                the instance's acting_on list

            Reasons:
                This method is called prior to computing non-zero mels using get_conn_...().
                We want to compute non-zero mels of differently marked operators separately
                and so we collect the identifiers for every operators to loop through them
                later in get_conn_...() by identifier
            """
            self._identifiers = data["identifiers"]
            self._uids = data["uids"]

    """
    Status:
        Completed

    Changes:
        A list of identifiers and uids are now being passed for the computation
        of connected components to be grouped by identifier
    """

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            pad (bool): Whether to use zero-valued matrix elements in order to return all equal
                        sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        self._setup()

        """
        Changes:
            Added a line to prep the _identifiers and _uids for numba.jit friendly
            execution

        Reasons:
            The gccf() is numba jitted. That means no lists and no dicts. The operator
            lookup based on their group needs to be done in a format which is strictly
            numpy arrays. The following code formats the _identifiers and _uids into a
            structure that ensures that
        """
        uids, ops_grouped_by_uid, ops_start, ops_count = self._aom.prep_for_gccf(
            self._identifiers, self._uids
        )

        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        """
        Changes:
            Added the _identifiers and _uids to the list of params of the
            _get_conn_flattened_kernel

        Reasons:
            _get_conn_flattened_kernel is now going to compute connected
            components in a grouped manner based on their identifiers. The
            list of identifiers and unique identifiers are needed for that
            process and are generated in _setup()
        """
        return self._get_conn_flattened_kernel(
            x,
            sections,
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
            self._nonzero_diagonal,
            uids,
            ops_grouped_by_uid,
            ops_start,
            ops_count,
            pad,
        )

    """
    Status:
        Completed

    Changes:
        A list of identifiers and uids are now being passed for the computation
        of connected components to be grouped by identifier
    """

    def _get_conn_flattened_closure(self):
        self._setup()
        _local_states = self._local_states
        _basis = self._basis
        _constant = self._constant
        _diag_mels = self._diag_mels
        _n_conns = self._n_conns
        _mels = self._mels
        _x_prime = self._x_prime
        _acting_on = self._acting_on
        _acting_size = self._acting_size

        """
        Changes:
            Added two lines to store the identifiers and unique identifiers

        Reasons:
            The gccf_fun (_get_conn_flattened_kernel()) now computes the connected
            components identifier-wise and these will be passed as params to it
        """
        _identifiers = self._identifiers
        _uids = self._uids

        """
        Changes:
            Added a line to prep the _identifiers and _uids for numba.jit friendly
            execution

        Reasons:
            The gccf() is numba jitted. That means no lists and no dicts. The operator
            lookup based on their group needs to be done in a format which is strictly
            numpy arrays. The following code formats the _identifiers and _uids into a
            structure that ensures that
        """
        uids, ops_grouped_by_uid, ops_start, ops_count = self._aom.prep_for_gccf(
            _identifiers, _uids
        )

        # workaround my painfully discovered Numba#6979 (cannot use NumPy bools in closures)
        _nonzero_diagonal = bool(self._nonzero_diagonal)

        fun = self._get_conn_flattened_kernel

        def gccf_fun(x, sections):
            """
            Changes:
                Added the _identifiers and _uids to the list of params of the
                _get_conn_flattened_kernel

            Reasons:
                _get_conn_flattened_kernel is now going to compute connected
                components in a grouped manner based on their identifiers. The
                list of identifiers and unique identifiers are needed for that
                process and are generated in _setup()
            """
            return fun(
                x,
                sections,
                _local_states,
                _basis,
                _constant,
                _diag_mels,
                _n_conns,
                _mels,
                _x_prime,
                _acting_on,
                _acting_size,
                _nonzero_diagonal,
                uids,
                ops_grouped_by_uid,
                ops_start,
                ops_count,
            )

        return jit(nopython=True)(gccf_fun)

    """
    Status:
        Completed

    Changes:
        This method now has been modified to compute the connected components for
        a given batch of basis states x differently from the NetKet implementation.
        Essentially, instead of looping through n_operators blindly, it loops through
        operators by identifier, computes the mels per identifier, and then groups
        everything at the end

    Comments:
        The implementation is different from NetKet. The NetKet code is removed to
        avoid clutter. Refer to line 144 in numba.py found in
        netket.operators._local_operator for the original implementation
    """

    @staticmethod
    @numba.jit(nopython=True)
    def _get_conn_flattened_kernel(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        nonzero_diagonal,
        uids,
        ops_grouped_by_uid,
        ops_start,
        ops_count,
        pad=False,
    ):

        # collect the batch_size based on the given input
        batch_size = x.shape[0]

        # collect the number of sites in the graph based on the input
        n_sites = x.shape[1]

        # get the dtype of the mels
        dtype = all_mels.dtype

        # get the number of unique identifiers to loop through every one of them
        num_uids = uids.shape[0]

        # TODO remove this line when numba 0.53 is dropped 0.54 is minimum version
        # workaround a bug in Numba arising when NUMBA_BOUNDSCHECK=1
        constant = constant.item()

        # assert that the batch size and sections dimensions match
        assert sections.shape[0] == batch_size

        # get the total number of operators
        n_operators = n_conns.shape[0]

        # an array to store the row index
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        # precompute all the xs_n
        # iterate over each state in the input batch of states
        for b in range(batch_size):

            # extract the state corresponding to this batch index
            x_b = x[b]

            # iterate over each operator
            for i in range(n_operators):

                # get the number of sites that the current operator acts upon
                acting_size_i = acting_size[i]

                # initialise the "row index" (xs_n) for this operator and this
                # batch state to zero. xs_n[b, i] will hold a computed integer
                # index corresponding to the local configuration of the subset
                # of sites that the operator acts on
                xs_n[b, i] = 0

                # extract the portion of the current state's configuration on which
                # the operator acts. acting_on[i, :acting_size_i] gives the indices
                # of the sites that the operator acts upon. x_b[...] picks out the
                # values of these sites from the full state.
                x_i = x_b[acting_on[i, :acting_size_i]]

                # for each site the operator acts upon, compute a contribution
                # the linear index xs_n. The operator's local states are enumerated
                # according to 'local_states', and 'basis' defines how to convert
                # the multi-site configuration into a single integer index.
                for k in range(acting_size_i):
                    # the indexing "acting_size_i - k - 1" is used to handle
                    # the site's position in a reversed order, which aligns with how
                    # the states are numbered. np.searchsorted(...) finds the
                    # position of the current site's state (x_i[...])
                    # in the sorted array of possible local states
                    # (local_states[i, ...]). Multiplying by basis[i, k]
                    # converts this position into a weighted offset, contributing to
                    # the final integer index representing the entire local configuration
                    # of these sites.
                    xs_n[b, i] += (
                        np.searchsorted(
                            local_states[i, acting_size_i - k - 1],
                            x_i[acting_size_i - k - 1],
                        )
                    ) * basis[i, k]

        # initialise the total number of connections (tot_conn) found so far to zero
        # this will accumulate the count of all diagonal and off-diagonal connections
        # for every state in the batch
        tot_conn = 0

        # initialise max_conn to zero. This will track the maximum number connections
        # (diagonal + off-diagonal) found for any single state, used for padding
        # if necessary.
        max_conn = 0

        # first pass: determine the total number of connections and fill out 'sections'
        # array. 'sections' is used to know where each batch element's connections end
        # in the flattened arrays.
        for b in range(batch_size):

            # initialise the connection count for this batch element.
            # If diagonal terms are nonzero, start from 1 (to count the diagonal
            # element),otherwise start from 0.
            conn_b = 1 if nonzero_diagonal else 0

            # add off-diagonal connections from all operators
            # the number of off-diagonal connections for state b and operator i is
            # stored in n_conns[i, xs_n[b, i]]
            for i in range(n_operators):
                conn_b += n_conns[i, xs_n[b, i]]

            # update the total connections count by adding the connections found
            # for this state
            tot_conn += conn_b

            # store the cumulative total into sections[b], which marks the end of this
            # batch element's connections in the flattened arrays
            sections[b] = tot_conn

            # if padding is requested, update max_conn to track the largest number of
            # connections found so far among all states.
            if pad:
                max_conn = max(conn_b, max_conn)

        # if padding is enabled, adjust tot_conn to be the total number of elements
        # for the entire batch, assuming each state has exactly max_conn connections,
        # resulting in a uniformly sized structure.
        if pad:
            tot_conn = batch_size * max_conn

        # initialise arrays for the x_prime and mels to be computed
        x_prime = np.empty((tot_conn, n_sites), dtype=x.dtype)
        mels = np.empty(tot_conn, dtype=dtype)

        # initialises a variable that will hold the maximum number of operators associated with any
        # single uid group. Initially, we assume the largest group size is zero
        max_ops_per_uid = 0

        # iterate over each uid index u. The variable num_uids represents the total number of unique
        # identifiers, each corresponding to a group of operators
        for u in range(num_uids):
            # for the current uid u, we check the size of its operator group, ops_count[u]
            # If this count exceeds the previously recorded maximum, we update max_ops_per_uid. By
            # doing this for every uid, we ensure that after this loop completes, max_ops_per_uid
            # will represent the largest operator group size among all uids
            if ops_count[u] > max_ops_per_uid:
                # we assign the number of operators in the current uid group to max_ops_per_uid
                # this effectively keeps track of the largest group seen so far
                max_ops_per_uid = ops_count[u]

        # allocate a one-dimensional NumPy array of length max_ops_per_uid and with the same
        # data type as mels (dtype). This array will be used to temporarily hold the diagonal
        # contributions of any single uid’s operators. By making it this large, we ensure it can
        # handle the biggest uid group without needing dynamic resizing
        diag_contribs = np.empty(max_ops_per_uid, dtype=dtype)

        # initialise a counter 'c' which will track the current position within
        # the flattened arrays 'x_prime' and 'mels'. This 'c' moves forward as we
        # add diagonal and off-diagonal connections for each batch element
        c = 0

        # loop over each state in the batch
        for b in range(batch_size):

            # store the starting index for the current state's diagonal element
            # if a diagonal element is present, it will be placed at position c_diag
            c_diag = c

            # extract the full configuration for the current state
            x_batch = x[b]

            # if the operator has a non-zero diagonal, place the diagonal element here
            # this is the first connection for this state, representing the state
            # itself
            if nonzero_diagonal:
                # set the matrix element at the diagonal position to the constant
                # diagonal value.
                mels[c_diag] = constant

                # copy the current batch state into x_prime at the diagonal position,
                # indicating the state connects to itself
                x_prime[c_diag] = np.copy(x_batch)

                # increment 'c' by one to account for this newly added diagonal
                # element.
                c += 1

            # now, we process operators grouped by their unique identifier 'uid_idx'
            # each 'uid_idx' represents a group of operators that we handle together
            for uid_idx in range(num_uids):

                # mark the start of this uid's contributions
                # this will be used to manipulate every uids grouped operators'
                # off-diagonal mels separately later
                c_before_uid = c

                # if there is a diagonal part, we add the diagonal contributions from
                # all operators in this uid group to the previously placed diagonal
                # element
                # The diagonal counting here is done uid-wise
                if nonzero_diagonal:
                    # retrieve the number of operators belonging to the current uid group identified
                    # by uid_idx
                    # this tells us how many diagonal contributions we will collect and manipulate
                    uid_op_count = ops_count[uid_idx]

                    # starting index in ops_grouped_by_uid where this uid’s operators begin
                    start = ops_start[uid_idx]

                    # the exclusive ending index
                    end = start + uid_op_count

                    for jj, idx2 in enumerate(range(start, end)):

                        # retrieves the global sub-operator index (e.g. retrieve the sub-operator
                        # from the list of all subops of this MarkedLocalOperator which sits in
                        # index i as it now belongs to the uid we are in)
                        i = ops_grouped_by_uid[idx2]

                        # the diagonal matrix element for operator i given the current state b
                        # store these diagonal contributions into diag_contribs[jj], effectively
                        # creating a contiguous array of diagonal contributions for this uid’s
                        # operators. We store them and not sum them immediately as we need to
                        # readjust them later
                        diag_contribs[jj] = diag_mels[i, xs_n[b, i]]

                    # once we are done collecting, we can now process the diagonal mels of the
                    # current uid seperately

                    # extract the relevant subset once to avoid repeated lookups
                    sub_diag_contribs = diag_contribs[:uid_op_count]

                    # get the real part of the diagonal contribs
                    sub_diag_contribs_r = sub_diag_contribs.real

                    # get the imaginary part of the diagonal contribs
                    sub_diag_contribs_i = sub_diag_contribs.imag

                    # sum the imaginary contributions independently to get S_uid
                    i_sum = sub_diag_contribs_i.sum()

                    # now adjust the sum to be considered as the diagonal contribution

                    # first check if we are dealing with a real valued operator
                    if i_sum == 0 and np.all(sub_diag_contribs_i == 0):
                        # if we are here, we have a real operator, return the real part only
                        sub_diag_contribs_sum = sub_diag_contribs_r

                    elif i_sum == 0 and not np.all(sub_diag_contribs_i == 0):
                        # if we are here, it means we have a complex operator but for some reason
                        # the imaginary parts summed to zero, so we need to return an array of zeros
                        # with the same shape as the real part
                        sub_diag_contribs_sum = np.abs(sub_diag_contribs_r * 0.0)
                    else:
                        # if we are here, it means we have a complex operator with non-trivial
                        # correction factor, so we perform the correction

                        # compose the factor to multiply every real diagonal mel
                        factor = (1 / i_sum) * np.sqrt(np.abs(i_sum))

                        # adjust the real parts of all diagonal contributions by this factor
                        sub_diag_contribs_r = sub_diag_contribs_r * factor

                        # variable to hold the final result to be summed later
                        sub_diag_contribs_sum = sub_diag_contribs_r

                    # sum all adjusted contributions and append to mels array
                    mels[c_diag] += sub_diag_contribs_sum.sum()

                # now handle the off-diagonal contributions of all operators in this
                # uid group

                # get the indexing related information
                start = ops_start[uid_idx]
                end = start + ops_count[uid_idx]

                for idx in range(start, end):
                    # here, we are looping through every subop in this given uid

                    # get the subop's index
                    i = ops_grouped_by_uid[idx]

                    # n_conn_i is the number of off-diagonal connections that the subop
                    # i introduces for the current state b
                    n_conn_i = n_conns[i, xs_n[b, i]]

                    # only proceed if this operator contributes any off-diagonal
                    # connections
                    if n_conn_i > 0:
                        # the indices of the sites that operator i acts upon
                        sites = acting_on[i]

                        # how many sites operator i acts on
                        acting_size_i = acting_size[i]

                        # loop through each off-diagonal connection introduced by this
                        # operator
                        for cc in range(n_conn_i):

                            # assign the corresponding matrix element from all_mels
                            # this reflects the strength of the transition from the
                            # original state to the new connected state
                            mels[c + cc] = all_mels[i, xs_n[b, i], cc]

                            # copy the original batch state into x_prime at the
                            # correct offset. We'll then modify the relevant sites
                            # to form the connected state.
                            x_prime[c + cc] = np.copy(x_batch)

                            # update the states at the sites on which this operator
                            # acts, using the data from all_x_prime to reflect the
                            # new local configuration
                            for k in range(acting_size_i):
                                x_prime[c + cc, sites[k]] = all_x_prime[
                                    i, xs_n[b, i], cc, k
                                ]

                        # after processing these n_conn_i connections, move 'c'
                        # forward by n_conn_i positions so that the next additions
                        # occur after these off-diagonal states.
                        c += n_conn_i

                # here, right after finishing all subops in this uid, we have a contiguous
                # chunk of mels (and x_prime) corresponding to this uid
                c_after_uid = c

                # we can now process the mels of every uid seperately

                # extract the relevant subset once to avoid repeated lookups
                # these are all the mels of the subops in this uid
                sub_mels = mels[c_before_uid:c_after_uid]

                # sum the imaginary part to form S_uid
                imag_contrib = sub_mels.imag.sum()

                # apply the trick if the imaginary part sum contribution is not zero

                # first check if we have a real operator
                if imag_contrib == 0 and np.all(sub_mels.imag == 0):
                    # if we are here, we have a real operator, return the real part only
                    sub_mels = sub_mels.real

                elif imag_contrib == 0 and not np.all(sub_mels.imag == 0):
                    # if we are here, it means we have a complex operator but for some reason
                    # the imaginary parts summed to zero, so we need to return an array of zeros
                    # with the same shape as the real part since now S_u is zero
                    sub_mels = np.abs(sub_mels.real * 0.0)

                else:
                    # if we are here, it means we have a complex operator with non-trivial
                    # correction factor, so we perform the correction

                    # compose the correction factor
                    factor = (1.0 / imag_contrib) * np.sqrt(np.abs(imag_contrib))

                    # adjust the real parts of all mels by this factor
                    sub_mels = sub_mels.real * factor

                # add back the adjusted non-diagonal mels
                mels[c_before_uid:c_after_uid] = sub_mels

            # if padding is enabled, we need to ensure that every batch state
            # has the same number of connections, even if some states have fewer
            if pad:

                # calculate how many extra "dummy" connections must be added to match
                # the maximum number of connections found for any state (max_conn).
                # 'c - c_diag' is how many connections we have so far (including diagonal) for this
                # state, and 'max_conn' is the uniform desired length
                delta_conn = max_conn - (c - c_diag)

                # only perform padding if this state has fewer connections than max_conn
                if delta_conn > 0:
                    # fill the extra space in 'mels' with zeros, indicating no actual connection
                    mels[c : c + delta_conn].fill(0)

                    # for each padded connection, copy the original state into x_prime
                    # this is a placeholder and ensures consistent dimensions
                    for p in range(delta_conn):
                        x_prime[c + p] = np.copy(x_batch)

                    # increment 'c' by the number of padded connections to maintain correct indexing
                    c += delta_conn

                # update sections[b] to mark where the connections for this batch element end,
                # after accounting for the padding
                sections[b] = c

        # return the x_prime and mels (as real numbers now)
        return x_prime, mels.real.astype(np.float64)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def get_conn_flattened_kernel_optimized(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        nonzero_diagonal,
        uids,
        ops_grouped_by_uid,
        ops_start,
        ops_count,
        pad=False,
    ):
        # start by getting dimensions and type information

        # number of states in the batch
        batch_size = x.shape[0]

        # number of sites per state
        n_sites = x.shape[1]

        # data type for matrix elements
        dtype = all_mels.dtype

        # number of uid groups
        num_uids = uids.shape[0]

        # total number of operators
        n_operators = n_conns.shape[0]

        # TODO remove this line when numba 0.53 is dropped 0.54 is minimum version
        # workaround a bug in Numba arising when NUMBA_BOUNDSCHECK=1
        constant = constant.item()

        # now precompute the local configuration indices (xs_n) for every state and operator
        # each xs_n[b, i] is computed by converting the local configuration into a single integer
        # the conversion is performed by iterating over the operator's acting sites in reverse order

        # allocate xs_n array
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        # parallelize over states
        for b in prange(batch_size):

            # extract the b-th state
            x_b = x[b]

            # loop over each operator
            for i in range(n_operators):

                # get the number of sites operator i acts on
                acting_size_i = acting_size[i]

                # initialize the local configuration index for this operator
                xs_val = 0

                # loop over each acting site (in reverse order)
                for k in range(acting_size_i):

                    # calculate reverse index
                    idx = acting_size_i - k - 1

                    # use np.searchsorted to find the index of the state value in the
                    # sorted local_states
                    pos = np.searchsorted(local_states[i, idx], x_b[acting_on[i, idx]])

                    # multiply the found position by the corresponding basis weight and accumulate
                    xs_val += pos * basis[i, k]

                # store the computed index for state b and operator i
                xs_n[b, i] = xs_val

        # first pass: compute the number of connections for each state and the total connections
        #
        # for each state, start with 1 if nonzero_diagonal is True (for the self-connection),
        # then add the off-diagonal connection counts from all operators
        # we also, update the cumulative sections array

        # total number of connections across the entire batch
        tot_conn = 0

        # maximum connections for a single state (used if padding is enabled)
        max_conn = 0

        # loop over each state
        for b in range(batch_size):

            # start with 1 if including the diagonal, else 0
            conn_b = 1 if nonzero_diagonal else 0

            # now add off-diagonal connections from each operator
            for i in range(n_operators):
                conn_b += n_conns[i, xs_n[b, i]]

            # accumulate the total connections
            tot_conn += conn_b

            # record the cumulative connection index for state b
            sections[b] = tot_conn

            # if padding is enabled, track the maximum connection count
            if pad and conn_b > max_conn:
                max_conn = conn_b

        # with padding, total connections is uniform per state
        if pad:
            tot_conn = batch_size * max_conn

        # allocate flattened output arrays
        x_prime = np.empty((tot_conn, n_sites), dtype=x.dtype)
        mels = np.empty(tot_conn, dtype=dtype)

        # precompute the maximum number of operators in any uid group
        # this determines the size of the temporary array for diagonal contributions
        max_ops_per_uid = 0
        for u in range(num_uids):
            if ops_count[u] > max_ops_per_uid:
                max_ops_per_uid = ops_count[u]

        # temporary array for diagonal contributions
        diag_contribs = np.empty(max_ops_per_uid, dtype=dtype)

        # the main accumulation loop
        # process each state sequentially to preserve the ordering of connections

        # global counter for the flattened output arrays
        c = 0

        # process each state in the batch
        for b in range(batch_size):

            # mark the index where the state's diagonal element will be stored
            c_diag = c

            # extract the b-th state's full configuration
            x_batch = x[b]

            # if a diagonal (self-connection) is required
            if nonzero_diagonal:

                # set the matrix element at the diagonal position to the constant
                mels[c_diag] = constant

                # copy the full state into x_prime at the diagonal position
                for s in range(n_sites):
                    x_prime[c_diag, s] = x_batch[s]

                # increment the global counter after processing the diagonal
                c += 1

            # process contributions for each uid group
            # we are not about to use prange here because we cannot run in parallel as there
            # will be race conditions for the global counter...

            # loop over each uid group
            for uid_idx in range(num_uids):

                # mark the starting index for this uid group's contributions
                c_before_uid = c

                # for the diagonal contributions from this uid group
                if nonzero_diagonal:

                    # the number of operators in this group
                    uid_op_count = ops_count[uid_idx]

                    # where do they start
                    start = ops_start[uid_idx]

                    # loop through as many operators in this group
                    for jj in range(uid_op_count):

                        # retrieve the operator index
                        i = ops_grouped_by_uid[start + jj]
                        # store the diagonal contribution from operator i
                        # (indexed by xs_n for state b)
                        diag_contribs[jj] = diag_mels[i, xs_n[b, i]]

                    # in what follows we apply the correction to the diagonal mels collected
                    # to know what that is, go through the _get_conn_flattened_kernel() above

                    # extract real and imaginary parts
                    sub_diag_contribs_r = diag_contribs[:uid_op_count].real
                    sub_diag_contribs_i = diag_contribs[:uid_op_count].imag

                    # local variable to hold the sum of the imaginary part
                    i_sum = 0.0

                    # sum the imaginary parts
                    for j in range(uid_op_count):
                        i_sum += sub_diag_contribs_i[j]

                    # handle the case that the sum of imaginary parts is exactly zero
                    if i_sum == 0.0:

                        # assume initially that all imag parts are zero
                        all_imag_zero = True

                        # check every contribution
                        for j in range(uid_op_count):

                            # found a nonzero imaginary part
                            if sub_diag_contribs_i[j] != 0.0:
                                all_imag_zero = False
                                break

                        if all_imag_zero:
                            # use the real parts directly
                            sub_diag_contribs_sum = sub_diag_contribs_r
                        else:
                            # if some imaginary parts are nonzero but cancel exactly, force zero
                            sub_diag_contribs_sum = np.empty(
                                uid_op_count, dtype=np.float64
                            )

                            for j in range(uid_op_count):
                                sub_diag_contribs_sum[j] = 0.0
                    else:
                        # if the imaginary sum is nonzero, compute the correction factor
                        factor = (1.0 / i_sum) * np.sqrt(np.abs(i_sum))

                        # multiply each real part by the factor
                        for j in range(uid_op_count):
                            sub_diag_contribs_r[j] = sub_diag_contribs_r[j] * factor

                        sub_diag_contribs_sum = sub_diag_contribs_r

                    tmp_sum = 0.0

                    # sum the corrected diagonal contributions
                    for j in range(uid_op_count):
                        tmp_sum += sub_diag_contribs_sum[j]

                    # add this sum to the diagonal matrix element
                    mels[c_diag] += tmp_sum

                # now time to process off-diagonal contributions

                # get the count of operators in this uid and where they start
                uid_op_count = ops_count[uid_idx]
                start = ops_start[uid_idx]

                # loop through as many operators in this uid
                for jj in range(uid_op_count):

                    # get the index of the operators
                    i = ops_grouped_by_uid[start + jj]

                    # and the number of off-diagonal connections for operator i in state b
                    n_conn_i = n_conns[i, xs_n[b, i]]

                    # proceed only if there are off-diagonal connections
                    if n_conn_i > 0:

                        # number of sites operator i acts on
                        acting_size_i = acting_size[i]

                        # loop over each off-diagonal connection
                        for cc in range(n_conn_i):

                            # assign the corresponding off-diagonal matrix element
                            mels[c + cc] = all_mels[i, xs_n[b, i], cc]

                            # copy the current state into x_prime
                            for s in range(n_sites):
                                x_prime[c + cc, s] = x_batch[s]

                            # update the sites that operator i acts upon
                            for k in range(acting_size_i):

                                # retrieve the site index
                                site = acting_on[i, k]
                                # replace the state value at that site with the new value from
                                # all_x_prime
                                x_prime[c + cc, site] = all_x_prime[
                                    i, xs_n[b, i], cc, k
                                ]

                        # increment the global counter by the number of off-diagonal connections
                        c += n_conn_i

                # adjust off-diagonal contributions, again, see the _get_conn_flattened_kernel()

                # mark the end index for the current uid group contributions
                c_after_uid = c

                # lovcal variable to hold the imag sum
                imag_sum = 0.0

                # sum the imaginary parts for this uid group
                for idx in range(c_before_uid, c_after_uid):
                    imag_sum += mels[idx].imag

                # if the imaginary sum is zero
                if imag_sum == 0.0:
                    all_imag_zero = True

                    # check each contribution to see if all imaginary parts were zero
                    for idx in range(c_before_uid, c_after_uid):

                        # if not, stop the loop and change the flag value
                        if mels[idx].imag != 0.0:
                            all_imag_zero = False
                            break

                    for idx in range(c_before_uid, c_after_uid):
                        # if there were no imaginary parts, use the real part as is
                        if all_imag_zero:
                            mels[idx] = mels[idx].real
                        else:
                            # else set them to zero
                            mels[idx] = 0.0
                else:
                    # otherwise, correct the mels
                    factor = (1.0 / imag_sum) * np.sqrt(np.abs(imag_sum))

                    for idx in range(c_before_uid, c_after_uid):
                        mels[idx] = mels[idx].real * factor

            # if pad is enabled, pad the current state's connection list to have a uniform number of
            # entries (equal to max_conn)
            if pad:

                # compute how many dummy connections are needed
                delta_conn = max_conn - (c - c_diag)

                if delta_conn > 0:
                    for p in range(delta_conn):

                        # fill extra mels with 0.0
                        mels[c + p] = 0.0
                        for s in range(n_sites):

                            # duplicate the state in x_prime
                            x_prime[c + p, s] = x_batch[s]

                    # increment counter by the padding amount
                    c += delta_conn

                # update sections for state b to reflect padded size
                sections[b] = c

        return x_prime, mels.real.astype(np.float64)

    """
    Status:
        Pending

    Changes:
        None needed as this is used for netket.operator.LocalLiouvillian or 
        CustomRuleNumpy samplers which we do not use. 
        Nevertheless, uid-wise computation will be implemented to streamline the process.
    """

    def get_conn_filtered(self, x, sections, filters):
        r"""Finds the connected elements of the Operator using only a subset of operators. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            filters (array): Only operators op(filters[i]) are used to find the connected elements
                            of x[i].

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        self._setup()

        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        return self._get_conn_filtered_kernel(
            x,
            sections,
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
            filters,
        )

    """
    Status:
        Pending

    Changes:
        None needed as this is used for netket.operator.LocalLiouvillian or 
        CustomRuleNumpy samplers which we do not use. 
        Nevertheless, uid-wise computation will be implemented to streamline the process.
    """

    @staticmethod
    @jit(nopython=True)
    def _get_conn_filtered_kernel(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        filters,
    ):
        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        assert filters.shape[0] == batch_size and sections.shape[0] == batch_size

        # TODO remove this line when numba 0.53 is dropped 0.54 is minimum version
        # workaround a bug in Numba arising when NUMBA_BOUNDSCHECK=1
        constant = constant.item()

        n_operators = n_conns.shape[0]
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0

        for b in range(batch_size):
            # diagonal element
            tot_conn += 1

            # counting the off-diagonal elements
            i = filters[b]

            assert i < n_operators and i >= 0
            acting_size_i = acting_size[i]

            xs_n[b, i] = 0
            x_b = x[b]
            x_i = x_b[acting_on[i, :acting_size_i]]
            for k in range(acting_size_i):
                xs_n[b, i] += (
                    np.searchsorted(
                        local_states[i, acting_size_i - k - 1],
                        x_i[acting_size_i - k - 1],
                    )
                    * basis[i, k]
                )

            tot_conn += n_conns[i, xs_n[b, i]]
            sections[b] = tot_conn

        x_prime = np.empty((tot_conn, n_sites))
        mels = np.empty(tot_conn, dtype=dtype)

        c = 0
        for b in range(batch_size):
            c_diag = c
            mels[c_diag] = constant
            x_batch = x[b]
            x_prime[c_diag] = np.copy(x_batch)
            c += 1

            i = filters[b]
            # Diagonal part
            mels[c_diag] += diag_mels[i, xs_n[b, i]]
            n_conn_i = n_conns[i, xs_n[b, i]]

            if n_conn_i > 0:
                sites = acting_on[i]
                acting_size_i = acting_size[i]

                for cc in range(n_conn_i):
                    mels[c + cc] = all_mels[i, xs_n[b, i], cc]
                    x_prime[c + cc] = np.copy(x_batch)

                    for k in range(acting_size_i):
                        x_prime[c + cc, sites[k]] = all_x_prime[i, xs_n[b, i], cc, k]
                c += n_conn_i

        return x_prime, mels

    """
    Status:
        Pending

    Changes:
        We never cast our types to Jax, so this will probably be left as is after review.
        Changing this will require all changes done in this class to be redone for the Jax version.
        
    Update (20.07.2025):
        This will become necessary for Jax sharding when parallelising as sharding Numba operators
        is no longer supported.
        
    Update (25.07.2025):
        This is now production ready, as NetKet will drop MPI support and parallelisation will only
        be available via Jax distributed
    """

    def to_jax_operator(self) -> "MarkedLocalOperatorJax":  # noqa: F821
        """
        Returns the JAX-compatible version of this operator, which is an
        instance of :class:`netket.operator.LocalOperatorJax`.
        """

        from .jax import MarkedLocalOperatorJax

        return MarkedLocalOperatorJax(
            self.hilbert,
            self.operators,
            self.acting_on,
            self.constant,
            dtype=self.dtype,
            mel_cutoff=self.mel_cutoff,
            ident=self.__ident__,
        )

    @property
    def uids(self):
        if not hasattr(self, "_uids"):
            self._uids = self._aom.get_identifiers(self.acting_on)[1]

        return self._uids


# ---------------------- WARNING WARNING WARNING ----------------------------
