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
NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

from typing import Dict, List, Tuple, Callable

import numpy as np
import jax.numpy as jnp

from netket.hilbert import AbstractHilbert
from netket.utils.types import DType

from neuralqx.utils.misc import ActingOnModifier


def encode_string_uids(uids_list):
    """
    A function which will take the list of unique ids and then map them to a unique integer
    representation such that every unique id is mapped to an integer. Essentially, converting string
    based uids to integer based uids in a unique manner.

    Returns:
      - int_uids: list or array of the same length as `uids_list`, where int_uids[i] is the integer
       label for uids_list[i]
      - uid_to_int: dict mapping uid_string -> integer_label
      - int_to_uid: list where int_to_uid[integer_label] = original uid_string
    """

    unique_strings = sorted(set(uids_list))
    uid_to_int = {}
    int_to_uid = []

    for i, s in enumerate(unique_strings):
        uid_to_int[s] = i
        int_to_uid.append(s)

    int_uids = [uid_to_int[s] for s in uids_list]

    return int_uids, uid_to_int, int_to_uid


def _pack_internals_jax(
    hilbert: AbstractHilbert,
    operators_dict: dict,
    constant,
    dtype: DType,
    mel_cutoff: float,
    aom: ActingOnModifier,
    pack_internals: Callable,
):
    """
    This implementation follows NetKet's logic of grouping by acting_on size, but instead
    we group together operators based on their unique ID and then call pack_internals on
    each group
    """

    """
    Changes:
        Collect the unique IDs as this is what we need

    Reason:
        We are about to loop through operators and group them by unique IDs, so
        we have to collect all the information through the aom now
    """
    # get the uids and all ids
    instance_identifiers, instance_u_identifiers = aom.get_identifiers(
        list(operators_dict.keys())
    )
    # get JAX friendly representations
    uids, ops_grouped_by_uid, ops_start, ops_count = aom.prep_for_gccf(
        instance_identifiers, instance_u_identifiers
    )

    """
    Changes:
        The op_acting_on are left MODIFIED!

    Reason:
        The base pack_internals will take care of purifying them. If we do that now, an assertion
        error will be thrown when extracting identifiers and uids in the base pack_internals.
    """
    op_acting_on = list(operators_dict.keys())

    operators = list(operators_dict.values())

    # a global data dict to be returned
    data = {}
    data["nonzero_diagonal"] = np.abs(constant) >= mel_cutoff
    data["max_conn_size"] = 0

    """
    Changes:
        Iteratore over groups of operators with the same unique ID

    Reason:
        We intend to group by UID not by acting_size, so we replace
        NetKet's logic with this one
    """
    # get the number of UIDS
    num_uids = uids.shape[0]

    # loop through as many UIDS as there are
    for uid_idx in range(num_uids):
        # get the operator count
        uid_op_count = ops_count[uid_idx]

        # we now need the indices of these operators so we can fetch them from `operators`
        # the starting index
        start = ops_start[uid_idx]
        # the ending index = start + count_of_ops_in_this_uid
        end = start + uid_op_count

        # get the concrete list of indices of the operators in this uid
        indices = tuple(int(ops_grouped_by_uid[i]) for i in range(start, end))

        # now collect them all in a dict
        operators_dict_uid = {op_acting_on[i]: operators[i] for i in indices}

        # now pack internals for this group
        data_uid = pack_internals(
            hilbert, operators_dict_uid, 0, dtype, mel_cutoff, jax=True
        )

        # get the nonzero_diagonal
        nonzero_diagonal = bool(data_uid.pop("nonzero_diagonal"))

        # get the max conn size
        max_conn_size = int(data_uid.pop("max_conn_size"))

        # if we have a non-zero diagonal, NetKet only counts the diag elem once a little
        # later, so we subtract it here
        if nonzero_diagonal:
            max_conn_size = max_conn_size - 1

        # append to the big data structure to be returned
        data["nonzero_diagonal"] = data["nonzero_diagonal"] or nonzero_diagonal
        data["max_conn_size"] = data["max_conn_size"] + max_conn_size

        # append the remaining elements from this specific uid pack_internals result
        for k, v in data_uid.items():
            if k == "identifiers" or k == "uids":
                continue
            data[k] = data.pop(k, []) + [jnp.asarray(v)]

    # account for the diag elem we subtracted above
    if data["nonzero_diagonal"]:
        data["max_conn_size"] = data["max_conn_size"] + 1

    # convert to JAX arrays, otherwise expect() will scream...
    # we need to encode the uids as string, so we map into a unique int representations
    # because JAX does not support str arrays
    data["uids"] = jnp.asarray(encode_string_uids(uids)[0])
    data["ops_grouped_by_uid"] = jnp.asarray(ops_grouped_by_uid)
    data["ops_start"] = jnp.asarray(ops_start)
    data["ops_count"] = jnp.asarray(ops_count)

    return data


def pack_internals_jax_uid_aon(
    hilbert: AbstractHilbert,
    operators_dict: dict,
    constant,
    dtype: DType,
    mel_cutoff: float,
    aom: ActingOnModifier,
    pack_internals: Callable,
):
    """
    This implementation follows NetKet's logic of grouping by acting_on size, but instead we first
    group together operators based on their unique ID and then further subgroup them based on their
    acting_on size. We then call pack_internals on the per-uid acting_on group. It returns a list,
    instead of a dict, of the 'packed internals' for each uid group.
    """

    """
    Changes:
        Collect the unique IDs as this is what we need

    Reason:
        We are about to loop through operators and group them by unique IDs, so
        we have to collect all the information through the aom now
    """
    # get the uids and all ids
    instance_identifiers, instance_u_identifiers = aom.get_identifiers(
        list(operators_dict.keys())
    )
    # get JAX friendly representations
    uids, ops_grouped_by_uid, ops_start, ops_count = aom.prep_for_gccf(
        instance_identifiers, instance_u_identifiers
    )

    """
    Changes:
        The op_acting_on are left MODIFIED!

    Reason:
        The base pack_internals will take care of purifying them. If we do that now, an assertion
        error will be thrown when extracting identifiers and uids in the base pack_internals.
    """
    op_acting_on = list(operators_dict.keys())

    operators = list(operators_dict.values())

    # a global data list to be returned, this will house the per-uid group information
    data = []

    """
    Changes:
        Iterate over groups of operators with the same unique id

    Reason:
        We intend to group by uid not by acting_size, so we replace
        NetKet's logic with this one
    """
    # get the number of uids
    num_uids = uids.shape[0]

    # loop through as many uid as there are
    for uid_idx in range(num_uids):
        # get the operator count
        uid_op_count = ops_count[uid_idx]

        # we now need the indices of these operators so we can fetch them from `operators`
        # the starting index
        start = ops_start[uid_idx]
        # the ending index = start + count_of_ops_in_this_uid
        end = start + uid_op_count

        # get the concrete list of indices of the operators in this uid
        indices = tuple(int(ops_grouped_by_uid[i]) for i in range(start, end))

        # get the list of acting_on for this uid
        op_acting_on_s = [op_acting_on[i] for i in indices]

        # get the acting_size for this set of acting ons of this uid
        # we purify when counting to ensure proper grouping per aons which actually matter
        acting_size_uid = np.array(
            [len(aom.get_pure_aon(aon)) for aon in op_acting_on_s],
            dtype=np.intp,
        )

        # get list of the operators for this uid
        operators_s = [operators[i] for i in indices]

        # data structures for the uid which will hold the pack_internals resulting from every
        # acting_on subgroup
        data_uid = {}
        data_uid["nonzero_diagonal"] = np.abs(constant) >= mel_cutoff
        data_uid["max_conn_size"] = 0

        # ata structures for the acting_on process, s for support
        data_s = {}
        data_s["nonzero_diagonal"] = np.abs(constant) >= mel_cutoff
        data_s["max_conn_size"] = 0

        # iterate over operators in this uid which have the same acting_size_uid
        for s in np.unique(acting_size_uid) if len(acting_size_uid) > 0 else [0]:

            # get the indices
            (indices_s,) = np.where(acting_size_uid == s)

            # build from the indices the specific operator_dict in this uid group for operators with
            # supports of length acting_size_uid
            operators_dict_s = {op_acting_on_s[i]: operators_s[i] for i in indices_s}

            # perform pack internals for this group of operators in this acting_size_uid group in this uid
            # the JAX flag must be True to instruct the pack_internals to use NetKet's new indexing
            data_s = pack_internals(
                hilbert, operators_dict_s, 0, dtype, mel_cutoff, jax=True
            )

            # get the non-zero diagonal flag and the maximum number of connections
            nonzero_diagonal = bool(data_s.pop("nonzero_diagonal"))
            max_conn_size = int(data_s.pop("max_conn_size"))

            # if we have a non-zero diagonal, NetKet only counts the diag elem once a little
            # later, so we subtract it here
            if nonzero_diagonal:
                max_conn_size = max_conn_size - 1

            # append to the uid data structure this aon group's information
            data_uid["nonzero_diagonal"] = bool(
                data_uid["nonzero_diagonal"] or nonzero_diagonal
            )
            data_uid["max_conn_size"] = data_uid["max_conn_size"] + max_conn_size

            # here we append the rest of the data, except the identifiers and uids returned from
            # the pack_internals as they are of dtype str, incompatible with JAX arrays...
            for k, v in data_s.items():
                if k == "identifiers" or k == "uids":
                    continue
                data_uid[k] = data_uid.pop(k, []) + [jnp.asarray(v)]

        # account for the diag elem we subtracted above
        if data_uid["nonzero_diagonal"]:
            data_uid["max_conn_size"] = data_uid["max_conn_size"] + 1

        # now we append to the overall data list
        data.append(data_uid)

    # now get global information that we need

    # and then the max_conn_size overall
    # sum up all the off-diagonal parts
    offdiag_sum_ = sum(
        d["max_conn_size"] - (1 if d["nonzero_diagonal"] else 0) for d in data
    )

    # check if ANY group is nonzero_diagonal
    any_diagonal_ = np.any(d["nonzero_diagonal"] for d in data)

    if any_diagonal_:
        total_max_conn_size_ = offdiag_sum_ + 1
    else:
        total_max_conn_size_ = offdiag_sum_

    # build a dict of everything we want to return
    r_data = {
        "data": data,
        "nonzero_diagonal": bool(any_diagonal_),
        "max_conn_size": total_max_conn_size_,
        "uids": jnp.asarray(encode_string_uids(uids)[0]),
        "ops_grouped_by_uid": jnp.asarray(ops_grouped_by_uid),
        "ops_start": jnp.asarray(ops_start),
        "ops_count": jnp.asarray(ops_count),
    }

    # return the data
    return r_data


def pack_internals_jax_uid(
    hilbert: AbstractHilbert,
    operators_dict: Dict[Tuple[int, ...], np.ndarray],
    constant,
    dtype,
    mel_cutoff: float,
    *,
    aom: ActingOnModifier,
    pack_internals_fn: Callable,
):
    """
    Pack the internal lazy representation of a LocalOperator grouping only by `uid`.
    All arrays are padded to the largest shape occurring inside that uid,
    so that every entry has statically-known shape – JAX-JIT friendly.
    """

    # figure out which operator belongs to which uid
    identifiers, uids_raw = aom.get_identifiers(list(operators_dict.keys()))
    uids, ops_grouped_by_uid, ops_start, ops_count = aom.prep_for_gccf(
        identifiers, uids_raw
    )

    num_uids = int(uids.shape[0])
    op_acting_on = list(operators_dict.keys())
    operators = list(operators_dict.values())

    # for each uid collect all its operators (no sub-grouping),
    # run the standard pack_internals once, and remember the padding
    acting_on_: List[jnp.ndarray] = []
    acting_size_: List[jnp.ndarray] = []
    n_conns_: List[jnp.ndarray] = []
    diag_mels_: List[jnp.ndarray] = []
    mels_: List[jnp.ndarray] = []
    x_prime_: List[jnp.ndarray] = []
    basis_: List[jnp.ndarray] = []

    nonzero_diag_global = False
    max_conn_size_global = 0

    for uid_idx in range(num_uids):
        # slice out the indices of the operators that share this uid
        start, count = int(ops_start[uid_idx]), int(ops_count[uid_idx])
        idxs = tuple(int(ops_grouped_by_uid[start + k]) for k in range(count))

        sub_dict = {op_acting_on[i]: operators[i] for i in idxs}

        # the numba-style packer already knows how to pad *within* this group
        p = pack_internals_fn(hilbert, sub_dict, 0, dtype, mel_cutoff, jax=True)

        # store / pad at the uid level
        acting_on_.append(jnp.asarray(p["acting_on"]))
        acting_size_.append(jnp.asarray(p["acting_size"]))
        n_conns_.append(jnp.asarray(p["n_conns"]))
        diag_mels_.append(jnp.asarray(p["diag_mels"]))
        mels_.append(jnp.asarray(p["mels"]))
        x_prime_.append(jnp.asarray(p["x_prime"]))
        basis_.append(jnp.asarray(p["basis"]))

        # book-keeping for global metadata
        uid_nonzero = bool(p["nonzero_diagonal"])
        uid_maxconn = int(p["max_conn_size"]) - (1 if uid_nonzero else 0)

        nonzero_diag_global = nonzero_diag_global or uid_nonzero
        max_conn_size_global += uid_maxconn

    if nonzero_diag_global:
        # add the (single) diagonal slot
        max_conn_size_global += 1

    return dict(
        acting_on=acting_on_,
        acting_size=acting_size_,
        n_conns=n_conns_,
        diag_mels=diag_mels_,
        mels=mels_,
        x_prime=x_prime_,
        basis=basis_,
        nonzero_diagonal=nonzero_diag_global,
        max_conn_size=max_conn_size_global,
        constant=constant,
        # (the four arrays below are handy diagnostics, not needed by the kernel)
        uids=jnp.asarray(encode_string_uids(uids)[0]),
        ops_grouped_by_uid=jnp.asarray(ops_grouped_by_uid),
        ops_start=jnp.asarray(ops_start),
        ops_count=jnp.asarray(ops_count),
    )


def pack_internals_jax(
    hilbert: AbstractHilbert,
    operators_dict: dict,
    constant,
    dtype: DType,
    mel_cutoff: float,
    aom: ActingOnModifier,
    pack_internals: Callable,
):
    # we groups together operators which act on the same number of sites
    # and then call pack_internals on each group
    #
    # TODO in the future consider separating also operators with different sizes
    # to avoid excessive padding
    # (only relevant for non-uniform number of local states)

    """
    Changes:
        Collect the unique IDs as this is what we need

    Reason:
        We are about to loop through operators and group them by unique IDs, so
        we have to collect all the information through the aom now
    """
    # get the uids and all ids
    instance_identifiers, instance_u_identifiers = aom.get_identifiers(
        list(operators_dict.keys())
    )
    # get JAX friendly representations
    uids, ops_grouped_by_uid, ops_start, ops_count = aom.prep_for_gccf(
        instance_identifiers, instance_u_identifiers
    )

    """
    Changes:
        The op_acting_on are left MODIFIED!

    Reason:
        The base pack_internals will take care of purifying them. If we do that now, an assertion
        error will be thrown when extracting identifiers and uids in the base pack_internals.
    """
    op_acting_on = list(operators_dict.keys())

    operators = list(operators_dict.values())

    # how many sites each operator is acting on
    acting_size = np.array([len(aon) for aon in op_acting_on], dtype=np.intp)

    data = {}
    data["nonzero_diagonal"] = np.abs(constant) >= mel_cutoff
    data["max_conn_size"] = 0

    # iterate over groups of operators with same number of sites
    # special case for empty operator
    for s in np.unique(acting_size) if len(acting_size) > 0 else [0]:
        (indices,) = np.where(acting_size == s)
        operators_dict_s = {op_acting_on[i]: operators[i] for i in indices}
        data_s = pack_internals(
            hilbert, operators_dict_s, 0, dtype, mel_cutoff, jax=False
        )
        nonzero_diagonal = bool(data_s.pop("nonzero_diagonal"))
        max_conn_size = int(data_s.pop("max_conn_size"))
        if nonzero_diagonal:
            # we only count the diag elem once below, so we subtract it here
            max_conn_size = max_conn_size - 1
        data["nonzero_diagonal"] = data["nonzero_diagonal"] or nonzero_diagonal
        data["max_conn_size"] = data["max_conn_size"] + max_conn_size
        # append other elements to lists:
        for k, v in data_s.items():
            data[k] = data.pop(k, []) + [jnp.asarray(v)]

    # count the diagonal once at the end
    if data["nonzero_diagonal"]:
        data["max_conn_size"] = data["max_conn_size"] + 1

    return data
