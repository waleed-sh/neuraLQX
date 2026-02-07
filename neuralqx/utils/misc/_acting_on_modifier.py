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
This is a class responsible for handling the modification of acting_on sites for the Thiemann
regularised constraint in 4d
"""

from numpy import ndarray
from typing import Iterable, Optional, Union, Any
import numpy as np


class ActingOnModifier:
    """
    A class which is responsible for modifying the acting_on of a LocalOperator
    """

    def __init__(self, modifier: str):
        self.modifier = modifier

    @staticmethod
    def ensure_pure_aon(ao: tuple) -> bool:
        """
        A method that ensures that the acting_on is pure (no identifiers have
        been performed). Returns True if the acting_on is pure.
        """
        return not isinstance(ao[0], str)

    def mark_aon(self, ao: tuple) -> tuple:
        """
        A method to mark an acting_on tuple with the unique identifier of the
        LocalOperator
        """
        if self.ensure_pure_aon(ao):
            ao = (self.modifier,) + ao

        return ao

    def modify_aon(self, ao: tuple) -> tuple:
        """
        A method to modify a tuple of actin_on sites by adding -1 to the end of the tuple
        """

        # first ensure that the acting_on has been identified
        ao = self.mark_aon(ao)

        return ao + (-1,)

    @staticmethod
    def _strip_aon_tuple(ao: tuple) -> tuple:
        """
        A method which strips any acting_on sites which are `-1` but leaves the identifier
        """

        return tuple(a for a in ao if a != -1)

    def get_pure_aon(self, ao: tuple) -> tuple:
        """
        A method which returns the pure acting_on, with no `-1` sites or identifiers
        """
        ao = self._strip_aon_tuple(ao)

        if self.ensure_pure_aon(ao):
            return ao

        return ao[1:]

    def get_pure_aon_iter_marked(self, ao: Iterable) -> tuple:
        """
        A method which acts exactly as get_pure_aon_iter but additionally returns the
        modifiers as well
        """

        if not isinstance(ao, list):
            ao = list(ao)

        markings = (
            [a[0] for a in ao] if all(isinstance(a[0], str) for a in ao) else None
        )

        return self.get_pure_aon_iter(ao), markings

    def get_pure_aon_iter(self, ao: Iterable) -> Iterable:
        """
        A method which returns pure acting_on sites for a list of acting_on sites
        """

        if not isinstance(ao, list):
            ao = list(ao)

        return [self.get_pure_aon(a) for a in ao]

    def get_count_minus(
        self, ao: Union[Iterable, Any]
    ) -> tuple[Union[Iterable, Any], int]:
        """
        A method which returns how many `-1` sites were present in an acting_on site and
        returns that number as well as the pure acting_on site
        """

        count_minus = ao.count(-1)

        ao = tuple(a for a in ao if a != -1)

        if not self.ensure_pure_aon(ao):
            ao = ao[1:]

        return ao, count_minus

    def append_minus(self, ao: tuple, count_minus: int) -> tuple:
        """
        A method which appends back the `-1` sites back to the acting_on but not the identifier
        """

        # check if the identifier is missing
        ao = self.mark_aon(ao)

        return ao + count_minus * (-1,)

    def append_ident(self, ao: tuple, count_minus: int) -> tuple:
        """
        A method which appends back the `-1` and the identifier to the acting_on
        """

        return (self.modifier,) + ao + count_minus * (-1,)

    def mark_canonicalized_acting_on(
        self,
        canonical_acting_on: list,
        markings: Optional[list] = None,
    ) -> list:
        """
        A method which loops through a list of canonicalized acting_on sites
        and prepends to them an identifier and as many `-1` sites for duplicate
        acting_on sites present in the list
        """
        modified_canonical_acting_on = []
        for i, acting_on in enumerate(canonical_acting_on):
            # add the modifier, while taking into account any previous markings
            modifier = self.modifier
            if markings is not None:
                modifier = modifier + markings[i]

            acting_on = (modifier,) + acting_on

            # check for duplicates
            while acting_on in modified_canonical_acting_on:
                acting_on = acting_on + (-1,)
            modified_canonical_acting_on.append(acting_on)

        return modified_canonical_acting_on

    def insert_highest_modification(
        self, acting_on_list: list, acting_on: tuple
    ) -> tuple:
        """
        A method which is ensured to return a modified acting_on which does not exist in
        acting_on_list by appending `-1` sites as needed. It is assumed that acting_on
        is pure
        """

        # first add the modifier to ensure correct search in now modified acting_on_list
        acting_on = self.mark_aon(acting_on)

        while acting_on in acting_on_list:
            acting_on = acting_on + (-1,)

        return self.mark_aon(self._ensure_int(acting_on))

    def get_count_minus_iter(self, acting_on_list: list) -> tuple[list, list]:
        """
        A method which takes a list of modified acting_on sites and returns two
        lists:
            (a) a list of pure acting_on sites
            (b) a list of count_minus for every acting_on site
        """
        pure_acting_on = []
        count_minus = []

        for i, acting_on in enumerate(acting_on_list):
            get_count_minus_aon = self.get_count_minus(acting_on)
            pure_acting_on.append(get_count_minus_aon[0])
            count_minus.append(get_count_minus_aon[1])

        return pure_acting_on, count_minus

    @staticmethod
    def _ensure_int(acting_on: tuple) -> tuple:
        """
        A method that ensures that the sites are int valued. Assumes marked acting_on.
        """
        return tuple(int(a) for a in acting_on[1:])

    @staticmethod
    def get_identifiers(acting_on_list: list) -> tuple[list, list]:
        """
        A method which returns the list of identifiers for every operator as well
        as a list of all identifiers (a set of identifiers)
        """
        idents = []
        for aon in acting_on_list:
            # ensure that the current acting_on has an identifier
            assert isinstance(
                aon[0], str
            ), f"Acting on modifier is of type {type(aon[0])}: {aon}"
            # append to the idents list
            idents.append(aon[0])

        return idents, list(set(idents))

    # ---------------------------- WARNING WARNING WARNING ----------------------------
    # The logic in the following code should not be altered under any circumstances.
    # This is for internal use only and modifying the logic may result in incorrect
    # computation of matrix elements of some constraints

    @staticmethod
    def prep_for_gccf(
        identifiers_list: list, uids: list
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """
         A method to transform a list of operator identifiers and their unique IDs into a form
         suitable for use in Numba's noPython mode.

         This method takes:
             - identifiers_list: A list of identifiers (one per operator) that link each operator
               to a particular "group" identified by a unique ID
             - uids: A list of unique IDs that represent distinct operator groups.

         Both `identifiers_list` and `uids` are assumed to represent some kind of numeric or
         sortable identifiers. The function will:
             1. Convert these lists into NumPy arrays and ensure `uids` is sorted.
             2. Compute how many operators belong to each uid.
             3. Determine the start positions for each uid within a flattened array of all operators
             4. Create a single flat array (`ops_grouped_by_uid`) that stores all operator indices
        grouped by their corresponding uid.

         It will then return:
             - uids (np.ndarray): The sorted array of unique IDs.
             - ops_grouped_by_uid (np.ndarray): flat array of operator indices grouped first by uid
             - ops_start (np.ndarray): array where ops_start[uid_idx] gives the starting index of
                                       that uid's operators in ops_grouped_by_uid.
             - ops_count (np.ndarray): array where ops_count[uid_idx] gives the number of operators
                                       belonging to that uid.
        """

        # cast into NumPy arrays
        identifiers_list = np.array(identifiers_list)
        uids = np.array(uids)

        # ensure that uids are sorted so that np.searchsorted works correctly
        uids.sort()

        # number of unique IDs
        num_uids = uids.shape[0]

        # number of operators
        n_operators = identifiers_list.shape[0]

        # now count how many operators belong to each uid
        # ops_count will store the size of each uid group
        ops_count = np.zeros(num_uids, dtype=np.intp)
        for i in range(n_operators):
            # find the index in uids where identifiers_list[i] would fit
            uid_index = np.searchsorted(uids, identifiers_list[i])

            # increment the count for the corrresponding uid.
            ops_count[uid_index] += 1

        # compute start indices for each uid in a flattened operator list
        # ops_start[uid_idx] tells us where that uid's operators begin in ops_grouped_by_uid
        ops_start = np.zeros(num_uids, dtype=np.intp)
        for u in range(1, num_uids):
            ops_start[u] = ops_start[u - 1] + ops_count[u - 1]

        # now create a single flat array with all operators grouped by uid
        # we'll fill this array so that all operators belonging to uid u are placed
        # contiguously, starting at ops_start[u]
        ops_grouped_by_uid = np.empty(n_operators, dtype=np.intp)

        # temp_count will track how many operators of each uid we've placed so far
        temp_count = np.zeros(num_uids, dtype=np.intp)
        for i in range(n_operators):
            uid_index = np.searchsorted(uids, identifiers_list[i])

            # now compute the exact position in ops_grouped_by_uid for this operator based
            # on how many we've already placed for this uid
            pos = ops_start[uid_index] + temp_count[uid_index]
            ops_grouped_by_uid[pos] = i
            temp_count[uid_index] += 1

        # Return the arrays:
        #   - uids: the sorted unique IDs
        #   - ops_grouped_by_uid: flattened array of operator indices grouped by uid
        #   - ops_start: start indices of each uid's group
        #   - ops_count: number of operators for each uid
        return uids, ops_grouped_by_uid, ops_start, ops_count

    # ---------------------------- WARNING WARNING WARNING ----------------------------
