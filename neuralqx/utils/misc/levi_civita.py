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


def get_ijk_key(
    o: tuple,
    p: tuple,
) -> tuple:
    """
    Map a reference ordered triple to indices within another triple.

    Given a reference triple ``o = (e1, e2, e3)`` and another triple ``p`` containing the same
    elements (typically a permutation), this function returns the index positions of ``o``'s
    elements inside ``p`` as a tuple ``(i, j, k)``.

    Example:
        ``o = ('a','b','c')``, ``p = ('b','c','a')`` returns ``(2, 0, 1)``.

    :param o: Reference triple whose element order defines the canonical ordering.
    :param p: Target triple (typically a permutation of ``o``) in which indices are looked up.
    :returns: Tuple ``(i, j, k)`` giving the positions of ``o[0]``, ``o[1]``, ``o[2]`` in ``p``.
    :raises ValueError: If any element of ``o`` is not present in ``p``.
    """

    return p.index(o[0]), p.index(o[1]), p.index(o[2])


def get_sign_of_tuple_key(indices: tuple) -> int:
    """
    Compute the parity (+1/-1) of a permutation specified by index positions.

    This function returns the sign of the permutation encoded by ``indices`` by counting the
    number of inversions. An even number of inversions yields ``+1`` and an odd number yields
    ``-1``. This is commonly used to obtain the Levi-Civita sign for a permuted triple.

    :param indices: Tuple of indices (e.g. ``(i, j, k)``) encoding a permutation.
    :returns: ``+1`` if the permutation is even, otherwise ``-1``.
    """

    # dev: handle the case if i = j, j = k or k = i (should return 0) in 3d LV case?
    #  check the following
    if len(indices) != len(set(indices)):
        return 0

    count = 0
    n = len(indices)

    for i in range(n):
        for j in range(i + 1, n):
            if indices[i] > indices[j]:
                count += 1

    return 1 if count % 2 == 0 else -1


def compute_levi_civita_key(
    combinations: list,
    lc: dict,
) -> None:
    """
    Populate a Levi-Civita lookup dictionary for permutations of a reference triple.

    Given a list of permutations of a fixed triple ``(e1, e2, e3)``, this function computes the
    Levi-Civita sign ``ε(e1, e2, e3)`` for each permuted triple and stores it in ``lc``. The
    dictionary is modified in-place and uses ``str(permuted_edge_triple)`` as the key.

    The sign is computed by:
    1) taking the first triple in ``combinations`` as the reference ordering,
    2) mapping the reference elements into the permuted triple to obtain index positions, and
    3) computing the parity of that permutation.

    :param combinations: List of triples (tuples) representing permutations of the same three items.
                         The first element is treated as the reference ordering.
    :param lc: Dictionary to be filled/updated with Levi-Civita signs. Modified in-place.
    :returns: ``None``.
    :raises IndexError: If ``combinations`` is empty.
    """

    first_pair = combinations[0]

    # for every permutation, save its epsilon contribution
    for permuted_edge_triple in combinations:
        key = str(permuted_edge_triple)
        if key not in lc:
            lc[key] = get_sign_of_tuple_key(
                get_ijk_key(first_pair, permuted_edge_triple)
            )
