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


import importlib
import itertools
import pytest


@pytest.fixture
def lcmod():
    return importlib.import_module("neuralqx.utils.misc.levi_civita")


def test_get_ijk_key_returns_indices_in_parent_tuple(lcmod):
    p = ("e1", "e2", "e3")
    o = ("e2", "e1", "e3")
    assert lcmod.get_ijk_key(o, p) == (1, 0, 2)


def test_get_ijk_key_raises_if_element_not_in_parent(lcmod):
    p = ("e1", "e2", "e3")
    o = ("e1", "eX", "e3")
    with pytest.raises(ValueError):
        lcmod.get_ijk_key(o, p)


def test_get_ijk_key_raises_if_parent_has_duplicates_and_missing_index(lcmod):
    p = ("e1", "e2", "e2")
    o = ("e2", "e1", "e2")
    assert lcmod.get_ijk_key(o, p) == (1, 0, 1)


@pytest.mark.parametrize(
    "indices, expected",
    [
        ((0, 1, 2), 1),
        ((0, 2, 1), -1),
        ((1, 0, 2), -1),
        ((1, 2, 0), 1),
        ((2, 0, 1), 1),
        ((2, 1, 0), -1),
    ],
)
def test_get_sign_of_tuple_key_parity_for_all_S3(lcmod, indices, expected):
    assert lcmod.get_sign_of_tuple_key(indices) == expected


def test_get_sign_of_tuple_key_handles_non_3_len_tuples(lcmod):
    assert lcmod.get_sign_of_tuple_key((0, 1, 2, 3)) == 1
    assert lcmod.get_sign_of_tuple_key((1, 0, 2, 3)) == -1


def test_get_sign_of_tuple_key_duplicate_indices_current_behavior(lcmod):
    assert lcmod.get_sign_of_tuple_key((0, 0, 1)) == 0
    assert lcmod.get_sign_of_tuple_key((1, 0, 0)) == 0


def test_compute_levi_civita_key_populates_all_permutations_with_correct_signs(lcmod):
    edges = ("a", "b", "c")
    perms = list(itertools.permutations(edges, 3))
    lc = {}

    lcmod.compute_levi_civita_key(perms, lc)

    assert len(lc) == 6

    first = perms[0]
    assert lc[str(first)] == 1

    for perm in perms:
        ijk = lcmod.get_ijk_key(first, perm)
        expected = lcmod.get_sign_of_tuple_key(ijk)
        assert lc[str(perm)] == expected


def test_compute_levi_civita_key_does_not_overwrite_existing_values(lcmod):
    edges = ("a", "b", "c")
    perms = list(itertools.permutations(edges, 3))
    lc = {}

    lcmod.compute_levi_civita_key(perms, lc)

    target_key = str(perms[1])
    lc[target_key] = 12345

    lcmod.compute_levi_civita_key(perms, lc)
    assert lc[target_key] == 12345


def test_compute_levi_civita_key_reference_order_changes_sign_convention(lcmod):
    edges = ("a", "b", "c")
    perms = list(itertools.permutations(edges, 3))

    lc1 = {}
    lcmod.compute_levi_civita_key(perms, lc1)

    perms_rot = perms[1:] + perms[:1]
    lc2 = {}
    lcmod.compute_levi_civita_key(perms_rot, lc2)

    ref2 = perms_rot[0]
    factor = lc1[str(ref2)]

    for perm in perms:
        assert lc2[str(perm)] == lc1[str(perm)] * factor


def test_compute_levi_civita_key_partial_existing_dict_only_adds_missing(lcmod):
    edges = ("a", "b", "c")
    perms = list(itertools.permutations(edges, 3))
    lc = {}

    lc[str(perms[0])] = 999
    lcmod.compute_levi_civita_key(perms, lc)

    assert len(lc) == 6
    assert lc[str(perms[0])] == 999


def test_compute_levi_civita_key_requires_combinations_nonempty(lcmod):
    lc = {}
    with pytest.raises(IndexError):
        lcmod.compute_levi_civita_key([], lc)
