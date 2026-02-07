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


import numpy as np
import pytest

from neuralqx.utils.misc._acting_on_modifier import ActingOnModifier


@pytest.fixture
def mod():
    return ActingOnModifier("M")


def test_ensure_pure_aon(mod):
    assert mod.ensure_pure_aon((1, 2)) is True
    assert mod.ensure_pure_aon(("M", 1, 2)) is False


def test_mark_aon_idempotent(mod):
    pure = (1, 2, 3)
    marked = mod.mark_aon(pure)
    assert marked == ("M", 1, 2, 3)

    already = ("X", 4, 5)

    assert mod.mark_aon(already) == already


def test_modify_aon_marks_then_appends_minus(mod):
    assert mod.modify_aon((1, 2)) == ("M", 1, 2, -1)
    assert mod.modify_aon(("M", 1, 2)) == ("M", 1, 2, -1)


def test_strip_aon_tuple_removes_all_minus_preserves_rest(mod):
    ao = ("M", 1, -1, 2, -1, -1, 3)
    stripped = mod._strip_aon_tuple(ao)
    assert stripped == ("M", 1, 2, 3)


def test_get_pure_aon_strips_minus_and_identifier(mod):
    assert mod.get_pure_aon((1, 2, -1, -1)) == (1, 2)
    assert mod.get_pure_aon(("M", 1, 2, -1, -1)) == (1, 2)
    assert mod.get_pure_aon(("X", 9, -1, 10)) == (9, 10)


def test_get_pure_aon_iter_accepts_iterables(mod):
    data = (("M", 1, -1), (2, 3, -1, -1), ("M", 9, 10))
    out = mod.get_pure_aon_iter(data)
    assert out == [(1,), (2, 3), (9, 10)]


def test_get_pure_aon_iter_marked_all_marked_returns_markings(mod):
    data = [("A", 1, -1), ("B", 2, 3), ("C", 9, -1)]
    pure, markings = mod.get_pure_aon_iter_marked(data)
    assert pure == [(1,), (2, 3), (9,)]
    assert markings == ["A", "B", "C"]


def test_get_pure_aon_iter_marked_mixed_returns_none(mod):
    data = [("A", 1, -1), (2, 3), ("C", 9)]
    pure, markings = mod.get_pure_aon_iter_marked(data)
    assert pure == [(1,), (2, 3), (9,)]
    assert markings is None


def test_get_count_minus_returns_pure_and_count(mod):
    ao = ("M", 1, 2, -1, -1)
    pure, count = mod.get_count_minus(ao)
    assert pure == (1, 2)
    assert count == 2

    ao2 = (5, -1, -1)
    pure2, count2 = mod.get_count_minus(ao2)
    assert pure2 == (5,)
    assert count2 == 2


def test_append_minus_marks_if_needed_and_appends(mod):
    assert mod.append_minus((1, 2), 0) == ("M", 1, 2)
    assert mod.append_minus((1, 2), 2) == ("M", 1, 2, -1, -1)

    assert mod.append_minus(("M", 1, 2), 1) == ("M", 1, 2, -1)


def test_append_ident_always_prefixes_modifier(mod):
    assert mod.append_ident((1, 2), 2) == ("M", 1, 2, -1, -1)


def test_get_count_minus_then_append_ident_roundtrip_for_matching_modifier(mod):
    original = ("M", 7, 8, -1, -1, -1)
    pure, cnt = mod.get_count_minus(original)
    rebuilt = mod.append_ident(pure, cnt)
    assert rebuilt == original


def test_get_count_minus_iter(mod):
    acting_on_list = [("M", 1, -1), ("M", 2, 3, -1, -1), (9,)]
    pure_list, counts = mod.get_count_minus_iter(acting_on_list)
    assert pure_list == [(1,), (2, 3), (9,)]
    assert counts == [1, 2, 0]


def test_mark_canonicalized_acting_on_duplicates_get_unique_minus_tails(mod):
    canonical = [(0, 1), (0, 1), (0, 1)]
    out = mod.mark_canonicalized_acting_on(canonical)

    assert out[0] == ("M", 0, 1)
    assert out[1] == ("M", 0, 1, -1)
    assert out[2] == ("M", 0, 1, -1, -1)
    assert len(set(out)) == 3

    assert [mod.get_pure_aon(a) for a in out] == canonical


def test_mark_canonicalized_acting_on_with_markings_changes_identifier(mod):
    canonical = [(0, 1), (0, 1)]
    markings = ["A", "B"]
    out = mod.mark_canonicalized_acting_on(canonical, markings=markings)

    assert out[0][0] == "MA"
    assert out[1][0] == "MB"
    assert out[0] == ("MA", 0, 1)
    assert out[1] == ("MB", 0, 1)


def test_insert_highest_modification_returns_unique_and_preserves_pure(mod):
    base_pure = (1, 2)
    acting_on_list = [
        ("M", 1, 2),
        ("M", 1, 2, -1),
        ("M", 1, 2, -1, -1),
    ]

    out = mod.insert_highest_modification(acting_on_list, base_pure)
    assert out not in acting_on_list
    assert out == ("M", 1, 2, -1, -1, -1)

    assert mod.get_pure_aon(out) == base_pure


def test_insert_highest_modification_ensures_int_sites(mod):
    base_pure = (np.int64(3), np.int32(4))
    acting_on_list = [("M", 3, 4)]
    out = mod.insert_highest_modification(acting_on_list, base_pure)

    assert out[0] == "M"
    assert isinstance(out[1], int)
    assert isinstance(out[2], int)
    assert mod.get_pure_aon(out) == (3, 4)


def test__ensure_int_casts_all_sites(mod):
    ao = ("M", np.int64(1), np.int32(2), -1)
    pure_ints = mod._ensure_int(ao)
    assert pure_ints == (1, 2, -1)
    assert all(isinstance(x, int) for x in pure_ints)


def test_get_identifiers_returns_ordered_and_unique(mod):
    acting_on_list = [("A", 1), ("B", 2), ("A", 3)]
    idents, uniq = mod.get_identifiers(acting_on_list)
    assert idents == ["A", "B", "A"]
    assert set(uniq) == {"A", "B"}


def test_get_identifiers_asserts_if_missing_identifier(mod):
    with pytest.raises(AssertionError):
        mod.get_identifiers([(1, 2), ("A", 3)])


def test_prep_for_gccf_groups_indices_correctly_and_stably():
    identifiers_list = [10, 20, 10, 30, 20, 10]
    uids = [30, 10, 20]

    uids_arr, ops_grouped, ops_start, ops_count = ActingOnModifier.prep_for_gccf(
        identifiers_list, uids
    )

    assert np.array_equal(uids_arr, np.array([10, 20, 30]))
    assert uids_arr.dtype == np.array(uids).dtype

    assert np.array_equal(ops_count, np.array([3, 2, 1], dtype=np.intp))

    assert np.array_equal(ops_start, np.array([0, 3, 5], dtype=np.intp))

    assert ops_grouped.dtype == np.intp
    assert sorted(ops_grouped.tolist()) == list(range(len(identifiers_list)))

    g10 = ops_grouped[ops_start[0] : ops_start[0] + ops_count[0]].tolist()
    assert g10 == [0, 2, 5]

    g20 = ops_grouped[ops_start[1] : ops_start[1] + ops_count[1]].tolist()
    assert g20 == [1, 4]

    g30 = ops_grouped[ops_start[2] : ops_start[2] + ops_count[2]].tolist()
    assert g30 == [3]


def test_prep_for_gccf_randomized_invariants():
    rng = np.random.default_rng(0)
    n_ops = 50
    uids = np.array([5, 7, 9, 11], dtype=np.int64)
    identifiers_list = rng.choice(uids, size=n_ops, replace=True).tolist()

    uids_in = [11, 5, 9, 7]

    uids_arr, ops_grouped, ops_start, ops_count = ActingOnModifier.prep_for_gccf(
        identifiers_list, uids_in
    )

    assert np.array_equal(uids_arr, np.sort(uids))
    assert ops_grouped.shape == (n_ops,)
    assert ops_start.shape == (len(uids),)
    assert ops_count.shape == (len(uids),)

    assert sorted(ops_grouped.tolist()) == list(range(n_ops))

    for uid_idx, uid in enumerate(uids_arr.tolist()):
        group = ops_grouped[
            ops_start[uid_idx] : ops_start[uid_idx] + ops_count[uid_idx]
        ].tolist()
        expected = [i for i, u in enumerate(identifiers_list) if u == uid]
        assert group == expected


def test_prep_for_gccf_errors_if_identifier_not_in_uids():
    identifiers_list = [1, 2, 99]
    uids = [1, 2]
    with pytest.raises(IndexError):
        ActingOnModifier.prep_for_gccf(identifiers_list, uids)
