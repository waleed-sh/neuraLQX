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
import pytest


@pytest.fixture
def gsym():
    return importlib.import_module("neuralqx.utils.symmetries.graph_symmetries")


@pytest.fixture
def su():
    return importlib.import_module("neuralqx.utils.symmetries.utils")


@pytest.fixture
def types_utils():
    return importlib.import_module("neuralqx.utils.symmetries.types.utils")


def _rev(e):
    a, b, k = e
    return (b, a, k)


def test_product_of_bucket_permutations_counts_and_values(su):
    buckets = [[1, 2], [3]]
    choices = list(su._product_of_bucket_permutations(buckets))
    assert len(choices) == 2
    assert choices == [
        [[1, 2], [3]],
        [[2, 1], [3]],
    ]


def test_product_of_bucket_permutations_empty_buckets_is_single_empty_choice(su):
    assert list(su._product_of_bucket_permutations([])) == [[]]


def test_vmap_name_identity_and_cycles(su):
    assert su._vmap_name({0: 0, 1: 1, 2: 2}) == "auto_id"
    assert su._vmap_name({0: 1, 1: 0}) == "auto_(0,1)"
    assert su._vmap_name({0: 1, 1: 2, 2: 0}) == "auto_(0,1,2)"
    assert su._vmap_name({0: 1, 1: 0, 2: 3, 3: 2}) == "auto_(0,1)_(2,3)"


def test_vmap_name_ignores_fixed_points_and_is_deterministic(su):
    assert su._vmap_name({0: 0, 1: 2, 2: 1}) == "auto_(1,2)"


def test_graph_symmetries_requires_nonempty_edges(gsym):
    GS = gsym.GraphSymmetries
    with pytest.raises(ValueError, match="No edges provided"):
        GS([])


def test_graph_symmetries_domain_dedup_preserves_orientation_order(gsym):

    GS = gsym.GraphSymmetries

    raw = [
        (0, 1, 0),
        (0, 1, 0),
        (1, 0, 0),
    ]
    G = GS(raw)

    assert "(0, 1, 0)" in repr(G)
    assert "(1, 0, 0)" in repr(G)

    assert G.edges == [(0, 1, 0)]
    assert G.vertices == [0, 1]


def test_graph_symmetries_edges_and_vertices_are_sorted_and_are_copies(gsym):
    GS = gsym.GraphSymmetries

    raw = [(2, 1, 0), (0, 2, 1), (1, 0, 0)]
    G = GS(raw)

    assert G.edges == sorted(G.edges)
    assert G.vertices == sorted(G.vertices)

    edges_copy = G.edges
    edges_copy.append((99, 99, 99))
    assert (99, 99, 99) not in G.edges


def test_permutations_contains_identity_and_is_cached(gsym):
    GS = gsym.GraphSymmetries

    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw, max_permutation_edges=10)

    p1 = G.permutations
    p2 = G.permutations

    assert p1 is p2

    assert p1[0].name == "id"

    e = (0, 1, 0)
    assert p1[0][e] == e
    assert p1[0][_rev(e)] == _rev(e)


def test_permutations_respects_max_permutation_edges_cap(gsym):
    GS = gsym.GraphSymmetries

    raw = [(0, 1, 0), (0, 2, 0), (1, 2, 0)]
    G = GS(raw, max_permutation_edges=2)

    perms = G.permutations
    assert len(perms) == 1
    assert perms[0].name == "id"


def test_permutations_nontrivial_exists_when_small(gsym):

    GS = gsym.GraphSymmetries

    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw, max_permutation_edges=10)

    perms = G.permutations
    assert len(perms) == 2
    assert perms[0].name == "id"
    assert perms[1].name.startswith("perm_")

    moved = any(k != v for k, v in perms[1].mapping_canon.items())
    assert moved


def test_cycles_count_and_shifts(gsym):
    GS = gsym.GraphSymmetries

    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw)

    cycles = G.cycles
    assert len(cycles) == 2
    assert [c.shift for c in cycles] == [0, 1]

    assert all(k == v for k, v in cycles[0].mapping_canon.items())

    assert any(k != v for k, v in cycles[1].mapping_canon.items())


def test_automorphisms_parallel_edge_keys_allow_key_permutations_multiplies(gsym):

    GS = gsym.GraphSymmetries

    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw, allow_key_permutations=True, max_total_key_permutations=10_000)

    autos = G.automorphisms
    assert len(autos) == 4

    id_like = [
        a
        for a in autos
        if a.name == "auto_id"
        and a.vertex_map == {0: 0, 1: 1}
        and all(k == v for k, v in a.edge_map_canon.items())
    ]
    assert len(id_like) == 1

    assert any(a.vertex_map == {0: 1, 1: 0} for a in autos)

    assert any(
        a.vertex_map == {0: 0, 1: 1}
        and any(k != v for k, v in a.edge_map_canon.items())
        for a in autos
    )


def test_automorphisms_parallel_edges_without_key_permutations(gsym):
    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw, allow_key_permutations=False)

    autos = G.automorphisms

    assert len(autos) == 2
    assert any(a.vertex_map == {0: 0, 1: 1} for a in autos)
    assert any(a.vertex_map == {0: 1, 1: 0} for a in autos)

    assert all(all(k == v for k, v in a.edge_map_canon.items()) for a in autos)


def test_automorphisms_key_permutation_cap_falls_back_to_identity_per_pair(gsym):

    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw, allow_key_permutations=True, max_total_key_permutations=1)

    autos = G.automorphisms

    assert len(autos) == 2


def test_automorphisms_chain_graph_has_expected_nontrivial_vertex_swap(gsym, su):

    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (1, 2, 0)]
    G = GS(raw, allow_key_permutations=False)

    autos = G.automorphisms
    assert len(autos) == 2

    assert any(a.name == "auto_id" for a in autos)

    assert any(a.name == "auto_(0,2)" for a in autos)


def test_private_is_automorphism_basic_checks(gsym):
    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (1, 2, 0)]
    G = GS(raw, allow_key_permutations=False)

    assert G._is_automorphism({0: 2, 1: 1, 2: 0}) is True

    assert G._is_automorphism({0: 0, 1: 1}) is False
    assert G._is_automorphism({0: 0, 1: 1, 2: 2, 3: 3}) is False


def test_graph_symmetries_property_concatenates_and_is_cached(gsym):
    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw)

    all1 = G.graph_symmetries
    all2 = G.graph_symmetries
    assert all1 is all2

    assert len(all1) == len(G.permutations) + len(G.cycles) + len(G.automorphisms)


def test_make_permutation_from_order_accepts_oriented_edges_and_validates(
    gsym, types_utils
):
    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw)

    new_order = [_rev((0, 1, 1)), _rev((0, 1, 0))]
    p = G.make_permutation_from_order(new_order, name="customX")

    assert p.name == "customX"

    canon_edges = G.edges
    canon_new = [types_utils._canon_edge(e) for e in new_order]
    expected = {canon_edges[i]: canon_new[i] for i in range(len(canon_edges))}
    assert p.mapping_canon == expected


def test_make_permutation_from_order_rejects_invalid_order(gsym):
    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw)

    with pytest.raises(ValueError, match="new_order must be a permutation"):
        G.make_permutation_from_order([(0, 1, 0)], name="bad")

    with pytest.raises(ValueError, match="new_order must be a permutation"):
        G.make_permutation_from_order([(0, 1, 0), (9, 9, 9)], name="bad2")


def test_repr_mentions_counts_consistent_with_properties(gsym):
    GS = gsym.GraphSymmetries
    raw = [(0, 1, 0), (0, 1, 1)]
    G = GS(raw, max_permutation_edges=10)

    r = repr(G)
    assert "#Permutations" in r and "#Cycles" in r and "#Automorphisms" in r
    assert f"#Permutations = {len(G.permutations)}" in r
    assert f"#Cycles = {len(G.cycles)}" in r
    assert f"#Automorphisms = {len(G.automorphisms)}" in r
