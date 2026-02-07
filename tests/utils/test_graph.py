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
def gmod():
    pytest.importorskip("jax")
    return importlib.import_module("neuralqx.utils.misc.graph")


def _t(edge):
    return tuple(edge)


def _loop_as_tuples(loop):
    return [_t(e) for e in loop]


class DummyGraph:
    def __init__(self, edges):
        self.edges = set(tuple(e) for e in edges)


def test_find_corner_vertex_keyed_exactly_one_common(gmod):
    ok, v = gmod.find_corner_vertex_keyed(((0, 1, 9), (1, 2, 7)))
    assert ok is True
    assert v == 1


def test_find_corner_vertex_keyed_no_common_returns_false(gmod):
    ok, v = gmod.find_corner_vertex_keyed(((0, 1, 0), (2, 3, 0)))
    assert ok is True
    assert v is False


def test_find_corner_vertex_keyed_two_common_returns_false(gmod):
    ok, v = gmod.find_corner_vertex_keyed(((0, 1, 0), (1, 0, 9)))
    assert ok is True
    assert v is False


def test_get_vertices_from_minimal_loop_single_and_multiple(gmod):
    L1 = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    L2 = [(10, 11, 7), (11, 12, 7), (12, 10, 7)]

    out = gmod.get_vertices_from_minimal_loop([L1, L2])
    assert out == [[0, 1, 2], [10, 11, 12]]


def test_get_common_base_vertex_intersection(gmod):
    L1 = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    L2 = [(2, 3, 0), (3, 0, 0), (0, 2, 0)]
    common = gmod.get_common_base_vertex((L1, L2))
    assert set(common) == {0, 2}


def test_get_edges_at_base_vertex_finds_pair(gmod):
    loop = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    pair = gmod.get_edges_at_base_vertex(loop, base_vertex=1)
    assert pair is not None
    pair_t = tuple(_t(e) for e in pair)
    assert set(pair_t) == {(0, 1, 0), (1, 2, 0)}


def test_get_edges_at_base_vertex_returns_none_if_not_found(gmod):
    loop = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    assert gmod.get_edges_at_base_vertex(loop, base_vertex=99) is None


def test_get_reversed_keyed_edge_preserves_key_position(gmod):
    e = (0, 1, 123)
    r = gmod.get_reversed_keyed_edge(e)
    assert r == (1, 0, 123)

    r_list = gmod.get_reversed_keyed_edge(e, as_list=True)
    assert r_list == [1, 0, 123]


def test_get_triplets_key_returns_all_permutations_of_connected_edges(gmod):
    edges = [
        [0, 1, 0],
        [1, 2, 0],
        [1, 3, 0],
        [4, 5, 0],
    ]
    triplets = gmod.get_triplets_key(edges, vertex=1)
    assert len(triplets) == 6
    for t in triplets:
        assert len(t) == 3
        for e in t:
            assert 1 in e[:-1]


def test_reduce_permutations_removes_swapped_first_two_when_third_same(gmod):
    a, b, c, d = "a", "b", "c", "d"
    perms = [
        [a, b, c],
        [b, a, c],
        [a, b, c],
        [a, b, d],
        [b, a, d],
    ]
    reduced = gmod.reduce_permutations(perms)
    assert reduced == [[a, b, c], [a, b, d]]


def test_reorder_edge_triplet_key_orients_edges_and_tracks_sign(gmod):
    base = 1

    e0 = (2, 1, 0)

    e1 = (1, 3, 0)

    e2 = (4, 1, 0)
    triplet = [e0, e1, e2]

    reordered, sgn = gmod.reorder_edge_triplet_key(triplet, base_vertex=base)
    reordered = [_t(e) for e in reordered]

    assert reordered[0][0] == base
    assert reordered[1][1] == base
    assert reordered[2][0] == base
    assert sgn == -1


def test_reorder_edge_triplet_key_no_reversal_sign_is_plus_one(gmod):
    base = 1
    e0 = (1, 2, 0)
    e1 = (3, 1, 0)
    e2 = (1, 4, 0)
    reordered, sgn = gmod.reorder_edge_triplet_key([e0, e1, e2], base_vertex=base)
    assert [_t(e) for e in reordered] == [e0, e1, e2]
    assert sgn == 1


def test_find_minimal_loops_with_edges_key_includes_loops_with_both_edges_orientation_free(
    gmod,
):

    L1 = [(0, 1, 0), (2, 1, 0), (2, 0, 0)]
    L2 = [(10, 11, 0), (11, 12, 0), (12, 10, 0)]
    loops = [L1, L2]

    edge_pair = [[0, 1, 0], [1, 2, 0]]
    out = gmod.find_minimal_loops_with_edges_key(loops, edge_pair)
    assert out == [L1]


def test_find_minimal_loops_with_edges_key_rejects_same_edge_or_reverse(gmod):
    loops = [[(0, 1, 0), (1, 2, 0), (2, 0, 0)]]

    with pytest.raises(ValueError, match="contains the same edge"):
        gmod.find_minimal_loops_with_edges_key(loops, [[0, 1, 0], [0, 1, 0]])

    with pytest.raises(ValueError, match="contains the same edge"):
        gmod.find_minimal_loops_with_edges_key(loops, [[0, 1, 0], [1, 0, 0]])


def test_reorder_minimal_loop_key_reorders_and_orients_path(gmod):
    minimal_loop = [(1, 0, 0), (1, 2, 0), (2, 0, 0)]

    edge_pair = [
        [0, 1, 0],
        [2, 0, 0],
    ]

    path = gmod.reorder_minimal_loop_key(minimal_loop, edge_pair)
    path_t = _loop_as_tuples(path)

    assert path_t == [(0, 1, 0), (1, 2, 0), (2, 0, 0)]


def test_reorder_minimal_loop_key_raises_on_disconnected_edges(gmod):
    minimal_loop = [(0, 1, 0), (2, 3, 0), (4, 5, 0)]
    edge_pair = [[0, 1, 0], [2, 3, 0]]

    with pytest.raises(ValueError, match="No connected edge found"):
        gmod.reorder_minimal_loop_key(minimal_loop, edge_pair)


def test_detect_edge_orientation_key_creation_vs_annihilation(gmod):
    G = DummyGraph(edges=[(0, 1, 7), (1, 2, 7)])
    graph_shift = 10

    assert gmod.detect_edge_orientation_key(G, [10, 11, 7], graph_shift) == "creation"
    assert (
        gmod.detect_edge_orientation_key(G, [10, 99, 7], graph_shift) == "annihilation"
    )


def test_create_dressed_minimal_loop_key_dresses_each_edge(gmod):
    G = DummyGraph(edges=[(0, 1, 0), (1, 2, 0)])
    loop = [
        [10, 11, 0],
        [11, 12, 0],
        [12, 10, 0],
    ]
    dressed = gmod.create_dressed_minimal_loop_key(G, loop, graph_shift=10)

    assert len(dressed) == 3

    assert dressed[0][0] == (10, 11, 0)
    assert dressed[0][1]["key"] == 0
    assert dressed[0][1]["type"] == "creation"
    assert dressed[1][1]["type"] == "creation"
    assert dressed[2][1]["type"] == "annihilation"


def test_count_contributing_triplets_key_counts_only_eps_plus_one(gmod):
    e1, e2, e3 = (0, 1, 0), (1, 2, 0), (2, 0, 0)
    t_good = [e1, e2, e3]
    t_bad = [e2, e1, e3]

    levi = {
        str(tuple(t_good)): 1,
        str(tuple(t_bad)): -1,
    }
    assert gmod.count_contributing_triplets_key(levi, [t_good, t_bad]) == 1


def test_get_ml_graph_shift_key_splits_h_and_ha_and_adjoint_swaps(gmod):
    dressed = [
        ((10, 11, 0), {"type": "creation", "key": 0}),
        ((11, 12, 0), {"type": "annihilation", "key": 0}),
        ((12, 10, 0), {"type": "creation", "key": 0}),
    ]

    mapping = lambda e: (e[0], e[1])

    out = gmod.get_ml_graph_shift_key(dressed, mapping=mapping, adjoint=False)
    assert out["h"] == [(10, 11), (12, 10)]
    assert out["ha"] == [(11, 12)]

    out_adj = gmod.get_ml_graph_shift_key(dressed, mapping=mapping, adjoint=True)
    assert out_adj["h"] == [(11, 12)]
    assert out_adj["ha"] == [(10, 11), (12, 10)]


def test_get_true_edge_triplet_orients_edges_to_match_graph(gmod):
    G = DummyGraph(edges=[(0, 1, 0), (1, 2, 0), (2, 0, 0)])

    edge_list = [(1, 0, 0), (1, 2, 0), (0, 2, 0)]
    out = gmod.get_true_edge_triplet(edge_list, G)
    assert out == ((0, 1, 0), (1, 2, 0), (2, 0, 0))


def test_detect_edge_orientation_key_uses_minus_key_shift_semantics(gmod):
    G = DummyGraph(edges=[(5, 6, 3)])
    s = 100
    assert gmod.detect_edge_orientation_key(G, [105, 106, 3], s) == "creation"


def test_find_corner_vertex_keyed_ignores_keys(gmod):
    ok, v = gmod.find_corner_vertex_keyed(((0, 1, 0), (1, 2, 99)))
    assert ok is True
    assert v == 1


def test_get_vertices_from_minimal_loop_handles_duplicates(gmod):
    loop = [(0, 1, 0), (1, 2, 0), (2, 0, 0), (0, 1, 0)]
    out = gmod.get_vertices_from_minimal_loop([loop])
    assert out == [[0, 1, 2]]


def test_get_common_base_vertex_empty_intersection(gmod):
    L1 = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    L2 = [(10, 11, 0), (11, 12, 0), (12, 10, 0)]
    assert gmod.get_common_base_vertex((L1, L2)) == []


def test_get_edges_at_base_vertex_ambiguous_returns_valid_pair(gmod):
    loop = [(0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 2, 0)]
    pair = gmod.get_edges_at_base_vertex(loop, base_vertex=0)
    assert pair is not None
    assert gmod.find_corner_vertex_keyed(pair)[1] == 0


def test_get_triplets_key_requires_three_connected_edges(gmod):
    edges = [[0, 1, 0], [1, 2, 0]]
    triplets = gmod.get_triplets_key(edges, vertex=1)
    assert triplets == []


def test_reorder_edge_triplet_key_preserves_keys_on_reversal(gmod):
    base = 1
    e0 = (2, 1, 7)
    e1 = (1, 3, 8)
    e2 = (4, 1, 9)
    reordered, _ = gmod.reorder_edge_triplet_key([e0, e1, e2], base_vertex=base)
    assert [r[-1] for r in reordered] == [7, 8, 9]


def test_find_minimal_loops_with_edges_key_respects_keys(gmod):
    L = [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
    loops = [L]
    out = gmod.find_minimal_loops_with_edges_key(loops, [[0, 1, 0], [1, 2, 0]])
    assert out == []


def test_reorder_minimal_loop_key_flips_end_edge_if_needed(gmod):
    minimal_loop = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    edge_pair = [[0, 1, 0], [0, 2, 0]]
    path = gmod.reorder_minimal_loop_key(minimal_loop, edge_pair)
    assert tuple(path[-1]) == (2, 0, 0)


def test_reorder_minimal_loop_key_preserves_edges_and_length(gmod):
    minimal_loop = [(0, 1, 0), (2, 1, 0), (2, 0, 0)]
    edge_pair = [[0, 1, 0], [2, 0, 0]]

    out = gmod.reorder_minimal_loop_key(minimal_loop, edge_pair)

    out_t = [tuple(e) for e in out]
    in_t = [tuple(e) for e in minimal_loop]

    assert len(out_t) == len(in_t)

    for i in range(len(out_t) - 1):
        assert out_t[i][1] == out_t[i + 1][0]

    assert out_t[0] == (0, 1, 0)
    assert out_t[-1] == (2, 0, 0)

    def undirected(e):
        a, b, k = e
        return (min(a, b), max(a, b), k)

    assert sorted(undirected(e) for e in out_t) == sorted(undirected(e) for e in in_t)


def test_reorder_minimal_loop_key_edge_pair_can_make_remainder_disconnected(gmod):
    minimal_loop = [(0, 1, 0), (2, 1, 0), (2, 0, 0)]

    edge_pair = [[0, 1, 0], [1, 2, 0]]

    with pytest.raises(ValueError, match="No connected edge found"):
        gmod.reorder_minimal_loop_key(minimal_loop, edge_pair)


def test_detect_edge_orientation_key_negative_shift(gmod):
    G = DummyGraph(edges=[(0, 1, 0)])
    assert (
        gmod.detect_edge_orientation_key(G, [-10, -9, 0], graph_shift=-10) == "creation"
    )


def test_create_dressed_minimal_loop_key_metadata_integrity(gmod):
    G = DummyGraph(edges=[(0, 1, 5)])
    loop = [[10, 11, 5]]
    dressed = gmod.create_dressed_minimal_loop_key(G, loop, graph_shift=10)
    assert dressed[0][1].keys() == {"type", "key"}
    assert dressed[0][1]["key"] == 5


def test_count_contributing_triplets_key_ignores_missing_or_non_one(gmod):
    t1 = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    t2 = [(0, 1, 0), (2, 0, 0), (1, 2, 0)]
    levi = {str(tuple(t1)): 1, str(tuple(t2)): 0}
    assert gmod.count_contributing_triplets_key(levi, [t1, t2, ["not", "present"]]) == 1


def test_get_ml_graph_shift_key_mapping_receives_keyed_edge_tuple(gmod):
    dressed = [((10, 11, 7), {"type": "creation", "key": 7})]

    def mapping(e):
        assert isinstance(e, tuple) and len(e) == 3
        return e[:2]

    out = gmod.get_ml_graph_shift_key(dressed, mapping=mapping)
    assert out["h"] == [(10, 11)]
    assert out["ha"] == []


def test_get_true_edge_triplet_no_orientation_in_graph_returns_reversed_anyway(gmod):
    G = DummyGraph(edges=[(0, 1, 0)])
    edge_list = [(2, 3, 9)]
    out = gmod.get_true_edge_triplet(edge_list, G)
    assert out == ((3, 2, 9),)


def test_reduce_permutations_keeps_first_seen_canonical_representative(gmod):
    a, b, c = "a", "b", "c"
    perms = [[b, a, c], [a, b, c]]
    reduced = gmod.reduce_permutations(perms)
    assert reduced == [[b, a, c]]
