#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
import pytest


def _norm_edge(e):
    if len(e) == 2:
        u, v = e
        return (u, v, 0)
    if len(e) == 3:
        return tuple(e)
    raise ValueError(f"Unexpected edge format: {e}")


def _canon_keyless(u, v):
    return tuple(sorted((u, v)))


def _canon_keyed(u, v, k):
    a, b = sorted((u, v))
    return (a, b, k)


def _edge_chain_is_closed(loop):

    if not loop:
        return False
    for i in range(len(loop)):
        _, v, _ = loop[i]
        u2, _, _ = loop[(i + 1) % len(loop)]
        if v != u2:
            return False
    return True


def _rotate_to_start_at_min_tail(loop):
    tails = [e[0] for e in loop]
    m = min(tails)
    i = tails.index(m)
    return loop[i:] + loop[:i]


def test_planar_key_insertion_and_original_storage(
    GraphHandler, theta_planar_keyless_edges
):
    h = GraphHandler(theta_planar_keyless_edges, plot=False)

    assert h.is_planar is True
    assert h._original_edges == theta_planar_keyless_edges

    assert len(h.edges) == 5
    for e in h.edges:
        u, v, k = _norm_edge(e)
        assert k == 0

    assert set(h.vertices) == {0, 1, 2, 3}

    assert len(h._edges_keyed) == len(h.edges)
    assert len(h._edges_keyless) == 5
    assert all(len(x) == 2 for x in h._edges_keyless)


def test_planar_keyed_edges_preserved(GraphHandler, theta_planar_keyed_edges):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)
    assert h.is_planar is True

    assert len(h.edges) == 5
    keys = [(_norm_edge(e)[2]) for e in h.edges]
    assert all(k == 0 for k in keys)


def test_nonplanar_relabeling_and_convert_vertex(
    GraphHandler, theta_nonplanar_keyed_edges
):
    h = GraphHandler(theta_nonplanar_keyed_edges, plot=False)

    assert h.is_planar is False

    assert all(isinstance(v, tuple) for v in h._original_vertices)
    assert all(isinstance(v, int) for v in h.vertices)

    assert hasattr(h, "_nonplanar_int2orig_map")
    assert hasattr(h, "_nonplanar_orig2int_map")
    assert len(h._nonplanar_int2orig_map) == len(h.vertices)

    for orig in h._original_vertices:
        vi = h.convert_vertex_to_int(orig)
        assert isinstance(vi, int)
        assert h._nonplanar_int2orig_map[vi] == orig


def test_convert_vertex_to_int_raises_for_planar(
    GraphHandler, theta_planar_keyed_edges
):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)
    assert h.is_planar is True
    with pytest.raises(NotImplementedError):
        _ = h.convert_vertex_to_int((0, 0, 0))


def test_mapping_and_inverse_mapping_are_bijective_and_handle_reversal(
    GraphHandler, theta_planar_keyed_edges
):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)

    assert isinstance(h.mapping, dict)
    assert len(h.mapping) == len(h.edges)

    vals = list(h.mapping.values())
    assert len(set(vals)) == len(vals)
    assert all(isinstance(v, int) for v in vals)

    for e in h.edges:
        u, v, k = _norm_edge(e)
        dv = h.map_edge_to_dual_vertex((u, v, k))
        assert dv is not None
        inv = h.inverse_mapping(dv)
        assert inv is not None
        iu, iv, ik = _norm_edge(inv)
        assert ik == k
        assert _canon_keyless(iu, iv) == _canon_keyless(u, v)

        dv2 = h.map_edge_to_dual_vertex((v, u, k))
        assert dv2 == dv


def test_dual_nx_line_graph_nodes_correspond_to_primal_edges(
    GraphHandler, theta_planar_keyed_edges
):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)

    primal = {_canon_keyed(*_norm_edge(e)) for e in h.nx_graph.edges(keys=True)}
    dual_nodes = {_canon_keyed(*_norm_edge(n)) for n in h.dual_nx_graph.nodes()}

    assert primal == dual_nodes


def test_node_connectivity_is_consistent_orientationwise(
    GraphHandler, theta_planar_keyed_edges
):

    h = GraphHandler(theta_planar_keyed_edges, plot=False)

    conn = getattr(h, "list_of_node_connectivity", None)
    assert isinstance(conn, dict)
    assert set(conn.keys()) == set(h.vertices)

    incident_global = {_canon_keyed(*_norm_edge(e)) for e in h.edges}
    appearances = {e: 0 for e in incident_global}

    for v, attrs in conn.items():
        inc = [tuple(x) for x in attrs.get("incoming", [])]
        out = [tuple(x) for x in attrs.get("outgoing", [])]

        seen = [_canon_keyed(*_norm_edge(e)) for e in (inc + out)]
        assert len(seen) == len(set(seen))

        for e in inc + out:
            u, w, k = _norm_edge(e)
            assert v in (u, w)
            ce = _canon_keyed(u, w, k)
            assert ce in incident_global
            appearances[ce] += 1

    assert all(c == 2 for c in appearances.values())

    for e in h.edges:
        u, v, k = _norm_edge(e)
        cu = conn[u]
        cv = conn[v]
        cu_in = {_canon_keyed(*_norm_edge(x)) for x in cu["incoming"]}
        cu_out = {_canon_keyed(*_norm_edge(x)) for x in cu["outgoing"]}
        cv_in = {_canon_keyed(*_norm_edge(x)) for x in cv["incoming"]}
        cv_out = {_canon_keyed(*_norm_edge(x)) for x in cv["outgoing"]}
        ce = _canon_keyed(u, v, k)

        ok = (ce in cu_out and ce in cv_in) or (ce in cu_in and ce in cv_out)
        assert ok, f"Edge {e} orientation not split incoming/outgoing across endpoints."


def test_register_graph_data_planar_pair_count(GraphHandler, theta_planar_keyed_edges):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)

    h._register_graph_data(h.nx_graph, "graph")
    data = h.graph_edges_data["graph"]["connectivities"]

    for v in h.nx_graph.nodes():
        connected_edges = list(h.nx_graph.edges(v, keys=True))
        d = len(connected_edges)
        pairs = data[str(v)]["edges"]

        expected = 2 * (d * (d - 1) // 2)
        assert len(pairs) == expected

        s = set(map(tuple, pairs))
        for e1, e2 in itertools.combinations(connected_edges, 2):
            assert (e1, e2) in s
            assert (e2, e1) in s


def test_register_graph_data_nonplanar_permutation_count(
    GraphHandler, theta_nonplanar_keyed_edges
):
    h = GraphHandler(theta_nonplanar_keyed_edges, plot=False)
    h._register_graph_data(h.nx_graph, "graph")
    data = h.graph_edges_data["graph"]["connectivities"]

    for v in h.nx_graph.nodes():
        connected_edges = list(h.nx_graph.edges(v, keys=True))
        d = len(connected_edges)
        triples = data[str(v)]["edges"]

        expected = d * (d - 1) * (d - 2) if d >= 3 else 0
        assert len(triples) == expected


def test_get_edge_direction_factor_semantics(GraphHandler, theta_planar_keyed_edges):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)

    e1 = tuple(h.edges[0])
    e2 = tuple(h.edges[1])

    f = h._get_edge_direction_factor((e1, e2), vertex=0)
    assert f == 1

    u1, v1, k1 = _norm_edge(e1)
    u2, v2, k2 = _norm_edge(e2)
    f2 = h._get_edge_direction_factor(((v1, u1, k1), (v2, u2, k2)), vertex=0)
    assert f2 == 1

    alien = (999, 1000, 0)
    f3 = h._get_edge_direction_factor((alien, e1), vertex=0)
    assert f3 == -1


def test_smallest_loops_theta_graph_two_triangles(
    GraphHandler, theta_planar_keyed_edges
):
    h = GraphHandler(theta_planar_keyed_edges, plot=False)
    loops = h.get_smallest_loops()

    assert isinstance(loops, list)
    assert len(loops) == 2
    assert all(len(loop) == 3 for loop in loops)

    primal = {_canon_keyed(*_norm_edge(e)) for e in h.edges}

    for loop in loops:
        loop = [tuple(e) for e in loop]
        assert _edge_chain_is_closed(loop), f"Loop not closed: {loop}"

        rotated = _rotate_to_start_at_min_tail(loop)
        assert rotated == loop, "Loop is not canonical-rotated to smallest tail."

        for u, v, k in loop:
            ce = _canon_keyed(u, v, k)
            assert ce in primal, f"Loop uses edge not in graph: {(u,v,k)}"


def test_smallest_loops_parallel_edges_yield_two_edge_loop(
    GraphHandler, edges_parallel_two_edge_loop
):
    h = GraphHandler(edges_parallel_two_edge_loop, plot=False)
    loops = h.get_smallest_loops()

    assert len(loops) >= 1
    assert any(len(loop) == 2 for loop in loops)

    found = [loop for loop in loops if len(loop) == 2][0]
    assert _edge_chain_is_closed(found)
    keys = sorted([_norm_edge(e)[2] for e in found])
    assert keys == [0, 1]


def test_smallest_loops_filters_to_min_length_regression(
    GraphHandler, edges_triangle_and_square_disconnected
):

    h = GraphHandler(edges_triangle_and_square_disconnected, plot=False)
    loops = h.get_smallest_loops()

    assert len(loops) >= 1
    lengths = [len(l) for l in loops]
    assert all(L == min(lengths) for L in lengths)
    assert min(lengths) == 3
