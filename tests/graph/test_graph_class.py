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


import pytest


def undirected_edge(e):
    u, v, k = e
    a, b = (u, v) if u <= v else (v, u)
    return (a, b, k)


def strip_dressed_edge(item):

    if isinstance(item, (tuple, list)) and len(item) == 3:
        u, v, k = item
        if isinstance(k, (int,)) and isinstance(u, (int,)) and isinstance(v, (int,)):
            return (u, v, k)

    if isinstance(item, (tuple, list)) and len(item) >= 1:
        first = item[0]
        if isinstance(first, (tuple, list)) and len(first) == 3:
            u, v, k = first
            if (
                isinstance(k, (int,))
                and isinstance(u, (int,))
                and isinstance(v, (int,))
            ):
                return (u, v, k)

    if isinstance(item, dict) and "edge" in item:
        e = item["edge"]
        if isinstance(e, (tuple, list)) and len(e) == 3:
            u, v, k = e
            return (u, v, k)

    return None


def test_graph_minimal_loops_theta_are_two_triangles(Graph, theta_planar_keyless_edges):
    g = Graph(theta_planar_keyless_edges, plot=False)
    loops = g.minimal_loops()

    assert isinstance(loops, list)
    assert (
        len(loops) == 2
    ), f"Expected 2 minimal loops for theta graph, got {len(loops)}"

    assert sorted(len(L) for L in loops) == [3, 3]

    loops_sets = {frozenset(undirected_edge(tuple(e)) for e in L) for L in loops}

    tri1 = frozenset({(0, 1, 0), (0, 2, 0), (1, 2, 0)})
    tri2 = frozenset({(0, 2, 0), (0, 3, 0), (2, 3, 0)})

    assert loops_sets == {tri1, tri2}, f"Unexpected minimal loops: {loops_sets}"


def test_graph_dressed_minimal_loops_cover_real_edges(
    Graph, theta_planar_keyless_edges
):
    g = Graph(theta_planar_keyless_edges, plot=False)
    loops = g.minimal_loops()
    dressed = g.dressed_minimal_loops()

    assert isinstance(dressed, list)
    assert len(dressed) == len(loops)

    primal_edges_undirected = {undirected_edge(tuple(e)) for e in g.edges}

    for loop in dressed:
        assert isinstance(loop, list)
        for item in loop:
            e = strip_dressed_edge(item)
            assert e is not None, f"Could not parse dressed edge item: {item}"
            assert undirected_edge(e) in primal_edges_undirected


def test_minimal_dual_loops_is_not_implemented(Graph, theta_planar_keyless_edges):
    g = Graph(theta_planar_keyless_edges, plot=False)

    _ = g.minimal_dual_loops(edge_rep=False)
    assert isinstance(_[0][0][0], tuple)

    _ = g.minimal_dual_loops(edge_rep=True)
    assert isinstance(_[0][0][0], int)


def test_dressed_minimal_dual_loops_is_present_or_skipped(
    Graph, theta_planar_keyless_edges
):
    g = Graph(theta_planar_keyless_edges, plot=False)

    if not hasattr(g.handler, "dual_minimal_cycles"):
        pytest.skip("dual_minimal_cycles not implemented on GraphHandler yet.")
    out = g.dressed_minimal_dual_loops()
    assert isinstance(out, list)


def test_graph_repr_is_stable(Graph, theta_planar_keyless_edges):
    g = Graph(theta_planar_keyless_edges, plot=False)
    r = repr(g)
    assert "n_edges=" in r
    assert "n_vertices=" in r
