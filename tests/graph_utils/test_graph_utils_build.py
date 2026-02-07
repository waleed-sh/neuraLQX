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


def _import_build():
    from neuralqx.graph.core.utils import build as _build

    return _build


def _nk_edges(g):
    if hasattr(g, "edges") and callable(getattr(g, "edges")):
        return [tuple(e) for e in g.edges()]
    if hasattr(g, "edges"):
        e = g.edges
        return [tuple(x) for x in (e() if callable(e) else e)]
    raise RuntimeError("Cannot extract edges from netket graph.")


def test_map_distinct_vertices_and_map_edges_are_consistent(
    theta_planar_keyed_edges, nx
):
    build = _import_build()

    G = nx.MultiGraph()
    for u, v, k in theta_planar_keyed_edges:
        G.add_edge(u, v, key=k)
    LG = nx.line_graph(G)
    lg_edges = list(LG.edges())

    distinct, mapping = build.map_distinct_vertices(lg_edges)
    assert len(distinct) == len(mapping)
    assert list(range(len(distinct))) == sorted(mapping.values())

    mapped = build.map_edges(lg_edges, mapping)
    assert len(mapped) == len(lg_edges)
    assert all(isinstance(x, list) and len(x) == 2 for x in mapped)


def test_create_dual_nk_graph_matches_networkx_line_graph(
    theta_planar_keyed_edges, nx, netket
):
    build = _import_build()

    class Dummy:
        pass

    handler = Dummy()
    handler.nx_graph = nx.MultiGraph()
    handler.edges = theta_planar_keyed_edges
    for u, v, k in theta_planar_keyed_edges:
        handler.nx_graph.add_edge(u, v, key=k)

    dual_nk, mapping = build.create_dual_nk_graph(handler)
    assert len(mapping) == len(theta_planar_keyed_edges)

    LG = nx.line_graph(handler.nx_graph)
    expected_edges = set()
    for e1, e2 in LG.edges():
        expected_edges.add(tuple(sorted((mapping[e1], mapping[e2]))))

    got_edges = set(tuple(sorted(e)) for e in _nk_edges(dual_nk))
    assert got_edges == expected_edges


def test_create_dual_nk_graph_nonplanar_inverts_mapping(
    theta_planar_keyed_edges, nx, netket
):
    build = _import_build()

    class Dummy:
        pass

    handler = Dummy()
    handler.nx_graph = nx.MultiGraph()
    handler.edges = theta_planar_keyed_edges
    for u, v, k in theta_planar_keyed_edges:
        handler.nx_graph.add_edge(u, v, key=k)

    dual_nk, mapping = build.create_dual_nk_graph(handler, is_nonplanar=True)
    assert set(mapping.keys()) == set(range(len(theta_planar_keyed_edges)))
    assert all(isinstance(v, tuple) and len(v) == 3 for v in mapping.values())


def test_create_dressed_nx_graph_sets_directional_attrs(theta_planar_keyed_edges, nx):
    build = _import_build()

    class Dummy:
        pass

    handler = Dummy()
    handler.edges = theta_planar_keyed_edges
    handler.nx_graph = nx.MultiGraph()

    build.create_dressed_nx_graph(handler)

    for u, v, k, data in handler.nx_graph.edges(keys=True, data=True):
        assert "s" in data and "e" in data
        s = data["s"]
        e = data["e"]

        assert {s, e} == {u, v}

        assert s in handler.nx_graph.nodes()
        assert e in handler.nx_graph.nodes()

        if u != v:
            assert s != e


def test_init_node_connectivity_is_incidence_complete_and_partitioned(
    theta_planar_keyed_edges, nx
):

    build = _import_build()

    class Dummy:
        pass

    handler = Dummy()
    handler.edges = theta_planar_keyed_edges
    handler.nx_graph = nx.MultiGraph()
    build.create_dressed_nx_graph(handler)

    build.init_node_connectivity(handler)
    conn = handler.list_of_node_connectivity

    primal_edges = list(handler.nx_graph.edges(keys=True))
    primal_undirected = set()
    for u, v, k in primal_edges:
        a, b = sorted((u, v))
        primal_undirected.add((a, b, k))

    appearances = {e: 0 for e in primal_undirected}

    for v in handler.nx_graph.nodes():
        assert v in conn, f"Missing connectivity entry for vertex {v}"
        inc = [tuple(e) for e in conn[v].get("incoming", [])]
        out = [tuple(e) for e in conn[v].get("outgoing", [])]

        assert len(inc) == len(set(inc)), f"Duplicate in incoming[{v}]"
        assert len(out) == len(set(out)), f"Duplicate in outgoing[{v}]"

        union = inc + out
        assert len(union) == len(
            set(union)
        ), f"Edge appears in both incoming and outgoing at v={v}"

        for a, b, k in union:
            assert v in (a, b), f"Non-incident edge {(a,b,k)} listed at vertex {v}"
            x, y = sorted((a, b))
            ce = (x, y, k)
            assert ce in primal_undirected, f"Edge {(a,b,k)} not in primal graph edges"
            appearances[ce] += 1

        incident_truth = set()
        for a, b, k in primal_edges:
            if v in (a, b):
                x, y = sorted((a, b))
                incident_truth.add((x, y, k))

        union_undirected = set((tuple(sorted((a, b))) + (k,)) for (a, b, k) in union)
        assert union_undirected == incident_truth, (
            f"Incoming∪Outgoing at vertex {v} does not match incident edges.\n"
            f"Missing: {incident_truth - union_undirected}\n"
            f"Extra: {union_undirected - incident_truth}"
        )

    assert all(
        c == 2 for c in appearances.values()
    ), f"Edge appearance counts: {appearances}"


def test_dress_minimal_cycles_edges_produces_types(theta_planar_keyed_edges, nx):
    build = _import_build()

    class Dummy:
        pass

    handler = Dummy()
    handler.edges = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    handler.nx_graph = nx.MultiGraph()
    build.create_dressed_nx_graph(handler)

    handler.minimal_cycles = []

    def smallest_loops():
        return [[(0, 1, 0), (1, 2, 0), (2, 0, 0)]]

    handler.get_smallest_loops = smallest_loops

    build.dress_minimal_cycles_edges(handler)
    assert len(handler.minimal_cycles) == 1
    dressed = handler.minimal_cycles[0]
    assert len(dressed) == 3

    for edge, meta in dressed:
        assert "type" in meta and meta["type"] in {
            "creation",
            "annihilation",
            "unknown",
        }
        assert meta["key"] == edge[-1]

    assert all(meta["type"] == "creation" for _, meta in dressed)


def test_index_to_label_mapping_is_correct():
    build = _import_build()

    assert build.index_to_label(0) == (0, 0, 0)
    assert build.index_to_label(1) == (0, 0, 1)
    assert build.index_to_label(2) == (0, 1, 0)
    assert build.index_to_label(3) == (0, 1, 1)
    assert build.index_to_label(4) == (1, 0, 0)
    assert build.index_to_label(7) == (1, 1, 1)


def test_relabel_edges_to_nonplanar_is_bijective_and_preserves_keys():
    build = _import_build()

    edges = [(10, 20, 0), (20, 30, 5), (10, 30, 2)]
    new_edges, mapping = build.relabel_edges_to_nonplanar(edges)

    assert set(mapping.keys()) == {10, 20, 30}
    assert len(set(mapping.values())) == 3

    assert sorted([k for _, _, k in new_edges]) == [0, 2, 5]

    for (u, v, k), (nu, nv, nk) in zip(edges, new_edges):
        assert nk == k
        assert nu == mapping[u]
        assert nv == mapping[v]


def test_relabel_nx_edges_produces_compact_int_labels(nx):
    build = _import_build()

    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")

    edges = build.relabel_nx_edges(G)

    flat = {x for e in edges for x in e}
    assert flat == {0, 1, 2}
    assert len(edges) == 2


def test_canonical_rotate_picks_min_start_and_requires_closure():
    build = _import_build()

    loop = [(2, 0, 0), (0, 1, 0), (1, 2, 0)]
    out = build.canonical_rotate(loop)
    assert out[0][0] == 0
    assert all(out[i][1] == out[(i + 1) % 3][0] for i in range(3))

    bad = [(0, 1, 0), (2, 3, 0)]
    assert build.canonical_rotate(bad) == bad
