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


def _import_plotting():
    from neuralqx.graph.core.utils import plotting as _plot

    return _plot


def test_reorder_edges_respects_networkx_edge_order(theta_planar_keyed_edges):
    plot = _import_plotting()

    nx_edges = list(theta_planar_keyed_edges)
    shuffled = list(reversed(theta_planar_keyed_edges))

    out = plot.reorder_edges(shuffled, nx_edges)
    assert out == nx_edges


def test_reorder_edges_raises_if_edge_missing(theta_planar_keyed_edges):
    plot = _import_plotting()

    nx_edges = list(theta_planar_keyed_edges)
    bad = nx_edges + [(999, 1000, 0)]

    with pytest.raises(KeyError):
        _ = plot.reorder_edges(bad, nx_edges)


def test__draw_graph_and_dual_sets_positions_and_dual_positions(nx):
    plot = _import_plotting()

    class Dummy:
        pass

    h = Dummy()
    h.is_planar = True

    h.nx_graph = nx.MultiGraph()
    h.edges = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    for u, v, k in h.edges:
        h.nx_graph.add_edge(u, v, key=k)

    h.dual_nx_graph = nx.line_graph(h.nx_graph)

    calls = []

    def _register_graph_data(graph, graph_type):
        calls.append(graph_type)
        if not hasattr(h, "graph_edges_data"):
            h.graph_edges_data = {}
        h.graph_edges_data.setdefault(graph_type, {"connectivities": {}})
        for v in graph.nodes():
            h.graph_edges_data[graph_type]["connectivities"][str(v)] = {"edges": []}

    h._register_graph_data = _register_graph_data

    fig = plot._draw_graph_and_dual(h, plot=False, overlay_dual=False, return_fig=True)
    assert fig is not None

    assert hasattr(h, "positions")
    assert isinstance(h.positions, dict)
    assert set(h.positions.keys()) == set(h.nx_graph.nodes())

    assert hasattr(h, "dual_positions")
    assert set(h.dual_positions.keys()) == set(h.nx_graph.edges(keys=True))

    for (u, v, k), mid in h.dual_positions.items():
        pu = h.positions[u]
        pv = h.positions[v]
        assert (mid == (pu + pv) / 2).all()

    assert "graph" in calls and "dualGraph" in calls


def test_draw_nonplanar_graph_raises_without_coords():
    plot = _import_plotting()

    class Dummy:
        pass

    h = Dummy()
    h.edges = [(0, 1, 0)]
    with pytest.raises(ValueError):
        _ = plot.draw_nonplanar_graph(h, plot=False, return_fig=True)


def test_draw_nonplanar_graph_returns_figure(nx):
    plot = _import_plotting()

    class Dummy:
        pass

    h = Dummy()
    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
    }
    h.edges = [(0, 1, 0), (0, 2, 0), (1, 2, 0)]

    fig = plot.draw_nonplanar_graph(h, plot=False, return_fig=True, show_labels=False)
    assert fig is not None
