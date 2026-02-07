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


def _import_signs():
    from neuralqx.graph.core.utils import signs as _signs

    return _signs


def test__compute_angles_right_angle():
    s = _import_signs()
    node_pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}
    a = s._compute_angles((0, 1, 0), (0, 2, 0), node_pos)
    assert abs(a - 90.0) < 1e-12


def test__rotation_matrix_orthonormal(np):
    s = _import_signs()
    R = s._rotation_matrix(0.123)
    I = R.T @ R
    assert np.allclose(I, np.eye(2), atol=1e-12)


def test__find_sign_known_vectors():
    s = _import_signs()

    assert s._find_sign((1.0, 0.0), (0.0, 1.0), show_plots=False) == -1
    assert s._find_sign((0.0, 1.0), (1.0, 0.0), show_plots=False) == 1


def test_compute_sgn_det_orientation_flip_changes_sign(np):
    s = _import_signs()

    class Dummy:
        pass

    h = Dummy()
    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
    }
    edge_set = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]

    h._original_edges = list(edge_set)
    v = s.compute_sgn_det(h, edge_set)
    assert v == 1

    h._original_edges = [(1, 0, 0), (0, 2, 0), (0, 3, 0)]
    v2 = s.compute_sgn_det(h, edge_set)
    assert v2 == -1


def test_compute_signs_planar_populates_tables(np):
    s = _import_signs()

    class Dummy:
        pass

    h = Dummy()
    h.is_planar = True

    h.positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
    }

    h.graph_signs = {}
    h.graph_positions = {}
    h.dual_graph_signs = {}
    h.dual_graph_positions = {}

    e01 = (0, 1, 0)
    e02 = (0, 2, 0)
    h.graph_edges_data = {
        "graph": {"connectivities": {"0": {"edges": [(e01, e02), (e02, e01)]}}}
    }

    h._get_edge_direction_factor = lambda edge_pair, vertex: 1

    s.compute_signs(h, graph_type="graph", show_plots=False)

    assert "0" in h.graph_signs
    k1 = str((e01, e02))
    k2 = str((e02, e01))
    assert k1 in h.graph_signs["0"]
    assert k2 in h.graph_signs["0"]

    origin = h.positions[0]
    f1 = h.positions[1] - origin
    f2 = h.positions[2] - origin
    expected = s._find_sign(f1, f2, False)
    assert int(h.graph_signs["0"][k1]) == expected

    assert int(h.graph_signs["0"][k2]) == -expected


def test_compute_non_planar_graph_signs_populates_using_nonplanar_signs(
    tripod_nonplanar_edges,
):
    s = _import_signs()

    class Dummy:
        pass

    h = Dummy()
    h.is_planar = False
    h.graph_signs = {}
    h.dual_graph_signs = {}

    import networkx as nx

    h.nx_graph = nx.MultiGraph()
    for u, v, k in tripod_nonplanar_edges:
        h.nx_graph.add_edge(u, v, key=k)

    h.dual_nx_graph = nx.line_graph(h.nx_graph)

    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
    }
    h.edges = tripod_nonplanar_edges

    s.compute_non_planar_graph_signs(h, graph_type="graph")

    assert all(str(v) in h.graph_signs for v in h.nx_graph.nodes())
    assert len(h.graph_signs["0"]) > 0
