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


import numpy as np
import pytest


def test_core_counts_and_mapping_invariants(Graph, theta_planar_keyless_edges):
    g = Graph(theta_planar_keyless_edges, plot=False)

    assert g.is_planar is True
    assert g.n_edges == len(theta_planar_keyless_edges)

    assert g.n_dual_vertices == g.n_edges
    assert g.dual_nx_graph.number_of_nodes() == g.n_edges

    mapping = g.mapping
    assert isinstance(mapping, dict)
    assert len(mapping) == g.n_edges
    vals = sorted(mapping.values())
    assert vals == list(
        range(g.n_edges)
    ), f"Expected contiguous mapping [0...E-1], got {vals}"


def test_edge_to_index_and_index_to_edge_roundtrip_and_reverse(
    Graph, theta_planar_keyless_edges, errors
):
    g = Graph(theta_planar_keyless_edges, plot=False)
    edges = list(g.edges)

    for e in edges:
        idx = g.edge_to_index(e)
        back = g.index_to_edge(idx)

        assert g.edge_to_index(back) == idx

        u, v, k = e
        rev = (v, u, k)
        assert g.edge_to_index(rev) == idx


def test_edge_to_index_raises_for_unknown_edge(
    Graph, theta_planar_keyless_edges, errors
):
    g = Graph(theta_planar_keyless_edges, plot=False)

    bad = (123, 456, 0)
    with pytest.raises(Exception) as excinfo:
        g.edge_to_index(bad)

    if errors is not None and hasattr(errors, "InvalidIndexError"):
        assert isinstance(excinfo.value, errors.InvalidIndexError)
    else:
        assert excinfo.value.__class__.__name__ in {"InvalidIndexError", "KeyError"}


def test_index_to_edge_raises_for_out_of_range(
    Graph, theta_planar_keyless_edges, errors
):
    g = Graph(theta_planar_keyless_edges, plot=False)

    with pytest.raises(Exception) as excinfo:
        g.index_to_edge(g.n_edges + 10)

    if errors is not None and hasattr(errors, "OutOfRangeIndexError"):
        assert isinstance(excinfo.value, errors.OutOfRangeIndexError)
    else:
        assert excinfo.value.__class__.__name__ in {
            "OutOfRangeIndexError",
            "ValueError",
        }


def test_valence_matches_networkx_degree(Graph, theta_planar_keyless_edges):
    g = Graph(theta_planar_keyless_edges, plot=False)
    nxg = g.nx_graph

    for v in nxg.nodes():
        assert g.valence(v) == nxg.degree(v)


def test_edges_at_and_edges_from_have_expected_semantics_on_theta(
    Graph, theta_planar_keyless_edges
):
    g = Graph(theta_planar_keyless_edges, plot=False)

    from0 = g.edges_from(0)
    assert len(from0) == 3
    assert {e[1] for e in from0} == {1, 2, 3}

    at2 = g.edges_at(2)
    assert len(at2) == 3
    assert {e[0] for e in at2} == {0, 1, 3}


def test_nonplanar_accessors_raise_on_planar(Graph, theta_planar_keyless_edges, errors):
    g = Graph(theta_planar_keyless_edges, plot=False)

    with pytest.raises(Exception) as exc1:
        _ = g.nonplanar_edges
    with pytest.raises(Exception) as exc2:
        _ = g.nonplanar_vertex_mapping

    if errors is not None and hasattr(errors, "NonExistentNonPlanarEdgesError"):
        assert isinstance(exc1.value, errors.NonExistentNonPlanarEdgesError)
    if errors is not None and hasattr(errors, "NonExistentNonPlanarVerticesError"):
        assert isinstance(exc2.value, errors.NonExistentNonPlanarVerticesError)


def test_export_creates_nonempty_file(Graph, theta_planar_keyless_edges, tmp_path):
    g = Graph(theta_planar_keyless_edges, plot=False)
    out = tmp_path / "theta_graph.png"
    g.export(str(out), dpi=120)

    assert out.exists()
    assert out.stat().st_size > 0


def test_random_embedding_on_planar_graph_is_disallowed(
    Graph, theta_planar_keyed_edges
):
    from neuralqx.utils.errors import (
        NonExistentNonPlanarEdgesError,
        RandomEmbeddingForPlanarGraphWarning,
        NonExistentNonPlanarVerticesError,
    )

    with pytest.warns(RandomEmbeddingForPlanarGraphWarning):
        g = Graph(
            theta_planar_keyed_edges,
            plot=False,
            random_embedding=True,
            random_embedding_seed=7,
        )

    assert g.is_planar is True

    with pytest.raises(NonExistentNonPlanarEdgesError):
        _ = g.nonplanar_edges

    with pytest.raises(NonExistentNonPlanarVerticesError):
        _ = g.nonplanar_vertex_mapping
