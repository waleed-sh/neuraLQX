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


def test_half_ladder_nonplanar_is_not_supported(HalfLadderGraph):
    with pytest.raises(NotImplementedError):
        _ = HalfLadderGraph(length=3, connected=False, non_planar=True, plot=False)


@pytest.mark.parametrize("connected", [False, True])
def test_half_ladder_counts_and_structure(HalfLadderGraph, connected):
    L = 4
    g = HalfLadderGraph(length=L, connected=connected, plot=False)

    assert len(g.vertices_k) == L + 1
    assert len(g.vertices_mu) == L + 1
    assert g.n_vertices == 2 * (L + 1)

    expected_k = L + (1 if connected else 0)
    assert len(g.edges_k) == expected_k

    assert len(g.edges_mu) == L + 1

    assert g.n_edges == len(g.edges_k) + len(g.edges_mu)

    assert g.is_connected is connected


def test_half_ladder_get_edges_at_k_vertex_invariants(HalfLadderGraph):
    L = 5
    g = HalfLadderGraph(length=L, connected=False, plot=False)

    mu, ks = g.get_edges_at_k_vertex(2, dual_rep=False)
    assert len(mu) == 1
    assert len(ks) == 2

    mu0, ks0 = g.get_edges_at_k_vertex(0, dual_rep=False)
    assert len(mu0) == 1
    assert len(ks0) == 1

    with pytest.raises(ValueError):
        _ = g.get_edges_at_k_vertex(g.vertices_mu[0], dual_rep=False)


def test_half_ladder_get_edges_at_k_vertex_dual_rep_matches_edge_to_index(
    HalfLadderGraph,
):
    g = HalfLadderGraph(length=4, connected=False, plot=False)

    mu_e, k_e = g.get_edges_at_k_vertex(2, dual_rep=False)
    mu_i, k_i = g.get_edges_at_k_vertex(2, dual_rep=True)

    assert mu_i == [g.edge_to_index(e) for e in mu_e]
    assert k_i == [g.edge_to_index(e) for e in k_e]
    assert all(isinstance(x, int) for x in mu_i + k_i)


def test_half_ladder_has_k_vertex_neighbours_semantics(HalfLadderGraph):
    g = HalfLadderGraph(length=4, connected=False, plot=False)

    assert g.has_k_vertex_neighbours(0, "left") is False
    assert g.has_k_vertex_neighbours(0, "right") is True

    last = g.vertices_k[-1]
    assert g.has_k_vertex_neighbours(last, "right") is False
    assert g.has_k_vertex_neighbours(last, "left") is True


def test_half_ladder_get_kminus_kplus_splits_left_right(HalfLadderGraph):
    g = HalfLadderGraph(length=4, connected=False, plot=False)

    out = g.get_kminus_kplus(2, dual_rep=False)
    assert isinstance(out, list) and len(out) == 2
    left, right = out

    assert len(left) == 1
    assert len(right) == 1

    assert left[0][1] == 2
    assert right[0][0] == 2


def test_half_ladder_edges_mu_idx_matches_edges_mu(HalfLadderGraph):
    import jax.numpy as jnp

    g = HalfLadderGraph(length=4, connected=False, plot=False)

    idx = g.edges_mu_idx
    assert isinstance(idx, jnp.ndarray)
    assert idx.dtype == jnp.int32
    assert idx.shape[0] == len(g.edges_mu)

    expected = jnp.array([g.edge_to_index(e) for e in g.edges_mu], dtype=jnp.int32)
    assert jnp.all(idx == expected)


@pytest.mark.parametrize("connected, expected_loops", [(False, 0), (True, 1)])
def test_half_ladder_minimal_loops_connected_ring_only(
    HalfLadderGraph, connected, expected_loops
):
    L = 5
    g = HalfLadderGraph(length=L, connected=connected, plot=False)
    loops = g.minimal_loops()

    assert len(loops) == expected_loops
    if connected:
        assert len(loops[0]) == L + 1
