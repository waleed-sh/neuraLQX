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


def test_k5_graph_planar_mode_counts(K5Graph):
    g = K5Graph(non_planar=False, plot=False)

    assert g.is_planar is True
    assert g.n_vertices == 5
    assert g.n_edges == 10

    assert len(g.mapping) == g.n_edges
    assert sorted(g.mapping.values()) == list(range(g.n_edges))


def test_k5_graph_nonplanar_mode_plumbing(K5Graph):
    g = K5Graph(non_planar=True, plot=False)

    assert g.is_planar is False
    assert g.n_vertices == 5
    assert g.n_edges == 10

    np_edges = g.nonplanar_edges
    assert len(np_edges) == g.n_edges

    vmap = g.nonplanar_vertex_mapping
    assert len(vmap) == g.n_vertices

    assert all(isinstance(k, int) for k in vmap.keys())
    assert all(isinstance(v, tuple) for v in vmap.values())

    assert len(set(vmap.values())) == g.n_vertices

    for i, orig in vmap.items():
        assert g.nonplanar_vertex_to_index(orig) == i


def test_k5_minimal_cycles_include_triangles(K5Graph):
    g = K5Graph(non_planar=False, plot=False)
    loops = g.minimal_loops()

    assert len(loops) >= 1
    assert min(len(L) for L in loops) == 3
