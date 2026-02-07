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


def test_triangular_lattice_counts_match_networkx(TriangularLatticeGraph, nx):
    m, n, periodic = 3, 3, False
    g = TriangularLatticeGraph(m, n, periodic=periodic, plot=False)
    expected = nx.generators.triangular_lattice_graph(m, n, periodic=periodic)

    assert g.n_vertices == expected.number_of_nodes()
    assert g.n_edges == expected.number_of_edges()
    assert g.is_planar is True


def test_triangular_lattice_has_triangles_as_min_cycles(TriangularLatticeGraph):
    g = TriangularLatticeGraph(3, 3, periodic=False, plot=False)
    loops = g.minimal_loops()

    assert len(loops) >= 1
    min_len = min(len(L) for L in loops)
    assert min_len == 3
    assert any(len(L) == 3 for L in loops)
