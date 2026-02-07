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


def test_grid2d_counts_match_networkx(Grid2D, nx):
    params = [
        (2, 2, False, False),
        (3, 2, False, False),
        (3, 3, True, False),
        (3, 3, False, True),
    ]

    for m, n, periodic, non_planar in params:
        g = Grid2D(m, n, periodic=periodic, non_planar=non_planar, plot=False)
        expected = nx.generators.grid_2d_graph(m, n, periodic=periodic)

        assert g.n_vertices == expected.number_of_nodes()
        assert g.n_edges == expected.number_of_edges()

        if non_planar:
            assert g.is_planar is False
            np_edges = g.nonplanar_edges
            assert len(np_edges) == g.n_edges
            vmap = g.nonplanar_vertex_mapping
            assert len(vmap) == g.n_vertices
        else:
            assert g.is_planar is True


def test_grid2d_minimal_loop_2x2_is_one_square(Grid2D):
    g = Grid2D(2, 2, periodic=False, plot=False)
    loops = g.minimal_loops()

    assert len(loops) == 1
    assert len(loops[0]) == 4
