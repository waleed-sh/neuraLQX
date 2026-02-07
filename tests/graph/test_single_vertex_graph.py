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


def test_single_vertex_orientation_is_respected(SingleVertexGraph):
    g = SingleVertexGraph(3, orientation=[1, -1, 1], plot=False)

    assert g.n_vertices == 4
    assert g.n_edges == 3

    edges = {tuple(e) for e in g.edges}
    assert (0, 1, 0) in edges
    assert (2, 0, 0) in edges
    assert (0, 3, 0) in edges

    assert g.minimal_loops() == []


def test_single_vertex_orientation_mismatch_raises(SingleVertexGraph, errors):
    with pytest.raises(Exception) as excinfo:
        _ = SingleVertexGraph(3, orientation=[1, -1], plot=False)

    if errors is not None and hasattr(errors, "OrientationValenceMismatchError"):
        assert isinstance(excinfo.value, errors.OrientationValenceMismatchError)
    else:
        assert excinfo.value.__class__.__name__ == "OrientationValenceMismatchError"


def test_single_vertex_nonplanar_option_enables_nonplanar_edges(SingleVertexGraph):
    g = SingleVertexGraph(3, orientation=[1, 1, 1], non_planar=True, plot=False)

    assert g.is_planar is False
    np_edges = g.nonplanar_edges
    assert len(np_edges) == g.n_edges
    assert len(g.nonplanar_vertex_mapping) == g.n_vertices
