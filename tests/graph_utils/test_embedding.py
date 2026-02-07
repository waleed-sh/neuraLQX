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


def _import_embedding():
    from neuralqx.graph.core.utils import embedding as _emb

    return _emb


def test_randomize_graph_coordinates_is_deterministic(theta_nonplanar_keyed_edges):
    emb = _import_embedding()

    e1, m1 = emb.randomize_graph_coordinates(theta_nonplanar_keyed_edges, seed=123)
    e2, m2 = emb.randomize_graph_coordinates(theta_nonplanar_keyed_edges, seed=123)

    assert e1 == e2
    assert m1 == m2

    for v, xyz in m1.items():
        assert isinstance(xyz, tuple) and len(xyz) == 3
        assert all(isinstance(x, float) for x in xyz)

    for a, b, k in e1:
        assert a in m1.values()
        assert b in m1.values()
        assert isinstance(k, int)


def test_randomize_graph_coordinates_changes_with_seed(theta_nonplanar_keyed_edges):
    emb = _import_embedding()

    e1, m1 = emb.randomize_graph_coordinates(theta_nonplanar_keyed_edges, seed=1)
    e2, m2 = emb.randomize_graph_coordinates(theta_nonplanar_keyed_edges, seed=2)

    assert m1 != m2
    assert e1 != e2
