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


def _import_nonplanar_signs():
    try:
        from neuralqx.graph.core.utils.sign_utils import NonplanarSigns

        return NonplanarSigns
    except Exception:
        from neuralqx.graph.core.utils.sign_utils._nonplanar import NonplanarSigns

        return NonplanarSigns


def test_edgegeom_first_point_after_basic(np):
    from neuralqx.graph.core.utils.sign_utils._nonplanar import EdgeGeom

    eg = EdgeGeom(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
    nxt = eg.first_point_after((0.0, 0.0, 0.0))
    assert np.allclose(nxt, np.array([1.0, 0.0, 0.0]))


def test_edgegeom_first_point_after_polyline_skips_vertex(np):
    from neuralqx.graph.core.utils.sign_utils._nonplanar import EdgeGeom

    eg = EdgeGeom(
        start=(0.0, 0.0, 0.0),
        end=(2.0, 0.0, 0.0),
        polyline=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    )
    nxt = eg.first_point_after((0.0, 0.0, 0.0))
    assert np.allclose(nxt, np.array([1.0, 0.0, 0.0]))


def test_edgegeom_first_point_after_degenerate_raises():
    from neuralqx.graph.core.utils.sign_utils._nonplanar import EdgeGeom

    eg = EdgeGeom(
        start=(0.0, 0.0, 0.0),
        end=(0.0, 0.0, 0.0),
        polyline=[(0.0, 0.0, 0.0)],
    )
    with pytest.raises(NotImplementedError):
        _ = eg.first_point_after((0.0, 0.0, 0.0))


def test_nonplanar_tangents_are_unit_and_deterministic_for_seed(
    tripod_nonplanar_edges, np
):
    NonplanarSigns = _import_nonplanar_signs()

    class Dummy:
        pass

    h = Dummy()
    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
    }
    h.edges = tripod_nonplanar_edges

    nps1 = NonplanarSigns(h, seed=123)
    nps2 = NonplanarSigns(h, seed=123)

    for e in h.edges:
        t1 = nps1._tangent(e, h._nonplanar_int2orig_map[0])
        t2 = nps2._tangent(e, h._nonplanar_int2orig_map[0])
        assert np.allclose(t1, t2)
        assert abs(np.linalg.norm(t1) - 1.0) < 1e-12


def test_nonplanar_duplicate_edges_are_perturbed(parallel_edges_two_keys, np):
    NonplanarSigns = _import_nonplanar_signs()

    class Dummy:
        pass

    h = Dummy()
    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
    }
    h.edges = parallel_edges_two_keys

    nps = NonplanarSigns(h, seed=999)
    v0 = h._nonplanar_int2orig_map[0]

    t0 = nps._tangent((0, 1, 0), v0)
    t1 = nps._tangent((0, 1, 1), v0)

    assert not np.allclose(t0, t1, atol=1e-10)
    assert abs(np.linalg.norm(t0) - 1.0) < 1e-12
    assert abs(np.linalg.norm(t1) - 1.0) < 1e-12


def test_vertex_signs_antisymmetry_non_coplanar(tripod_nonplanar_edges):
    NonplanarSigns = _import_nonplanar_signs()

    class Dummy:
        pass

    h = Dummy()
    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
    }
    h.edges = tripod_nonplanar_edges

    nps = NonplanarSigns(h, seed=0, tol=1e-14)
    table = nps.vertex_signs(0)

    e1, e2, e3 = tripod_nonplanar_edges
    assert table[(e1, e2, e3)] == 1
    assert table[(e2, e1, e3)] == -1
    assert table[(e1, e3, e2)] == -1
    assert table[(e2, e3, e1)] == 1


def test_vertex_signs_all_zero_if_coplanar(np):
    NonplanarSigns = _import_nonplanar_signs()

    class Dummy:
        pass

    h = Dummy()
    h._nonplanar_int2orig_map = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (1.0, 1.0, 0.0),
    }
    h.edges = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]

    nps = NonplanarSigns(h, seed=0, tol=1e-12)
    table = nps.vertex_signs(0)
    assert all(v == 0 for v in table.values())
