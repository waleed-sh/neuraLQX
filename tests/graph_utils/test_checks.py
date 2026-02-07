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


def _import_checks():
    from neuralqx.graph.core.utils import checks as _checks

    return _checks


def test_validate_and_insert_keys_keyless_unique(theta_planar_keyless_edges):
    checks = _import_checks()
    out = checks.validate_and_insert_keys(theta_planar_keyless_edges)
    assert len(out) == len(theta_planar_keyless_edges)
    assert all(len(e) == 3 for e in out)
    assert all(e[2] == 0 for e in out)


def test_validate_and_insert_keys_duplicate_keyless_raises():
    checks = _import_checks()
    edges = [(0, 1), (0, 1)]
    with pytest.raises(ValueError):
        _ = checks.validate_and_insert_keys(edges)


def test_validate_and_insert_keys_all_keyed_ok(parallel_edges_two_keys):
    checks = _import_checks()
    out = checks.validate_and_insert_keys(parallel_edges_two_keys)
    assert set(out) == set(parallel_edges_two_keys)


def test_validate_and_insert_keys_mixed_keyed_unkeyed_raises():
    checks = _import_checks()
    edges = [(0, 1, 0), (0, 2)]
    with pytest.raises(TypeError):
        _ = checks.validate_and_insert_keys(edges)


def test_validate_and_insert_keys_non_int_key_raises():
    checks = _import_checks()
    edges = [(0, 1, "k")]
    with pytest.raises(TypeError):
        _ = checks.validate_and_insert_keys(edges)


def test_validate_and_insert_keys_bad_edge_len_raises():
    checks = _import_checks()
    with pytest.raises(ValueError):
        _ = checks.validate_and_insert_keys([(0,)])


def test_check_vertex_type_consistency_planar(theta_planar_keyless_edges):
    checks = _import_checks()
    assert checks.check_vertex_type_consistency(theta_planar_keyless_edges) == "planar"


def test_check_vertex_type_consistency_nonplanar(theta_nonplanar_keyless_edges):
    checks = _import_checks()
    assert (
        checks.check_vertex_type_consistency(theta_nonplanar_keyless_edges)
        == "non-planar"
    )


def test_check_vertex_type_consistency_mixed_raises():
    checks = _import_checks()
    edges = [(0, (0, 0, 0))]
    with pytest.raises(ValueError):
        _ = checks.check_vertex_type_consistency(edges)


def test_get_vertices_from_edges_ignores_keys(theta_planar_keyed_edges):
    checks = _import_checks()
    verts = checks.get_vertices_from_edges(theta_planar_keyed_edges)
    assert verts == [0, 1, 2, 3]


def test_get_vertices_from_edges_invalid_edge_raises():
    checks = _import_checks()
    with pytest.raises(ValueError):
        _ = checks.get_vertices_from_edges([(0,)])


def test__check_planarity_empty_raises():
    checks = _import_checks()
    with pytest.raises(ValueError):
        _ = checks._check_planarity([])


def test__check_planarity_planar(theta_planar_keyless_edges):
    checks = _import_checks()
    assert checks._check_planarity(theta_planar_keyless_edges) == "planar"


def test__check_planarity_nonplanar(theta_nonplanar_keyless_edges):
    checks = _import_checks()
    assert checks._check_planarity(theta_nonplanar_keyless_edges) == "non-planar"


def test__check_planarity_incorrect_format_raises(neuralqx):
    checks = _import_checks()
    IncorrectEdgeFormatError = neuralqx.utils.errors.IncorrectEdgeFormatError
    edges = [(["x"], ["y"])]
    with pytest.raises(IncorrectEdgeFormatError):
        _ = checks._check_planarity(edges)


def test__relabel_non_planar_edges_bijection(theta_nonplanar_keyed_edges):
    checks = _import_checks()
    vertices, relabeled, int2orig, orig2int = checks._relabel_non_planar_edges(
        theta_nonplanar_keyed_edges
    )

    assert set(vertices) == set(int2orig.keys())
    assert set(int2orig.keys()) == set(range(len(int2orig)))
    assert set(orig2int.values()) == set(range(len(orig2int)))

    for i, orig in int2orig.items():
        assert orig2int[orig] == i

    for u, v, k in relabeled:
        ou, ov, ok = (
            theta_nonplanar_keyed_edges[0][0],
            theta_nonplanar_keyed_edges[0][1],
            theta_nonplanar_keyed_edges[0][2],
        )
        assert isinstance(u, int) and isinstance(v, int)
        assert isinstance(k, int)
