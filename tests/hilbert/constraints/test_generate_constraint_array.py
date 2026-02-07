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


def _incidence_matrix(edges):
    verts = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
    v2i = {v: i for i, v in enumerate(verts)}
    n, m = len(verts), len(edges)
    B = np.zeros((n, m), dtype=int)
    for j, (u, v, _k) in enumerate(edges):
        B[v2i[u], j] = 1
        B[v2i[v], j] = -1
    return B, verts


def test_generate_constraint_array_triangle_is_deterministic_and_valid(parse_edge_term):
    from neuralqx.hilbert.constraints.utils._common import generate_constraint_array

    edges = [
        (0, 1, 0),
        (0, 2, 0),
        (1, 2, 0),
    ]
    cons = generate_constraint_array(edges)

    assert isinstance(cons, list)
    assert len(cons) == 2

    edge_to_idx = {e: i for i, e in enumerate(edges)}

    fixed_idxs = []
    rhs_map = []
    for lhs_list, rhs_list in cons:
        assert len(lhs_list) == 1
        fixed_edge = parse_edge_term(lhs_list[0])[1]
        fixed_idx = edge_to_idx[fixed_edge]
        fixed_idxs.append(fixed_idx)

        d = {}
        for term in rhs_list:
            coef, et = parse_edge_term(term)
            d[edge_to_idx[et]] = d.get(edge_to_idx[et], 0) + coef
        rhs_map.append(d)

    assert fixed_idxs == [0, 1]

    B, _verts = _incidence_matrix(edges)
    Br = np.delete(B, B.shape[0] - 1, axis=0)

    tree = fixed_idxs
    chord = [j for j in range(len(edges)) if j not in tree]
    assert chord == [2]

    for val in [-2, -1, 0, 1, 2, 3]:
        f = np.zeros((len(edges),), dtype=int)
        f[chord[0]] = val

        for fixed_idx, d in zip(tree, rhs_map):
            s = 0
            for j, coef in d.items():
                s += coef * f[j]
            f[fixed_idx] = s

        res = Br @ f
        assert np.all(
            res == 0
        ), f"constraint solve did not satisfy incidence: f={f}, Brf={res}"


def test_generate_constraint_array_parallel_edges_are_supported_and_deterministic(
    parse_edge_term,
):
    from neuralqx.hilbert.constraints.utils._common import generate_constraint_array

    edges = [
        (0, 1, 0),
        (0, 1, 1),
        (1, 2, 0),
    ]
    cons = generate_constraint_array(edges)

    assert len(cons) == 2

    fixed = [parse_edge_term(c[0][0])[1] for c in cons]
    assert (0, 1, 0) in fixed or (0, 1, 1) in fixed
    assert (0, 1, 0) in fixed or (0, 1, 1) in fixed
    assert (1, 2, 0) in fixed
    assert not ((0, 1, 0) in fixed and (0, 1, 1) in fixed)


def test_generate_constraint_array_disconnected_graph_raises():
    from neuralqx.hilbert.constraints.utils._common import generate_constraint_array

    edges = [
        (0, 1, 0),
        (2, 3, 0),
    ]
    with pytest.raises(Exception):
        generate_constraint_array(edges)
