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

import inspect


def _filter_kwargs_for_callable(fn, kwargs: dict):
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return accepted


def make_instance(cls, *args, **kwargs):

    kw = _filter_kwargs_for_callable(cls.__init__, kwargs)
    return cls(*args, **kw)


def _try_construct_hex(HexagonalLatticeGraph, m, n, **kwargs):
    attempts = [
        ((m, n), kwargs),
        ((), {**kwargs, "m": m, "n": n}),
        ((), {**kwargs, "Lx": m, "Ly": n}),
        ((), {**kwargs, "rows": m, "cols": n}),
        ((m, n), {**kwargs, "periodic": False}),
    ]
    last = None
    for args, kw in attempts:
        try:
            return make_instance(HexagonalLatticeGraph, *args, **kw)
        except TypeError as e:
            last = e
            continue
    raise last


def test_hexagonal_lattice_counts_match_networkx(HexagonalLatticeGraph, nx):
    m, n = 3, 3
    g = _try_construct_hex(HexagonalLatticeGraph, m, n, plot=False, periodic=False)

    ref = nx.hexagonal_lattice_graph(m, n, periodic=False)
    assert g.n_vertices == ref.number_of_nodes()
    assert g.n_edges == ref.number_of_edges()
    assert g.is_planar is True


def test_hexagonal_lattice_minimal_loops_are_hexagons_or_shortest_is_6(
    HexagonalLatticeGraph,
):

    m, n = 4, 4
    g = _try_construct_hex(HexagonalLatticeGraph, m, n, plot=False, periodic=False)

    loops = g.minimal_loops()
    assert len(loops) >= 1
    min_len = min(len(L) for L in loops)
    assert min_len == 6
    assert all(
        len(L) == 6 for L in loops
    ), "GraphHandler keeps only shortest loops, expected all returned loops to have length 6."


def test_hexagonal_lattice_periodic_option_if_supported(HexagonalLatticeGraph, nx):
    m, n = 3, 4
    try:
        g = _try_construct_hex(HexagonalLatticeGraph, m, n, plot=False, periodic=True)
    except TypeError:
        pytest.skip("HexagonalLatticeGraph does not support periodic=True.")

    ref = nx.hexagonal_lattice_graph(m, n, periodic=True)
    assert g.n_vertices == ref.number_of_nodes()
    assert g.n_edges == ref.number_of_edges()


def test_hexagonal_lattice_repr_contains_core_fields(HexagonalLatticeGraph):
    g = _try_construct_hex(HexagonalLatticeGraph, 3, 3, plot=False, periodic=False)
    r = repr(g)
    assert "n_edges=" in r
    assert "n_vertices=" in r
    assert "is_planar=" in r
