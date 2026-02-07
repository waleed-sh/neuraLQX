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


def _try_construct_circular(CircularLadderGraph, N, **kwargs):
    for args, kw in [
        ((N,), kwargs),
        ((), {**kwargs, "N": N}),
        ((), {**kwargs, "n": N}),
        ((), {**kwargs, "length": N}),
    ]:
        try:
            return make_instance(CircularLadderGraph, *args, **kw)
        except TypeError:
            continue
    raise


def test_circular_ladder_counts_match_networkx(CircularLadderGraph, nx):
    N = 6
    g = _try_construct_circular(CircularLadderGraph, N, plot=False)

    ref = nx.circular_ladder_graph(N)
    assert g.n_vertices == ref.number_of_nodes()
    assert g.n_edges == ref.number_of_edges()
    assert g.is_planar is True


def test_circular_ladder_minimal_loops_are_N_squares(CircularLadderGraph):

    N = 7
    g = _try_construct_circular(CircularLadderGraph, N, plot=False)

    loops = g.minimal_loops()
    assert len(loops) == N
    assert all(len(L) == 4 for L in loops)


def test_circular_ladder_nonplanar_option_if_supported(CircularLadderGraph):
    N = 5
    try:
        g = _try_construct_circular(CircularLadderGraph, N, plot=False, non_planar=True)
    except TypeError:
        pytest.skip("CircularLadderGraph does not support non_planar=True.")

    assert g.is_planar is False
    assert len(g.nonplanar_edges) == g.n_edges
    assert len(g.nonplanar_vertex_mapping) == g.n_vertices


def test_circular_ladder_repr_contains_core_fields(CircularLadderGraph):
    N = 4
    g = _try_construct_circular(CircularLadderGraph, N, plot=False)
    r = repr(g)
    assert "n_edges=" in r
    assert "n_vertices=" in r
    assert "is_planar=" in r
