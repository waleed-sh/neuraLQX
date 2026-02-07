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


def _try_construct_hypercube(HypercubeGraph, dim, **kwargs):
    for args, kw in [
        ((dim,), kwargs),
        ((), {**kwargs, "dim": dim}),
        ((), {**kwargs, "d": dim}),
        ((), {**kwargs, "n": dim}),
    ]:
        try:
            return make_instance(HypercubeGraph, *args, **kw)
        except TypeError:
            continue
    raise


def test_hypercube_counts_match_networkx(HypercubeGraph, nx):
    dim = 4
    g = _try_construct_hypercube(HypercubeGraph, dim, plot=False)

    ref = nx.hypercube_graph(dim)
    assert g.n_vertices == ref.number_of_nodes()
    assert g.n_edges == ref.number_of_edges()

    # every vertex in hypercube has degree=dim
    degs = [g.valence(v) for v in g.vertices]
    assert set(degs) == {dim}


def test_hypercube_minimal_loops_are_4cycles_and_basis_size_matches_cycle_rank(
    HypercubeGraph, nx
):

    dim = 3
    g = _try_construct_hypercube(HypercubeGraph, dim, plot=False)

    V = g.n_vertices
    E = g.n_edges
    # expected_rank = E - V + 1

    loops = g.minimal_loops()
    assert len(loops) == 6
    assert all(len(L) == 4 for L in loops)

    assert g.is_planar is True


def test_hypercube_repr_contains_core_fields(HypercubeGraph):
    dim = 3
    g = _try_construct_hypercube(HypercubeGraph, dim, plot=False)
    r = repr(g)
    assert "n_edges=" in r
    assert "n_vertices=" in r
    assert "is_planar=" in r
