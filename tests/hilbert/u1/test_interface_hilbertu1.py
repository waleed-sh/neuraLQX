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


def test_hilbertu1_builds_constrained_and_unconstrained(graph_small):
    from neuralqx.hilbert.u1.interface import HilbertU1
    from neuralqx.hilbert.u1.constrained_core import ConstrainedHilbertU1Core
    from neuralqx.hilbert.u1.unconstrained_core import UnconstrainedHilbertU1Core

    h_c = HilbertU1(
        graph=graph_small,
        cutoff=1,
        gauge_dimensions=2,
        is_gauge_invariant=True,
        auto_constraint=True,
    )
    assert isinstance(h_c.hilbert, ConstrainedHilbertU1Core)

    h_u = HilbertU1(
        graph=graph_small,
        cutoff=1,
        gauge_dimensions=2,
        is_gauge_invariant=False,
    )
    assert isinstance(h_u.hilbert, UnconstrainedHilbertU1Core)


@pytest.mark.xfail(
    strict=True,
    reason="HilbertU1 currently does not accept gauge_invariant keyword as alias.",
)
def test_hilbertu1_accepts_gauge_invariant_alias(graph_small):

    from neuralqx.hilbert.u1.interface import HilbertU1
    from neuralqx.hilbert.u1.constrained_core import ConstrainedHilbertU1Core

    h = HilbertU1(
        graph=graph_small,
        cutoff=1,
        gauge_dimensions=2,
        gauge_invariant=True,
        auto_constraint=True,
    )
    assert isinstance(h.hilbert, ConstrainedHilbertU1Core)


def test_edge_charges_matches_view_extraction(graph_small, jax, jnp):
    from neuralqx.hilbert.u1.interface import HilbertU1

    h = HilbertU1(
        graph=graph_small,
        cutoff=1,
        gauge_dimensions=2,
        is_gauge_invariant=True,
        auto_constraint=True,
    )

    sigma = h.hilbert.random_state(jax.random.PRNGKey(0), size=3)
    v = h.hilbert.view(sigma)

    e0 = (
        graph_small.edges[0]
        if getattr(graph_small, "is_planar", True)
        else graph_small.nonplanar_edges[0]
    )

    vals = h.edge_charges(sigma, e0)
    assert vals.shape == (3 * int(h.hilbert.gauge_dimensions),)

    site0 = h.edge_to_site(e0, gauge_copy=0)
    edge_idx = int(site0) % int(h.hilbert.tiny_size)

    expected = np.asarray(v[:, :, edge_idx]).reshape(-1)
    assert np.array_equal(np.asarray(vals), expected)


def test_edge_charges_collapses_leading_dims(graph_small, jax, jnp):
    from neuralqx.hilbert.u1.interface import HilbertU1

    h = HilbertU1(
        graph=graph_small,
        cutoff=1,
        gauge_dimensions=2,
        is_gauge_invariant=True,
        auto_constraint=True,
    )

    B = 2
    T = 4
    sigma = h.hilbert.random_state(jax.random.PRNGKey(1), size=B)
    sigma3 = jnp.broadcast_to(sigma, (T,) + sigma.shape)

    e0 = (
        graph_small.edges[0]
        if getattr(graph_small, "is_planar", True)
        else graph_small.nonplanar_edges[0]
    )
    vals = h.edge_charges(sigma3, e0)

    G = int(h.hilbert.gauge_dimensions)
    assert vals.shape == (T * B * G,)


def test_edge_charges_invalid_edge_raises(graph_small, jax):
    from neuralqx.hilbert.u1.interface import HilbertU1
    from neuralqx.utils.errors import InvalidIndexError

    h = HilbertU1(
        graph=graph_small,
        cutoff=1,
        gauge_dimensions=2,
        is_gauge_invariant=True,
        auto_constraint=True,
    )
    sigma = h.hilbert.random_state(jax.random.PRNGKey(0), size=1)

    bad_edge = (0, 1, 999999)

    with pytest.raises(InvalidIndexError):
        _ = h.edge_charges(sigma, bad_edge)
