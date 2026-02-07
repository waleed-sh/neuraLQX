# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_lqx_quickstart_smoke_builds_operators():

    import neuralqx as nqx

    CUTOFF = 1
    GD = 3

    graph = nqx.graph.K5Graph(non_planar=True, random_embedding=False, plot=False)

    H = nqx.hilbert.u1.HilbertU1(
        graph,
        CUTOFF,
        gauge_dimensions=GD,
        is_gauge_invariant=True,
        auto_constraint=True,
    )

    G = nqx.gauge_groups.u1.U1GaugeGroup(H, lazy=True, computational=True, jax=True)

    lqx = nqx.lqx.LqxWCL(
        H, G, lazy_load=False, spacetime_dimensions=4, computational=True
    )

    C = lqx.thiemann_quadratic_constraint(computational=True, jax=True, adjoint=False)
    lqx.constraint = C
    assert lqx.constraint is not None

    v0 = lqx.volume(0)
    F = lqx.curvature_constraint()

    assert v0 is not None
    assert F is not None
