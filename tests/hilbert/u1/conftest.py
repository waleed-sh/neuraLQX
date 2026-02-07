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


@pytest.fixture(scope="session")
def jax():
    return pytest.importorskip("jax")


@pytest.fixture(scope="session")
def jnp(jax):
    return jax.numpy


@pytest.fixture(scope="session")
def nk():
    return pytest.importorskip("netket")


@pytest.fixture(scope="session")
def graph_small():
    from neuralqx.graph.k5_graph import K5Graph

    return K5Graph(plot=False)


@pytest.fixture(scope="function")
def constrained_core_small(graph_small):

    from neuralqx.hilbert.u1.constrained_core import ConstrainedHilbertU1Core

    return ConstrainedHilbertU1Core(
        graph=graph_small,
        cutoff=1,
        step=1,
        gauge_dimensions=2,
        constraint=None,
        auto_constraint=True,
    )


@pytest.fixture(scope="function")
def unconstrained_core_small(graph_small):
    from neuralqx.hilbert.u1.unconstrained_core import UnconstrainedHilbertU1Core

    return UnconstrainedHilbertU1Core(
        graph=graph_small,
        cutoff=1,
        step=1,
        gauge_dimensions=2,
    )
