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


import os
import pytest


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


@pytest.fixture(scope="session", autouse=True)
def _silence_nqs_verbose():
    os.environ["NQS_VERBOSE"] = "False"
    yield


@pytest.fixture(scope="session")
def jax():
    return pytest.importorskip("jax")


@pytest.fixture(scope="session")
def jnp(jax):
    import jax.numpy as jnp

    return jnp


@pytest.fixture(scope="session")
def netket():
    return pytest.importorskip("netket")


@pytest.fixture(scope="session")
def neuralqx():
    return pytest.importorskip("neuralqx")


@pytest.fixture(scope="session")
def theta_graph(neuralqx):
    return neuralqx.graph.Graph([(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)])


def _get_u1_hilbert_ctor(neuralqx):
    return neuralqx.hilbert.u1.HilbertU1


@pytest.fixture(scope="function")
def hilbert_u1_full_1copy(neuralqx, theta_graph):
    Hilb = _get_u1_hilbert_ctor(neuralqx)
    return Hilb(
        theta_graph,
        cutoff=1,
        is_gauge_invariant=True,
        auto_constraint=True,
        gauge_dimensions=1,
    )


@pytest.fixture(scope="function")
def hilbert_u1_full_2copy(neuralqx, theta_graph):
    Hilb = _get_u1_hilbert_ctor(neuralqx)
    return Hilb(
        theta_graph,
        cutoff=1,
        is_gauge_invariant=False,
        auto_constraint=1,
        gauge_dimensions=2,
    )


@pytest.fixture(scope="function")
def hilbert_u1_full_4copy(neuralqx, theta_graph):
    Hilb = _get_u1_hilbert_ctor(neuralqx)
    return Hilb(
        theta_graph,
        cutoff=1,
        is_gauge_invariant=False,
        auto_constraint=1,
        gauge_dimensions=4,
    )
