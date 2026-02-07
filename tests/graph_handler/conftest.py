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


@pytest.fixture(scope="session", autouse=True)
def _mpl_headless_backend():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    yield


@pytest.fixture(scope="session")
def nx():
    return pytest.importorskip("networkx")


@pytest.fixture(scope="session")
def netket():
    return pytest.importorskip("netket")


@pytest.fixture(scope="session")
def neuralqx():
    return pytest.importorskip("neuralqx")


@pytest.fixture(scope="session")
def GraphHandler(neuralqx):
    try:
        from neuralqx.graph.core.graph_handler import GraphHandler

        return GraphHandler
    except Exception:
        from neuralqx.graph.core import GraphHandler

        return GraphHandler


@pytest.fixture(scope="session")
def theta_planar_keyless_edges():
    return [(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)]


@pytest.fixture(scope="session")
def theta_planar_keyed_edges():
    return [(0, 1, 0), (0, 2, 0), (0, 3, 0), (3, 2, 0), (1, 2, 0)]


@pytest.fixture(scope="session")
def theta_nonplanar_keyless_edges():
    return [
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
    ]


@pytest.fixture(scope="session")
def theta_nonplanar_keyed_edges():
    return [
        ((0, 0, 0), (0, 0, 1), 0),
        ((0, 0, 0), (0, 1, 1), 0),
        ((0, 0, 0), (1, 1, 1), 0),
        ((1, 1, 1), (0, 1, 1), 0),
        ((0, 0, 1), (0, 1, 1), 0),
    ]


@pytest.fixture(scope="session")
def edges_parallel_two_edge_loop():
    return [(0, 1, 0), (0, 1, 1)]


@pytest.fixture(scope="session")
def edges_triangle_and_square_disconnected():
    return [
        (0, 1, 0),
        (1, 2, 0),
        (2, 0, 0),
        (3, 4, 0),
        (4, 5, 0),
        (5, 6, 0),
        (6, 3, 0),
    ]
