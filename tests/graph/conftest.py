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


import importlib
import os
import pytest


def _import_any_attr(module_candidates: list[str], attr_candidates: list[str]):
    last = None
    for modname in module_candidates:
        try:
            mod = importlib.import_module(modname)
            for attr in attr_candidates:
                if hasattr(mod, attr):
                    return getattr(mod, attr)
        except Exception as e:
            last = e
    raise ImportError(
        f"Could not import any of {attr_candidates} from {module_candidates}. Last error: {last}"
    )


@pytest.fixture(scope="session", autouse=True)
def _force_matplotlib_headless():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def _import_attr(attr: str, candidates: list[str]):
    last_exc = None
    for modpath in candidates:
        try:
            mod = importlib.import_module(modpath)
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception as e:
            last_exc = e
            continue
    raise ImportError(
        f"Could not import {attr} from any of {candidates}. Last error: {last_exc}"
    )


@pytest.fixture(scope="session")
def nx():
    import networkx as networkx

    return networkx


@pytest.fixture(scope="session")
def Graph():
    return _import_attr("Graph", ["neuralqx.graph.graph", "neuralqx.graph"])


@pytest.fixture(scope="session")
def Grid2D():
    return _import_attr("Grid2D", ["neuralqx.graph.grid_2d", "neuralqx.graph"])


@pytest.fixture(scope="session")
def TriangularLatticeGraph():
    return _import_attr(
        "TriangularLatticeGraph",
        ["neuralqx.graph.triangular_lattice", "neuralqx.graph"],
    )


@pytest.fixture(scope="session")
def K5Graph():
    return _import_attr("K5Graph", ["neuralqx.graph.k5_graph", "neuralqx.graph"])


@pytest.fixture(scope="session")
def SingleVertexGraph():
    return _import_attr(
        "SingleVertexGraph", ["neuralqx.graph.single_vertex", "neuralqx.graph"]
    )


@pytest.fixture(scope="session")
def HalfLadderGraph():
    return _import_attr(
        "HalfLadderGraph", ["neuralqx.graph.half_ladder_graph", "neuralqx.graph"]
    )


@pytest.fixture(scope="session")
def errors():
    """
    Best-effort import of error types. We don't require these imports to exist as exact symbols,
    tests will fall back to checking exception class names if needed
    """
    try:
        return importlib.import_module("neuralqx.utils.errors")
    except Exception:
        return None


@pytest.fixture()
def theta_planar_keyless_edges():
    return [(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)]


@pytest.fixture()
def theta_planar_keyed_edges():
    return [(0, 1, 0), (0, 2, 0), (0, 3, 0), (3, 2, 0), (1, 2, 0)]


@pytest.fixture()
def theta_nonplanar_keyed_edges():
    return [
        ((0, 0, 0), (0, 0, 1), 0),
        ((0, 0, 0), (0, 1, 1), 0),
        ((0, 0, 0), (1, 1, 1), 0),
        ((1, 1, 1), (0, 1, 1), 0),
        ((0, 0, 1), (0, 1, 1), 0),
    ]


@pytest.fixture(scope="session")
def LadderGraph():
    return _import_any_attr(
        ["neuralqx.graph.ladder_graph", "neuralqx.graph"],
        ["LadderGraph"],
    )


@pytest.fixture(scope="session")
def CircularLadderGraph():
    return _import_any_attr(
        ["neuralqx.graph.circular_ladder", "neuralqx.graph"],
        ["CircularLadderGraph", "CircularLadder"],
    )


@pytest.fixture(scope="session")
def HypercubeGraph():
    return _import_any_attr(
        ["neuralqx.graph.hypercube", "neuralqx.graph"],
        ["HypercubeGraph", "HyperCubeGraph", "Hypercube"],
    )


@pytest.fixture(scope="session")
def DodecahedralGraph():
    return _import_any_attr(
        ["neuralqx.graph.dodecahedral", "neuralqx.graph"],
        ["DodecahedralGraph", "Dodecahedral"],
    )


@pytest.fixture(scope="session")
def HexagonalLatticeGraph():
    return _import_any_attr(
        ["neuralqx.graph.hexagonal_lattice", "neuralqx.graph"],
        ["HexagonalLatticeGraph", "HexagonalLattice"],
    )
