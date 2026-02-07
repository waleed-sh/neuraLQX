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

import shutil
import pytest
from types import SimpleNamespace


@pytest.fixture(scope="session", autouse=True)
def _set_headless_plot_backend():
    import matplotlib

    matplotlib.use("Agg")
    yield


@pytest.fixture
def tmp_output_dir(tmp_path):
    out = tmp_path / "solver_output"
    out.mkdir()
    yield str(out)
    shutil.rmtree(out, ignore_errors=True)


class _TestMPI:
    available = False
    n_nodes = 1

    @staticmethod
    def is_global_master():
        return True

    @staticmethod
    def barrier():
        return None


@pytest.fixture(scope="session")
def mpi_world():
    return _TestMPI()


@pytest.fixture(scope="session")
def lqx_system():
    import neuralqx as nqx

    CUTOFF = 1
    GD = 3

    graph = nqx.graph.K5Graph(
        non_planar=True,
        random_embedding=False,
        plot=False,
    )

    H = nqx.hilbert.u1.HilbertU1(
        graph,
        CUTOFF,
        gauge_dimensions=GD,
        is_gauge_invariant=True,
        auto_constraint=True,
    )

    G = nqx.gauge_groups.u1.U1GaugeGroup(
        H,
        lazy=True,
        computational=True,
        jax=True,
    )

    lqx = nqx.lqx.LqxWCL(
        H,
        G,
        spacetime_dimensions=4,
        computational=True,
        lazy_load=False,
    )

    lqx.constraint = lqx.thiemann_quadratic_constraint(
        computational=True,
        jax=True,
        adjoint=False,
    )

    return lqx


@pytest.fixture
def solver_env(monkeypatch, tmp_output_dir, lqx_system, mpi_world):

    import neuralqx.solver.solver as solver_mod
    import neuralqx.utils.mpi as mpi_mod

    monkeypatch.setattr(mpi_mod, "available", False)
    monkeypatch.setattr(mpi_mod, "n_nodes", 1)
    monkeypatch.setattr(mpi_mod, "is_global_master", lambda: True)
    monkeypatch.setattr(mpi_mod, "barrier", lambda: None)

    solver = solver_mod.Solver(
        lqx_system,
        output_path=tmp_output_dir,
    )

    return SimpleNamespace(
        solver=solver,
        Solver=solver_mod.Solver,
        AbstractSolver=solver_mod.AbstractSolver,
        lqx=lqx_system,
        mpi=mpi_world,
        output_path=tmp_output_dir,
    )
