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

nqx = pytest.importorskip("neuralqx")
nk = pytest.importorskip("netket")


@pytest.fixture
def wcl4d_system(tmp_path):

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
        lazy_load=False,
        spacetime_dimensions=4,
        computational=True,
    )

    solver = nqx.solver.Solver(lqx, output_path=str(tmp_path), clean_up=True)

    solver.set_sampler(
        sampler_type="U1 Gauge Sampler",
        number_of_samples=450,
        number_of_chains=10,
        number_of_sweeps=20,
    )

    solver.set_optimizer(optimizer_type="Adam", learning_rate=0.001, use_sr=True)

    solver.set_network(nk.models.RBM())

    solver.initialize_vmc()

    return graph, H, G, lqx, solver


@pytest.fixture
def spherical_system(tmp_path):

    CUTOFF = 1
    GD = 1

    graph = nqx.graph.HalfLadderGraph(
        length=10,
    )

    H = nqx.hilbert.u1.HilbertU1(
        graph,
        CUTOFF,
        gauge_dimensions=GD,
        is_gauge_invariant=False,
    )

    G = nqx.gauge_groups.u1.U1GaugeGroup(
        H,
        lazy=True,
        computational=True,
        jax=True,
    )

    lqx = nqx.lqx.LqxBojowaldSwiderski(
        H,
        G,
        lazy_load=False,
    )

    solver = nqx.solver.Solver(lqx, output_path=str(tmp_path), clean_up=True)

    solver.set_sampler(
        sampler_type="Metropolis Local",
        number_of_samples=450,
        number_of_chains=10,
        number_of_sweeps=20,
    )

    solver.set_optimizer(optimizer_type="Adam", learning_rate=0.001, use_sr=True)

    solver.set_network(nk.models.RBM())

    solver.initialize_vmc()

    return graph, H, G, lqx, solver


@pytest.fixture
def qr_system(tmp_path):

    CUTOFF = 1
    GD = 1

    graph = nqx.graph.SingleVertexGraph(3)

    H = nqx.hilbert.u1.HilbertU1(
        graph,
        CUTOFF,
        gauge_dimensions=GD,
        is_gauge_invariant=False,
        positive_qn=True,
    )

    G = nqx.gauge_groups.u1.U1GaugeGroup(
        H,
        lazy=True,
        computational=True,
        jax=True,
    )

    lqx = nqx.lqx.LqxSingleVertexQR(
        H,
        G,
        computational=True,
    )

    solver = nqx.solver.Solver(lqx, output_path=str(tmp_path), clean_up=True)

    solver.set_sampler(
        sampler_type="Metropolis Local",
        number_of_samples=450,
        number_of_chains=10,
        number_of_sweeps=20,
    )

    solver.set_optimizer(optimizer_type="Adam", learning_rate=0.001, use_sr=True)

    solver.set_network(nk.models.RBM(alpha=2))

    solver.initialize_vmc()

    return graph, H, G, lqx, solver
