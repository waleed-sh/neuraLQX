# tests/predefined_models/test_models_run_vmc.py

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


from __future__ import annotations

from pathlib import Path

import pytest


def assert_dict_keys(
    d: dict,
    *,
    required={"Mean", "Sigma", "R_hat", "Variance", "TauCorr"},
    optional=(),
    allow_extra=True,
):
    d_keys = set(d)
    required = set(required)
    optional = set(optional)

    missing = required - d_keys
    unexpected = d_keys - required - optional

    if missing:
        raise AssertionError(f"Missing keys: {missing}")

    if unexpected and not allow_extra:
        raise AssertionError(f"Unexpected keys: {unexpected}")


def _safe_solver_run(solver, n_iter: int = 1) -> None:

    try:
        solver.run(
            n_iter,
            silent_plot=True,
            silent_print=True,
            live_monitoring=False,
        )
    except TypeError:
        solver.run(n_iter)


def _run_case(case: str, output_path: str) -> None:
    import neuralqx as nqx
    import jax.numpy as jnp
    import netket as nk

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    if case == "qr":
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

        solver = nqx.solver.Solver(lqx, output_path=str(out))

        solver.set_sampler(
            sampler_type="Metropolis Local",
            number_of_samples=450,
            number_of_chains=10,
            number_of_sweeps=20,
        )

        solver.set_optimizer(optimizer_type="Adam", learning_rate=0.001, use_sr=True)
        solver.set_network(nk.models.RBM())
        solver.set_network(nk.models.RBM(alpha=2))

        _safe_solver_run(solver, 1)

        assert solver.is_solved
        assert_dict_keys(
            solver.expect(lqx.creation(0, 1, computational=True, jax=True)).to_dict()
        )
        assert_dict_keys(
            solver.expect(
                lqx.annihilation(0, 1, computational=True, jax=True)
            ).to_dict()
        )
        assert_dict_keys(
            solver.expect(lqx.s(0, no_i=True, computational=True, jax=True)).to_dict()
        )
        assert_dict_keys(
            solver.expect(lqx.c(0, computational=True, jax=True)).to_dict()
        )
        assert_dict_keys(
            solver.expect(lqx.E(0, 1, computational=True, jax=True)).to_dict()
        )
        assert_dict_keys(
            solver.expect(lqx.E_inv(0, 1, computational=True, jax=True)).to_dict()
        )
        return

    if case == "wcl":
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

        lqx.constraint = lqx.thiemann_quadratic_constraint(
            computational=True, jax=True, adjoint=False, lazy=True
        )

        solver = nqx.solver.Solver(lqx, output_path=str(out))

        solver.set_sampler(
            sampler_type="U1 Gauge Sampler",
            number_of_samples=450,
            number_of_chains=10,
            number_of_sweeps=20,
        )

        solver.set_optimizer(optimizer_type="Adam", learning_rate=0.001, use_sr=True)
        solver.set_network(nk.models.RBM())

        _safe_solver_run(solver, 1)

        assert solver.is_solved
        assert_dict_keys(solver.expect(lqx.volume(0, jax=True)).to_dict())
        assert_dict_keys(
            solver.expect(lqx.model.volume(0, computational=True, jax=True)).to_dict()
        )
        assert_dict_keys(
            solver.expect(
                lqx.model.volume(0, computational=False, jax=False, standalone=True)
            ).to_dict()
        )

        assert jnp.allclose(
            solver.expect(
                lqx.model.volume(0, computational=False, jax=False, standalone=True)
            ).Mean,
            solver.expect(lqx.model.volume(0, computational=True, jax=True)).Mean,
            atol=2e-2,
            rtol=1e-3,
        )

        assert jnp.allclose(
            solver.expect(lqx.volume(0, computational=True, jax=True)).Mean,
            solver.expect(lqx.model.volume(0, computational=True, jax=True)).Mean,
            atol=2e-2,
            rtol=1e-3,
        )

        assert jnp.allclose(
            solver.expect(lqx.volume(0, computational=False, jax=False)).Mean,
            solver.expect(
                lqx.model.volume(0, computational=False, jax=False, standalone=True)
            ).Mean,
            atol=2e-2,
            rtol=1e-3,
        )

        assert_dict_keys(
            solver.expect(lqx.area([(0, 1, 0), (0, 2, 0)], jax=True)).to_dict()
        )

        assert_dict_keys(
            solver.expect(
                lqx.model.area([(0, 1, 0), (0, 2, 0)], computational=False, jax=False)
            ).to_dict()
        )

        assert jnp.allclose(
            solver.expect(lqx.model.area([(0, 1, 0), (0, 2, 0)], jax=True)).Mean,
            solver.expect(
                lqx.model.area([(0, 1, 0), (0, 2, 0)], computational=False, jax=False)
            ).Mean,
            atol=2e-2,
            rtol=1e-3,
        )

        assert jnp.allclose(
            solver.expect(
                lqx.area([(0, 1, 0), (0, 2, 0)], computational=True, jax=True)
            ).Mean,
            solver.expect(
                lqx.model.area([(0, 1, 0), (0, 2, 0)], computational=False, jax=False)
            ).Mean,
            atol=2e-2,
            rtol=1e-3,
        )

        assert jnp.allclose(
            solver.expect(
                lqx.area([(0, 1, 0), (0, 2, 0)], computational=True, jax=True)
            ).Mean,
            solver.expect(
                lqx.model.area([(0, 1, 0), (0, 2, 0)], computational=True, jax=True)
            ).Mean,
            atol=2e-2,
            rtol=1e-3,
        )

        assert_dict_keys(
            solver.expect(lqx.holonomy(graph.edges[0], jax=True)).to_dict()
        )
        assert_dict_keys(solver.expect(lqx.curvature_constraint(jax=True)).to_dict())

        with pytest.raises(nqx.utils.errors.InvalidSurfaceError):
            solver.expect(
                lqx.area({(0, 1, 0), (0, 2, 0)}, computational=True, jax=True)
            )

        with pytest.raises(nqx.utils.errors.AreaDifferenceEdgesError):
            solver.expect(
                lqx.model.area_difference_squared_operator(
                    [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
                )
            )

        return

    if case == "spherical":
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

        solver = nqx.solver.Solver(lqx, output_path=str(out))

        solver.set_sampler(
            sampler_type="Metropolis Local",
            number_of_samples=450,
            number_of_chains=10,
            number_of_sweeps=20,
        )

        solver.set_optimizer(optimizer_type="Adam", learning_rate=0.001, use_sr=True)
        solver.set_network(nk.models.RBM())

        _safe_solver_run(solver, 1)

        assert solver.is_solved
        assert_dict_keys(solver.expect(lqx.volume(0, jax=True)).to_dict())
        assert_dict_keys(solver.expect(lqx.flux(0, jax=True)).to_dict())
        assert_dict_keys(
            solver.expect(lqx.holonomy(graph.edges[0], jax=True)).to_dict()
        )
        assert_dict_keys(solver.expect(lqx.number(graph.edges[0], jax=True)).to_dict())
        return

    raise ValueError(f"Unknown case: {case!r}")


pytestmark = pytest.mark.integration


def test_nqx_qr_runs_vmc(tmp_path, run_vmc_case):
    run_vmc_case("qr", tmp_path)


def test_nqx_wcl_runs_vmc(tmp_path, run_vmc_case):
    run_vmc_case("wcl", tmp_path)


def test_nqx_spherical_runs_vmc(tmp_path, run_vmc_case):
    run_vmc_case("spherical", tmp_path)
