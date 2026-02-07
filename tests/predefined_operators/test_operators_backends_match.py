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

jax = pytest.importorskip("jax")
import jax.numpy as jnp

nqx = pytest.importorskip("neuralqx")


def test_solver_expect_thiemann_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    vertex = 0

    op_comp = nqx.operators.computational.Euclidean4d.ThiemannRegularisedVertexConstraintOperator(
        lqx.model,
        vertex,
        fast=True,
        adjoint=False,
        jax=False,
    )
    op_jax = nqx.operators.computational.Euclidean4d.ThiemannRegularisedVertexConstraintOperator(
        lqx.model,
        vertex,
        fast=True,
        adjoint=False,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_area_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.Euclidean4d.AreaOperator(
        H,
        [(0, 1, 0), (0, 2, 0)],
        jax=False,
    )
    op_jax = nqx.operators.computational.Euclidean4d.AreaOperator(
        H,
        [(0, 1, 0), (0, 2, 0)],
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )

    op_comp = nqx.operators.computational.Euclidean4d.AreaOperator(
        H,
        [(0, 1, 0), (0, 2, 0)],
        jax=False,
        squared=True,
    )
    op_jax = nqx.operators.computational.Euclidean4d.AreaOperator(
        H,
        [(0, 1, 0), (0, 2, 0)],
        jax=True,
        squared=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_area_difference_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.Euclidean4d.AreaDifferenceSquaredOperator(
        H,
        [(0, 1, 0), (0, 2, 0)],
        jax=False,
    )
    op_jax = nqx.operators.computational.Euclidean4d.AreaDifferenceSquaredOperator(
        H,
        [(0, 1, 0), (0, 2, 0)],
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_area_difference_surfaces_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = (
        nqx.operators.computational.Euclidean4d.AreaDifferenceSquaredSurfacesOperator(
            H,
            [[(0, 1, 0), (0, 2, 0)], [(0, 1, 0), (0, 2, 0)]],
            jax=False,
        )
    )
    op_jax = (
        nqx.operators.computational.Euclidean4d.AreaDifferenceSquaredSurfacesOperator(
            H,
            [[(0, 1, 0), (0, 2, 0)], [(0, 1, 0), (0, 2, 0)]],
            jax=True,
        )
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_gauss_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.Euclidean4d.GaussConstraintOperator(
        H,
        modded=False,
        jax=False,
    )
    op_jax = nqx.operators.computational.Euclidean4d.GaussConstraintOperator(
        H,
        modded=False,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )

    op_comp = nqx.operators.computational.Euclidean4d.GaussConstraintOperator(
        H,
        modded=True,
        jax=False,
    )
    op_jax = nqx.operators.computational.Euclidean4d.GaussConstraintOperator(
        H,
        modded=True,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_volume_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.Euclidean4d.VolumeOperator(
        H,
        0,
        jax=False,
    )
    op_jax = nqx.operators.computational.Euclidean4d.VolumeOperator(
        H,
        0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_charge_coloring_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.misc.ChargeColoringOperator(
        H,
        0,
        1,
        jax=False,
    )
    op_jax = nqx.operators.computational.misc.ChargeColoringOperator(
        H,
        0,
        1,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_coloring_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.misc.ColoringOperator(
        H,
        0,
        (1, 1, 1),
        jax=False,
    )
    op_jax = nqx.operators.computational.misc.ColoringOperator(
        H,
        0,
        (1, 1, 1),
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_ndholonomy_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.misc.NDU1HolonomyOperator(
        H,
        0,
        jax=False,
    )
    op_jax = nqx.operators.computational.misc.NDU1HolonomyOperator(
        H,
        0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )

    op_comp = nqx.operators.computational.misc.NDU1HolonomyOperator(
        H,
        0,
        jax=False,
        adjoint=True,
    )
    op_jax = nqx.operators.computational.misc.NDU1HolonomyOperator(
        H,
        0,
        jax=True,
        adjoint=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_ndminimal_loop_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.misc.NDMinimalLoopHolonomyOperator(
        H,
        graph.dressed_minimal_loops()[0],
        jax=False,
    )
    op_jax = nqx.operators.computational.misc.NDMinimalLoopHolonomyOperator(
        H,
        graph.dressed_minimal_loops()[0],
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )

    op_comp = nqx.operators.computational.misc.NDMinimalLoopHolonomyOperator(
        H,
        graph.dressed_minimal_loops()[0],
        jax=False,
        adjoint=True,
    )
    op_jax = nqx.operators.computational.misc.NDMinimalLoopHolonomyOperator(
        H,
        graph.dressed_minimal_loops()[0],
        jax=True,
        adjoint=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_1mhh_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.misc.OneMinusHalfHolonomyOperator(
        H,
        graph.dressed_minimal_loops()[0],
        copy=0,
        jax=False,
    )
    op_jax = nqx.operators.computational.misc.OneMinusHalfHolonomyOperator(
        H,
        graph.dressed_minimal_loops()[0],
        copy=0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_number_comp_matches_jax(wcl4d_system):
    graph, H, G, lqx, solver = wcl4d_system

    op_comp = nqx.operators.computational.misc.NumberOperator(
        H,
        0,
        jax=False,
    )
    op_jax = nqx.operators.computational.misc.NumberOperator(
        H,
        0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_qreuclidean_comp_matches_jax(qr_system):
    graph, H, G, lqx, solver = qr_system

    op_comp = nqx.operators.computational.qr.EuclideanConstraintOperator(
        H,
        jax=False,
    )
    op_jax = nqx.operators.computational.qr.EuclideanConstraintOperator(
        H,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_qrcreation_comp_matches_jax(qr_system):
    graph, H, G, lqx, solver = qr_system

    op_comp = nqx.operators.computational.qr.QRCreationOperator(
        H,
        site=0,
        jax=False,
    )
    op_jax = nqx.operators.computational.qr.QRCreationOperator(
        H,
        site=0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_qrannihilation_comp_matches_jax(qr_system):
    graph, H, G, lqx, solver = qr_system

    op_comp = nqx.operators.computational.qr.QRAnnihilationOperator(
        H,
        site=0,
        jax=False,
    )
    op_jax = nqx.operators.computational.qr.QRAnnihilationOperator(
        H,
        site=0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_qrflux_comp_matches_jax(qr_system):
    graph, H, G, lqx, solver = qr_system

    op_comp = nqx.operators.computational.qr.QRFluxOperator(
        H,
        site=0,
        jax=False,
    )
    op_jax = nqx.operators.computational.qr.QRFluxOperator(
        H,
        site=0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )

    op_comp = nqx.operators.computational.qr.QRFluxOperator(
        H,
        site=0,
        jax=False,
        inverse=True,
    )
    op_jax = nqx.operators.computational.qr.QRFluxOperator(
        H,
        site=0,
        jax=True,
        inverse=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )

    op_comp = nqx.operators.computational.qr.QRFluxOperator(
        H,
        site=0,
        jax=False,
        inverse=True,
        power=0.31,
    )
    op_jax = nqx.operators.computational.qr.QRFluxOperator(
        H,
        site=0,
        jax=True,
        inverse=True,
        power=0.31,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_spex_comp_matches_jax(spherical_system):
    graph, H, G, lqx, solver = spherical_system

    op_comp = nqx.operators.computational.spherical.SphericalExOperator(
        H,
        0,
        jax=False,
    )
    op_jax = nqx.operators.computational.spherical.SphericalExOperator(
        H,
        0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_spvolume_comp_matches_jax(spherical_system):
    graph, H, G, lqx, solver = spherical_system

    op_comp = nqx.operators.computational.spherical.SphericalVolumeOperator(
        H,
        0,
        jax=False,
    )
    op_jax = nqx.operators.computational.spherical.SphericalVolumeOperator(
        H,
        0,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )


def test_solver_expect_spconstraint_comp_matches_jax(spherical_system):
    graph, H, G, lqx, solver = spherical_system

    op_comp = nqx.operators.computational.spherical.SphericalVertexConstraintBojowaldSwiderskiOperator(
        H,
        2,
        jax=False,
    )
    op_jax = nqx.operators.computational.spherical.SphericalVertexConstraintBojowaldSwiderskiOperator(
        H,
        2,
        jax=True,
    )

    res_comp = solver.expect(op_comp)
    res_jax = solver.expect(op_jax)

    assert jnp.allclose(res_comp.Mean, res_jax.Mean, rtol=1e-5, atol=1e-6), (
        res_comp.Mean,
        res_jax.Mean,
    )
