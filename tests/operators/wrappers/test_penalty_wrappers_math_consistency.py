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

import jax
import numpy as np
import pytest

from neuralqx.vqs.mc.common import force_to_grad


def _stats_mean_real(stats) -> float:
    return float(np.real(np.asarray(stats.Mean)))


def _set_fixed_samples(vstate):
    samples = np.asarray(vstate.samples)
    vstate._samples = samples
    return samples


def _reuse_samples(vstate, samples) -> None:
    vstate._samples = samples


def _tree_allclose(a, b, *, rtol: float, atol: float) -> None:
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    assert treedef_a == treedef_b, "Pytree structure mismatch."
    for la, lb in zip(leaves_a, leaves_b):
        assert np.allclose(np.asarray(la), np.asarray(lb), rtol=rtol, atol=atol), (
            la,
            lb,
        )


def _tree_scale(x, factor: float):
    return jax.tree_util.tree_map(lambda v: factor * v, x)


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


@pytest.mark.parametrize("fused_kernels", [False, True], ids=["unfused", "fused"])
def test_penalty_cost_single_matches_scaled_operator(vstate, nk, nqx, fused_kernels):
    """
    Mathematical identity:
        PenaltyCost(O, f) == f * O
    for expectation, forces and gradient.
    """

    hilb = vstate.hilbert
    base = nk.operator.spin.sigmax(hilb, 0) + 0.37 * nk.operator.spin.sigmaz(hilb, 1)
    factor = 0.7
    penalty = nqx.operators.PenaltyCost(base, factor=factor)
    scaled = factor * base

    with nqx.cfg.patch("FUSED_KERNELS", fused_kernels):
        samples = _set_fixed_samples(vstate)

        _reuse_samples(vstate, samples)
        stats_pen = vstate.expect(penalty)
        _reuse_samples(vstate, samples)
        stats_scaled = vstate.expect(scaled)
        assert np.allclose(
            _stats_mean_real(stats_pen),
            _stats_mean_real(stats_scaled),
            atol=2e-6,
            rtol=2e-5,
        )

        _reuse_samples(vstate, samples)
        stats_pen_f, forces_pen = vstate.expect_and_forces(penalty, mutable=False)
        _reuse_samples(vstate, samples)
        stats_scaled_f, forces_scaled = vstate.expect_and_forces(scaled, mutable=False)
        assert np.allclose(
            _stats_mean_real(stats_pen_f),
            _stats_mean_real(stats_scaled_f),
            atol=2e-6,
            rtol=2e-5,
        )
        _tree_allclose(forces_pen, forces_scaled, atol=2e-6, rtol=2e-5)

        _reuse_samples(vstate, samples)
        stats_pen_g, grad_pen = vstate.expect_and_grad(penalty, mutable=False)
        _reuse_samples(vstate, samples)
        stats_scaled_g, grad_scaled = vstate.expect_and_grad(scaled, mutable=False)
        assert np.allclose(
            _stats_mean_real(stats_pen_g),
            _stats_mean_real(stats_scaled_g),
            atol=2e-6,
            rtol=2e-5,
        )
        _tree_allclose(grad_pen, grad_scaled, atol=2e-6, rtol=2e-5)

        grad_from_forces = force_to_grad(forces_pen, vstate.parameters)
        _tree_allclose(grad_from_forces, grad_pen, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("fused_kernels", [False, True], ids=["unfused", "fused"])
def test_inverse_expectation_cost_single_chain_rule(vstate, nk, nqx, fused_kernels):
    """
    Mathematical identity for IEC:
        C(E) = f / (alpha + E + eps)^2
        dC/dθ = C'(E) * dE/dθ
    """

    hilb = vstate.hilbert
    base = nk.operator.spin.sigmax(hilb, 0) + 0.37 * nk.operator.spin.sigmaz(hilb, 1)
    iec = nqx.operators.InverseExpectationCost(base, factor=0.8, alpha=1.2)

    with nqx.cfg.patch("FUSED_KERNELS", fused_kernels):
        samples = _set_fixed_samples(vstate)

        _reuse_samples(vstate, samples)
        base_stats, base_forces = vstate.expect_and_forces(base, mutable=False)
        base_mean = _stats_mean_real(base_stats)

        denom = iec.alpha + base_mean + iec.eps
        fprime = (-2.0 * iec.factor) / (denom**3)
        expected_mean = iec.factor / (denom**2)
        expected_forces = _tree_scale(base_forces, fprime)
        expected_grad = force_to_grad(expected_forces, vstate.parameters)

        _reuse_samples(vstate, samples)
        stats_expect = vstate.expect(iec)
        assert np.allclose(
            _stats_mean_real(stats_expect), expected_mean, atol=2e-6, rtol=2e-5
        )

        _reuse_samples(vstate, samples)
        stats_forces, forces = vstate.expect_and_forces(iec, mutable=False)
        assert np.allclose(
            _stats_mean_real(stats_forces), expected_mean, atol=2e-6, rtol=2e-5
        )
        _tree_allclose(forces, expected_forces, atol=2e-4, rtol=7e-4)

        _reuse_samples(vstate, samples)
        stats_grad, grad = vstate.expect_and_grad(iec, mutable=False)
        assert np.allclose(
            _stats_mean_real(stats_grad), expected_mean, atol=2e-6, rtol=2e-5
        )
        _tree_allclose(grad, expected_grad, atol=3e-4, rtol=1e-3)

        # IEC single uses a dedicated gradient route, so this check uses a looser tolerance.
        grad_from_forces = force_to_grad(forces, vstate.parameters)
        _tree_allclose(grad_from_forces, grad, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("fused_kernels", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("wrapper_kind", ["penalty", "iec"])
def test_wrapper_sequence_additivity(vstate, nk, nqx, fused_kernels, wrapper_kind):
    """
    For sequence objective [A, B]:
        <A+B> = <A> + <B>
        F(A+B) = F(A) + F(B)
        g(A+B) = g(A) + g(B)
    where gradients are compared against the same dispatch family used by
    sequence-wrapper gradients (singleton sequence terms).
    """

    hilb = vstate.hilbert
    base = nk.operator.spin.sigmax(hilb, 0) + 0.37 * nk.operator.spin.sigmaz(hilb, 1)
    extra = nk.operator.spin.sigmax(hilb, 2)
    extra_penalty = nqx.operators.PenaltyCost(extra, factor=0.25)

    if wrapper_kind == "penalty":
        term_a = nqx.operators.PenaltyCost(base, factor=0.7)
    else:
        term_a = nqx.operators.InverseExpectationCost(base, factor=0.8, alpha=1.2)
    term_b = extra_penalty
    seq = [term_a, term_b]

    with nqx.cfg.patch("FUSED_KERNELS", fused_kernels):
        samples = _set_fixed_samples(vstate)

        _reuse_samples(vstate, samples)
        stats_a = vstate.expect(term_a)
        _reuse_samples(vstate, samples)
        stats_b = vstate.expect(term_b)
        expected_mean = _stats_mean_real(stats_a) + _stats_mean_real(stats_b)

        _reuse_samples(vstate, samples)
        stats_seq = vstate.expect(seq)
        assert np.allclose(
            _stats_mean_real(stats_seq), expected_mean, atol=2e-6, rtol=2e-5
        )

        _reuse_samples(vstate, samples)
        _, forces_a = vstate.expect_and_forces(term_a, mutable=False)
        _reuse_samples(vstate, samples)
        _, forces_b = vstate.expect_and_forces(term_b, mutable=False)
        expected_forces = _tree_add(forces_a, forces_b)

        _reuse_samples(vstate, samples)
        stats_seq_f, forces_seq = vstate.expect_and_forces(seq, mutable=False)
        assert np.allclose(
            _stats_mean_real(stats_seq_f), expected_mean, atol=2e-6, rtol=2e-5
        )
        _tree_allclose(forces_seq, expected_forces, atol=2e-5, rtol=2e-4)

        # Compare sequence-gradient against the sum of singleton-sequence gradients,
        # to stay on the same dispatch family used by sequence penalty wrappers.
        _reuse_samples(vstate, samples)
        _, grad_a = vstate.expect_and_grad([term_a], mutable=False)
        _reuse_samples(vstate, samples)
        _, grad_b = vstate.expect_and_grad([term_b], mutable=False)
        expected_grad = _tree_add(grad_a, grad_b)

        _reuse_samples(vstate, samples)
        stats_seq_g, grad_seq = vstate.expect_and_grad(seq, mutable=False)
        assert np.allclose(
            _stats_mean_real(stats_seq_g), expected_mean, atol=2e-6, rtol=2e-5
        )
        _tree_allclose(grad_seq, expected_grad, atol=2e-4, rtol=5e-4)
