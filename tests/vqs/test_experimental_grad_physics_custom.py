# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from functools import reduce

import flax.linen as nn
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from tests.operators.helpers import dense_from_get_conn_padded
from tests.operators.toy_ops import KetSigmap, KetSigmam


class AffineLogPsi(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        w = self.param("w", nn.initializers.normal(stddev=0.1), (x.shape[-1],))
        b = self.param("b", nn.initializers.zeros, ())
        y = jnp.dot(x, w) + b
        return y.astype(jnp.complex64)


class RealParamsComplexLogPsi(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        wr = self.param("wr", nn.initializers.normal(stddev=0.12), (x.shape[-1],))
        wi = self.param("wi", nn.initializers.normal(stddev=0.12), (x.shape[-1],))
        br = self.param("br", nn.initializers.normal(stddev=0.05), ())
        bi = self.param("bi", nn.initializers.normal(stddev=0.05), ())
        re = jnp.dot(x, wr) + br
        im = jnp.dot(x, wi) + bi
        return (re + 1j * im).astype(jnp.complex64)


class NonHolomorphicComplexLogPsi(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        w = self.param(
            "w",
            nn.initializers.normal(stddev=0.12),
            (x.shape[-1],),
            jnp.complex64,
        )
        b = self.param("b", nn.initializers.normal(stddev=0.05), (), jnp.complex64)
        reweighted = jnp.dot(x, w) + 0.37 * jnp.dot(x, jnp.conj(w))
        shifted = b + 0.13 * jnp.conj(b)
        return (reweighted + shifted).astype(jnp.complex64)


class KetSigmapAdjoint(KetSigmap):
    @property
    def adjoint(self):
        return KetSigmamAdjoint(self.hilbert, self.site)


class KetSigmamAdjoint(KetSigmam):
    @property
    def adjoint(self):
        return KetSigmapAdjoint(self.hilbert, self.site)


def _tree_max_abs_error(a, b) -> float:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb, "Pytree structures differ"
    return max(float(jnp.max(jnp.abs(xa - xb))) for xa, xb in zip(la, lb))


def _tree_real(a):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x).real, a)


def _compare_old_vs_experimental(vstate, op, *, nqx):
    _ = vstate.samples

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", False):
        stats_old, grad_old = vstate.expect_and_grad(op, mutable=False)

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        stats_new, grad_new = vstate.expect_and_grad(op, mutable=False)

    return stats_old, grad_old, stats_new, grad_new


def _dense_matrix(operator, *, nk):
    if isinstance(operator, nk.operator.Squared):
        parent = operator.parent
        m_parent = jnp.asarray(dense_from_get_conn_padded(parent))
        m_parent_dag = jnp.asarray(dense_from_get_conn_padded(parent.adjoint))
        return m_parent_dag @ m_parent
    return jnp.asarray(dense_from_get_conn_padded(operator))


def _exact_expect_and_grad_real(vstate, op_or_ops, *, nk):
    states = jnp.asarray(vstate.hilbert.all_states())
    model_state = vstate.model_state
    apply_fun = vstate._apply_fun

    if isinstance(op_or_ops, (list, tuple)):
        mats = [_dense_matrix(op, nk=nk) for op in op_or_ops]
        matrix = reduce(lambda a, b: a + b, mats)
    else:
        matrix = _dense_matrix(op_or_ops, nk=nk)

    def _expect(params):
        variables = {"params": params, **model_state}
        logpsi = jax.vmap(lambda s: apply_fun(variables, s))(states)
        psi = jnp.exp(logpsi)
        return jnp.vdot(psi, matrix @ psi) / jnp.vdot(psi, psi)

    mean = _expect(vstate.parameters)
    grad_real = jax.grad(lambda p: jnp.real(_expect(p)))(vstate.parameters)
    return mean, grad_real


@pytest.fixture
def exact_mcstate(nqx, nk):
    nk.config.netket_experimental = True

    hilbert = nk.hilbert.Spin(s=0.5, N=2)
    sampler = nk.sampler.ExactSampler(hilbert)
    st = nqx.vqs.MCState(
        sampler,
        AffineLogPsi(),
        n_samples=65536,
        n_discard_per_chain=0,
        seed=123,
        sampler_seed=456,
    )
    st.sample(n_samples=65536, n_discard_per_chain=0)
    return st


@pytest.fixture
def exact_mcstate_real_params_complex_out(nqx, nk):
    nk.config.netket_experimental = True

    hilbert = nk.hilbert.Spin(s=0.5, N=2)
    sampler = nk.sampler.ExactSampler(hilbert)
    st = nqx.vqs.MCState(
        sampler,
        RealParamsComplexLogPsi(),
        n_samples=65536,
        n_discard_per_chain=0,
        seed=321,
        sampler_seed=654,
    )
    st.sample(n_samples=65536, n_discard_per_chain=0)
    return st


@pytest.fixture
def exact_mcstate_complex_nonholomorphic(nqx, nk):
    nk.config.netket_experimental = True

    hilbert = nk.hilbert.Spin(s=0.5, N=2)
    sampler = nk.sampler.ExactSampler(hilbert)
    st = nqx.vqs.MCState(
        sampler,
        NonHolomorphicComplexLogPsi(),
        n_samples=65536,
        n_discard_per_chain=0,
        seed=222,
        sampler_seed=777,
    )
    st.sample(n_samples=65536, n_discard_per_chain=0)
    return st


def test_custom_operator_single_gradient_matches_legacy_and_exact(
    exact_mcstate, nqx, nk
):
    op = KetSigmapAdjoint(exact_mcstate.hilbert, site=0)

    stats_old, grad_old, stats_new, grad_new = _compare_old_vs_experimental(
        exact_mcstate,
        op,
        nqx=nqx,
    )
    mean_exact, grad_exact = _exact_expect_and_grad_real(exact_mcstate, op, nk=nk)

    assert jnp.allclose(jnp.asarray(stats_old.Mean), jnp.asarray(mean_exact), atol=4e-3)
    assert jnp.allclose(jnp.asarray(stats_new.Mean), jnp.asarray(mean_exact), atol=4e-3)

    grad_old_real = _tree_real(grad_old)
    grad_new_real = _tree_real(grad_new)

    assert _tree_max_abs_error(grad_old_real, grad_exact) <= 8e-3
    assert _tree_max_abs_error(grad_new_real, grad_exact) <= 8e-3
    assert _tree_max_abs_error(grad_old_real, grad_new_real) <= 8e-3


def test_custom_operator_sequence_complex_dtype_matches_legacy_fallback(
    exact_mcstate, nqx, nk
):
    op = KetSigmapAdjoint(exact_mcstate.hilbert, site=0)
    ops = [op, op]

    stats_old, grad_old, stats_new, grad_new = _compare_old_vs_experimental(
        exact_mcstate,
        ops,
        nqx=nqx,
    )
    mean_exact, grad_exact = _exact_expect_and_grad_real(exact_mcstate, ops, nk=nk)

    assert jnp.allclose(jnp.asarray(stats_old.Mean), jnp.asarray(mean_exact), atol=4e-3)
    assert jnp.allclose(jnp.asarray(stats_new.Mean), jnp.asarray(mean_exact), atol=4e-3)

    old_err = _tree_max_abs_error(_tree_real(grad_old), grad_exact)
    new_err = _tree_max_abs_error(_tree_real(grad_new), grad_exact)

    # Complex-dtype sequence objectives are routed to generic non-Hermitian VJP.
    assert _tree_max_abs_error(grad_new, grad_old) <= 1e-7
    assert jnp.allclose(
        jnp.asarray(stats_new.Mean), jnp.asarray(stats_old.Mean), atol=1e-7
    )
    assert abs(new_err - old_err) <= 1e-7


def test_custom_squared_expectation_is_nonnegative_and_real(exact_mcstate, nqx, nk):
    op = KetSigmapAdjoint(exact_mcstate.hilbert, site=0)
    sq = nk.operator.Squared(op)

    with nqx.cfg.patch("EXPERIMENTAL", True), nqx.cfg.patch("EXPERIMENTAL_GRAD", True):
        stats_sq, grad_sq = exact_mcstate.expect_and_grad(sq, mutable=False)

    mean_sq = complex(np.asarray(stats_sq.Mean))
    assert mean_sq.real >= -1e-8
    assert abs(mean_sq.imag) <= 1e-8

    # Keep a minimal API sanity check: gradient must be finite.
    leaves, _ = jax.tree_util.tree_flatten(grad_sq)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_custom_operator_adjoint_matrix_is_conjugate_transpose(exact_mcstate):
    op = KetSigmapAdjoint(exact_mcstate.hilbert, site=0)
    op_dag = op.adjoint

    matrix = dense_from_get_conn_padded(op)
    matrix_dag = dense_from_get_conn_padded(op_dag)

    assert np.allclose(matrix_dag, matrix.conj().T, rtol=0.0, atol=0.0)


def test_real_params_complex_objective_matches_generic_vjp(
    exact_mcstate_real_params_complex_out,
    nqx,
):
    op = KetSigmapAdjoint(exact_mcstate_real_params_complex_out.hilbert, site=0)

    stats_old, grad_old, stats_new, grad_new = _compare_old_vs_experimental(
        exact_mcstate_real_params_complex_out,
        op,
        nqx=nqx,
    )

    # Ensure this is genuinely a complex-valued objective, then verify the
    # experimental route preserves generic VJP semantics.
    assert abs(complex(np.asarray(stats_new.Mean)).imag) >= 1e-2
    assert _tree_max_abs_error(grad_new, grad_old) <= 1e-7


def test_real_params_complex_objective_sequence_matches_generic_vjp(
    exact_mcstate_real_params_complex_out,
    nqx,
):
    op = KetSigmapAdjoint(exact_mcstate_real_params_complex_out.hilbert, site=0)
    ops = [op, op]

    stats_old, grad_old, stats_new, grad_new = _compare_old_vs_experimental(
        exact_mcstate_real_params_complex_out,
        ops,
        nqx=nqx,
    )

    # Ensure this remains a genuinely complex-valued objective in sequence mode.
    assert abs(complex(np.asarray(stats_new.Mean)).imag) >= 1e-2
    assert _tree_max_abs_error(grad_new, grad_old) <= 1e-7
    assert jnp.allclose(
        jnp.asarray(stats_new.Mean), jnp.asarray(stats_old.Mean), atol=1e-7
    )


def test_complex_nonholomorphic_single_matches_generic_vjp(
    exact_mcstate_complex_nonholomorphic,
    nqx,
):
    op = KetSigmapAdjoint(exact_mcstate_complex_nonholomorphic.hilbert, site=0)

    _, grad_old, _, grad_new = _compare_old_vs_experimental(
        exact_mcstate_complex_nonholomorphic,
        op,
        nqx=nqx,
    )

    assert _tree_max_abs_error(grad_new, grad_old) <= 1e-7


def test_complex_nonholomorphic_sequence_matches_generic_vjp(
    exact_mcstate_complex_nonholomorphic,
    nqx,
):
    op = KetSigmapAdjoint(exact_mcstate_complex_nonholomorphic.hilbert, site=0)
    ops = [op, op.adjoint]

    _, grad_old, _, grad_new = _compare_old_vs_experimental(
        exact_mcstate_complex_nonholomorphic,
        ops,
        nqx=nqx,
    )

    assert _tree_max_abs_error(grad_new, grad_old) <= 1e-7
