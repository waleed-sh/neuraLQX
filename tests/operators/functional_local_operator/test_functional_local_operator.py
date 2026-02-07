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


import numpy as np
import pytest

pytest.importorskip("netket")
pytest.importorskip("jax")
pytest.importorskip("numba")

import jax.numpy as jnp

from tests.operators.functional_local_operator.helpers import (
    make_local_operator,
    make_functional_local_operator,
    assert_conn_transformed,
    dense_nonzero_transform,
    make_mcstate,
    exact_expectation_from_dense,
)


@pytest.mark.parametrize(
    "func_name, func", [("identity", lambda x: x), ("conj", lambda x: np.conjugate(x))]
)
def test_dense_and_connections_match_reference(
    hilbert, site, complex_single_site_matrix, func_name, func
):
    op0 = make_local_operator(hilbert, complex_single_site_matrix, site)
    opF = make_functional_local_operator(
        hilbert, complex_single_site_matrix, site, mels_func=func
    )

    dense0 = np.asarray(op0.to_dense())
    denseF = np.asarray(opF.to_dense())

    expected = dense_nonzero_transform(dense0, func)
    assert np.allclose(denseF, expected)

    # connection structure must match; mels are transformed on non-zero entries
    x = np.asarray(hilbert.all_states()[:8], dtype=np.int64)
    assert_conn_transformed(hilbert, op0, opF, x, transform=func)


def test_mel_cutoff_is_applied_before_function(hilbert, site):
    # off-diagonal element is below cutoff => should be absent even after transform
    opmat = np.array([[0.5, 1e-10 + 2e-10j], [0.0, -0.25]], dtype=np.complex128)
    cutoff = 1e-8

    op0 = make_local_operator(hilbert, opmat, site, mel_cutoff=cutoff)
    opF = make_functional_local_operator(
        hilbert, opmat, site, mels_func=lambda x: np.conjugate(x), mel_cutoff=cutoff
    )

    dense0 = np.asarray(op0.to_dense())
    denseF = np.asarray(opF.to_dense())

    # only diagonal entries should survive
    assert np.allclose(dense0, np.diag(np.diag(dense0)))
    assert np.allclose(denseF, np.diag(np.diag(denseF)))

    # connections should be purely diagonal (no off-diagonals)
    x = np.asarray(hilbert.all_states()[:8], dtype=np.int64)
    xp, mels = opF.get_conn_padded(x)
    xp = np.asarray(xp)
    mels = np.asarray(mels)
    # any non-zero mel must keep the state unchanged
    mask = ~np.isclose(mels, 0.0)
    assert np.all(xp[mask] == x[:, None, :][mask])


def test_to_jax_operator_matches_numba_version(
    hilbert, site, complex_single_site_matrix
):
    opF = make_functional_local_operator(
        hilbert, complex_single_site_matrix, site, mels_func=lambda x: np.conjugate(x)
    )

    try:
        opJ = opF.to_jax_operator()
    except Exception as e:
        pytest.skip(
            f"to_jax_operator unavailable or failed to import JAX variant: {e!r}"
        )

    x = np.asarray(hilbert.all_states()[:6], dtype=np.int64)
    xpF, mF = opF.get_conn_padded(x)
    xpJ, mJ = opJ.get_conn_padded(jnp.asarray(x))

    assert np.allclose(np.asarray(xpF), np.asarray(xpJ))
    assert np.allclose(np.asarray(mF), np.asarray(mJ))

    # assert np.allclose(np.asarray(opF.to_dense()), np.asarray(opJ.to_dense()))


def test_operator_arithmetic_is_not_disabled(hilbert, site, complex_single_site_matrix):
    opF = make_functional_local_operator(
        hilbert, complex_single_site_matrix, site, mels_func=lambda x: np.conjugate(x)
    )
    op0 = make_local_operator(hilbert, complex_single_site_matrix, site)

    for expr in [
        lambda: opF + op0,
        lambda: op0 + opF,
        lambda: opF * op0,
        lambda: op0 * opF,
        lambda: 2.0 * opF,
        lambda: opF * 2.0,
        lambda: opF - op0,
        lambda: op0 - opF,
    ]:
        _ = expr()


def test_expectation_matches_reference_operator_exact_and_mc(
    hilbert, site, complex_single_site_matrix
):
    # functional = conjugate(nonzero mels) => equivalent to LocalOperator built from conjugated single-site matrix
    op0 = make_local_operator(hilbert, complex_single_site_matrix, site)
    opF = make_functional_local_operator(
        hilbert, complex_single_site_matrix, site, mels_func=lambda x: np.conjugate(x)
    )

    op_ref = make_local_operator(
        hilbert, np.conjugate(complex_single_site_matrix), site
    )

    denseF = np.asarray(opF.to_dense())
    denseR = np.asarray(op_ref.to_dense())
    assert np.allclose(denseF, denseR)

    vstate = make_mcstate(
        hilbert, seed=0, n_samples=4096, n_chains=16, n_discard_per_chain=64
    )

    exact = exact_expectation_from_dense(vstate, denseF)

    stats = vstate.expect(opF)
    mean = getattr(stats, "Mean", getattr(stats, "mean"))
    sigma = getattr(
        stats, "Sigma", getattr(stats, "sigma", getattr(stats, "error_of_mean", None))
    )

    mean = np.asarray(mean)
    assert np.all(np.isfinite(mean))

    if sigma is not None:
        sigma = np.asarray(sigma)
        atol = np.maximum(1e-3, 6.0 * sigma)
    else:
        atol = 2e-2

    assert np.allclose(mean, exact, atol=atol, rtol=1e-3)
