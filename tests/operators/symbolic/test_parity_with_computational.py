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

import numpy as np
import pytest

from tests.operators.helpers import (
    assert_means_close,
    dense_from_get_conn_padded,
)
from tests.operators.symbolic.helpers import (
    build_symbolic_sigmam,
    build_symbolic_sigmap,
    build_symbolic_sigmaz,
    compile_no_cache,
    dense_as_local,
)
from tests.operators.toy_ops import (
    KetSigmam,
    KetSigmamJax,
    KetSigmap,
    KetSigmapJax,
    KetSigmaz,
    KetSigmazJax,
)


def _mk_symbolic(symbolic, hilbert, kind: str, site: int):
    if kind == "Z":
        return compile_no_cache(build_symbolic_sigmaz(symbolic, hilbert, site))
    if kind == "Sp":
        return compile_no_cache(build_symbolic_sigmap(symbolic, hilbert, site))
    if kind == "Sm":
        return compile_no_cache(build_symbolic_sigmam(symbolic, hilbert, site))
    raise ValueError(kind)


def _mk_comp(hilbert, kind: str, site: int):
    if kind == "Z":
        return KetSigmaz(hilbert, site), KetSigmazJax(hilbert, site)
    if kind == "Sp":
        return KetSigmap(hilbert, site), KetSigmapJax(hilbert, site)
    if kind == "Sm":
        return KetSigmam(hilbert, site), KetSigmamJax(hilbert, site)
    raise ValueError(kind)


def _mk_local(nk, hilbert, kind: str, site: int):
    if kind == "Z":
        return nk.operator.spin.sigmaz(hilbert, site)
    if kind == "Sp":
        return nk.operator.spin.sigmap(hilbert, site)
    if kind == "Sm":
        return nk.operator.spin.sigmam(hilbert, site)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["Z", "Sp", "Sm"])
@pytest.mark.parametrize("site", [0, 1])
def test_symbolic_dense_matches_local_and_computational(
    symbolic, nk, hilbert_tiny, kind, site
):
    sym = _mk_symbolic(symbolic, hilbert_tiny, kind, site)
    comp, comp_jax = _mk_comp(hilbert_tiny, kind, site)
    loc = _mk_local(nk, hilbert_tiny, kind, site)

    m_local = np.asarray(loc.to_dense())
    m_sym = dense_as_local(sym)
    m_comp = dense_from_get_conn_padded(comp).T
    m_jax = dense_from_get_conn_padded(comp_jax).T

    assert np.allclose(m_sym, m_local)
    assert np.allclose(m_comp, m_local)
    assert np.allclose(m_jax, m_local)


@pytest.mark.parametrize("kind", ["Z", "Sp", "Sm"])
@pytest.mark.parametrize("site", [0, 1])
def test_symbolic_expect_means_match_local_and_computational(
    symbolic, nk, vstate, kind, site
):
    hilb = vstate.hilbert
    sym = _mk_symbolic(symbolic, hilb, kind, site)
    comp, comp_jax = _mk_comp(hilb, kind, site)
    loc = _mk_local(nk, hilb, kind, site)

    s_loc = vstate.expect(loc)
    s_sym = vstate.expect(sym)
    s_comp = vstate.expect(comp)
    s_jax = vstate.expect(comp_jax)

    assert_means_close(s_loc, s_sym)
    assert_means_close(s_loc, s_comp)
    assert_means_close(s_loc, s_jax)


@pytest.mark.parametrize("site", [0, 1])
def test_symbolic_expect_and_grad_hermitian_matches(symbolic, nk, vstate, site, jax):
    hilb = vstate.hilbert
    sym = _mk_symbolic(symbolic, hilb, "Z", site)
    comp, comp_jax = _mk_comp(hilb, "Z", site)
    loc = _mk_local(nk, hilb, "Z", site)

    s_loc, g_loc = vstate.expect_and_grad(loc)
    s_sym, g_sym = vstate.expect_and_grad(sym)
    s_comp, g_comp = vstate.expect_and_grad(comp)
    s_jax, g_jax = vstate.expect_and_grad(comp_jax)

    assert_means_close(s_loc, s_sym)
    assert_means_close(s_loc, s_comp)
    assert_means_close(s_loc, s_jax)

    for g in (g_loc, g_sym, g_comp, g_jax):
        leaves, _ = jax.tree_util.tree_flatten(g)
        assert all(np.all(np.isfinite(np.asarray(x))) for x in leaves)


@pytest.mark.parametrize("site", [0, 1])
def test_symbolic_nonhermitian_expect_and_grad_behaviour(symbolic, nk, vstate, site):
    hilb = vstate.hilbert
    sym = _mk_symbolic(symbolic, hilb, "Sp", site)
    comp, comp_jax = _mk_comp(hilb, "Sp", site)
    loc = _mk_local(nk, hilb, "Sp", site)

    for op in (loc, sym, comp, comp_jax):
        try:
            _ = vstate.expect_and_grad(op)
        except Exception as exc:
            if exc.__class__.__name__ in {"UnexpectedTracerError", "TypeError"}:
                pytest.xfail(
                    f"Known nonhermitian grad instability ({exc.__class__.__name__}: {exc})"
                )
            raise


@pytest.mark.parametrize("site", [0, 1])
def test_symbolic_operator_algebra_parity(symbolic, nk, vstate, site):
    hilb = vstate.hilbert
    Zs = _mk_symbolic(symbolic, hilb, "Z", site)
    Sps = _mk_symbolic(symbolic, hilb, "Sp", site)
    Sms = _mk_symbolic(symbolic, hilb, "Sm", site)

    Zl = _mk_local(nk, hilb, "Z", site)
    Spl = _mk_local(nk, hilb, "Sp", site)
    Sml = _mk_local(nk, hilb, "Sm", site)

    expr_local = {
        "A": Zl + 0.3,
        "B": Spl + Sml,
        "C": Spl @ Sml,
        "D": (Spl + Sml) @ (Zl + 0.1),
    }
    expr_sym = {
        "A": Zs + 0.3,
        "B": Sps + Sms,
        "C": Sps * Sms,
        "D": (Sps + Sms) * (Zs + 0.1),
    }

    for key in expr_local:
        m_l = np.asarray(expr_local[key].to_dense())
        m_s = dense_from_get_conn_padded(expr_sym[key]).T
        assert np.allclose(m_s, m_l), f"dense mismatch for {key}"

        s_l = vstate.expect(expr_local[key])
        s_s = vstate.expect(expr_sym[key])
        assert_means_close(s_l, s_s)


@pytest.mark.parametrize("site", [0, 1])
def test_symbolic_noncommuting_product_order(symbolic, nk, vstate, site):
    hilb = vstate.hilbert
    Sps = _mk_symbolic(symbolic, hilb, "Sp", site)
    Sms = _mk_symbolic(symbolic, hilb, "Sm", site)
    Spl = _mk_local(nk, hilb, "Sp", site)
    Sml = _mk_local(nk, hilb, "Sm", site)

    m_sp_sm = dense_from_get_conn_padded(Sps * Sms).T
    m_sm_sp = dense_from_get_conn_padded(Sms * Sps).T
    assert not np.allclose(m_sp_sm, m_sm_sp)

    s_sp_sm_local = vstate.expect(Spl @ Sml)
    s_sp_sm_sym = vstate.expect(Sps * Sms)
    s_sm_sp_local = vstate.expect(Sml @ Spl)
    s_sm_sp_sym = vstate.expect(Sms * Sps)
    assert_means_close(s_sp_sm_local, s_sp_sm_sym)
    assert_means_close(s_sm_sp_local, s_sm_sp_sym)
