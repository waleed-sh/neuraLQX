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


from tests.operators.helpers import (
    dense_from_get_conn_padded,
    contains_comp_leaf,
    exact_wavefunction,
    exact_expectation_from_dense,
    assert_means_close,
    stats_as_dict,
)
from tests.operators.toy_ops import (
    KetSigmaz,
    KetSigmap,
    KetSigmam,
    KetSigmazJax,
    KetSigmapJax,
    KetSigmamJax,
)


def _dense(op):
    if hasattr(op, "to_dense"):
        return np.asarray(op.to_dense())
    M = dense_from_get_conn_padded(op)
    if contains_comp_leaf(op):
        M = M.T
    return M


@pytest.mark.parametrize("site", [0, 1])
def test_algebraic_expressions_match_means_and_exact(vstate, site, nk):
    hilb = vstate.hilbert

    Zl = nk.operator.spin.sigmaz(hilb, site)
    Spl = nk.operator.spin.sigmap(hilb, site)
    Sml = nk.operator.spin.sigmam(hilb, site)

    Zc = KetSigmaz(hilb, site)
    Spc = KetSigmap(hilb, site)
    Smc = KetSigmam(hilb, site)

    Zj = KetSigmazJax(hilb, site)
    Spj = KetSigmapJax(hilb, site)
    Smj = KetSigmamJax(hilb, site)

    exprs_local = {
        "A": Zl + 0.3,
        "B": 0.3 + Zl,
        "C": Zl - 0.7,
        "D": 0.7 - Zl,
        "E": Spl + Sml,
        "F": Spl - Sml,
        "G": Zl + Spl - 0.2,
        "H": (Spl + Sml) * (Zl + 0.1),
        "I": Spl * Sml,
        "J": Sml * Spl,
        "K": Zl * Zl,
    }
    exprs_comp = {
        "A": Zc + 0.3,
        "B": 0.3 + Zc,
        "C": Zc - 0.7,
        "D": 0.7 - Zc,
        "E": Spc + Smc,
        "F": Spc - Smc,
        "G": Zc + Spc - 0.2,
        "H": (Spc + Smc) * (Zc + 0.1),
        "I": Spc * Smc,
        "J": Smc * Spc,
        "K": Zc * Zc,
    }
    exprs_jax = {
        "A": Zj + 0.3,
        "B": 0.3 + Zj,
        "C": Zj - 0.7,
        "D": 0.7 - Zj,
        "E": Spj + Smj,
        "F": Spj - Smj,
        "G": Zj + Spj - 0.2,
        "H": (Spj + Smj) * (Zj + 0.1),
        "I": Spj * Smj,
        "J": Smj * Spj,
        "K": Zj * Zj,
    }

    psi, _ = exact_wavefunction(vstate)

    for key in exprs_local:
        L = exprs_local[key]
        C = exprs_comp[key]
        J = exprs_jax[key]

        exact = exact_expectation_from_dense(psi, _dense(L))

        sL = vstate.expect(L)
        sC = vstate.expect(C)
        sJ = vstate.expect(J)

        assert_means_close(sL, sC)
        assert_means_close(sL, sJ)

        for s in (sL, sC, sJ):
            d = stats_as_dict(s)
            mean = np.asarray(d["mean"])
            sigma = 0.0 if d["sigma"] is None else float(np.max(np.asarray(d["sigma"])))
            tol = max(3e-3, 8.0 * sigma)
            diff = mean - exact
            if np.iscomplexobj(diff):
                assert (
                    abs(diff.real) <= tol and abs(diff.imag) <= tol
                ), f"{key}: mean {mean} vs exact {exact} (tol={tol})"
            else:
                assert (
                    abs(float(diff)) <= tol
                ), f"{key}: mean {mean} vs exact {exact} (tol={tol})"
