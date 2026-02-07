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


def _dense_for_operator(op):
    import numpy as np

    if hasattr(op, "to_dense"):
        return np.asarray(op.to_dense())
    M = dense_from_get_conn_padded(op)
    if contains_comp_leaf(op):
        M = M.T
    return M


@pytest.mark.parametrize("site", [0, 1])
def test_expect_means_match_across_types_and_against_exact(vstate, site, nk):
    hilb = vstate.hilbert

    ops_local = {
        "Z": nk.operator.spin.sigmaz(hilb, site),
        "Sp": nk.operator.spin.sigmap(hilb, site),
        "Sm": nk.operator.spin.sigmam(hilb, site),
    }
    ops_comp = {
        "Z": KetSigmaz(hilb, site),
        "Sp": KetSigmap(hilb, site),
        "Sm": KetSigmam(hilb, site),
    }
    ops_jax = {
        "Z": KetSigmazJax(hilb, site),
        "Sp": KetSigmapJax(hilb, site),
        "Sm": KetSigmamJax(hilb, site),
    }

    psi, _ = exact_wavefunction(vstate)

    for k in ["Z", "Sp", "Sm"]:
        M_exact = _dense_for_operator(ops_local[k])
        exact = exact_expectation_from_dense(psi, M_exact)

        sL = vstate.expect(ops_local[k])
        sC = vstate.expect(ops_comp[k])
        sJ = vstate.expect(ops_jax[k])

        assert_means_close(sL, sC)
        assert_means_close(sL, sJ)
        assert_means_close(sC, sJ)

        for s in (sL, sC, sJ):
            d = stats_as_dict(s)
            mean = np.asarray(d["mean"])
            sigma = 0.0 if d["sigma"] is None else float(np.max(np.asarray(d["sigma"])))
            tol = max(3e-3, 8.0 * sigma)
            diff = mean - exact
            if np.iscomplexobj(diff):
                assert (
                    abs(diff.real) <= tol and abs(diff.imag) <= tol
                ), f"{k}: mean {mean} vs exact {exact} (tol={tol})"
            else:
                assert (
                    abs(float(diff)) <= tol
                ), f"{k}: mean {mean} vs exact {exact} (tol={tol})"
