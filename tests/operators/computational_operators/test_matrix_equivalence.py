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


from tests.operators.helpers import dense_from_get_conn_padded, contains_comp_leaf
from tests.operators.toy_ops import (
    KetSigmaz,
    KetSigmap,
    KetSigmam,
    KetSigmazJax,
    KetSigmapJax,
    KetSigmamJax,
)


@pytest.mark.parametrize("site", [0, 1])
def test_base_operator_dense_matches_local_up_to_transpose(hilbert_tiny, site, nk):
    hilb = hilbert_tiny

    local = {
        "Z": nk.operator.spin.sigmaz(hilb, site).to_dense(),
        "Sp": nk.operator.spin.sigmap(hilb, site).to_dense(),
        "Sm": nk.operator.spin.sigmam(hilb, site).to_dense(),
    }
    comp = {
        "Z": KetSigmaz(hilb, site),
        "Sp": KetSigmap(hilb, site),
        "Sm": KetSigmam(hilb, site),
    }
    jaxop = {
        "Z": KetSigmazJax(hilb, site),
        "Sp": KetSigmapJax(hilb, site),
        "Sm": KetSigmamJax(hilb, site),
    }

    for k in local:
        M_c = dense_from_get_conn_padded(comp[k]).T
        M_j = dense_from_get_conn_padded(jaxop[k]).T
        assert np.allclose(M_c, local[k])
        assert np.allclose(M_j, local[k])


@pytest.mark.parametrize("site", [0, 1])
def test_composite_dense_matches_local(hilbert_tiny, site, nk):
    hilb = hilbert_tiny

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
        "B": Zl - 0.7,
        "C": Spl + Sml,
        "D": Spl - Sml,
        "E": Zl * Zl,
        "F": Spl * Sml,
        "G": Sml * Spl,
        "H": (Spl + Sml) * (Zl + 0.1),
    }
    exprs_comp = {
        "A": Zc + 0.3,
        "B": Zc - 0.7,
        "C": Spc + Smc,
        "D": Spc - Smc,
        "E": Zc * Zc,
        "F": Spc * Smc,
        "G": Smc * Spc,
        "H": (Spc + Smc) * (Zc + 0.1),
    }
    exprs_jax = {
        "A": Zj + 0.3,
        "B": Zj - 0.7,
        "C": Spj + Smj,
        "D": Spj - Smj,
        "E": Zj * Zj,
        "F": Spj * Smj,
        "G": Smj * Spj,
        "H": (Spj + Smj) * (Zj + 0.1),
    }

    for key in exprs_local:
        M_L = exprs_local[key].to_dense()
        M_C = dense_from_get_conn_padded(exprs_comp[key])
        M_J = dense_from_get_conn_padded(exprs_jax[key])

        if contains_comp_leaf(exprs_comp[key]):
            M_C = M_C.T
        if contains_comp_leaf(exprs_jax[key]):
            M_J = M_J.T

        assert np.allclose(M_C, M_L), f"comp matrix mismatch for {key}"
        assert np.allclose(M_J, M_L), f"jax matrix mismatch for {key}"
