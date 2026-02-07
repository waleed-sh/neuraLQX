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
    exact_wavefunction,
    exact_expectation_from_dense,
    dense_from_get_conn_padded,
    contains_comp_leaf,
    assert_means_close,
)
from tests.operators.toy_ops import KetSigmap, KetSigmam, KetSigmapJax, KetSigmamJax


def _dense(op):
    if hasattr(op, "to_dense"):
        return np.asarray(op.to_dense())
    M = dense_from_get_conn_padded(op)
    if contains_comp_leaf(op):
        M = M.T
    return M


@pytest.mark.parametrize("site", [0, 1])
def test_noncommuting_products_order_and_means_match(vstate, site, nk):
    hilb = vstate.hilbert

    SpL = nk.operator.spin.sigmap(hilb, site)
    SmL = nk.operator.spin.sigmam(hilb, site)

    SpC = KetSigmap(hilb, site)
    SmC = KetSigmam(hilb, site)

    SpJ = KetSigmapJax(hilb, site)
    SmJ = KetSigmamJax(hilb, site)

    psi, _ = exact_wavefunction(vstate)

    exact_SpSm = exact_expectation_from_dense(psi, _dense(SpL * SmL))
    exact_SmSp = exact_expectation_from_dense(psi, _dense(SmL * SpL))
    assert not np.allclose(
        exact_SpSm, exact_SmSp
    ), "Expected non-commuting products to differ"

    for opL, opC, opJ, exact in [
        (SpL * SmL, SpC * SmC, SpJ * SmJ, exact_SpSm),
        (SmL * SpL, SmC * SpC, SmJ * SpJ, exact_SmSp),
    ]:
        sL = vstate.expect(opL)
        sC = vstate.expect(opC)
        sJ = vstate.expect(opJ)

        assert_means_close(sL, sC)
        assert_means_close(sL, sJ)
