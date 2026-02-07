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

from tests.operators.helpers import assert_means_close
from tests.operators.toy_ops import KetSigmaz, KetSigmazJax


@pytest.mark.parametrize("site", [0, 1])
def test_expect_and_grad_hermitian_matches(vstate, site, nk, jax):
    hilb = vstate.hilbert

    Zl = nk.operator.spin.sigmaz(hilb, site)
    Zc = KetSigmaz(hilb, site)
    Zj = KetSigmazJax(hilb, site)

    sL, gL = vstate.expect_and_grad(Zl)
    sC, gC = vstate.expect_and_grad(Zc)
    sJ, gJ = vstate.expect_and_grad(Zj)

    assert_means_close(sL, sC)
    assert_means_close(sL, sJ)

    def _tree_all_finite(pytree):
        leaves, _ = jax.tree_util.tree_flatten(pytree)
        return all(np.all(np.isfinite(np.asarray(x))) for x in leaves)

    assert _tree_all_finite(gL)
    assert _tree_all_finite(gC)
    assert _tree_all_finite(gJ)
