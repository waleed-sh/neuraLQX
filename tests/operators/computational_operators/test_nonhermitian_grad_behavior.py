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

import os
import pytest


from tests.operators.toy_ops import KetSigmap, KetSigmapJax


@pytest.mark.parametrize("site", [0, 1])
def test_nonhermitian_expect_and_grad_is_gated_or_xfails(vstate, site, nk):
    hilb = vstate.hilbert
    SpL = nk.operator.spin.sigmap(hilb, site)
    SpC = KetSigmap(hilb, site)
    SpJ = KetSigmapJax(hilb, site)

    experimental = os.environ.get("NETKET_EXPERIMENTAL", "0") == "1"

    if not experimental:
        with pytest.raises(RuntimeError):
            _ = vstate.expect_and_grad(SpL)
        with pytest.raises(RuntimeError):
            _ = vstate.expect_and_grad(SpC)
        with pytest.raises(RuntimeError):
            _ = vstate.expect_and_grad(SpJ)
        return

    _ = vstate.expect_and_grad(SpL)
    _ = vstate.expect_and_grad(SpC)

    try:
        _ = vstate.expect_and_grad(SpJ)
    except TypeError as e:
        pytest.xfail(
            f"Known limitation: nonhermitian grad traces operator through JAX: {e}"
        )
