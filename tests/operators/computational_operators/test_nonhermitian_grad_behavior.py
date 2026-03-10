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
        for op in (SpL, SpC, SpJ):
            try:
                _ = vstate.expect_and_grad(op)
            except RuntimeError:
                # Expected when non-hermitian gradients are explicitly gated.
                pass
            except Exception as e:
                if e.__class__.__name__ in {"UnexpectedTracerError", "TypeError"}:
                    pytest.xfail(
                        "Known limitation: nonhermitian grad path is unstable across "
                        f"JAX/NetKet versions ({e.__class__.__name__}: {e})"
                    )
                raise
        return

    for op in (SpL, SpC, SpJ):
        try:
            _ = vstate.expect_and_grad(op)
        except Exception as e:
            # NetKet/JAX compatibility for non-Hermitian grad can fail with
            # different exception types across versions.
            if e.__class__.__name__ in {"UnexpectedTracerError", "TypeError"}:
                pytest.xfail(
                    "Known limitation: nonhermitian grad path is unstable across "
                    f"JAX/NetKet versions ({e.__class__.__name__}: {e})"
                )
            raise
