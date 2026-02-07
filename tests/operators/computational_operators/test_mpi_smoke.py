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
import numpy as np
import pytest

from tests.operators.helpers import make_mcstate
from tests.operators.toy_ops import KetSigmaz


@pytest.mark.mpi
def test_mpi_expect_smoke(nk):
    if (
        os.environ.get("NQX_MPI_TESTS", "0") != "1"
        and os.environ.get("NQX_MPI", "0") != "1"
    ):
        pytest.skip("MPI tests not enabled (set NQX_MPI_TESTS=1).")

    pytest.importorskip("mpi4py")

    hilb = nk.hilbert.Spin(s=1 / 2, N=4)
    st = make_mcstate(hilb, seed=0, n_samples=2048, n_chains=32)
    _ = np.asarray(st.samples)

    op = KetSigmaz(hilb, 0)
    stats = st.expect(op)
    mean = getattr(stats, "Mean", getattr(stats, "mean"))
    assert np.all(np.isfinite(np.asarray(mean)))
