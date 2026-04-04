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

import os

import numpy as np
import pytest

os.environ.setdefault("NQX_EXPERIMENTAL", "1")
os.environ.setdefault("NETKET_EXPERIMENTAL", "1")


@pytest.fixture(scope="session")
def symbolic(nqx):
    return nqx.experimental.operators.symbolic


@pytest.fixture(scope="session")
def hilbert_spin1(nk):
    return nk.hilbert.Spin(s=1, N=3)


@pytest.fixture(scope="session")
def hilbert_fock_small(nk):
    return nk.hilbert.Fock(n_max=2, N=2)


@pytest.fixture(scope="session")
def hilbert_fock_one(nk):
    return nk.hilbert.Fock(n_max=2, N=1)


@pytest.fixture(scope="session")
def spin1_all_states(hilbert_spin1):
    return np.asarray(hilbert_spin1.all_states(), dtype=np.int64)
