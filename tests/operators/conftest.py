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

from tests.operators.helpers import make_mcstate


@pytest.fixture(scope="session")
def seed():
    return 1234


@pytest.fixture(scope="session")
def hilbert_small(nk):
    return nk.hilbert.Spin(s=1 / 2, N=4)


@pytest.fixture(scope="session")
def hilbert_tiny(nk):
    return nk.hilbert.Spin(s=1 / 2, N=3)


@pytest.fixture(autouse=True)
def _jit_policy(request, jax):

    prev = jax.config.jax_disable_jit
    want_jit = request.node.get_closest_marker("jaxjit") is not None

    jax.config.update("jax_disable_jit", not want_jit)
    try:
        yield
    finally:
        jax.config.update("jax_disable_jit", prev)


@pytest.fixture(scope="function")
def vstate(hilbert_small, seed):
    st = make_mcstate(hilbert_small, seed=seed, n_samples=8192, n_chains=64)
    _ = np.asarray(st.samples)
    return st


@pytest.fixture(scope="function")
def vstate_tiny(hilbert_tiny, seed):
    st = make_mcstate(hilbert_tiny, seed=seed, n_samples=4096, n_chains=32)
    _ = np.asarray(st.samples)
    return st
