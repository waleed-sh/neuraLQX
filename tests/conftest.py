# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np
import pytest
from pathlib import Path
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Setting `jax_pmap_shmap_merge` is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Implicit conversion of an array to a dtype is deprecated.*",
    category=DeprecationWarning,
)


def _enable_experimental():
    os.environ["NQX_EXPERIMENTAL"] = "1"
    os.environ["NETKET_EXPERIMENTAL"] = "1"


if os.getenv("NQX_TESTING") == "1":
    _enable_experimental()


@pytest.fixture(scope="session")
def nk():
    nk = pytest.importorskip("netket")
    nk.config.netket_experimental = True
    return nk


@pytest.fixture(scope="session")
def jax():
    return pytest.importorskip("jax")


@pytest.fixture(scope="session")
def jnp(jax):
    import jax.numpy as jnp

    return jnp


@pytest.fixture(scope="session")
def nqx(jax):
    import neuralqx as nqx

    return nqx


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path_writable(tmp_path: Path) -> Path:
    return tmp_path
