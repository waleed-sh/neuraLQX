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


import ast
from dataclasses import dataclass
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def jax():
    return pytest.importorskip("jax")


@pytest.fixture(scope="session")
def jnp(jax):
    return pytest.importorskip("jax.numpy")


@pytest.fixture(scope="session")
def nk():
    return pytest.importorskip("netket")


def _parse_edge_term(term: str):

    s = term.strip()
    coef = 1

    if "*" in s:
        left, right = s.split("*", 1)
        coef = int(left.strip())
        tup = ast.literal_eval(right.strip())
        return coef, tup

    if s.startswith("-"):
        coef = -1
        tup = ast.literal_eval(s[1:].strip())
        return coef, tup

    if s.startswith("+"):
        tup = ast.literal_eval(s[1:].strip())
        return 1, tup

    tup = ast.literal_eval(s)
    return 1, tup


@pytest.fixture(scope="session")
def parse_edge_term():

    return _parse_edge_term


@dataclass(frozen=True)
class DummyLocalStates:

    values: np.ndarray

    def all_states(self):
        return self.values
