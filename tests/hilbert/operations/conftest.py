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


class DummyLocalStates:
    def __init__(self, values):
        self._values = np.asarray(values)

    def all_states(self):
        return self._values


def make_concrete_space(
    AbstractHilbertSpace,
    *,
    allowed_basis_states,
    size: int,
    dtype,
    q_min=None,
    q_max=None,
    q_step=None,
):
    size_val = int(size)
    dtype_val = dtype
    allowed_val = allowed_basis_states

    abstract = getattr(AbstractHilbertSpace, "__abstractmethods__", set())

    def __init__(self):
        self._allowed_basis_states = allowed_val
        self._size = size_val
        self._dtype = dtype_val

        self._dimensions = size_val

        if q_min is not None:
            self._q_min = q_min
        if q_max is not None:
            self._q_max = q_max
        if q_step is not None:
            self._q_step = q_step

    def _stub_factory(name: str):
        def _stub(self, *a, **k):
            raise NotImplementedError(f"DummySpace stub '{name}' called")

        return _stub

    def _qprop(attrname: str):
        return property(lambda self: getattr(self, f"_{attrname}"))

    ns = {
        "__init__": __init__,
        "allowed_basis_states": property(lambda self: self._allowed_basis_states),
        "size": property(lambda self: self._size),
        "dtype": property(lambda self: self._dtype),
        "q_min": _qprop("q_min"),
        "q_max": _qprop("q_max"),
        "q_step": _qprop("q_step"),
    }

    for name in abstract:
        if name in ns:
            continue
        ns[name] = _stub_factory(name)

    DummySpace = type("DummySpace", (AbstractHilbertSpace,), ns)
    return DummySpace()


@pytest.fixture(scope="function")
def unconstrained_space(jnp):
    from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace

    local = DummyLocalStates(values=[-1, 0, 1])
    return make_concrete_space(
        AbstractHilbertSpace,
        allowed_basis_states=local,
        size=8,
        dtype=jnp.int32,
    )


@pytest.fixture(scope="function")
def u1_like_space(jnp):
    from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace

    local = DummyLocalStates(values=[-1, 0, 1])
    return make_concrete_space(
        AbstractHilbertSpace,
        allowed_basis_states=local,
        size=8,
        dtype=jnp.int32,
        q_min=-1,
        q_max=1,
        q_step=1,
    )
