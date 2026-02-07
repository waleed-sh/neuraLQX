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


import itertools
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pytest


def _it_product(*iters: Iterable[int]) -> Iterable[Tuple[int, ...]]:
    return itertools.product(*iters)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def nk():
    return pytest.importorskip("netket")


@pytest.fixture(scope="session")
def jax():
    return pytest.importorskip("jax")


@pytest.fixture(scope="session")
def jnp(jax):
    return pytest.importorskip("jax.numpy")


@pytest.fixture(scope="session")
def static_range_small(nk):
    from netket.utils import StaticRange

    return StaticRange(start=0, step=1, length=3)


@dataclass(frozen=True)
class _DummyUnconstrainedSpace:

    hilbert: object
    allowed_basis_states: object
    size: int
    dtype: object


@pytest.fixture(scope="function")
def unconstrained_space(nk, static_range_small, jnp):

    N = 5
    h = nk.hilbert.HomogeneousHilbert(local_states=static_range_small, N=N)
    return _DummyUnconstrainedSpace(
        hilbert=h,
        allowed_basis_states=static_range_small,
        size=N,
        dtype=jnp.int32,
    )


@dataclass(frozen=True)
class _GaugeFixing:
    free: np.ndarray


class _DummyGaugeFixedSpace:

    def __init__(
        self, *, local_states, gauge_dimensions: int, edges_per_copy: int, free_idx, jnp
    ):
        self.allowed_basis_states = local_states
        self.gauge_dimensions = int(gauge_dimensions)
        self.tiny_size = int(edges_per_copy)
        self.size = self.gauge_dimensions * self.tiny_size
        self.dtype = jnp.int32
        self.gauge_fixing = _GaugeFixing(free=np.asarray(free_idx, dtype=np.int64))

        class _NoIndexHilbert:
            def states_to_numbers(self, *_args, **_kwargs):
                raise RuntimeError(
                    "This dummy gauge-fixed space is not NetKet-indexable."
                )

            def numbers_to_states(self, *_args, **_kwargs):
                raise RuntimeError(
                    "This dummy gauge-fixed space is not NetKet-indexable."
                )

        self.hilbert = _NoIndexHilbert()

    @property
    def _L(self) -> int:
        return int(self.allowed_basis_states.length)

    def reimpose_gauge_fixing(self, sigma):
        import jax.numpy as jnp

        v = jnp.asarray(sigma, dtype=self.dtype).reshape(
            (-1, self.gauge_dimensions, self.tiny_size)
        )
        free = self.gauge_fixing.free
        if free.shape[0] < 2:
            return v.reshape((-1, self.size))

        L = self._L
        f0 = v[:, :, int(free[0])]
        f1 = v[:, :, int(free[1])]

        v = v.at[:, :, 1].set((f0 + f1) % L)
        if self.tiny_size > 3:
            v = v.at[:, :, 3].set((f0 - f1) % L)

        return v.reshape((-1, self.size))

    def check_states(self, sigma):
        import jax.numpy as jnp

        v = jnp.asarray(sigma, dtype=self.dtype).reshape(
            (-1, self.gauge_dimensions, self.tiny_size)
        )
        free = self.gauge_fixing.free
        if free.shape[0] < 2:
            return jnp.ones((v.shape[0],), dtype=bool)

        L = self._L
        f0 = v[:, :, int(free[0])]
        f1 = v[:, :, int(free[1])]

        c1 = v[:, :, 1] == ((f0 + f1) % L)
        if self.tiny_size > 3:
            c2 = v[:, :, 3] == ((f0 - f1) % L)
            ok = jnp.all(c1 & c2, axis=1)
        else:
            ok = jnp.all(c1, axis=1)
        return ok


@pytest.fixture(scope="function")
def gauge_fixed_space(nk, jnp):
    from netket.utils import StaticRange

    local_states = StaticRange(start=0, step=1, length=2)
    return _DummyGaugeFixedSpace(
        local_states=local_states,
        gauge_dimensions=2,
        edges_per_copy=4,
        free_idx=[0, 2],
        jnp=jnp,
    )


@pytest.fixture(scope="function")
def gauge_fixed_space_small(gauge_fixed_space):
    return gauge_fixed_space


def enumerate_all_digits(L: int, D: int):
    yield from _it_product(*([range(L)] * D))


def build_full_states_from_free_digits(
    space, free_digits_batch: np.ndarray
) -> np.ndarray:
    import jax.numpy as jnp

    sigma = space.reimpose_gauge_fixing(
        jnp.asarray(free_digits_batch, dtype=space.dtype)
    )
    return np.asarray(sigma)
