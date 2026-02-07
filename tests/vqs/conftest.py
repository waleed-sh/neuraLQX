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

import pytest
import functools
import flax.linen as nn


@functools.lru_cache(maxsize=None)
def _jax():
    return pytest.importorskip("jax")


@functools.lru_cache(maxsize=None)
def _jnp():
    jax = _jax()
    import jax.numpy as jnp

    return jnp


class AffineLogPsi(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(_jnp().float32)
        w = self.param("w", nn.initializers.normal(stddev=0.1), (x.shape[-1],))
        b = self.param("b", nn.initializers.zeros, ())
        y = _jnp().dot(x, w) + b
        return y.astype(_jnp().complex64)


def stats_mean(stats):
    return _jnp().asarray(stats.Mean)


def tree_allclose(a, b, *, rtol=1e-6, atol=1e-7):
    leaves_a, treedef_a = _jax().tree_util.tree_flatten(a)
    leaves_b, treedef_b = _jax().tree_util.tree_flatten(b)
    assert treedef_a == treedef_b, "Pytree structure mismatch"
    for la, lb in zip(leaves_a, leaves_b):
        assert _jnp().allclose(la, lb, rtol=rtol, atol=atol), (la, lb)


@pytest.fixture
def hilbert_spin_2(nk):
    return nk.hilbert.Spin(s=0.5, N=2)


@pytest.fixture
def sampler_local_2chains(hilbert_spin_2, nk):
    return nk.sampler.MetropolisLocal(hilbert_spin_2, n_chains=2)


@pytest.fixture
def mcstate(sampler_local_2chains, nqx):
    model = AffineLogPsi()
    st = nqx.vqs.MCState(
        sampler_local_2chains,
        model,
        n_samples=128,
        n_discard_per_chain=0,
        seed=123,
        sampler_seed=456,
    )
    st.sample(n_samples=128, n_discard_per_chain=0)
    assert st.samples is not None
    return st


@pytest.fixture
def ops_spin_2(hilbert_spin_2, nk):
    sx0 = nk.operator.spin.sigmax(hilbert_spin_2, 0)
    sz0 = nk.operator.spin.sigmaz(hilbert_spin_2, 0)
    sz1 = nk.operator.spin.sigmaz(hilbert_spin_2, 1)
    return sx0, sz0, sz1


@pytest.fixture
def helpers():
    return stats_mean, tree_allclose
