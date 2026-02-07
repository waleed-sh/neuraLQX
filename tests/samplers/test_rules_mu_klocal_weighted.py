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

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neuralqx.samplers.rules._mu_sampler import MuSampler
from neuralqx.samplers.rules._metropolis_samplers import MetropolisKLocalRule


class _DummySampler:
    def __init__(self, hilbert, *, n_chains_per_rank: int):
        self.hilbert = hilbert
        self.n_chains_per_rank = n_chains_per_rank


def test_mu_sampler_enforces_no_zero_on_forbidden_indices():

    class _Graph:
        edges_mu_idx = jnp.array([1, 3], dtype=jnp.int32)

    class _Hilb:
        size = 5
        local_states = (-1, 0, 1)
        graph = _Graph()

    B = 32
    hilb = _Hilb()
    sampler = _DummySampler(hilb, n_chains_per_rank=B)

    sigma = jnp.zeros((B, hilb.size), dtype=jnp.int32)

    rule = MuSampler()
    key = jax.random.PRNGKey(0)
    sigma_p, logcorr = rule.transition(
        sampler, machine=None, parameters=None, state=None, key=key, σ=sigma
    )

    assert logcorr is None
    assert sigma_p.shape == sigma.shape

    forb = hilb.graph.edges_mu_idx
    assert not bool(
        jnp.any(sigma_p[:, forb] == 0)
    ), "MuSampler failed to enforce non-zero values on forbidden indices"

    allowed = jnp.array(hilb.local_states, dtype=sigma_p.dtype)
    ok = jnp.any(sigma_p[..., None] == allowed[None, None, :], axis=-1)
    assert bool(jnp.all(ok)), "MuSampler produced values outside hilb.local_states"


def test_metropolis_klocal_rule_applies_flip_state_k_times(monkeypatch):

    import neuralqx.samplers.rules._metropolis_samplers as mmod

    calls = {"n": 0, "indices_shapes": []}

    def fake_flip_state(hilb, key, sigma, indices):
        calls["n"] += 1
        calls["indices_shapes"].append(tuple(indices.shape))

        v = sigma
        idx = indices.astype(jnp.int32)

        def update_one(chain_state, site):
            x = chain_state[site]
            x01 = x + 1
            x01 = (x01 + 1) % 3
            x_new = x01 - 1
            return chain_state.at[site].set(x_new)

        v2 = jax.vmap(update_one)(v, idx)
        return v2, None

    monkeypatch.setattr(
        mmod.nk.hilbert.random, "flip_state", fake_flip_state, raising=True
    )

    class _Hilb:
        size = 7

    B = 16
    hilb = _Hilb()
    sampler = _DummySampler(hilb, n_chains_per_rank=B)

    sigma = jnp.zeros((B, hilb.size), dtype=jnp.int32)
    rule = MetropolisKLocalRule(n_flips=5)

    key = jax.random.PRNGKey(0)
    sigma_p, logcorr = rule.transition(
        sampler, machine=None, params=None, sampler_state=None, key=key, σ=sigma
    )

    assert logcorr is None
    assert sigma_p.shape == sigma.shape
    assert calls["n"] == 5, "flip_state was not called exactly n_flips times"
    assert all(
        s == (B,) for s in calls["indices_shapes"]
    ), "indices shape mismatch per flip"


def test_metropolis_klocal_rule_identity_when_n_flips_zero(monkeypatch):

    class _Hilb:
        size = 4

    B = 8
    sampler = _DummySampler(_Hilb(), n_chains_per_rank=B)
    sigma = jnp.array([[0, 1, 0, -1]] * B, dtype=jnp.int32)

    rule = MetropolisKLocalRule(n_flips=0)
    sigma_p, logcorr = rule.transition(
        sampler,
        machine=None,
        params=None,
        sampler_state=None,
        key=jax.random.PRNGKey(0),
        σ=sigma,
    )
    assert logcorr is None
    assert jnp.array_equal(sigma_p, sigma)
