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

import os
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
neuralqx = pytest.importorskip("neuralqx")

from neuralqx.samplers.rules._gauge_samplers import (
    RandomU1GaugeSampler,
    U1GaugeSampler,
    U1GaugeSamplerNonzero,
    U1InvariantPlaquetteSampler,
)


class _DummySampler:
    def __init__(self, hilbert, *, n_chains_per_rank: int = 4, n_batches: int = 7):
        self.hilbert = hilbert
        self.n_chains_per_rank = n_chains_per_rank
        self.n_batches = n_batches


def _assert_within_cutoff(states, cutoff: int):
    assert jnp.all(states <= cutoff), "State exceeds +cutoff"
    assert jnp.all(states >= -cutoff), "State exceeds -cutoff"


def _assert_all_gi(H_nqx, states):
    gi = H_nqx.core.is_gauge_invariant(states)
    assert bool(jnp.all(gi)), "Proposed state left gauge-invariant subspace"


@pytest.fixture(scope="module")
def u1_theta_gi_system():
    os.environ["NQS_VERBOSE"] = "False"

    graph = neuralqx.graph.Graph([(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)])
    H = neuralqx.hilbert.u1.HilbertU1(
        graph,
        cutoff=1,
        is_gauge_invariant=True,
        auto_constraint=1,
        gauge_dimensions=1,
    )
    return H, H.hilbert


@pytest.mark.parametrize(
    "Rule",
    [
        U1GaugeSampler,
        U1InvariantPlaquetteSampler,
        RandomU1GaugeSampler,
    ],
)
def test_gauge_rules_transition_preserves_gauge_invariance_and_cutoff(
    u1_theta_gi_system, Rule
):

    H_nqx, hilb = u1_theta_gi_system
    rule = Rule()

    B = 8
    key = jax.random.PRNGKey(0)
    sigma = H_nqx.random_state(key, size=B)
    _assert_all_gi(H_nqx, sigma)

    sampler = _DummySampler(hilb, n_chains_per_rank=B)
    key, sub = jax.random.split(key)

    sigma_p, logcorr = rule.transition(
        sampler, machine=None, params=None, sampler_state=None, key=sub, σ=sigma
    )

    assert logcorr is None
    assert sigma_p.shape == sigma.shape

    _assert_all_gi(H_nqx, sigma_p)
    _assert_within_cutoff(sigma_p, cutoff=1)


@pytest.mark.parametrize(
    "Rule",
    [
        U1GaugeSampler,
        U1InvariantPlaquetteSampler,
    ],
)
def test_local_gauge_rules_usually_move_some_chain(u1_theta_gi_system, Rule):

    H_nqx, hilb = u1_theta_gi_system
    rule = Rule()
    B = 10

    key = jax.random.PRNGKey(123)
    sigma = H_nqx.random_state(key, size=B)

    sampler = _DummySampler(hilb, n_chains_per_rank=B)

    moved = False
    for _ in range(10):
        key, sub = jax.random.split(key)
        sigma_p, _ = rule.transition(
            sampler, machine=None, params=None, sampler_state=None, key=sub, σ=sigma
        )
        if bool(jnp.any(sigma_p != sigma)):
            moved = True
            break

    assert moved, f"{Rule.__name__} did not move any chain over several proposals"


def test_random_u1_gauge_rule_is_input_independent_in_practice(u1_theta_gi_system):

    H_nqx, hilb = u1_theta_gi_system
    rule = RandomU1GaugeSampler()

    B = 6
    key = jax.random.PRNGKey(7)
    sigma = H_nqx.random_state(key, size=B)
    sampler = _DummySampler(hilb, n_chains_per_rank=B)

    key, k1 = jax.random.split(key)
    s1, _ = rule.transition(sampler, None, None, None, k1, sigma)

    key, k2 = jax.random.split(key)
    s2, _ = rule.transition(sampler, None, None, None, k2, sigma)

    assert bool(jnp.any(s1 != s2)), "Two random proposals unexpectedly identical"


def test_u1_gauge_nonzero_init_and_random_state_never_return_vacuum(u1_theta_gi_system):

    H_nqx, hilb = u1_theta_gi_system
    rule = U1GaugeSamplerNonzero()

    sampler = _DummySampler(hilb, n_batches=12)

    key = jax.random.PRNGKey(0)
    x0 = rule.init_state(sampler, machine=None, params=None, key=key)
    assert x0.shape[0] == 1
    assert not bool(jnp.all(x0 == 0)), "init_state returned vacuum (all-zero) state"

    key, sub = jax.random.split(key)
    xs = rule.random_state(sampler, None, None, None, sub)
    assert xs.shape[0] == sampler.n_batches
    assert not bool(
        jnp.any(jnp.all(xs == 0, axis=1))
    ), "random_state returned vacuum for at least one chain"

    _assert_all_gi(H_nqx, xs)
    _assert_within_cutoff(xs, cutoff=1)


def test_u1_gauge_nonzero_transition_lazy_rejects_vacuum_proposals(monkeypatch):

    rule = U1GaugeSamplerNonzero()

    class _FakeHilb:
        def flip_gi_state_sgd(self, sigma, key, adjacency, number_of_edges):
            return jnp.zeros_like(sigma)

    sampler = _DummySampler(_FakeHilb(), n_chains_per_rank=3)

    key = jax.random.PRNGKey(0)
    sigma = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, -1, 0, 1],
            [0, 1, 0, -1, 1],
        ],
        dtype=jnp.int32,
    )

    sigma_p, logcorr = rule.transition(
        sampler, machine=None, params=None, sampler_state=None, key=key, σ=sigma
    )
    assert logcorr is None
    assert jnp.array_equal(
        sigma_p, sigma
    ), "When proposal is vacuum, rule should keep original state (lazy kernel)."


def test_gauge_random_state_uses_n_batches_for_parallel_tempering(monkeypatch):

    import neuralqx.samplers.rules._gauge_samplers as gsm

    class _FakePTBase:
        pass

    monkeypatch.setattr(gsm, "ParallelTemperingSampler", _FakePTBase, raising=True)

    class _FakePTSampler(_FakePTBase):
        def __init__(self, hilb, n_batches, n_chains_per_rank):
            self.hilbert = hilb
            self.n_batches = n_batches
            self.n_chains_per_rank = n_chains_per_rank

    class _SpyHilb:
        def __init__(self):
            self.last_size = None

        def random_state(self, *, key, size):
            self.last_size = size
            return jnp.zeros((size, 5), dtype=jnp.int32)

    hilb = _SpyHilb()
    sampler = _FakePTSampler(hilb, n_batches=11, n_chains_per_rank=3)

    rule = U1GaugeSampler()
    _ = rule.random_state(sampler, None, None, None, jax.random.PRNGKey(0))
    assert hilb.last_size == 11, "Expected random_state to use sampler.n_batches for PT"
