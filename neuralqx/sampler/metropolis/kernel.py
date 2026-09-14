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


"""Metropolis-Hastings transition kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.sampler.metropolis.validation import assert_sample_shape
from neuralqx.sampler.rules import normalize_transition_output
from neuralqx.sampler.state import MetropolisSamplerState


def metropolis_sweep(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
) -> MetropolisSamplerState:
    """Run one vectorized Metropolis-Hastings proposal over all chains."""
    rng, key_transition, key_accept = jax.random.split(state.rng, 3)
    transition = normalize_transition_output(
        sampler.rule.transition(
            sampler,
            apply_fn,
            parameters,
            state,
            key_transition,
            state.sigma,
        ),
        state.rule_state,
    )

    sigma_p = jnp.asarray(transition.sigma, dtype=state.sigma.dtype)
    assert_sample_shape(sigma_p, state.sigma.shape[0], sampler.hilbert.size)

    proposal_log_prob = sampler._evaluate_log_probability(
        apply_fn,
        parameters,
        sigma_p,
    )
    log_ratio = proposal_log_prob - state.log_prob
    if transition.log_correction is not None:
        correction = jnp.asarray(transition.log_correction, dtype=log_ratio.dtype)
        log_ratio = log_ratio + correction

    accept = jax.random.uniform(key_accept, shape=log_ratio.shape) < jnp.exp(log_ratio)
    sigma = jnp.where(accept[:, None], sigma_p, state.sigma)
    log_prob = jnp.where(accept, proposal_log_prob, state.log_prob)
    step_increment = jnp.asarray(
        sampler.n_chains, dtype=jnp.asarray(state.n_steps).dtype
    )
    accepted = accept.astype(jnp.asarray(state.n_accepted_per_chain).dtype)

    return state.replace(
        sigma=sigma,
        log_prob=log_prob,
        rng=rng,
        rule_state=transition.rule_state,
        n_steps=state.n_steps + step_increment,
        n_accepted_per_chain=state.n_accepted_per_chain + accepted,
    )


def sample_next(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
) -> tuple[MetropolisSamplerState, tuple[jax.Array, jax.Array]]:
    """Sample one stored chain point after ``sweep_size`` proposals."""

    def body(_idx: int, current: MetropolisSamplerState) -> MetropolisSamplerState:
        return metropolis_sweep(sampler, apply_fn, parameters, current)

    new_state = jax.lax.fori_loop(0, int(sampler.sweep_size), body, state)
    return new_state, (new_state.sigma, new_state.log_prob)


def sample_chain(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
    chain_length: int,
) -> tuple[MetropolisSamplerState, tuple[jax.Array, jax.Array]]:
    """Scan ``sample_next`` for a fixed chain length."""
    state, (samples, log_probabilities) = jax.lax.scan(
        lambda carry, _unused: sample_next(sampler, apply_fn, parameters, carry),
        state,
        xs=None,
        length=int(chain_length),
    )
    return state, (jnp.swapaxes(samples, 0, 1), jnp.swapaxes(log_probabilities, 0, 1))


__all__ = ["metropolis_sweep", "sample_chain", "sample_next"]
