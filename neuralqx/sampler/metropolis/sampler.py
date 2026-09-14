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


"""Metropolis-Hastings sampler implementation."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.sharding import place_sample_batch
from neuralqx.sampler._samples import n_batches as resolve_n_batches
from neuralqx.sampler._samples import n_chains_per_device as resolve_n_chains_per_device
from neuralqx.sampler._samples import resolve_n_chains
from neuralqx.sampler.base import AbstractSampler
from neuralqx.sampler.metropolis.kernel import sample_chain
from neuralqx.sampler.metropolis.kernel import sample_next
from neuralqx.sampler.metropolis.runtime import initialize_sharded_state_arrays
from neuralqx.sampler.metropolis.runtime import random_state_sharded
from neuralqx.sampler.metropolis.runtime import reset_sharded
from neuralqx.sampler.metropolis.runtime import sample_chain_sharded
from neuralqx.sampler.metropolis.runtime import use_sample_shard_map
from neuralqx.sampler.metropolis.validation import assert_sample_shape
from neuralqx.sampler.metropolis.validation import ensure_metropolis_state
from neuralqx.sampler.metropolis.validation import validate_rule_for_hilbert
from neuralqx.sampler.rules import AbstractTransitionRule
from neuralqx.sampler.rules import LocalRule
from neuralqx.sampler.state import AbstractSamplerState
from neuralqx.sampler.state import MetropolisSamplerState
from neuralqx.utils.struct import field


class MetropolisSampler(AbstractSampler):
    """Metropolis-Hastings sampler with pluggable transition rules."""

    rule: AbstractTransitionRule | None = field(pytree=False, default=None)
    sweep_size: int | None = field(static=True, default=None)
    reset_chains: bool = field(static=True, default=False)
    n_chains: int | None = field(static=True, default=None)
    n_chains_per_device: int | None = field(static=True, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rule is None:
            object.__setattr__(self, "rule", LocalRule())
        if not isinstance(self.rule, AbstractTransitionRule):
            raise TypeError(
                "rule must be an AbstractTransitionRule; "
                f"got {type(self.rule).__name__}."
            )
        validate_rule_for_hilbert(self.rule, self.hilbert.size)
        if not isinstance(self.reset_chains, bool):
            raise TypeError("reset_chains must be a boolean.")

        sweep_size = (
            self.hilbert.size if self.sweep_size is None else int(self.sweep_size)
        )
        if sweep_size <= 0:
            raise ValueError("sweep_size must be positive.")

        n_chains = resolve_n_chains(
            n_chains=self.n_chains,
            n_chains_per_device=self.n_chains_per_device,
            sample_sharding=self.sample_sharding_enabled,
            device_count=jax.device_count(),
        )
        n_chains_per_device = resolve_n_chains_per_device(
            n_chains=n_chains,
            sample_sharding=self.sample_sharding_enabled,
            device_count=jax.device_count(),
        )
        object.__setattr__(self, "sweep_size", sweep_size)
        object.__setattr__(self, "n_chains", n_chains)
        object.__setattr__(self, "n_chains_per_device", n_chains_per_device)

    @property
    def n_batches(self) -> int:
        return resolve_n_batches(
            n_chains=int(self.n_chains),
            sample_sharding=self.sample_sharding_enabled,
            device_count=jax.device_count(),
        )

    def _init_state(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        key: jax.Array,
    ) -> MetropolisSamplerState:
        if use_sample_shard_map(self):
            return self._init_state_sharded(apply_fn, parameters, key)
        return self._init_state_local(apply_fn, parameters, key)

    @partial(jax.jit, static_argnames=("apply_fn",))
    def _init_state_local(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        key: jax.Array,
    ) -> MetropolisSamplerState:
        key_state, key_rule, key_random = jax.random.split(key, 3)
        sigma = place_sample_batch(
            jnp.zeros((self.n_batches, self.hilbert.size), dtype=self.dtype)
        )
        log_prob = place_sample_batch(
            jnp.full((self.n_batches,), -jnp.inf, dtype=jnp.float64)
        )
        state = MetropolisSamplerState(
            sigma=sigma,
            log_prob=log_prob,
            rng=key_state,
            rule_state=self.rule.init_state(self, apply_fn, parameters, key_rule),
            n_steps=jnp.zeros((), dtype=jnp.int32),
            n_accepted_per_chain=place_sample_batch(
                jnp.zeros((self.n_batches,), dtype=jnp.int32)
            ),
        )
        if self.reset_chains:
            return state
        sigma = self.rule.random_state(self, apply_fn, parameters, state, key_random)
        assert_sample_shape(sigma, self.n_batches, self.hilbert.size)
        return state.replace(sigma=place_sample_batch(jnp.asarray(sigma)))

    def _init_state_sharded(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        key: jax.Array,
    ) -> MetropolisSamplerState:
        sigma, log_prob, accepted, rng, key_rule, key_random = (
            initialize_sharded_state_arrays(
                self,
                key,
            )
        )
        state = MetropolisSamplerState(
            sigma=sigma,
            log_prob=log_prob,
            rng=rng,
            rule_state=self.rule.init_state(self, apply_fn, parameters, key_rule),
            n_steps=jnp.zeros((), dtype=jnp.int32),
            n_accepted_per_chain=accepted,
        )
        if self.reset_chains:
            return state
        sigma = random_state_sharded(self, apply_fn, parameters, state, key_random)
        return state.replace(sigma=sigma)

    def _reset(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState,
    ) -> MetropolisSamplerState:
        if use_sample_shard_map(self):
            return reset_sharded(
                self,
                apply_fn,
                parameters,
                ensure_metropolis_state(state),
            )
        return self._reset_local(apply_fn, parameters, state)

    @partial(jax.jit, static_argnames=("apply_fn",))
    def _reset_local(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState,
    ) -> MetropolisSamplerState:
        state = ensure_metropolis_state(state)
        rng = state.rng
        if self.reset_chains:
            rng, key = jax.random.split(rng)
            sigma = self.rule.random_state(self, apply_fn, parameters, state, key)
            assert_sample_shape(sigma, self.n_batches, self.hilbert.size)
            sigma = place_sample_batch(jnp.asarray(sigma))
        else:
            sigma = place_sample_batch(jnp.asarray(state.sigma))

        log_prob = self._evaluate_log_probability(apply_fn, parameters, sigma)
        return state.replace(
            sigma=sigma,
            log_prob=place_sample_batch(log_prob),
            rng=rng,
            rule_state=self.rule.reset(self, apply_fn, parameters, state),
            n_steps=jnp.zeros_like(state.n_steps),
            n_accepted_per_chain=place_sample_batch(
                jnp.zeros_like(state.n_accepted_per_chain)
            ),
        )

    def sample_next(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: MetropolisSamplerState,
    ) -> tuple[MetropolisSamplerState, tuple[jax.Array, jax.Array]]:
        """Sample one stored chain point after ``sweep_size`` proposals."""
        return self._sample_next(apply_fn, parameters, state)

    def _sample_next(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: MetropolisSamplerState,
    ) -> tuple[MetropolisSamplerState, tuple[jax.Array, jax.Array]]:
        return sample_next(self, apply_fn, parameters, state)

    def _sample_chain(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState,
        chain_length: int,
        *,
        return_log_probabilities: bool = False,
    ):
        if use_sample_shard_map(self):
            (samples, log_probabilities), state = sample_chain_sharded(
                self,
                apply_fn,
                parameters,
                ensure_metropolis_state(state),
                int(chain_length),
            )
            if return_log_probabilities:
                return (samples, log_probabilities), state
            return samples, state
        return self._sample_chain_local(
            apply_fn,
            parameters,
            state,
            chain_length,
            return_log_probabilities=return_log_probabilities,
        )

    @partial(
        jax.jit,
        static_argnames=("apply_fn", "chain_length", "return_log_probabilities"),
    )
    def _sample_chain_local(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState,
        chain_length: int,
        *,
        return_log_probabilities: bool = False,
    ):
        state = ensure_metropolis_state(state)
        state, (samples, log_probabilities) = sample_chain(
            self,
            apply_fn,
            parameters,
            state,
            int(chain_length),
        )
        if return_log_probabilities:
            return (samples, log_probabilities), state
        return samples, state

    def _evaluate_log_probability(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sigma: jax.Array,
    ) -> jax.Array:
        out = self.log_probability(apply_fn, parameters, sigma)
        expected = (int(sigma.shape[0]),)
        if out.shape != expected:
            raise ValueError(
                "apply_fn must return one scalar per chain; "
                f"got shape {out.shape}, expected {expected}."
            )
        return out


__all__ = ["MetropolisSampler"]
