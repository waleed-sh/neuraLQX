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


"""Execution helpers for local and sample-sharded Metropolis samplers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

import jax
from jax.sharding import PartitionSpec
from neuralqx.jax.shard_map import shard_map
from neuralqx.jax.sharding import can_shard_samples
from neuralqx.jax.sharding import make_sample_zeros
from neuralqx.jax.sharding import place_sample_batch
from neuralqx.jax.sharding import sample_mesh
from neuralqx.jax.sharding import sample_partition_count
from neuralqx.jax.sharding import sample_partition_spec
from neuralqx.jax.sharding import shard_sample_keys
from neuralqx.sampler.metropolis.kernel import sample_chain
from neuralqx.sampler.rules.exchange import ExchangeRule
from neuralqx.sampler.rules.exchange.kernel import exchange_batch_spmd
from neuralqx.sampler.state import MetropolisSamplerState


class LocalBatchSamplerView:
    """Sampler view whose ``n_batches`` is local to one sample shard."""

    def __init__(self, sampler: Any, n_batches: int, *, rule: Any | None = None):
        self._sampler = sampler
        self._n_batches = int(n_batches)
        self._rule = rule

    @property
    def n_batches(self) -> int:
        return self._n_batches

    @property
    def rule(self) -> Any:
        return self._sampler.rule if self._rule is None else self._rule

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sampler, name)


class ShardedExchangeRuleView:
    """Runtime exchange rule with replicated metadata passed through shard-map."""

    def __init__(
        self,
        *,
        site_i: jax.Array,
        site_j: jax.Array,
        incident_clusters: jax.Array,
        incident_valid: jax.Array,
        probabilities: jax.Array | None,
    ):
        self.site_i = site_i
        self.site_j = site_j
        self.incident_clusters = incident_clusters
        self.incident_valid = incident_valid
        self.probabilities = None if probabilities.shape[0] == 0 else probabilities

    def transition(self, sampler, apply_fn, parameters, sampler_state, key, sigma):
        del sampler, apply_fn, parameters, sampler_state
        return exchange_batch_spmd(
            site_i=self.site_i,
            site_j=self.site_j,
            incident_clusters=self.incident_clusters,
            incident_valid=self.incident_valid,
            probabilities=self.probabilities,
            key=key,
            sigma=sigma,
        )


def use_sample_shard_map(sampler: Any) -> bool:
    """Return whether this sampler should use explicit sample shard-map execution."""
    return can_shard_samples(int(sampler.n_chains), ndim=1)


def initialize_sharded_state_arrays(sampler: Any, key: jax.Array):
    """Create sharded empty state arrays and per-shard RNG keys."""
    partitions = sample_partition_count(1)
    key_state, key_rule, key_random = jax.random.split(key, 3)
    sigma = make_sample_zeros((sampler.n_batches, sampler.hilbert.size), sampler.dtype)
    log_prob = make_sample_zeros((sampler.n_batches,), jnp.float64) - jnp.inf
    accepted = make_sample_zeros((sampler.n_batches,), jnp.int32)
    rng = shard_sample_keys(key_state, partitions)
    return sigma, log_prob, accepted, rng, key_rule, key_random


def ensure_sharded_state(state: MetropolisSamplerState) -> MetropolisSamplerState:
    """Place state arrays and promote scalar RNGs for sample shard-map execution."""
    partitions = sample_partition_count(1)
    rng = state.rng
    if getattr(rng, "ndim", None) == 0:
        rng = shard_sample_keys(rng, partitions)
    return state.replace(
        sigma=place_sample_batch(jnp.asarray(state.sigma)),
        log_prob=place_sample_batch(jnp.asarray(state.log_prob)),
        rng=rng,
        n_accepted_per_chain=place_sample_batch(
            jnp.asarray(state.n_accepted_per_chain)
        ),
    )


def random_state_sharded(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
    key: jax.Array,
) -> jax.Array:
    """Generate initial chain states with one independent key per sample shard."""
    mesh = _sample_mesh_or_error()
    keys = shard_sample_keys(key, sample_partition_count(1))

    def local_random_state(
        sigma,
        log_prob,
        local_rng,
        rule_state,
        n_steps,
        accepted,
        params,
    ):
        local_key = local_rng[0]
        local_sampler = LocalBatchSamplerView(sampler, sigma.shape[0])
        local_state = MetropolisSamplerState(
            sigma=sigma,
            log_prob=log_prob,
            rng=local_key,
            rule_state=rule_state,
            n_steps=n_steps,
            n_accepted_per_chain=accepted,
        )
        return jnp.asarray(
            local_sampler.rule.random_state_local(
                local_sampler,
                apply_fn,
                params,
                local_state,
                local_key,
            )
        )

    mapped = shard_map(
        local_random_state,
        mesh=mesh,
        in_specs=(
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            sample_partition_spec(1),
            PartitionSpec(),
        ),
        out_specs=sample_partition_spec(2),
    )
    return mapped(
        state.sigma,
        state.log_prob,
        keys,
        state.rule_state,
        state.n_steps,
        state.n_accepted_per_chain,
        parameters,
    )


def reset_sharded(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
) -> MetropolisSamplerState:
    """Reset log-probabilities and counters using explicit sample shard-map."""
    mesh = _sample_mesh_or_error()
    state = ensure_sharded_state(state)

    def local_reset(sigma, log_prob, rng, rule_state, n_steps, accepted, params):
        local_rng = rng[0]
        local_sampler = LocalBatchSamplerView(sampler, sigma.shape[0])
        local_state = MetropolisSamplerState(
            sigma=sigma,
            log_prob=log_prob,
            rng=local_rng,
            rule_state=rule_state,
            n_steps=n_steps,
            n_accepted_per_chain=accepted,
        )
        if sampler.reset_chains:
            local_rng, key = jax.random.split(local_rng)
            sigma = jnp.asarray(
                local_sampler.rule.random_state_local(
                    local_sampler,
                    apply_fn,
                    params,
                    local_state,
                    key,
                )
            )
        log_prob = local_sampler.log_probability(apply_fn, params, sigma)
        return sigma, log_prob, local_rng[None], jnp.zeros_like(accepted)

    mapped = shard_map(
        local_reset,
        mesh=mesh,
        in_specs=(
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            sample_partition_spec(1),
            PartitionSpec(),
        ),
        out_specs=(
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            sample_partition_spec(1),
        ),
    )
    sigma, log_prob, rng, accepted = mapped(
        state.sigma,
        state.log_prob,
        state.rng,
        state.rule_state,
        state.n_steps,
        state.n_accepted_per_chain,
        parameters,
    )
    return state.replace(
        sigma=sigma,
        log_prob=log_prob,
        rng=rng,
        rule_state=sampler.rule.reset(sampler, apply_fn, parameters, state),
        n_steps=jnp.zeros_like(state.n_steps),
        n_accepted_per_chain=accepted,
    )


def sample_chain_sharded(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
    chain_length: int,
) -> tuple[tuple[jax.Array, jax.Array], MetropolisSamplerState]:
    """Run a fixed-length sample chain with explicit sample shard-map execution."""
    if isinstance(sampler.rule, ExchangeRule):
        return _sample_chain_sharded_exchange(
            sampler,
            apply_fn,
            parameters,
            state,
            chain_length,
        )
    return _sample_chain_sharded_generic(
        sampler,
        apply_fn,
        parameters,
        state,
        chain_length,
    )


def _sample_chain_sharded_generic(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
    chain_length: int,
) -> tuple[tuple[jax.Array, jax.Array], MetropolisSamplerState]:
    mesh = _sample_mesh_or_error()
    state = ensure_sharded_state(state)

    def local_sample_chain(sigma, log_prob, rng, rule_state, n_steps, accepted, params):
        local_sampler = LocalBatchSamplerView(sampler, sigma.shape[0])
        local_state = MetropolisSamplerState(
            sigma=sigma,
            log_prob=log_prob,
            rng=rng[0],
            rule_state=rule_state,
            n_steps=n_steps,
            n_accepted_per_chain=accepted,
        )
        new_state, (samples, log_probabilities) = sample_chain(
            local_sampler,
            apply_fn,
            params,
            local_state,
            int(chain_length),
        )
        return (
            samples,
            log_probabilities,
            new_state.sigma,
            new_state.log_prob,
            new_state.rng[None],
            new_state.rule_state,
            new_state.n_steps,
            new_state.n_accepted_per_chain,
        )

    mapped = shard_map(
        local_sample_chain,
        mesh=mesh,
        in_specs=(
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            sample_partition_spec(1),
            PartitionSpec(),
        ),
        out_specs=(
            sample_partition_spec(3),
            sample_partition_spec(2),
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            sample_partition_spec(1),
        ),
    )
    (
        samples,
        log_probabilities,
        sigma,
        log_prob,
        rng,
        rule_state,
        n_steps,
        accepted,
    ) = mapped(
        state.sigma,
        state.log_prob,
        state.rng,
        state.rule_state,
        state.n_steps,
        state.n_accepted_per_chain,
        parameters,
    )
    new_state = state.replace(
        sigma=sigma,
        log_prob=log_prob,
        rng=rng,
        rule_state=rule_state,
        n_steps=n_steps,
        n_accepted_per_chain=accepted,
    )
    return (samples, log_probabilities), new_state


def _sample_chain_sharded_exchange(
    sampler: Any,
    apply_fn: Callable[[Any, jax.Array], jax.Array],
    parameters: Any,
    state: MetropolisSamplerState,
    chain_length: int,
) -> tuple[tuple[jax.Array, jax.Array], MetropolisSamplerState]:
    mesh = _sample_mesh_or_error()
    state = ensure_sharded_state(state)
    probabilities = _exchange_probabilities(sampler.rule)

    def local_sample_chain(
        sigma,
        log_prob,
        rng,
        rule_state,
        n_steps,
        accepted,
        params,
        site_i,
        site_j,
        incident_clusters,
        incident_valid,
        probabilities,
    ):
        rule = ShardedExchangeRuleView(
            site_i=site_i,
            site_j=site_j,
            incident_clusters=incident_clusters,
            incident_valid=incident_valid,
            probabilities=probabilities,
        )
        local_sampler = LocalBatchSamplerView(sampler, sigma.shape[0], rule=rule)
        local_state = MetropolisSamplerState(
            sigma=sigma,
            log_prob=log_prob,
            rng=rng[0],
            rule_state=rule_state,
            n_steps=n_steps,
            n_accepted_per_chain=accepted,
        )
        new_state, (samples, log_probabilities) = sample_chain(
            local_sampler,
            apply_fn,
            params,
            local_state,
            int(chain_length),
        )
        return (
            samples,
            log_probabilities,
            new_state.sigma,
            new_state.log_prob,
            new_state.rng[None],
            new_state.rule_state,
            new_state.n_steps,
            new_state.n_accepted_per_chain,
        )

    mapped = shard_map(
        local_sample_chain,
        mesh=mesh,
        in_specs=(
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
        ),
        out_specs=(
            sample_partition_spec(3),
            sample_partition_spec(2),
            sample_partition_spec(2),
            sample_partition_spec(1),
            sample_partition_spec(1),
            PartitionSpec(),
            PartitionSpec(),
            sample_partition_spec(1),
        ),
    )
    (
        samples,
        log_probabilities,
        sigma,
        log_prob,
        rng,
        rule_state,
        n_steps,
        accepted,
    ) = mapped(
        state.sigma,
        state.log_prob,
        state.rng,
        state.rule_state,
        state.n_steps,
        state.n_accepted_per_chain,
        parameters,
        sampler.rule.site_i,
        sampler.rule.site_j,
        sampler.rule.incident_clusters,
        sampler.rule.incident_valid,
        probabilities,
    )
    new_state = state.replace(
        sigma=sigma,
        log_prob=log_prob,
        rng=rng,
        rule_state=rule_state,
        n_steps=n_steps,
        n_accepted_per_chain=accepted,
    )
    return (samples, log_probabilities), new_state


def _exchange_probabilities(rule: ExchangeRule) -> jax.Array:
    if rule.probabilities is None:
        return jnp.asarray([], dtype=jnp.float32)
    return jnp.asarray(rule.probabilities, dtype=jnp.float32)


def _sample_mesh_or_error():
    mesh = sample_mesh()
    if mesh is None:
        raise RuntimeError("Sample shard-map execution requires an active sample mesh.")
    return mesh


__all__ = [
    "LocalBatchSamplerView",
    "ensure_sharded_state",
    "initialize_sharded_state_arrays",
    "random_state_sharded",
    "reset_sharded",
    "sample_chain_sharded",
    "use_sample_shard_map",
]
