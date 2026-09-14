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


"""Local Metropolis transition rule."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.sampler.rules.base import AbstractTransitionRule

from .domain import all_local_sizes_equal
from .domain import homogeneous_binary_flip_parameters
from .domain import local_range_parameters
from .domain import local_value_table
from .domain import values_equal


class LocalRule(AbstractTransitionRule):
    """Change one randomly selected flat site to a different local value."""

    def transition(
        self,
        sampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state,
        key: jax.Array,
        sigma: jax.Array,
    ) -> tuple[jax.Array, None]:
        del apply_fn, parameters, sampler_state
        n_chains = sigma.shape[0]
        range_parameters = local_range_parameters(sampler.hilbert, sigma.dtype)
        binary_range = range_parameters is not None and all_local_sizes_equal(
            sampler.hilbert,
            2,
        )
        if binary_range:
            key_sites = key
        else:
            key_sites, key_values = jax.random.split(key)

        sites = jax.random.randint(
            key_sites,
            shape=(n_chains,),
            minval=0,
            maxval=sampler.hilbert.size,
            dtype=jnp.int32,
        )
        rows = jnp.arange(n_chains, dtype=jnp.int32)

        if range_parameters is not None:
            starts, steps, local_sizes = range_parameters
            current_values = sigma[rows, sites]
            if binary_range:
                update_values = _binary_update_values(
                    sampler.hilbert,
                    sigma.dtype,
                    starts,
                    steps,
                    sites,
                    current_values,
                )
                proposed = sigma.at[rows, sites].set(update_values.astype(sigma.dtype))
                return _enforce_constraints(sampler.hilbert, sigma, proposed), None
            current = jnp.rint((current_values - starts[sites]) / steps[sites]).astype(
                jnp.int32
            )
        else:
            values_table, valid_table, local_sizes = local_value_table(
                sampler.hilbert,
                sigma.dtype,
            )
            current_values = sigma[rows, sites]
            candidate_values = values_table[sites]
            candidate_valid = valid_table[sites]
            matches = values_equal(current_values, candidate_values)
            current = jnp.argmax(matches & candidate_valid, axis=-1).astype(jnp.int32)

        selected_sizes = local_sizes[sites]
        span = jnp.maximum(selected_sizes - 1, 1)
        raw = jax.random.randint(
            key_values,
            shape=(n_chains,),
            minval=0,
            maxval=span,
            dtype=jnp.int32,
        )
        updates = jnp.where(selected_sizes <= 1, current, raw + (raw >= current))
        if range_parameters is not None:
            update_values = starts[sites] + updates.astype(sigma.dtype) * steps[sites]
        else:
            update_values = values_table[sites, updates].astype(sigma.dtype)

        proposed = sigma.at[rows, sites].set(update_values)
        return _enforce_constraints(sampler.hilbert, sigma, proposed), None

    def __hash__(self) -> int:
        return hash(type(self))

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self)


def _binary_update_values(
    hilbert,
    dtype,
    starts: jax.Array,
    steps: jax.Array,
    sites: jax.Array,
    current_values: jax.Array,
) -> jax.Array:
    flip_parameters = homogeneous_binary_flip_parameters(hilbert, dtype)
    if flip_parameters is None:
        return 2 * starts[sites] + steps[sites] - current_values
    start, step = flip_parameters
    return 2 * start + step - current_values


def _enforce_constraints(hilbert, current: jax.Array, proposed: jax.Array) -> jax.Array:
    if not hilbert.constrained:
        return proposed
    return jnp.where(hilbert.is_valid(proposed)[:, None], proposed, current)


__all__ = ["LocalRule"]
