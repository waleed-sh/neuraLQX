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


"""Metropolis transition-rule abstractions."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.sharding import place_sample_batch
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta

if TYPE_CHECKING:
    from neuralqx.sampler.metropolis import MetropolisSampler
    from neuralqx.sampler.state import MetropolisSamplerState


class TransitionOutput(Struct):
    """Normalized output of a Metropolis transition rule."""

    sigma: Any
    log_correction: Any | None = None
    rule_state: Any | None = None


class AbstractTransitionRule(Struct, metaclass=StructABCMeta):
    """Base class for Metropolis transition rules."""

    def init_state(
        self,
        sampler: MetropolisSampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        key: jax.Array,
    ) -> Any | None:
        """Initialize optional transition-rule state."""
        del sampler, apply_fn, parameters, key
        return None

    def reset(
        self,
        sampler: MetropolisSampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state: MetropolisSamplerState,
    ) -> Any | None:
        """Reset optional transition-rule state."""
        del sampler, apply_fn, parameters
        return sampler_state.rule_state

    def random_state(
        self,
        sampler: MetropolisSampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state: MetropolisSamplerState,
        key: jax.Array,
    ) -> jax.Array:
        """Generate a random state compatible with this rule."""
        states = self.random_state_local(
            sampler,
            apply_fn,
            parameters,
            sampler_state,
            key,
        )
        return place_sample_batch(jnp.asarray(states))

    def random_state_local(
        self,
        sampler: MetropolisSampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state: MetropolisSamplerState,
        key: jax.Array,
    ) -> jax.Array:
        """Generate an unplaced random batch for a local sampler shard."""
        del apply_fn, parameters, sampler_state
        return jnp.asarray(
            sampler.hilbert.random_state(
                key,
                size=sampler.n_batches,
                dtype=sampler.dtype,
            )
        )

    @abc.abstractmethod
    def transition(
        self,
        sampler: MetropolisSampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state: MetropolisSamplerState,
        key: jax.Array,
        sigma: jax.Array,
    ) -> (
        TransitionOutput
        | tuple[jax.Array, Any | None]
        | tuple[jax.Array, Any | None, Any]
    ):
        """Propose a new batch and optional log proposal correction."""


def normalize_transition_output(
    result: Any, previous_rule_state: Any
) -> TransitionOutput:
    """Convert user/built-in rule return values into ``TransitionOutput``."""
    if isinstance(result, TransitionOutput):
        rule_state = (
            previous_rule_state if result.rule_state is None else result.rule_state
        )
        return result.replace(rule_state=rule_state)
    if not isinstance(result, tuple):
        return TransitionOutput(
            sigma=result,
            log_correction=None,
            rule_state=previous_rule_state,
        )
    if len(result) == 2:
        sigma, log_correction = result
        return TransitionOutput(
            sigma=sigma,
            log_correction=log_correction,
            rule_state=previous_rule_state,
        )
    if len(result) == 3:
        sigma, log_correction, rule_state = result
        return TransitionOutput(
            sigma=sigma,
            log_correction=log_correction,
            rule_state=previous_rule_state if rule_state is None else rule_state,
        )
    raise TypeError(
        "Transition rules must return sigma, (sigma, correction), or a TransitionOutput."
    )


__all__ = [
    "AbstractTransitionRule",
    "TransitionOutput",
    "normalize_transition_output",
]
