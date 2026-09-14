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


"""Sampler state containers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.struct import field


class AbstractSamplerState(Struct, metaclass=StructABCMeta):
    """Base class for sampler states."""


class MetropolisSamplerState(AbstractSamplerState):
    """State of a Metropolis-Hastings Markov chain ensemble."""

    sigma: Any
    log_prob: Any
    rng: jax.Array
    rule_state: Any = field(default=None)
    n_steps: Any = field(default_factory=lambda: jnp.zeros((), dtype=jnp.int32))
    n_accepted_per_chain: Any | None = field(default=None)

    def __post_init__(self) -> None:
        if self.n_accepted_per_chain is None:
            n_chains = int(jnp.asarray(self.sigma).shape[0])
            n_accepted_per_chain = jnp.zeros((n_chains,), dtype=jnp.int32)
        else:
            n_accepted_per_chain = jnp.asarray(self.n_accepted_per_chain)
        object.__setattr__(self, "n_accepted_per_chain", n_accepted_per_chain)

    @property
    def n_accepted(self) -> jax.Array:
        """Total accepted proposals since the last reset."""
        return jnp.sum(jnp.asarray(self.n_accepted_per_chain))

    @property
    def acceptance(self) -> jax.Array | None:
        """Fraction of accepted proposals since the last reset."""
        n_steps = jnp.asarray(self.n_steps)
        accepted = self.n_accepted / n_steps
        return jnp.where(
            n_steps == 0, jnp.asarray(jnp.nan, dtype=accepted.dtype), accepted
        )


__all__ = ["AbstractSamplerState", "MetropolisSamplerState"]
