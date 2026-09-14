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


"""Abstract sampler interfaces."""

from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.jax.random import as_prng_key
from neuralqx.jax.sharding import sample_sharding_enabled
from neuralqx.jax.streaming import apply_batched
from neuralqx.sampler._samples import chain_length_from_samples
from neuralqx.sampler.state import AbstractSamplerState
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.struct import field
from neuralqx.utils.typing import Key


class AbstractSampler(Struct, metaclass=StructABCMeta):
    """Base API for Monte Carlo samplers."""

    hilbert: DiscreteHilbertSpace = field(static=True)
    machine_pow: Any = field(default=2.0)
    dtype: Any = field(static=True, default=None)
    chunk_size: int | None = field(static=True, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.hilbert, DiscreteHilbertSpace):
            raise TypeError(
                "Samplers currently require a DiscreteHilbertSpace; "
                f"got {type(self.hilbert).__name__}."
            )
        machine_pow = jnp.asarray(self.machine_pow)
        if jnp.issubdtype(machine_pow.dtype, jnp.complexfloating):
            raise ValueError("machine_pow must be real.")
        dtype = self.hilbert.dtype if self.dtype is None else np.dtype(self.dtype)
        object.__setattr__(self, "machine_pow", machine_pow)
        object.__setattr__(self, "dtype", dtype)
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided.")

    @property
    def is_exact(self) -> bool:
        """Whether samples are independent exact draws."""
        return False

    @property
    def sample_sharding_enabled(self) -> bool:
        """Current sample-sharding policy from neuraLQX config."""
        return bool(sample_sharding_enabled())

    @property
    def sample_device_count(self) -> int:
        """Number of devices used for sample-axis work."""
        return int(jax.device_count()) if self.sample_sharding_enabled else 1

    def log_probability(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        states: Any,
    ) -> jax.Array:
        """Evaluate the real log target used by Metropolis-Hastings."""
        values = apply_batched(apply_fn, parameters, states, chunk_size=self.chunk_size)
        return jnp.asarray(jnp.asarray(self.machine_pow) * jnp.real(values))

    def init_state(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        key: Key | None = None,
    ) -> AbstractSamplerState:
        """Initialize sampler state from a deterministic key or seed."""
        return self._init_state(apply_fn, parameters, as_prng_key(key))

    def reset(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState | None = None,
        *,
        key: Key | None = None,
    ) -> AbstractSamplerState:
        """Reset or initialize the sampler state."""
        if state is None:
            state = self.init_state(apply_fn, parameters, key)
        return self._reset(apply_fn, parameters, state)

    def sample(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        *,
        state: AbstractSamplerState | None = None,
        key: Key | None = None,
        n_samples: int | None = None,
        chain_length: int | None = None,
        return_log_probabilities: bool = False,
    ):
        """Sample a chain batch.

        ``n_samples`` is the total requested sample count across all chains.
        When it is not divisible by ``n_chains``, it is rounded to the nearest
        positive divisible value and the chain length is adjusted accordingly.
        """
        if n_samples is not None and chain_length is not None:
            raise ValueError("Specify either n_samples or chain_length, not both.")
        if chain_length is None:
            total = 1000 if n_samples is None else int(n_samples)
            chain_length = chain_length_from_samples(total, self.n_chains)
        if chain_length <= 0:
            raise ValueError("chain_length must be positive.")
        if state is None:
            state = self.reset(apply_fn, parameters, key=key)
        return self._sample_chain(
            apply_fn,
            parameters,
            state,
            int(chain_length),
            return_log_probabilities=return_log_probabilities,
        )

    def samples(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        *,
        state: AbstractSamplerState | None = None,
        key: Key | None = None,
        chain_length: int = 1,
    ) -> Iterator[jax.Array]:
        """Yield one sample batch at a time."""
        if state is None:
            state = self.reset(apply_fn, parameters, key=key)
        for _ in range(chain_length):
            samples, state = self._sample_chain(
                apply_fn,
                parameters,
                state,
                1,
                return_log_probabilities=False,
            )
            yield samples[:, 0, :]

    @abc.abstractmethod
    def _init_state(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        key: jax.Array,
    ) -> AbstractSamplerState:
        """Implementation of ``init_state``."""

    @abc.abstractmethod
    def _reset(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState,
    ) -> AbstractSamplerState:
        """Implementation of ``reset``."""

    @abc.abstractmethod
    def _sample_chain(
        self,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        state: AbstractSamplerState,
        chain_length: int,
        *,
        return_log_probabilities: bool = False,
    ):
        """Implementation of ``sample``."""

    @property
    def n_batches(self) -> int:
        """Leading chain batch shape visible to the current process."""
        raise NotImplementedError


__all__ = ["AbstractSampler"]
