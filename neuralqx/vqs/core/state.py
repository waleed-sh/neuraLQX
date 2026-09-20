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


"""Concrete sampled variational state."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.jax.random import as_prng_key
from neuralqx.jax.tree import tree_size
from neuralqx.sampler import MetropolisLocal
from neuralqx.sampler import MetropolisSampler
from neuralqx.sampler import MetropolisSamplerState
from neuralqx.utils.frameworks import AbstractModelFramework
from neuralqx.utils.frameworks import FrameworkApply
from neuralqx.utils.frameworks import FrameworkMerge
from neuralqx.utils.frameworks import as_framework
from neuralqx.utils.frameworks import static_training_kwargs
from neuralqx.utils.typing import Key

from .base import AbstractVariationalState
from .evaluation import amplitudes_from_log_values
from .evaluation import evaluate_log_values
from .evaluation import flatten_state_batch
from .evaluation import restore_log_values


class VariationalState(AbstractVariationalState):
    """Sampled variational state backed by a neuraLQX sampler and model framework."""

    def __init__(
        self,
        sampler: MetropolisSampler | None = None,
        model: Any | None = None,
        *,
        hilbert: DiscreteHilbertSpace | None = None,
        variables: Any | None = None,
        parameters: Any | None = None,
        model_state: Any | None = None,
        apply_fn: Any | None = None,
        init_fn: Any | None = None,
        expects_variables: bool = False,
        seed: Key | None = None,
        sampler_seed: Key | None = None,
        n_samples: int = 1000,
        n_discard_per_chain: int = 0,
        chunk_size: int | None = None,
        mutable: Any = False,
        training_kwargs: dict[str, Any] | None = None,
    ) -> None:
        framework, framework_variables = as_framework(
            model,
            apply_fn=apply_fn,
            init_fn=init_fn,
            expects_variables=expects_variables,
        )
        if sampler is None:
            if hilbert is None:
                raise ValueError("Pass either sampler or hilbert.")
            sampler = MetropolisLocal(hilbert)
        elif hilbert is not None and sampler.hilbert != hilbert:
            raise ValueError("sampler.hilbert and hilbert disagree.")

        self._sampler = sampler
        self._hilbert = sampler.hilbert
        self._framework = framework
        self._chunk_size = chunk_size
        self._training_kwargs = dict(training_kwargs or {})
        self._apply_variables_kernel = FrameworkApply(
            framework,
            static_training_kwargs(self._training_kwargs),
        )
        self._merge_variables_kernel = FrameworkMerge(framework)
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "Mutable model collections are not supported by VariationalState yet. "
                "Pass mutable=False or update model_state explicitly."
            )
        self._mutable = False
        self._n_samples = int(n_samples)
        self._n_discard_per_chain = int(n_discard_per_chain)
        if self._n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self._n_discard_per_chain < 0:
            raise ValueError("n_discard_per_chain must be non-negative.")

        if variables is None and framework_variables is not None:
            variables = framework_variables
        if variables is None and parameters is not None:
            variables = framework.merge_variables(parameters, model_state)
        if variables is None:
            variables = framework.init(
                as_prng_key(seed),
                (1, self.hilbert.size),
                dtype=self.sampler.dtype,
            )
        self._parameters_structure = None
        self._variables = framework.place_variables(variables)
        self._parameters, self._model_state = framework.split_variables(self._variables)
        self._parameters_structure = jax.tree_util.tree_structure(self._parameters)

        self._sampler_key = as_prng_key(seed if sampler_seed is None else sampler_seed)
        self._sampler_state: MetropolisSamplerState | None = None
        self._sampler_state_stale = True
        self._samples: jax.Array | None = None
        self.reset()

    @property
    def hilbert(self) -> DiscreteHilbertSpace:
        return self._hilbert

    @property
    def sampler(self) -> MetropolisSampler:
        return self._sampler

    @property
    def sampler_state(self) -> MetropolisSamplerState:
        if self._sampler_state is None or self._sampler_state_stale:
            self.reset()
        assert self._sampler_state is not None
        return self._sampler_state

    @property
    def framework(self) -> AbstractModelFramework:
        return self._framework

    @property
    def apply_variables_kernel(self) -> FrameworkApply:
        """Stable static callable used by JAX kernels."""
        return self._apply_variables_kernel

    @property
    def merge_variables_kernel(self) -> FrameworkMerge:
        """Stable static variable-merge callable used by JAX kernels."""
        return self._merge_variables_kernel

    @property
    def machine_pow(self) -> Any:
        return self.sampler.machine_pow

    @property
    def chunk_size(self) -> int | None:
        return self._chunk_size

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise ValueError("n_samples must be positive.")
        self._n_samples = value
        self._samples = None

    @property
    def n_discard_per_chain(self) -> int:
        return self._n_discard_per_chain

    @n_discard_per_chain.setter
    def n_discard_per_chain(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError("n_discard_per_chain must be non-negative.")
        self._n_discard_per_chain = value

    @property
    def parameters(self) -> Any:
        return self._parameters

    @parameters.setter
    def parameters(self, value: Any) -> None:
        structure = jax.tree_util.tree_structure(value)
        if (
            self._parameters_structure is not None
            and structure != self._parameters_structure
        ):
            raise ValueError(
                "New parameters do not match the original pytree structure."
            )
        variables = self.framework.place_variables(
            self.framework.merge_variables(value, self.model_state)
        )
        value, model_state = self.framework.split_variables(variables)
        self._variables = variables
        self._model_state = model_state
        self._parameters = value
        self._parameters_structure = structure
        self._invalidate_sampler_state()

    @property
    def model_state(self) -> Any:
        return self._model_state

    @model_state.setter
    def model_state(self, value: Any) -> None:
        variables = self.framework.place_variables(
            self.framework.merge_variables(self.parameters, value)
        )
        self._parameters, self._model_state = self.framework.split_variables(variables)
        self._variables = variables
        self._invalidate_sampler_state()

    @property
    def variables(self) -> Any:
        return self._variables

    @variables.setter
    def variables(self, value: Any) -> None:
        value = self.framework.place_variables(value)
        parameters, model_state = self.framework.split_variables(value)
        structure = jax.tree_util.tree_structure(parameters)
        if (
            self._parameters_structure is not None
            and structure != self._parameters_structure
        ):
            raise ValueError(
                "New variables contain a different parameter pytree structure."
            )
        self._variables = value
        self._parameters = parameters
        self._model_state = model_state
        self._parameters_structure = structure
        self._invalidate_sampler_state()

    @property
    def samples(self) -> jax.Array:
        if self._samples is None:
            self.sample()
        assert self._samples is not None
        return self._samples

    @samples.setter
    def samples(self, value: Any) -> None:
        arr = jnp.asarray(value)
        if arr.ndim == 0 or arr.shape[-1] != self.hilbert.size:
            raise ValueError("Samples have the wrong trailing Hilbert dimension.")
        self._samples = arr

    @property
    def n_parameters(self) -> int:
        return tree_size(self.parameters)

    @property
    def training_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to model evaluations."""
        return dict(self._training_kwargs)

    def apply_variables(self, variables: Any, states: Any) -> Any:
        """Evaluate provided variables with this state's model-call policy."""
        return self.apply_variables_kernel(variables, states)

    def apply(self, states: Any, *, mutable: Any | None = None) -> Any:
        """Evaluate the variational log-amplitude model."""
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "Mutable model collections are not supported yet."
            )
        return self.apply_variables(self.variables, states)

    def log_value(
        self,
        states: Any,
        *,
        chunk_size: int | None = None,
        mutable: Any | None = None,
    ) -> jax.Array:
        """Evaluate ``log(psi(states))`` preserving the input batch shape."""
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "Mutable model collections are not supported yet."
            )
        flat, state_shape = flatten_state_batch(states, self.hilbert.size)
        values = evaluate_log_values(
            self.apply_variables_kernel,
            self.variables,
            flat,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )
        return restore_log_values(values, state_shape)

    def to_array(
        self,
        normalize: bool = True,
        *,
        max_states: int = 1_000_000,
        chunk_size: int | None = None,
    ) -> jax.Array:
        """Return amplitudes over the full indexable Hilbert basis."""
        if not self.hilbert.is_indexable:
            raise RuntimeError("to_array requires an indexable Hilbert space.")
        states = self.hilbert.all_states(max_states=max_states)
        log_values = evaluate_log_values(
            self.apply_variables_kernel,
            self.variables,
            states,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )
        return amplitudes_from_log_values(log_values, normalize=bool(normalize))

    def reset(self) -> None:
        """Reset sampler log probabilities and invalidate cached samples."""
        self._samples = None
        if hasattr(self, "_variables"):
            self._sampler_state = self.sampler.reset(
                self.apply_variables_kernel,
                self.variables,
                state=self._sampler_state,
                key=self._sampler_key,
            )
            self._sampler_state_stale = False

    def _invalidate_sampler_state(self) -> None:
        self._samples = None
        self._sampler_state_stale = True

    def sample(self) -> tuple[jax.Array, MetropolisSamplerState]:
        """Generate and cache fresh samples."""
        state = self.sampler_state
        if self.n_discard_per_chain:
            _discarded, state = self.sampler.sample(
                self.apply_variables_kernel,
                self.variables,
                state=state,
                chain_length=self.n_discard_per_chain,
            )
        samples, state = self.sampler.sample(
            self.apply_variables_kernel,
            self.variables,
            state=state,
            n_samples=self.n_samples,
        )
        self._samples = samples
        self._sampler_state = state
        return samples, state


__all__ = ["VariationalState"]
