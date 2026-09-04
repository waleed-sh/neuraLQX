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


"""Framework adapter interfaces for variational models."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import jax
from neuralqx.jax.sharding import place_parameters
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta
from neuralqx.utils.typing import Variables


class AbstractModelFramework(Struct, metaclass=StructABCMeta):
    """Framework-neutral interface used by variational states."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short framework name."""

    @abc.abstractmethod
    def init(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        *,
        dtype: Any,
    ) -> Variables:
        """Initialise framework variables for a dummy input shape."""

    @abc.abstractmethod
    def apply_variables(
        self,
        variables: Variables,
        states: Any,
        *,
        mutable: Any = False,
        training_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Evaluate the model on a batch of flat Hilbert states."""

    @abc.abstractmethod
    def split_variables(self, variables: Variables) -> tuple[Any, Any]:
        """Return ``(parameters, model_state)`` from framework variables."""

    @abc.abstractmethod
    def merge_variables(
        self, parameters: Any, model_state: Any | None = None
    ) -> Variables:
        """Build framework variables from parameters and non-parameter state."""

    def place_variables(self, variables: Variables) -> Variables:
        """Place parameter leaves according to neuraLQX parameter-sharding config."""
        parameters, model_state = self.split_variables(variables)
        return self.merge_variables(place_parameters(parameters), model_state)

    def parameters_from_variables(self, variables: Variables) -> Any:
        """Extract only trainable parameters."""
        parameters, _model_state = self.split_variables(variables)
        return parameters

    def model_state_from_variables(self, variables: Variables) -> Any:
        """Extract non-trainable framework state."""
        _parameters, model_state = self.split_variables(variables)
        return model_state

    def jit_cache_key(self) -> Any:
        """Return a hashable key for JAX static-callable caching."""
        return (type(self), hash(self))


def variables_dict(parameters: Any, model_state: Any | None = None) -> dict[str, Any]:
    """Build a Flax-style variable dictionary."""
    out = {"params": parameters}
    if model_state is not None:
        out.update(dict(model_state))
    return out


@dataclass(frozen=True)
class FrameworkApply:
    """Stable static callable for framework model evaluation inside JIT kernels."""

    framework: AbstractModelFramework
    training_kwargs: tuple[tuple[str, Any], ...] = ()

    def __call__(self, variables: Variables, states: Any) -> Any:
        return self.framework.apply_variables(
            variables,
            states,
            mutable=False,
            training_kwargs=dict(self.training_kwargs),
        )

    def __hash__(self) -> int:
        return hash((type(self), self.framework.jit_cache_key(), self.training_kwargs))

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.framework.jit_cache_key() == other.framework.jit_cache_key()
            and self.training_kwargs == other.training_kwargs
        )


@dataclass(frozen=True)
class FrameworkMerge:
    """Stable static callable for framework variable assembly inside JIT kernels."""

    framework: AbstractModelFramework

    def __call__(self, parameters: Any, model_state: Any | None = None) -> Variables:
        return self.framework.merge_variables(parameters, model_state)

    def __hash__(self) -> int:
        return hash((type(self), self.framework.jit_cache_key()))

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.framework.jit_cache_key() == other.framework.jit_cache_key()
        )


def static_training_kwargs(
    mapping: dict[str, Any] | None,
) -> tuple[tuple[str, Any], ...]:
    """Return a hashable, deterministic representation of training kwargs."""
    if not mapping:
        return ()
    return tuple(
        sorted((str(key), _static_value(value)) for key, value in mapping.items())
    )


def _static_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


__all__ = [
    "AbstractModelFramework",
    "FrameworkApply",
    "FrameworkMerge",
    "Variables",
    "static_training_kwargs",
    "variables_dict",
]
