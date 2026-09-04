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


"""Flax Linen and NNX framework adapters."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.utils.struct import field

from .base import AbstractModelFramework
from .base import Variables
from .base import variables_dict


class FlaxLinenFramework(AbstractModelFramework):
    """Adapter for ``flax.linen.Module`` instances."""

    model: Any = field(pytree=False, compare=False)

    @property
    def name(self) -> str:
        return "flax.linen"

    def __hash__(self) -> int:
        return hash((type(self), type(self.model), repr(self.model)))

    def init(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        *,
        dtype: Any,
    ) -> Variables:
        dummy = jnp.zeros(input_shape, dtype=dtype)
        variables = self.model.init(key, dummy)
        return self.place_variables(variables)

    def apply_variables(
        self,
        variables: Variables,
        states: Any,
        *,
        mutable: Any = False,
        training_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = {} if training_kwargs is None else dict(training_kwargs)
        if mutable is False or mutable is None:
            return self.model.apply(variables, states, **kwargs)
        return self.model.apply(variables, states, mutable=mutable, **kwargs)

    def split_variables(self, variables: Variables) -> tuple[Any, Any]:
        if not isinstance(variables, dict) and not hasattr(variables, "keys"):
            raise TypeError(
                "Flax Linen variables must be a mapping with a 'params' collection."
            )
        if "params" not in variables:
            raise ValueError("Flax Linen variables must contain a 'params' collection.")
        params = variables["params"]
        state = {key: value for key, value in variables.items() if key != "params"}
        return params, state

    def merge_variables(
        self, parameters: Any, model_state: Any | None = None
    ) -> Variables:
        return variables_dict(parameters, model_state)


class FlaxNNXFramework(AbstractModelFramework):
    """Adapter for ``flax.nnx.Module`` instances.

    NNX modules are split into a static graph definition plus a parameter state
    and a non-parameter state. The state objects remain pytrees, so gradients can
    flow through parameter leaves while the graph definition stays static.
    """

    graphdef: Any = field(pytree=False, compare=False)

    @property
    def name(self) -> str:
        return "flax.nnx"

    def __hash__(self) -> int:
        return hash((type(self), id(self.graphdef)))

    @classmethod
    def from_model(cls, model: Any) -> tuple[FlaxNNXFramework, Variables]:
        from flax import nnx

        graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
        framework = cls(graphdef)
        variables = framework.merge_variables(params, model_state)
        return framework, framework.place_variables(variables)

    def init(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        *,
        dtype: Any,
    ) -> Variables:
        del key, input_shape, dtype
        raise RuntimeError(
            "Flax NNX modules are initialized by constructing the module. "
            "Pass the initialized NNX module to as_framework(...)."
        )

    def apply_variables(
        self,
        variables: Variables,
        states: Any,
        *,
        mutable: Any = False,
        training_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del mutable
        from flax import nnx

        params, model_state = self.split_variables(variables)
        model = nnx.merge(self.graphdef, params, model_state)
        kwargs = {} if training_kwargs is None else dict(training_kwargs)
        return model(states, **kwargs)

    def split_variables(self, variables: Variables) -> tuple[Any, Any]:
        if not isinstance(variables, dict):
            raise TypeError(
                "Flax NNX variables must be a {'params': ..., 'state': ...} dict."
            )
        if "params" not in variables:
            raise ValueError("Flax NNX variables must contain a 'params' state.")
        return variables["params"], variables.get("state")

    def merge_variables(
        self, parameters: Any, model_state: Any | None = None
    ) -> Variables:
        return {"params": parameters, "state": model_state}


__all__ = ["FlaxLinenFramework", "FlaxNNXFramework"]
