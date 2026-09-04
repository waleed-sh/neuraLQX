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


"""Adapter for explicit init/apply callables."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from neuralqx.utils.struct import field

from .base import AbstractModelFramework
from .base import Variables
from .base import variables_dict


class CallableFramework(AbstractModelFramework):
    """Adapter around explicit apply/init callables.

    ``apply_fn`` may either accept ``(variables, states)`` or
    ``(parameters, states)``. Variables are always stored externally as a
    Flax-style mapping with a ``"params"`` collection.
    """

    apply_fn: Callable[..., Any] = field(pytree=False, compare=False)
    init_fn: Callable[..., Any] | None = field(
        pytree=False, default=None, compare=False
    )
    expects_variables: bool = field(static=True, default=False)
    framework_name: str = field(static=True, default="callable")

    @property
    def name(self) -> str:
        return self.framework_name

    def __hash__(self) -> int:
        return hash(
            (type(self), self.framework_name, id(self.apply_fn), id(self.init_fn))
        )

    def init(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
        *,
        dtype: Any,
    ) -> Variables:
        if self.init_fn is None:
            raise RuntimeError(
                "CallableFramework needs an init_fn when variables are omitted."
            )
        variables = self.init_fn(key, input_shape, dtype)
        if isinstance(variables, dict) and "params" in variables:
            return self.place_variables(variables)
        return self.place_variables({"params": variables})

    def apply_variables(
        self,
        variables: Variables,
        states: Any,
        *,
        mutable: Any = False,
        training_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "CallableFramework does not manage mutable collections."
            )
        kwargs = {} if training_kwargs is None else dict(training_kwargs)
        if self.expects_variables:
            return self.apply_fn(variables, states, **kwargs)
        return self.apply_fn(variables["params"], states, **kwargs)

    def split_variables(self, variables: Variables) -> tuple[Any, Any]:
        if isinstance(variables, dict) and "params" in variables:
            return variables["params"], {
                key: value for key, value in variables.items() if key != "params"
            }
        return variables, {}

    def merge_variables(
        self, parameters: Any, model_state: Any | None = None
    ) -> Variables:
        return variables_dict(parameters, model_state)


__all__ = ["CallableFramework"]
