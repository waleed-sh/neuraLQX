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


"""User-defined transition rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from neuralqx.utils.struct import field

from ..base import AbstractTransitionRule


class FunctionalRule(AbstractTransitionRule):
    """Transition rule backed by user-provided callables."""

    transition_fn: Callable[..., Any] = field(pytree=False, compare=False)
    random_state_fn: Callable[..., Any] | None = field(
        pytree=False,
        default=None,
        compare=False,
    )
    init_state_fn: Callable[..., Any] | None = field(
        pytree=False,
        default=None,
        compare=False,
    )
    reset_fn: Callable[..., Any] | None = field(
        pytree=False,
        default=None,
        compare=False,
    )
    name: str | None = field(static=True, default=None)

    def init_state(self, sampler, apply_fn, parameters, key: jax.Array) -> Any | None:
        if self.init_state_fn is None:
            return None
        return self.init_state_fn(sampler, apply_fn, parameters, key)

    def reset(self, sampler, apply_fn, parameters, sampler_state) -> Any | None:
        if self.reset_fn is None:
            return sampler_state.rule_state
        return self.reset_fn(sampler, apply_fn, parameters, sampler_state)

    def random_state_local(
        self, sampler, apply_fn, parameters, sampler_state, key: jax.Array
    ):
        if self.random_state_fn is None:
            return super().random_state_local(
                sampler,
                apply_fn,
                parameters,
                sampler_state,
                key,
            )
        return self.random_state_fn(sampler, apply_fn, parameters, sampler_state, key)

    def transition(
        self,
        sampler,
        apply_fn,
        parameters,
        sampler_state,
        key: jax.Array,
        sigma,
    ):
        return self.transition_fn(
            sampler, apply_fn, parameters, sampler_state, key, sigma
        )

    def __hash__(self) -> int:
        return hash((type(self), self.name, id(self.transition_fn)))


__all__ = ["FunctionalRule"]
