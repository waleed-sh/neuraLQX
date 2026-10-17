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


"""Optimizer helpers for neuraLQX drivers."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
import optax

import jax
from neuralqx.optimizer import identity_preconditioner


def cast_gradient_like_parameters(gradient: Any, parameters: Any) -> Any:
    """Cast an update/gradient pytree to parameter-compatible leaves."""

    def cast_leaf(update: Any, parameter: Any) -> Any:
        update_arr = jnp.asarray(update)
        parameter_arr = jnp.asarray(parameter)
        if not jnp.iscomplexobj(parameter_arr):
            update_arr = jnp.real(update_arr)
        if hasattr(parameter_arr, "dtype"):
            return update_arr.astype(parameter_arr.dtype)
        return update_arr

    return jax.tree_util.tree_map(cast_leaf, gradient, parameters)


def apply_optax_update(
    optimizer: Any,
    optimizer_state: Any,
    parameters: Any,
    gradient: Any,
) -> tuple[Any, Any]:
    """Apply one Optax-style update and return ``(new_state, new_parameters)``."""
    return _apply_optax_update(optimizer.update, optimizer_state, parameters, gradient)


@partial(jax.jit, static_argnums=0)
def _apply_optax_update(
    update_fn: Any,
    optimizer_state: Any,
    parameters: Any,
    gradient: Any,
) -> tuple[Any, Any]:
    updates, optimizer_state = update_fn(gradient, optimizer_state, parameters)
    return optimizer_state, optax.apply_updates(parameters, updates)


def resolve_preconditioner(preconditioner: Any | None) -> Any:
    """Return the identity preconditioner when no preconditioner is supplied."""
    return identity_preconditioner if preconditioner is None else preconditioner


__all__ = [
    "apply_optax_update",
    "cast_gradient_like_parameters",
    "resolve_preconditioner",
]
