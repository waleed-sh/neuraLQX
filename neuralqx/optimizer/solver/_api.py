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


"""Shared helpers for linear-system solvers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from functools import wraps
from typing import Any

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import jax


def solver_with_kwargs(fn: Callable[..., tuple[Any, Any]]) -> Callable[..., Any]:
    """Allow solvers to be partially configured by keyword-only calls."""

    @wraps(fn)
    def wrapped(operator: Any | None = None, rhs: Any | None = None, **kwargs: Any):
        if operator is None and rhs is None:

            def configured(op, b, **call_kwargs):
                return fn(op, b, **(kwargs | call_kwargs))

            configured._neuralqx_solver_base = fn
            configured._neuralqx_solver_kwargs = dict(kwargs)
            return configured
        if operator is None or rhs is None:
            raise TypeError("Pass both operator and rhs, or only keyword options.")
        return fn(operator, rhs, **kwargs)

    return wrapped


def dense_matrix(operator: Any) -> jax.Array:
    """Return a dense matrix from an array or LinearOperator-like object."""
    if isinstance(operator, jax.Array):
        return operator
    return operator.to_dense()


def flatten_rhs(rhs: Any) -> tuple[jax.Array, Callable[[jax.Array], Any]]:
    """Flatten a dense or pytree right-hand side."""
    return ravel_pytree(rhs)


def has_non_finite(tree: Any) -> jax.Array:
    """Return whether any leaf in a pytree contains NaN or Inf."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    flags = [jnp.any(~jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.any(jnp.asarray(flags))


def solver_base(solver: Any | None) -> Any | None:
    """Return the undecorated solver identity when available."""
    if solver is None:
        return None
    if isinstance(solver, partial):
        solver = solver.func
    return getattr(solver, "_neuralqx_solver_base", solver)


def solver_config(solver: Any | None) -> dict[str, Any]:
    """Return keyword configuration captured by ``solver_with_kwargs``."""
    return dict(getattr(solver, "_neuralqx_solver_kwargs", {}) or {})


__all__ = [
    "dense_matrix",
    "flatten_rhs",
    "has_non_finite",
    "solver_base",
    "solver_config",
    "solver_with_kwargs",
]
