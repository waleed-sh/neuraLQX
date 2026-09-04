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


"""Adaptive solver policies."""

from __future__ import annotations

from typing import Any

from neuralqx.jax.tree import tree_size

from ._api import solver_with_kwargs
from .direct import solve as dense_solve
from .iterative import cg


@solver_with_kwargs
def auto(
    operator: Any,
    rhs: Any,
    *,
    x0: Any | None = None,
    dense_threshold: int = 2048,
    dense_solver: Any = dense_solve,
    iterative_solver: Any = cg,
    dense_kwargs: dict[str, Any] | None = None,
    iterative_kwargs: dict[str, Any] | None = None,
    **iterative_options: Any,
):
    """Choose a dense or matrix-free solver from the system size.

    Dense solves are vastly faster for small/medium parameter counts because
    they avoid repeated QGT matvecs. Above ``dense_threshold`` the solver keeps
    the QGT matrix-free and delegates to ``iterative_solver``.
    """
    n_parameters = _n_parameters(operator, rhs)
    if n_parameters <= int(dense_threshold):
        solution, info = dense_solver(
            operator,
            rhs,
            x0=x0,
            **dict(dense_kwargs or {}),
        )
        return solution, {
            "method": "auto",
            "selected": "dense",
            "n_parameters": n_parameters,
            "inner": info,
        }

    options = dict(iterative_kwargs or {}) | iterative_options
    solution, info = iterative_solver(operator, rhs, x0=x0, **options)
    return solution, {
        "method": "auto",
        "selected": "iterative",
        "n_parameters": n_parameters,
        "inner": info,
    }


def _n_parameters(operator: Any, rhs: Any) -> int:
    if hasattr(operator, "n_parameters"):
        return int(operator.n_parameters)
    return tree_size(rhs)


__all__ = ["auto"]
