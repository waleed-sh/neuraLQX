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


"""Matrix-free iterative solvers."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.scipy as jsp

import jax

from ..linear_operator import LinearOperator
from ._api import solver_with_kwargs


@solver_with_kwargs
def cg(
    operator: Any,
    rhs: Any,
    *,
    x0: Any | None = None,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Any | None = None,
):
    """Conjugate-gradient solve for Hermitian positive-definite systems."""
    solution, info = _cg_solve(
        _compiled_matvec(operator),
        rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=_compiled_matvec(M),
    )
    return solution, {"method": "cg", "info": info}


def _compiled_matvec(operator):
    if callable(operator) and not isinstance(
        operator, (LinearOperator, jax.tree_util.Partial)
    ):
        return jax.tree_util.Partial(operator)
    return operator


@partial(jax.jit, static_argnames=("maxiter",))
def _cg_solve(operator, rhs, *, x0, tol, atol, maxiter, M):
    # Operator arrays are dynamic arguments, rather than constants captured by
    # a fresh callable on every solve. This lets changing SR systems reuse CG's
    # compiled loop and keeps custom_linear_solve tracing off the hot path.
    return jsp.sparse.linalg.cg(
        operator, rhs, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M
    )


@solver_with_kwargs
def gmres(
    operator: Any,
    rhs: Any,
    *,
    x0: Any | None = None,
    tol: float = 1e-5,
    atol: float = 0.0,
    restart: int = 20,
    maxiter: int | None = None,
    M: Any | None = None,
    solve_method: str = "batched",
):
    """GMRES solve for general matrix-free linear systems."""
    matvec = jax.tree_util.Partial(operator)
    solution, info = jsp.sparse.linalg.gmres(
        matvec,
        rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        solve_method=solve_method,
    )
    return solution, {"method": "gmres", "info": info}


__all__ = ["cg", "gmres"]
