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

from typing import Any

import jax.scipy as jsp

import jax

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
    matvec = jax.tree_util.Partial(operator)
    solution, info = jsp.sparse.linalg.cg(
        matvec,
        rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
    )
    return solution, {"method": "cg", "info": info}


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
