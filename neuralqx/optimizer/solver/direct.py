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


"""Dense direct solvers for small SR/QGT systems."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
import jax.scipy as jsp

import jax
from ._api import dense_matrix
from ._api import flatten_rhs
from ._api import solver_with_kwargs


@solver_with_kwargs
def solve(operator: Any, rhs: Any, *, assume_a: str = "pos", x0: Any | None = None):
    """Solve a dense linear system with ``jax.scipy.linalg.solve``."""
    del x0
    if isinstance(operator, jax.Array):
        return _solve_array_kernel(operator, rhs, assume_a=assume_a), {
            "method": "solve"
        }
    return _solve_operator_kernel(operator, rhs, assume_a=assume_a), {"method": "solve"}


@partial(jax.jit, static_argnames=("assume_a",))
def _solve_array_kernel(matrix: Any, rhs: Any, *, assume_a: str) -> Any:
    vector, unravel = flatten_rhs(rhs)
    solution = jsp.linalg.solve(matrix, vector, assume_a=assume_a)
    return unravel(solution)


@partial(jax.jit, static_argnames=("assume_a",))
def _solve_operator_kernel(operator: Any, rhs: Any, *, assume_a: str) -> Any:
    matrix = dense_matrix(operator)
    vector, unravel = flatten_rhs(rhs)
    solution = jsp.linalg.solve(matrix, vector, assume_a=assume_a)
    return unravel(solution)


@solver_with_kwargs
def cholesky(operator: Any, rhs: Any, *, lower: bool = False, x0: Any | None = None):
    """Solve a positive-definite dense system by Cholesky factorisation."""
    del x0
    if isinstance(operator, jax.Array):
        return _cholesky_array_kernel(operator, rhs, lower=lower), {
            "method": "cholesky"
        }
    return _cholesky_operator_kernel(operator, rhs, lower=lower), {"method": "cholesky"}


@partial(jax.jit, static_argnames=("lower",))
def _cholesky_array_kernel(matrix: Any, rhs: Any, *, lower: bool) -> Any:
    vector, unravel = flatten_rhs(rhs)
    factor = jsp.linalg.cho_factor(matrix, lower=lower)
    solution = jsp.linalg.cho_solve(factor, vector)
    return unravel(solution)


@partial(jax.jit, static_argnames=("lower",))
def _cholesky_operator_kernel(operator: Any, rhs: Any, *, lower: bool) -> Any:
    matrix = dense_matrix(operator)
    vector, unravel = flatten_rhs(rhs)
    factor = jsp.linalg.cho_factor(matrix, lower=lower)
    solution = jsp.linalg.cho_solve(factor, vector)
    return unravel(solution)


@solver_with_kwargs
def pinv(operator: Any, rhs: Any, *, rtol: float = 1e-12, x0: Any | None = None):
    """Solve with a Hermitian pseudo-inverse."""
    del x0
    if isinstance(operator, jax.Array):
        return _pinv_array_kernel(operator, rhs, rtol=rtol), {"method": "pinv"}
    return _pinv_operator_kernel(operator, rhs, rtol=rtol), {"method": "pinv"}


@jax.jit
def _pinv_array_kernel(matrix: Any, rhs: Any, *, rtol: float) -> Any:
    vector, unravel = flatten_rhs(rhs)
    inverse = jnp.linalg.pinv(matrix, rtol=rtol, hermitian=True)
    return unravel(inverse @ vector)


@jax.jit
def _pinv_operator_kernel(operator: Any, rhs: Any, *, rtol: float) -> Any:
    matrix = dense_matrix(operator)
    vector, unravel = flatten_rhs(rhs)
    inverse = jnp.linalg.pinv(matrix, rtol=rtol, hermitian=True)
    return unravel(inverse @ vector)


@solver_with_kwargs
def pinv_smooth(
    operator: Any,
    rhs: Any,
    *,
    rtol: float = 1e-12,
    rtol_smooth: float = 1e-12,
    x0: Any | None = None,
    return_eigvals: bool = False,
):
    """Solve with a smoothed Hermitian eigendecomposition pseudo-inverse."""
    del x0
    if isinstance(operator, jax.Array):
        solution, info = _pinv_smooth_array_kernel(
            operator,
            rhs,
            rtol=rtol,
            rtol_smooth=rtol_smooth,
            return_eigvals=return_eigvals,
        )
        return solution, {"method": "pinv_smooth"} | info
    solution, info = _pinv_smooth_operator_kernel(
        operator,
        rhs,
        rtol=rtol,
        rtol_smooth=rtol_smooth,
        return_eigvals=return_eigvals,
    )
    return solution, {"method": "pinv_smooth"} | info


@partial(jax.jit, static_argnames=("return_eigvals",))
def _pinv_smooth_array_kernel(
    matrix: Any,
    rhs: Any,
    *,
    rtol: float,
    rtol_smooth: float,
    return_eigvals: bool,
) -> tuple[Any, dict[str, Any]]:
    return _pinv_smooth_dense_kernel(
        matrix,
        rhs,
        rtol=rtol,
        rtol_smooth=rtol_smooth,
        return_eigvals=return_eigvals,
    )


@partial(jax.jit, static_argnames=("return_eigvals",))
def _pinv_smooth_operator_kernel(
    operator: Any,
    rhs: Any,
    *,
    rtol: float,
    rtol_smooth: float,
    return_eigvals: bool,
) -> tuple[Any, dict[str, Any]]:
    matrix = dense_matrix(operator)
    return _pinv_smooth_dense_kernel(
        matrix,
        rhs,
        rtol=rtol,
        rtol_smooth=rtol_smooth,
        return_eigvals=return_eigvals,
    )


def _pinv_smooth_dense_kernel(
    matrix: Any,
    rhs: Any,
    *,
    rtol: float,
    rtol_smooth: float,
    return_eigvals: bool,
) -> tuple[Any, dict[str, Any]]:
    vector, unravel = flatten_rhs(rhs)
    evals, evecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(evals))
    relative = jnp.where(scale > 0, jnp.abs(evals) / scale, 0.0)
    inv = jnp.where(relative > rtol, jnp.reciprocal(evals), 0.0)
    safe_relative = jnp.where(relative > 0, relative, 1.0)
    smooth = 1.0 / (1.0 + (jnp.asarray(rtol_smooth) / safe_relative) ** 6)
    inv = inv * jnp.where(relative > 0, smooth, 0.0)
    solution = evecs @ (inv * (jnp.conjugate(evecs).T @ vector))
    rank = jnp.sum(relative > rtol)
    nonzero = jnp.where(relative > 0, relative, jnp.inf)
    min_relative = jnp.min(nonzero)
    info = {
        "eval_min": evals[0],
        "eval_max": evals[-1],
        "rank": rank,
        "cond_number": jnp.where(min_relative < jnp.inf, 1.0 / min_relative, jnp.inf),
    }
    if return_eigvals:
        info["evals"] = evals
    return unravel(solution), info


__all__ = ["cholesky", "pinv", "pinv_smooth", "solve"]
