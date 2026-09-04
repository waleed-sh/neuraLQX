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


"""Linear-operator abstractions consumed by optimizers and SR solvers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field
from neuralqx.utils.typing import Solver


class LinearOperator(Struct):
    """Minimal matrix-free linear-operator interface.

    Optimizer-side objects implement ``A @ x`` for either a parameter pytree or a
    flattened vector. Solvers can call :meth:`solve` with any compatible iterative
    method, for example a conjugate-gradient wrapper.
    """

    diag_shift: Any = field(default=0.0, kw_only=True)

    def __matmul__(self, vector: Any) -> Any:
        raise NotImplementedError

    def __call__(self, vector: Any) -> Any:
        return self @ vector

    def __add__(self, shift: Any) -> LinearOperator:
        return self.replace(diag_shift=self.diag_shift + shift)

    def _solve(
        self,
        solve_fn: Solver,
        rhs: Any,
        *,
        x0: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if x0 is not None:
            kwargs["x0"] = x0
        return solve_fn(self, rhs, **kwargs)

    def solve(
        self,
        solve_fn: Solver,
        rhs: Any,
        *,
        x0: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Solve ``self @ x = rhs`` with a user-provided solver."""
        return self._solve(solve_fn, rhs, x0=x0, **kwargs)

    def to_dense(self):
        """Materialise a dense matrix representation."""
        raise NotImplementedError


class DenseLinearOperator(LinearOperator):
    """Dense matrix wrapper implementing the ``LinearOperator`` API."""

    matrix: Any

    def __matmul__(self, vector: Any) -> Any:
        if hasattr(vector, "ndim"):
            result = self.matrix @ vector
            return result + self.diag_shift * vector
        flat, unravel = ravel_pytree(vector)
        result = self.matrix @ flat
        return unravel(result + self.diag_shift * flat)

    def to_dense(self):
        matrix = jnp.asarray(self.matrix)
        return matrix + jnp.asarray(self.diag_shift, dtype=matrix.dtype) * jnp.eye(
            matrix.shape[0],
            dtype=matrix.dtype,
        )

    def __repr__(self) -> str:
        return f"DenseLinearOperator(shape={self.matrix.shape}, diag_shift={self.diag_shift})"


__all__ = ["DenseLinearOperator", "LinearOperator", "Solver"]
