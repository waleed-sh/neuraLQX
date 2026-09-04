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


"""Gradient preconditioner abstractions."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from .linear_operator import LinearOperator
from .linear_operator import Solver


class IdentityPreconditioner:
    """Preconditioner that leaves gradients unchanged."""

    info: Any = None

    def __call__(
        self,
        vstate: Any,
        gradient: Any,
        *,
        step: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        del vstate, step, kwargs
        return gradient


identity_preconditioner = IdentityPreconditioner()


class AbstractLinearPreconditioner(abc.ABC):
    """Base class for preconditioners solving ``A x = gradient``."""

    def __init__(
        self,
        solver: Solver,
        *,
        solver_restart: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.solver = solver
        self.solver_restart = bool(solver_restart)
        self.solver_kwargs = dict(solver_kwargs or {})
        self.x0: Any | None = None
        self.info: Any = None
        self.lhs: LinearOperator | None = None

    def __call__(
        self,
        vstate: Any,
        gradient: Any,
        *,
        step: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        lhs = self.lhs_constructor(vstate, step=step, **kwargs)
        x0 = None if self.solver_restart else self.x0
        result = lhs.solve(
            self.solver,
            gradient,
            x0=x0,
            **self.solver_kwargs,
        )
        if isinstance(result, tuple) and len(result) == 2:
            solution, info = result
        else:
            solution, info = result, None
        self.lhs = lhs
        self.x0 = solution
        self.info = info
        return solution

    @abc.abstractmethod
    def lhs_constructor(
        self,
        vstate: Any,
        *,
        step: Any | None = None,
        **kwargs: Any,
    ) -> LinearOperator:
        """Construct the linear operator used by this preconditioner."""


class LinearPreconditioner(AbstractLinearPreconditioner):
    """Preconditioner from a user-provided linear-operator constructor."""

    def __init__(
        self,
        lhs_constructor: Callable[..., LinearOperator],
        solver: Solver,
        *,
        solver_restart: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._lhs_constructor = lhs_constructor
        super().__init__(
            solver,
            solver_restart=solver_restart,
            solver_kwargs=solver_kwargs,
        )

    def lhs_constructor(
        self,
        vstate: Any,
        *,
        step: Any | None = None,
        **kwargs: Any,
    ) -> LinearOperator:
        return self._lhs_constructor(vstate, step=step, **kwargs)


__all__ = [
    "AbstractLinearPreconditioner",
    "IdentityPreconditioner",
    "LinearPreconditioner",
    "identity_preconditioner",
]
