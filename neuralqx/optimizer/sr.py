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


"""Stochastic reconfiguration / natural-gradient preconditioner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import solver as default_solvers
from .preconditioner import AbstractLinearPreconditioner
from .qgt_auto import qgt_auto


class SR(AbstractLinearPreconditioner):
    """QGT-backed stochastic-reconfiguration preconditioner."""

    def __init__(
        self,
        qgt: Callable[..., Any] | None = None,
        solver: Callable[..., Any] = default_solvers.auto,
        *,
        diag_shift: Any = 0.01,
        diag_scale: Any | None = None,
        solver_restart: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
        **qgt_kwargs: Any,
    ) -> None:
        if qgt is None:
            qgt = qgt_auto
            self._uses_auto_qgt = True
        else:
            self._uses_auto_qgt = False
        self.qgt_constructor = qgt
        self.diag_shift = diag_shift
        self.diag_scale = diag_scale
        self.qgt_kwargs = dict(qgt_kwargs)
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
        samples: Any | None = None,
        **kwargs: Any,
    ):
        diag_shift = _resolve_schedule(self.diag_shift, step, name="diag_shift")
        diag_scale = _resolve_schedule(self.diag_scale, step, name="diag_scale")
        if diag_scale is not None:
            raise NotImplementedError(
                "diag_scale is not supported by the built-in SR QGTs yet. "
                "Use diag_shift or a custom QGT constructor exposing diagonal scaling."
            )
        qgt_kwargs = self.qgt_kwargs | kwargs
        if samples is not None:
            qgt_kwargs["samples"] = samples
        if self._uses_auto_qgt:
            qgt_kwargs["solver"] = self.solver
            qgt_kwargs["solver_kwargs"] = self.solver_kwargs
        return self.qgt_constructor(
            vstate,
            diag_shift=diag_shift,
            **qgt_kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"SR(qgt_constructor={self.qgt_constructor}, "
            f"diag_shift={self.diag_shift}, solver={self.solver}, "
            f"solver_restart={self.solver_restart})"
        )


def _resolve_schedule(value: Any, step: Any | None, *, name: str) -> Any:
    if callable(value):
        if step is None:
            raise TypeError(f"Scheduled {name} requires a step argument.")
        return value(step)
    return value


__all__ = ["SR"]
