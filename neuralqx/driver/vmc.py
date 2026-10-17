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


"""Variational Monte Carlo optimization driver."""

from __future__ import annotations

from typing import Any

from neuralqx.vqs.estimators.utils import validate_operator_hilbert

from .base import AbstractVariationalDriver
from .optim import apply_optax_update
from .optim import cast_gradient_like_parameters
from .optim import resolve_preconditioner


class VMC(AbstractVariationalDriver):
    """Energy minimization driver for a single variational state."""

    def __init__(
        self,
        hamiltonian: Any,
        optimizer: Any,
        *,
        variational_state: Any,
        preconditioner: Any | None = None,
        loss_name: str = "Energy",
    ) -> None:
        validate_operator_hilbert(variational_state, hamiltonian)
        super().__init__(variational_state, loss_name=loss_name)
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.optimizer_state = optimizer.init(variational_state.parameters)
        self.preconditioner = resolve_preconditioner(preconditioner)

    @property
    def energy(self) -> Any:
        return self.loss_stats

    def compute_loss_and_update(self) -> tuple[Any, Any]:
        stats, gradient = self.state.expect_and_grad(self.hamiltonian)
        self._loss_grad = gradient
        update = self.preconditioner(
            self.state,
            gradient,
            step=self.step_count,
            samples=self.state.samples,
        )
        update = cast_gradient_like_parameters(update, self.state.parameters)
        return stats, update

    def update_parameters(self, update: Any) -> None:
        self.optimizer_state, parameters = apply_optax_update(
            self.optimizer,
            self.optimizer_state,
            self.state.parameters,
            update,
        )
        self.state.parameters = parameters

    def __repr__(self) -> str:
        return f"VMC(step_count={self.step_count}, state={self.state!r})"


__all__ = ["VMC"]
