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


"""Multi-state VMC driver with an orthogonality penalty."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.jax.differentiation import vjp as neuralqx_vjp
from neuralqx.jax.streaming import apply_batched
from neuralqx.utils.stats import Stats
from neuralqx.utils.stats import statistics
from neuralqx.vqs.estimators.utils import flatten_samples
from neuralqx.vqs.estimators.utils import validate_operator_hilbert

from .base import AbstractVariationalDriver
from .optim import apply_optax_update
from .optim import cast_gradient_like_parameters
from .optim import resolve_preconditioner


class MultiVMCDriver(AbstractVariationalDriver):
    """Joint VMC driver for several variational states."""

    def __init__(
        self,
        hamiltonian: Any,
        optimizer: Any,
        *,
        variational_states: Any,
        preconditioner: Any | None = None,
        orthogonality_strength: float = 1.0,
        loss_name: str = "Energy",
    ) -> None:
        states = tuple(variational_states)
        if not states:
            raise ValueError("MultiVMCDriver requires at least one variational state.")
        for state in states:
            validate_operator_hilbert(state, hamiltonian)
            if state.hilbert != states[0].hilbert:
                raise ValueError(
                    "All variational states must use the same Hilbert space."
                )

        super().__init__(states, loss_name=loss_name)
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.optimizer_states = [optimizer.init(state.parameters) for state in states]
        self.preconditioner = resolve_preconditioner(preconditioner)
        self.orthogonality_strength = float(orthogonality_strength)
        self._last_energies: list[Any] = []
        self._last_overlaps: dict[tuple[int, int], Any] = {}

    @property
    def states(self) -> tuple[Any, ...]:
        return self.state

    @property
    def energy(self) -> Any:
        return self.loss_stats

    def reset_step(self) -> None:
        for state in self.states:
            state.reset()

    def compute_loss_and_update(self) -> tuple[Any, Any]:
        energies: list[Any] = []
        gradients: list[Any] = []
        for state in self.states:
            stats, gradient = state.expect_and_grad(self.hamiltonian)
            energies.append(stats)
            gradients.append(gradient)

        overlaps: dict[tuple[int, int], Any] = {}
        if self.orthogonality_strength != 0.0 and len(self.states) > 1:
            for i in range(len(self.states)):
                for j in range(i + 1, len(self.states)):
                    stats, pair_grad = fidelity_expect_and_grad_joint(
                        self.states[i],
                        self.states[j],
                    )
                    overlaps[(i, j)] = stats
                    gradients[i] = _tree_add_scaled(
                        gradients[i],
                        pair_grad["i"],
                        self.orthogonality_strength,
                    )
                    gradients[j] = _tree_add_scaled(
                        gradients[j],
                        pair_grad["j"],
                        self.orthogonality_strength,
                    )

        updates = []
        for state, gradient in zip(self.states, gradients, strict=True):
            update = self.preconditioner(
                state,
                gradient,
                step=self.step_count,
                samples=state.samples,
            )
            updates.append(cast_gradient_like_parameters(update, state.parameters))

        self._last_energies = energies
        self._last_overlaps = overlaps
        self._loss_grad = gradients
        return _sum_independent_stats(energies), updates

    def update_parameters(self, update: Any) -> None:
        new_optimizer_states = []
        for idx, (state, state_update) in enumerate(
            zip(self.states, update, strict=True)
        ):
            opt_state, parameters = apply_optax_update(
                self.optimizer,
                self.optimizer_states[idx],
                state.parameters,
                state_update,
            )
            new_optimizer_states.append(opt_state)
            state.parameters = parameters
        self.optimizer_states = new_optimizer_states

    def estimate(self, observables: dict[str, Any] | None = None) -> dict[str, Any]:
        if observables is None:
            return {}
        return {
            name: [state.expect(operator) for state in self.states]
            for name, operator in observables.items()
        }

    def _log_additional_data(self, log_data: dict[str, Any]) -> None:
        super()._log_additional_data(log_data)
        for idx, stats in enumerate(self._last_energies):
            log_data[f"{self.loss_name}/state_{idx}"] = stats
        for (i, j), stats in self._last_overlaps.items():
            log_data[f"orthogonality/{i}_{j}"] = stats

    def _default_store_history(
        self, *, has_callbacks: bool, has_observables: bool
    ) -> bool:
        del has_callbacks, has_observables
        return True

    def __repr__(self) -> str:
        return (
            f"MultiVMCDriver(step_count={self.step_count}, n_states={len(self.states)})"
        )


def fidelity_expect_and_grad_joint(state_i: Any, state_j: Any) -> tuple[Stats, Any]:
    """Estimate the pair overlap penalty and gradients for two states."""
    if state_i.hilbert != state_j.hilbert:
        raise ValueError("Orthogonality penalties require matching Hilbert spaces.")
    sigma_i = state_i.samples
    sigma_j = state_j.samples
    return _fidelity_expect_and_grad_joint_kernel(
        state_i.apply_variables_kernel,
        state_i.merge_variables_kernel,
        state_j.apply_variables_kernel,
        state_j.merge_variables_kernel,
        state_i.parameters,
        state_i.model_state,
        state_j.parameters,
        state_j.model_state,
        sigma_i,
        sigma_j,
        state_i.hilbert.size,
        state_i.machine_pow,
        state_j.machine_pow,
        state_i.chunk_size,
        state_j.chunk_size,
    )


@partial(
    jax.jit,
    static_argnames=(
        "apply_i",
        "merge_i",
        "apply_j",
        "merge_j",
        "hilbert_size",
        "chunk_size_i",
        "chunk_size_j",
    ),
)
def _fidelity_expect_and_grad_joint_kernel(
    apply_i: Any,
    merge_i: Any,
    apply_j: Any,
    merge_j: Any,
    params_i: Any,
    model_state_i: Any,
    params_j: Any,
    model_state_j: Any,
    sigma_i: Any,
    sigma_j: Any,
    hilbert_size: int,
    machine_pow_i: Any,
    machine_pow_j: Any,
    chunk_size_i: int | None,
    chunk_size_j: int | None,
) -> tuple[Stats, Any]:
    flat_i, _shape_i = flatten_samples(sigma_i, hilbert_size)
    flat_j, _shape_j = flatten_samples(sigma_j, hilbert_size)
    n_samples = min(flat_i.shape[0], flat_j.shape[0])
    flat_i = flat_i[:n_samples]
    flat_j = flat_j[:n_samples]
    parameters = {"i": params_i, "j": params_j}

    def logpsi_i(params: Any, states: Any) -> jax.Array:
        variables = merge_i(params, model_state_i)
        return apply_batched(apply_i, variables, states, chunk_size=chunk_size_i)

    def logpsi_j(params: Any, states: Any) -> jax.Array:
        variables = merge_j(params, model_state_j)
        return apply_batched(apply_j, variables, states, chunk_size=chunk_size_j)

    def values_for_params(params: Any) -> jax.Array:
        li_x = logpsi_i(params["i"], flat_i)
        li_y = logpsi_i(params["i"], flat_j)
        lj_x = logpsi_j(params["j"], flat_i)
        lj_y = logpsi_j(params["j"], flat_j)
        values = jnp.exp(lj_x + li_y - li_x - lj_y)
        return jnp.real(values)

    def log_pdf_for_params(params: Any) -> jax.Array:
        li_x = logpsi_i(params["i"], flat_i)
        lj_y = logpsi_j(params["j"], flat_j)
        return jnp.asarray(machine_pow_i) * jnp.real(li_x) + jnp.asarray(
            machine_pow_j
        ) * jnp.real(lj_y)

    values, values_pullback = neuralqx_vjp(
        values_for_params, parameters, conjugate=True
    )
    stats = statistics(values)
    direct_grad = values_pullback(jnp.ones_like(values) / values.size)[0]

    log_pdf, score_pullback = neuralqx_vjp(log_pdf_for_params, parameters)
    centered = values - stats.mean
    score_grad = score_pullback((centered / values.size).astype(log_pdf.dtype))[0]
    return stats, _tree_add(direct_grad, score_grad)


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_add_scaled(left: Any, right: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a + scale * b, left, right)


def _sum_independent_stats(stats: list[Any]) -> Stats:
    means = [jnp.asarray(stat.mean) for stat in stats]
    variances = [jnp.asarray(getattr(stat, "variance", jnp.nan)) for stat in stats]
    errors = [jnp.asarray(getattr(stat, "error_of_mean", 0.0)) for stat in stats]
    n_samples = [jnp.asarray(getattr(stat, "n_samples", 0)) for stat in stats]
    return Stats(
        mean=sum(means),
        variance=sum(variances),
        error_of_mean=jnp.sqrt(sum(error**2 for error in errors)),
        n_samples=jnp.min(jnp.stack(n_samples)) if n_samples else jnp.asarray(0),
        r_hat=jnp.asarray(jnp.nan),
    )


MultiStateVMC = MultiVMCDriver


__all__ = [
    "MultiStateVMC",
    "MultiVMCDriver",
    "fidelity_expect_and_grad_joint",
]
