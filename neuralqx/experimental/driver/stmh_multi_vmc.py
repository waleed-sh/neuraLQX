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

from __future__ import annotations

from functools import partial

from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.jax import tree_cast
from netket.operator import AbstractOperator
from netket.optimizer import PreconditionerT
from netket.optimizer import identity_preconditioner
from neuralqx.utils import distributed as _dist
from netket.utils.types import Optimizer

from neuralqx.driver import VMC

from ..vqs.mc.mc_state import STMultiMCState
from ..vqs.mc.mc_state.utils import tree_add_scaled
from ..vqs.mc.mc_state.utils import get_stats_mean
from ..vqs.mc.mc_state.utils import tree_scale
from ..vqs.mc.mc_state.utils import make_grad_qgt_compatible

from neuralqx.vqs import MCState
from neuralqx.profile import section as prof_section


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def fidelity_expect_and_grad_joint_tied(
    apply_fun_i,
    apply_fun_j,
    machine_pow_i,
    machine_pow_j,
    shared_params,
    model_state_i,
    model_state_j,
    sigma_i,
    sigma_j,
):
    r"""
    Estimate pairwise fidelity-like overlap and its gradient w.r.t **shared** parameters.

    This reuses the same estimator structure as the current multi-state ``fidelity_expect_and_grad_joint``
    kernel, but treats the two heads as tied copies of one parameter pytree. The gradient is
    computed by differentiating a surrogate ``f(p_i, p_j)`` and then evaluating on the diagonal
    ``p_i = p_j = shared_params``. The exact gradient w.r.t the shared parameters is

    .. math::
        \nabla_\theta f(\theta, \theta) = \partial_{p_i} f + \partial_{p_j} f.

    Therefore we return ``grads['i'] + grads['j']`` as the correct shared-parameter penalty gradient.

    Notes
    -----
    - For ``machine_pow == 2`` in both samplers this estimates the normalized quantum fidelity.
    """

    n_chains = sigma_i.shape[0]
    if sigma_i.ndim >= 3:
        sigma_i = jax.lax.collapse(sigma_i, 0, 2)
    if sigma_j.ndim >= 3:
        sigma_j = jax.lax.collapse(sigma_j, 0, 2)

    n = min(sigma_i.shape[0], sigma_j.shape[0])
    sigma_i = sigma_i[:n]
    sigma_j = sigma_j[:n]

    dim = sigma_i.shape[-1]
    sigma_xy = jnp.concatenate([sigma_i, sigma_j], axis=-1)

    def logpsi_i(pars, s):
        return apply_fun_i({"params": pars, **model_state_i}, s)

    def logpsi_j(pars, s):
        return apply_fun_j({"params": pars, **model_state_j}, s)

    def log_pdf(pars_pair, s_xy):
        x, y = s_xy[:, :dim], s_xy[:, dim:]
        return (
            machine_pow_i * logpsi_i(pars_pair["i"], x).real
            + machine_pow_j * logpsi_j(pars_pair["j"], y).real
        )

    def local_fidelity(pars_pair, s_xy):
        x, y = s_xy[:, :dim], s_xy[:, dim:]
        li_x = logpsi_i(pars_pair["i"], x)
        li_y = logpsi_i(pars_pair["i"], y)
        lj_x = logpsi_j(pars_pair["j"], x)
        lj_y = logpsi_j(pars_pair["j"], y)
        return jnp.exp(lj_x + li_y - li_x - lj_y)

    pars0 = {"i": shared_params, "j": shared_params}

    def closure(pars_pair):
        val, stats = nkjax.expect(
            log_pdf, local_fidelity, pars_pair, sigma_xy, n_chains=n_chains
        )
        return val.real, stats

    fid_val, vjp_fun, fid_stats = nkjax.vjp(
        closure, pars0, has_aux=True, conjugate=True
    )
    grads_pair = vjp_fun(jnp.ones_like(fid_val))[0]

    grad_shared = jax.tree_util.tree_map(
        lambda gi, gj: gi + gj,
        grads_pair["i"],
        grads_pair["j"],
    )

    return fid_val, fid_stats, grad_shared


class SingleTrunkMultiHeadVMC(VMC):
    r"""
    VMC driver for a shared-parameter ST-MH ensemble built from head-specific ``MCState`` views.

    The optimized objective is

    .. math::
        C(\Theta) = \sum_k w_k E_k(\Theta) + \lambda \sum_{i<j} F_{ij}(\Theta),

    where ``F_ij`` is the normalized pairwise fidelity estimator (for ``machine_pow=2``).
    The ``sum_{i<j}`` form is equivalent to ``(1/2) sum_{i!=j}`` used in the paper.


    The existing ``MultiStateVMC`` is correct for MT-MH (independent parameter sets) and preconditions each state
    separately. This driver is for ST-MH. It aggregates all energy and penalty gradients into one shared parameter
    pytree and applies the optimizer/preconditioner exactly once.

    Note
    ------
    The default ``identity_preconditioner`` is exact for Euclidean gradient descent on the chosen
    Monte Carlo objective. If you pass an SR/QGT preconditioner, the driver applies it using a single
    reference head state's geometry (``preconditioner_state_index``), which is a practical approximation.
    """

    def __init__(
        self,
        variational_state: STMultiMCState,
        hamiltonian: Union[AbstractOperator, Sequence[AbstractOperator]],
        optimizer: Optimizer,
        *,
        preconditioner: PreconditionerT = identity_preconditioner,
        lambda_ortho: float = 1.0,
        energy_weights: Optional[Sequence[float]] = None,
        enforce_machine_pow_2: bool = True,
        preconditioner_state_index: int = 0,
    ):
        if not isinstance(variational_state, STMultiMCState):
            raise TypeError(
                "SingleTrunkMultiHeadVMC requires a STMultiMCState (shared-parameter container)."
            )

        super().__init__(
            hamiltonian=hamiltonian,
            variational_state=variational_state,
            preconditioner=preconditioner,
            optimizer=optimizer,
        )

        self.lambda_ortho = float(lambda_ortho)
        self.enforce_machine_pow_2 = bool(enforce_machine_pow_2)
        self.preconditioner_state_index = int(preconditioner_state_index)

        self._ham = (
            [hamiltonian] if isinstance(hamiltonian, AbstractOperator) else hamiltonian
        )

        if energy_weights is None:
            self.energy_weights = [1.0 / self.state.n_states] * self.state.n_states
        else:
            w = [float(x) for x in energy_weights]
            if len(w) != self.state.n_states:
                raise ValueError(
                    f"energy_weights must have length {self.state.n_states}, got {len(w)}."
                )
            s = sum(w)
            if s == 0.0:
                raise ValueError("energy_weights must not sum to zero.")
            self.energy_weights = [x / s for x in w]

        self._last_pair_fidelities = {}
        self._last_pair_fidelity_stats = {}

    @property
    def states(self) -> list[MCState]:
        return self.state.states

    def _check_machine_pow(self):
        if not self.enforce_machine_pow_2:
            return
        bad = []
        for i, st in enumerate(self.states):
            mp = int(getattr(getattr(st, "sampler", None), "machine_pow", 2))
            if mp != 2:
                bad.append((i, mp))
        if bad:
            raise ValueError(
                "SingleTrunkMultiHeadVMC orthogonality penalty uses the standard normalized fidelity "
                "interpretation only when machine_pow == 2 for all heads. Found: "
                + ", ".join([f"head {i}: machine_pow={mp}" for i, mp in bad])
            )

    def _forward_and_backward(self):
        with prof_section("reset_all_states (dispatch)", cat="vmc"):
            self.state.reset()

        self._check_machine_pow()

        #
        #
        # Per-head energies and energy gradients (all wrt SAME shared params)

        energies: list[Any] = []
        energy_grads: list[Any] = []
        global_grad = None
        weighted_energy_sum = 0.0

        with prof_section("expect_and_grad_total (dispatch)", cat="vmc"):
            for i, st in enumerate(self.states):
                with prof_section(f"expect_and_grad_head_{i} (sync)", cat="vmc") as sec:
                    e_stats, e_grad = sec.sync(st.expect_and_grad(self._ham))

                wi = self.energy_weights[i]
                energies.append(e_stats)
                energy_grads.append(e_grad)
                weighted_energy_sum += wi * get_stats_mean(e_stats)

                if global_grad is None:
                    global_grad = tree_scale(e_grad, wi)
                else:
                    global_grad = tree_add_scaled(global_grad, e_grad, wi)

        if global_grad is None:
            raise RuntimeError("No heads found in STMultiMCState.")

        #
        #
        # Pairwise orthogonality penalty gradients wrt shared params

        pair_penalty_sum = 0.0
        self._last_pair_fidelities = {}
        self._last_pair_fidelity_stats = {}

        with prof_section("orthogonality_total (dispatch)", cat="vmc"):
            if self.lambda_ortho != 0.0 and self.state.n_states > 1:
                for i in range(self.state.n_states):
                    for j in range(i + 1, self.state.n_states):
                        si, sj = self.states[i], self.states[j]

                        with prof_section(
                            f"fidelity_expect_and_grad_heads_{i}_{j} (sync)", cat="vmc"
                        ) as sec:
                            fid_val, fid_stats, fid_grad_shared = sec.sync(
                                fidelity_expect_and_grad_joint_tied(
                                    si._apply_fun,
                                    sj._apply_fun,
                                    int(getattr(si.sampler, "machine_pow", 2)),
                                    int(getattr(sj.sampler, "machine_pow", 2)),
                                    self.state.parameters,
                                    si.model_state,
                                    sj.model_state,
                                    si.samples,
                                    sj.samples,
                                )
                            )

                        # Accumulate λ * ∇F_ij into the single shared parameter gradient.
                        global_grad = tree_add_scaled(
                            global_grad, fid_grad_shared, self.lambda_ortho
                        )

                        try:
                            f_ij = get_stats_mean(fid_stats)
                        except Exception:
                            f_ij = float(jnp.asarray(fid_val).real)
                        # numerical clipping of the metric estimate for logging only
                        f_ij = float(max(0.0, min(1.0, f_ij)))

                        pair_penalty_sum += f_ij
                        self._last_pair_fidelities[(i, j)] = f_ij
                        self._last_pair_fidelity_stats[(i, j)] = fid_stats

        #
        #
        # Preconditioning and casting (one shared update only)

        ref_state = self.states[self.preconditioner_state_index]
        with prof_section("make_qgt_compatible_shared (dispatch)", cat="vmc"):
            global_grad = make_grad_qgt_compatible(self.state.parameters, global_grad)
        with prof_section("preconditioner_shared (dispatch)", cat="vmc"):
            dp = self.preconditioner(ref_state, global_grad, self.step_count)
        with prof_section("tree_cast_shared (dispatch)", cat="vmc"):
            dp = tree_cast(dp, self.state.parameters)

        self._last_energies = energies
        self._last_energy_grads = energy_grads
        self._last_weighted_energy_sum = weighted_energy_sum
        self._last_pair_penalty_sum = pair_penalty_sum
        self._loss_stats = float(
            weighted_energy_sum + self.lambda_ortho * pair_penalty_sum
        )

        return dp

    def _log_additional_data(self, log_dict: dict, step: int):
        super()._log_additional_data(log_dict, step)

        if hasattr(self, "_last_energies"):
            for i, st in enumerate(self._last_energies):
                log_dict[f"{self._loss_name} (head {i})"] = st

        if hasattr(self, "_last_weighted_energy_sum"):
            log_dict["energy/weighted_sum_mean"] = float(self._last_weighted_energy_sum)

        if hasattr(self, "_last_pair_penalty_sum"):
            log_dict["orthogonality/pair_fidelity_sum"] = float(
                self._last_pair_penalty_sum
            )

        if getattr(self, "_last_pair_fidelities", None):
            for (i, j), fij in self._last_pair_fidelities.items():
                log_dict[f"fidelity({i},{j})"] = float(fij)

    def _estimate_stats(self, observable):
        return self.state.expect(observable)

    def estimate(self, observables):
        # Flatten per-head outputs into logging-friendly keys
        if observables is None:
            observables = {}

        raw = super().estimate(observables)
        n_states = self.state.n_states
        flat: dict[str, Any] = {}

        def _walk(prefix, node):
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(prefix + (str(k),), v)
                return

            name = "/".join(prefix) if prefix else "Observable"
            if isinstance(node, list) and len(node) == n_states:
                for i, st in enumerate(node):
                    flat[f"{name} (head {i})"] = st
                try:
                    flat[f"{name}/sum_mean"] = sum(get_stats_mean(st) for st in node)
                except Exception:
                    pass
            else:
                flat[name] = node

        _walk((), raw)
        return flat
