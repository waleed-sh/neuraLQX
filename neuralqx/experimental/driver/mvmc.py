#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Multi-state VMC driver with an optional orthogonality regularisation term.

This module extends a standard variational Monte Carlo (VMC) optimisation loop to train
multiple independent :class:`~neuralqx.vqs.MCState` instances jointly via a
:class:`~neuralqx.experimental.vqs.mc.mc_state.MultiMCState` container.

At each optimisation step the driver computes an energy gradient for every state. When
enabled, an additional pairwise penalty based on a fidelity-like overlap estimator is
added to encourage mutual orthogonality between different states.

A dedicated JAX kernel is provided to estimate the pairwise overlap quantity and its
gradients with respect to both states' parameters using a shared joint sample batch.
"""

from functools import partial
from typing import Union, Any

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
from neuralqx.experimental.vqs.mc.mc_state import MultiMCState
from neuralqx.vqs import MCState
from neuralqx.profile import section as prof_section


def _params_are_complex(params) -> bool:
    """
    Check whether a parameter pytree contains any complex-valued leaves.

    This helper is used to decide whether imaginary components of gradients should be kept
    (as required for complex parameters) or discarded (for real parameters and real-valued
    objectives, where imaginary parts are typically estimator noise).

    :param params: Parameter pytree.
    :return: ``True`` if at least one leaf is complex-valued, otherwise ``False``.
    """

    return any(jnp.iscomplexobj(x) for x in jax.tree_util.tree_leaves(params))


def _make_grad_qgt_compatible(params, grad):
    """
    Post-process a gradient pytree to be compatible with real-valued preconditioning.

    Some preconditioners (for example SR/QGT-based methods) may assume real parameters and
    operate on real-valued gradients. For a real-valued objective with real parameters, the
    true gradient is real and any imaginary part is typically Monte Carlo noise.

    This helper keeps gradients unchanged when the parameters are complex, and otherwise
    drops the imaginary component leaf-wise.

    :param params: Parameter pytree used to determine whether complex gradients are allowed.
    :param grad: Gradient pytree to post-process.
    :return: A gradient pytree where leaves are real if ``params`` are real-valued, otherwise
             the original ``grad``.
    """

    # if params are real, the gradient of any real loss must be real.
    # any residual imag part is Monte Carlo noise / complex estimator noise.
    if _params_are_complex(params):
        return grad
    return jax.tree_util.tree_map(lambda x: jnp.real(x), grad)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def fidelity_expect_and_grad_joint(
    apply_fun_i,
    apply_fun_j,
    machine_pow_i,
    machine_pow_j,
    params_i,
    model_state_i,
    params_j,
    model_state_j,
    sigma_i,
    sigma_j,
):
    """
    Estimate a fidelity-like overlap quantity and its gradients for two variational states.

    This JIT-compiled kernel forms a joint configuration batch by concatenating samples from
    two states and evaluates a local ratio estimator on crossed configurations. The result
    is a scalar overlap measure (real-valued) together with auxiliary statistics and
    gradients with respect to both parameter sets.

    For the common choice ``machine_pow == 2`` (Born sampling), the expectation value matches
    the normalised pure-state fidelity

    .. math::

       F_{ij} = \\frac{|\\langle \\psi_i | \\psi_j \\rangle|^2}
                     {\\langle \\psi_i | \\psi_i \\rangle\\,\\langle \\psi_j | \\psi_j \\rangle}.

    More generally (for ``machine_pow != 2``), the returned quantity is still a consistent
    expectation value under the chosen sampling distributions but is not equal to the
    Hilbert-space fidelity unless ``machine_pow == 2``.

    MPI semantics
    -------------
    The value and statistics are computed using :func:`netket.jax.expect`, which internally
    uses MPI-aware reductions for statistics.

    For gradients, note that :func:`netket.jax.expect` uses an MPI mean inside its custom VJP
    rule. As a consequence, gradients produced by automatic differentiation are scaled by
    ``1 / n_ranks`` on each rank. To obtain rank-independent gradients suitable for parameter
    updates, this function performs an MPI *sum* over ranks on the AD gradients.

    :param apply_fun_i: Apply function for state *i* mapping variables and samples to ``log(psi)``.
    :param apply_fun_j: Apply function for state *j* mapping variables and samples to ``log(psi)``.
    :param machine_pow_i: Sampling power used for state *i* in the joint log density.
    :param machine_pow_j: Sampling power used for state *j* in the joint log density.
    :param params_i: Parameter pytree for state *i*.
    :param model_state_i: Model-state pytree for state *i* (everything except parameters).
    :param params_j: Parameter pytree for state *j*.
    :param model_state_j: Model-state pytree for state *j* (everything except parameters).
    :param sigma_i: Samples from state *i* (may include chain dimensions).
    :param sigma_j: Samples from state *j* (may include chain dimensions).

    :return: ``(fid_val, fid_stats, grads)`` where ``fid_val`` is a real scalar, ``fid_stats``
             is a NetKet statistics object, and ``grads`` is a dict-like pytree with keys
             ``"i"`` and ``"j"`` containing gradients for state *i* and state *j*.
    """

    # collapse chain dims like NetKet kernels do
    n_chains = sigma_i.shape[0]
    if sigma_i.ndim >= 3:
        sigma_i = jax.lax.collapse(sigma_i, 0, 2)
    if sigma_j.ndim >= 3:
        sigma_j = jax.lax.collapse(sigma_j, 0, 2)

    n = min(sigma_i.shape[0], sigma_j.shape[0])
    sigma_i, sigma_j = sigma_i[:n], sigma_j[:n]

    dim = sigma_i.shape[-1]
    sigma_xy = jnp.concatenate([sigma_i, sigma_j], axis=-1)

    def logpsi_i(pars, s):
        return apply_fun_i({"params": pars, **model_state_i}, s)

    def logpsi_j(pars, s):
        return apply_fun_j({"params": pars, **model_state_j}, s)

    def log_pdf(pars, s_xy):
        x, y = s_xy[:, :dim], s_xy[:, dim:]
        return (
            machine_pow_i * logpsi_i(pars["i"], x).real
            + machine_pow_j * logpsi_j(pars["j"], y).real
        )

    def local_fidelity(pars, s_xy):
        x, y = s_xy[:, :dim], s_xy[:, dim:]
        li_x = logpsi_i(pars["i"], x)
        li_y = logpsi_i(pars["i"], y)
        lj_x = logpsi_j(pars["j"], x)
        lj_y = logpsi_j(pars["j"], y)
        return jnp.exp(lj_x + li_y - li_x - lj_y)

    pars0 = {"i": params_i, "j": params_j}

    def closure(pars):
        val, stats = nkjax.expect(
            log_pdf, local_fidelity, pars, sigma_xy, n_chains=n_chains
        )
        return val.real, stats

    fid_val, vjp_fun, fid_stats = nkjax.vjp(
        closure, pars0, has_aux=True, conjugate=True
    )
    grads = vjp_fun(jnp.ones_like(fid_val))[0]

    return fid_val, fid_stats, grads


"""
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]
"""


class MultiStateVMC(VMC):
    r"""
    VMC driver for :class:`~neuralqx.experimental.vqs.mc.mc_state.MultiMCState` with an optional orthogonality penalty.

    For each contained state, this driver computes energy statistics and gradients of the
    Hamiltonian objective. If ``lambda_ortho`` is nonzero and more than one state is present,
    it additionally applies a pairwise penalty based on a fidelity-like overlap estimator to
    encourage the states to become mutually orthogonal.

    Preconditioning is applied independently per state. Interpreted as a joint optimisation
    problem over all parameters, this corresponds to a block-diagonal preconditioner.

    :param variational_state: Multi-state variational object containing multiple independent states.
    :param hamiltonian: Operator or list of operators defining the objective for each state.
    :param optimizer: Optimiser used to update parameters from the (preconditioned) gradients.
    :param preconditioner: Preconditioner applied per state to transform raw gradients.
    :param lambda_ortho: Strength of the orthogonality penalty. Set to ``0`` to disable.

    :raises TypeError: If any operator acts on a Hilbert space incompatible with the variational state.
    """

    def __init__(
        self,
        variational_state: MultiMCState,
        hamiltonian: Union[AbstractOperator, list],
        optimizer: Optimizer,
        *,
        preconditioner: PreconditionerT = identity_preconditioner,
        lambda_ortho: float = 1.0,
    ):
        """
        Initialise the multi-state VMC driver.

        This configures the base :class:`~neuralqx.driver.VMC` machinery and stores the
        orthogonality penalty strength. The Hamiltonian is normalized to a list internally so
        that each state's ``expect_and_grad`` call can be handled uniformly.

        :param variational_state: Multi-state variational object used for sampling and gradients.
        :param hamiltonian: Operator or list of operators defining the objective.
        :param optimizer: Optimiser used for parameter updates.
        :param preconditioner: Preconditioner applied per state.
        :param lambda_ortho: Strength of the orthogonality penalty. Set to ``0`` to disable.
        """

        # Note: a Hilbert space check is carried out in MultiMCState init

        super().__init__(
            hamiltonian=hamiltonian,
            variational_state=variational_state,
            preconditioner=preconditioner,
            optimizer=optimizer,
        )

        self.lambda_ortho = lambda_ortho
        self._ham = (
            [hamiltonian] if isinstance(hamiltonian, AbstractOperator) else hamiltonian
        )

    @property
    def states(self) -> list[MCState]:
        """
        Return the list of underlying per-state :class:`~neuralqx.vqs.MCState` instances.

        This is a convenience view into the states managed by the multi-state variational state.

        :return: The list of contained Monte Carlo variational states.
        """

        return self.state.states

    def _forward_and_backward(self):
        """
        Compute per-state gradients and return preconditioned update directions.

        This method performs the core optimisation step logic:

        - Reset the multi-state sampling statistics.
        - Compute energy statistics and raw gradients for each contained state.
        - Optionally add pairwise overlap-penalty gradients scaled by ``lambda_ortho``.
        - Make gradients compatible with real-valued preconditioning when required.
        - Apply the configured preconditioner independently per state.
        - Cast update pytrees to match each state's parameter structure.

        The most recent per-state energy statistics are stored for logging, and the driver's
        loss statistic is set to the sum of per-state energy means.

        :return: A list of preconditioned update directions, ordered like :attr:`states`.
        """

        with prof_section("reset_all_states (dispatch)", cat="vmc"):
            self.state.reset()

        # energy + grad per state
        energies, grads = [], []
        with prof_section("expect_and_grad_total (dispatch)", cat="vmc"):
            for i, st in enumerate(self.states):
                with prof_section(
                    f"expect_and_grad_state_{i} (sync)", cat="vmc"
                ) as sec:
                    e_stats, e_grad = sec.sync(st.expect_and_grad(self._ham))
                energies.append(e_stats)
                grads.append(e_grad)

        # pairwise fidelity penalty
        with prof_section("orthogonality_total (dispatch)", cat="vmc"):
            if self.lambda_ortho != 0 and self.state.n_states > 1:
                for i in range(self.state.n_states):
                    for j in range(i + 1, self.state.n_states):
                        si, sj = self.states[i], self.states[j]
                        with prof_section(
                            f"fidelity_expect_and_grad_states_{i}_{j} (sync)", cat="vmc"
                        ) as sec:
                            # force jax to block until ready to get accurate measurements and not only dispatch time
                            # this is only going to sync if a user requests it explicitly
                            _, fid_stats, fid_grads = sec.sync(
                                fidelity_expect_and_grad_joint(
                                    si._apply_fun,
                                    sj._apply_fun,
                                    int(si.sampler.machine_pow),
                                    int(sj.sampler.machine_pow),
                                    si.parameters,
                                    si.model_state,
                                    sj.parameters,
                                    sj.model_state,
                                    si.samples,
                                    sj.samples,
                                )
                            )

                        grads[i] = jax.tree_util.tree_map(
                            lambda a, b: a + self.lambda_ortho * b,
                            grads[i],
                            fid_grads["i"],
                        )
                        grads[j] = jax.tree_util.tree_map(
                            lambda a, b: a + self.lambda_ortho * b,
                            grads[j],
                            fid_grads["j"],
                        )

        # block diagonal SR (precondition per state)
        dp = []
        with prof_section("preconditioner_total (dispatch)", cat="vmc"):
            for i, (st, g) in enumerate(zip(self.states, grads)):
                with prof_section(
                    f"make_qgt_compatible_state_{i} (dispatch)", cat="vmc"
                ):
                    g = _make_grad_qgt_compatible(st.parameters, g)
                with prof_section(f"preconditioner_state_{i} (dispatch)", cat="vmc"):
                    d = self.preconditioner(st, g, self.step_count)
                with prof_section(f"tree_cast_state_{i} (dispatch)", cat="vmc"):
                    d = tree_cast(d, st.parameters)
                dp.append(d)

        self._last_energies = energies  # energies is list[Stats]
        self._loss_stats = float(sum(float(e.mean) for e in energies))

        return dp

    def _log_additional_data(self, log_dict: dict, step: int):
        """
        Add per-state objective statistics to the driver log dictionary.

        If per-state energies were computed in the last forward pass, this method exposes them
        under keys derived from the driver loss name with a state-index suffix, for example::

            "<loss_name> (state 0)"
            "<loss_name> (state 1)"

        :param log_dict: Mutable logging dictionary updated in place.
        :param step: Current optimisation step index.
        """

        super()._log_additional_data(log_dict, step)

        # easiest: store them in self._last_energies inside _forward_and_backward
        if hasattr(self, "_last_energies"):
            for i, st in enumerate(self._last_energies):
                log_dict[f"{self._loss_name} (state {i})"] = st

    def _estimate_stats(self, observable):
        """
        Estimate an observable on the multi-state variational state.

        This helper returns the raw per-state result from ``self.state.expect(observable)``.
        It exists to satisfy the base driver interface when the standard estimate pipeline is
        used.

        :param observable: Observable or pytree of observables to estimate.
        :return: Per-state expectation statistics as returned by the multi-state variational state.
        """

        return self.state.expect(observable)

    def estimate(self, observables):
        """
        Estimate observables and flatten per-state results into a logging-friendly dictionary.

        This overrides the base driver's :meth:`~neuralqx.driver.VMC.estimate` method to return a
        flat mapping from string keys to scalar statistics objects or values.

        For each observable leaf that yields a list of per-state statistics, entries are created
        as::

            "<name> (state i)" -> Stats

        Optionally, a simple aggregate curve is added as ``"<name>/sum_mean"`` containing the sum
        of per-state means when available. This aggregate is intended for quick diagnostics and
        is not a statistically rigorous combination of per-state uncertainties.

        :param observables: Observable or pytree of observables to estimate. If ``None``, an empty
                            set of observables is assumed.
        :return: Flat dictionary mapping names to per-state statistics objects or scalar values.
        """

        if observables is None:
            observables = {}

        # compute per-state stats for each observable using MultiMCState.expect()
        # this returns a pytree with leaves that are list[Stats] (length n_states)
        raw = super().estimate(observables)

        n_states = self.state.n_states

        # flatten into a dict[str, Stats] with keys "name/ (state {i})"
        flat: dict[str, Any] = {}

        # we'll re-walk `raw` ourselves
        def _walk(prefix, node):
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(prefix + (str(k),), v)
            else:
                # leaf: should be list[Stats] for MultiMCState
                name = "/".join(prefix) if prefix else "Observable"
                if isinstance(node, list) and len(node) == n_states:
                    for i, st in enumerate(node):
                        flat[f"{name} (state {i})"] = st
                    # also optionally store aggregate as mean over states
                    try:
                        # stats mean/sigma are not trivially averaged, so store list or first
                        # here: store the sum of means as a simple scalar curve
                        m = sum(float(getattr(st, "Mean", st.mean)) for st in node)
                        # build a dummy Stats-like scalar? easiest: keep list out of flat
                        flat[f"{name}/sum_mean"] = m
                    except Exception:
                        pass
                else:
                    # if user passed a scalar observable or something else, just store it.
                    flat[name] = node

        _walk((), raw)
        return flat
