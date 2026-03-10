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
from typing import Any, Optional, Tuple, Union, Callable

import numpy as np
import jax
import jax.numpy as jnp
from flax.core.scope import CollectionFilter

from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.optimizer import LinearOperator
from netket.sampler import Sampler
from netket.stats import Stats
from netket.utils.types import NNInitFunc
from netket.utils.types import PRNGKeyT
from netket.utils.types import PyTree


from neuralqx.vqs.mc.mc_state.state import MCState
from neuralqx.vqs.mc.mc_state.state import jit_evaluate
from neuralqx.vqs.mc.mc_state.state import local_estimators

from neuralqx.utils import distributed as _dist


def _is_master() -> bool:
    """
    Determine whether the current process should act as the global master for user-facing output.

    The master process is the one that should print logs, progress bars, and diagnostics in
    distributed runs.

    :return: ``True`` if this process is the global master, otherwise ``False``.
    """

    return bool(_dist.is_global_master())


def _get_stats_mean_sigma(stats_obj: Any) -> Tuple[float, float]:
    """
    Extract a numeric mean and standard error (sigma) from a Stats-like object or from a scalar.

    This helper supports common NetKet conventions where a statistics object exposes either
    ``Mean``/``Sigma`` (capitalized) or ``mean``/``sigma`` (lowercase). If no sigma is available,
    it is treated as ``0.0``.

    If extraction fails, ``(nan, nan)`` is returned.

    :param stats_obj: A NetKet ``Stats``-like object, or a scalar/array leaf.
    :return: Tuple ``(mean, sigma)`` as Python floats.
    """

    if stats_obj is None:
        return float("nan"), float("nan")

    # NetKet Stats .Mean/.Sigma
    for mk in ("Mean", "mean"):
        if hasattr(stats_obj, mk):
            break
    else:
        mk = None

    for sk in ("Sigma", "sigma"):
        if hasattr(stats_obj, sk):
            break
    else:
        sk = None

    if mk is not None:
        m = getattr(stats_obj, mk)
        try:
            m = float(np.asarray(np.real(m)))
        except Exception:
            m = float(np.real(m))
    else:
        # scalar leaf
        try:
            m = float(np.asarray(np.real(stats_obj)))
        except Exception:
            m = float(np.real(stats_obj))

    if sk is not None:
        s = getattr(stats_obj, sk)
        try:
            s = float(np.asarray(np.real(s)))
        except Exception:
            s = float(np.real(s))
    else:
        s = 0.0

    return m, s


@partial(jax.jit, static_argnums=(0, 1))
def _fidelity_expect_joint(
    apply_fun_i,
    apply_fun_j,
    machine_pow_i: jax.Array,
    machine_pow_j: jax.Array,
    params_i,
    model_state_i,
    params_j,
    model_state_j,
    sigma_i: jax.Array,
    sigma_j: jax.Array,
):
    """
    Estimate a fidelity-like overlap quantity between two variational states using a joint sample batch.

    This kernel builds a joint configuration batch ``sigma_xy`` by concatenating samples from
    state *i* (``x``) and state *j* (``y``), and then evaluates an estimator on independent draws
    ``x ~ p_i`` and ``y ~ p_j`` where

    .. math::

       p_i(x) \\propto |\\psi_i(x)|^{\\text{machine\\_pow}_i}, \\qquad
       p_j(y) \\propto |\\psi_j(y)|^{\\text{machine\\_pow}_j}.

    For ``machine_pow_i = machine_pow_j = 2`` this corresponds to the standard Born distribution
    and the expectation value equals the *normalized* quantum fidelity

    .. math::

       F_{ij}
       = \\frac{|\\langle \\psi_i | \\psi_j \\rangle|^2}{\\langle \\psi_i | \\psi_i \\rangle\\,\\langle \\psi_j | \\psi_j \\rangle}.

    More generally (for ``machine_pow != 2``), the returned quantity is still a consistent
    expectation value under the chosen sampling distributions but is **not** equal to the
    Hilbert-space fidelity unless ``machine_pow == 2``.

    Implementation notes
    --------------------
    - If samples include both chain and sweep dimensions, they are collapsed following NetKet
      conventions before forming the joint batch.
    - The two sample sets are truncated to the same number of configurations (the minimum of
      the two).
    - ``apply_fun_i`` and ``apply_fun_j`` are treated as static arguments under :func:`jax.jit`.

    :param apply_fun_i: Apply function for state *i* mapping variables and samples to ``log(psi)``.
    :param apply_fun_j: Apply function for state *j* mapping variables and samples to ``log(psi)``.
    :param machine_pow_i: Sampling power used for state *i* in the joint log density.
    :param machine_pow_j: Sampling power used for state *j* in the joint log density.
    :param params_i: Parameter pytree for state *i*.
    :param model_state_i: Model-state pytree for state *i* (everything except parameters).
    :param params_j: Parameter pytree for state *j*.
    :param model_state_j: Model-state pytree for state *j* (everything except parameters).
    :param sigma_i: Samples drawn from state *i* (may include chain dimensions).
    :param sigma_j: Samples drawn from state *j* (may include chain dimensions).

    :return: ``(fid_val, fid_stats)`` where ``fid_val`` is a real scalar and ``fid_stats`` is a
             NetKet ``Stats``-like object with uncertainty estimates.
    """

    # collapse chain dims like NetKet kernels do
    n_chains = sigma_i.shape[0]

    if sigma_i.ndim >= 3:
        sigma_i = jax.lax.collapse(sigma_i, 0, 2)
    if sigma_j.ndim >= 3:
        sigma_j = jax.lax.collapse(sigma_j, 0, 2)

    # Python int (static under jit)
    n = min(sigma_i.shape[0], sigma_j.shape[0])
    sigma_i, sigma_j = sigma_i[:n], sigma_j[:n]

    dim = sigma_i.shape[-1]
    sigma_xy = jnp.concatenate([sigma_i, sigma_j], axis=-1)

    def logpsi_i(pars, s):
        return apply_fun_i({"params": pars, **model_state_i}, s)

    def logpsi_j(pars, s):
        return apply_fun_j({"params": pars, **model_state_j}, s)

    # joint pdf for independent x~p_i and y~p_j (p ∝ |psi|^machine_pow)
    def log_pdf(pars, s_xy):
        x, y = s_xy[:, :dim], s_xy[:, dim:]
        return (
            machine_pow_i * logpsi_i(pars["i"], x).real
            + machine_pow_j * logpsi_j(pars["j"], y).real
        )

    # local estimator whose expectation gives fidelity (for machine_pow=2 it is the usual fidelity)
    def local_fidelity(pars, s_xy):
        x, y = s_xy[:, :dim], s_xy[:, dim:]
        li_x = logpsi_i(pars["i"], x)
        li_y = logpsi_i(pars["i"], y)
        lj_x = logpsi_j(pars["j"], x)
        lj_y = logpsi_j(pars["j"], y)
        return jnp.exp(lj_x + li_y - li_x - lj_y)

    pars0 = {"i": params_i, "j": params_j}

    fid_val, fid_stats = nkjax.expect(
        log_pdf, local_fidelity, pars0, sigma_xy, n_chains=n_chains
    )

    # make sure it's a real scalar (fidelity should be real)
    return fid_val.real, fid_stats


class MultiMCState:
    """
    Container for multiple :class:`~neuralqx.vqs.mc.mc_state.state.MCState` objects with overlap diagnostics.

    This class groups several independently sampled variational Monte Carlo states that share
    the same Hilbert space, and provides tools to compute pairwise overlap measures such as
    the fidelity matrix and derived quantities (overlap magnitude and orthogonality).

    All contained states **must** act on the same Hilbert space, a mismatch raises ``ValueError``.

    Parameters are managed per-state, no parameters are shared unless you explicitly tie them
    outside of this container.

    :param states: List of Monte Carlo variational states to manage.
    :raises ValueError: If ``states`` is empty or if any pair of states has a different Hilbert space.
    """

    def __init__(self, states: list[MCState]):
        """
        Create a multi-state container.

        This stores the provided list of states and validates that all states share the same
        Hilbert space (``state.hilbert``). The common Hilbert space is exposed through
        :attr:`hilbert`.

        :param states: List of :class:`~neuralqx.vqs.mc.mc_state.state.MCState` instances.
        :raises ValueError: If ``states`` is empty or if any pair of states has mismatching Hilbert spaces.
        """

        if len(states) == 0:
            raise ValueError("MultiMCState requires at least one MCState.")
        self.states = states

        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if state_i.hilbert != state_j.hilbert:
                    raise ValueError(
                        f"MultiMCState instances must share the same hilbert space. Your provided state at "
                        f"index {i} and {j} indices do not match."
                    )

        self._hilbert = states[0].hilbert

    def init_parameters(
        self, init_fun: NNInitFunc | None = None, *, seed: PRNGKeyT | None = None
    ):
        for state in self.states:
            state.init_parameters(init_fun=init_fun, seed=seed)

    @property
    def n_states(self) -> int:
        """
        Number of contained states.

        :return: The number of managed :class:`~neuralqx.vqs.mc.mc_state.state.MCState` instances.
        """

        return len(self.states)

    @property
    def n_samples(self) -> list[int]:
        """
        Return the number of samples for each state.

        :return: a list containing the number of samples for each state.
        """
        return [state.n_samples for state in self.states]

    @n_samples.setter
    def n_samples(self, values: int) -> None:
        """
        Set the number of samples for each state.

        :param values: the number of samples for each state.
        """
        for state in self.states:
            state.n_samples = values

    @property
    def sampler(self) -> list[Sampler]:
        """
        Return the sampler object for each state.

        :return: a list containing the sampler object for each state.
        """
        return [state.sampler for state in self.states]

    @property
    def model(self) -> Optional[list[Any]]:
        """
        Returns the model definition for each state.
        """
        return [state.model for state in self.states]

    @property
    def n_samples_per_rank(self) -> list[int]:
        """
        The number of samples generated per state on each MPI rank at each sampling step.
        """
        return [state.n_samples_per_rank for state in self.states]

    @n_samples_per_rank.setter
    def n_samples_per_rank(self, value: int) -> None:
        """
        Set the number of samples per rank at each sampling step for each state.
        """
        for state in self.states:
            state.n_samples_per_rank = value

    @property
    def chain_length(self) -> list[int]:
        """
        Length of the Markov chain used for sampling configurations for each state.

        Note: If running with MPI, the total samples will be n_nodes * chain_length * n_batches
        """
        return [state.chain_length for state in self.states]

    @chain_length.setter
    def chain_length(self, value: int) -> None:
        """
        Set the length of the Markov chain used for sampling configurations for each state.
        """
        for state in self.states:
            state.chain_length = value

    @property
    def n_discard_per_chain(self) -> list[int]:
        """
        Number of discarded samples at the beginning of the chain for each state.
        """
        return [state.n_discard_per_chain for state in self.states]

    @n_discard_per_chain.setter
    def n_discard_per_chain(self, value: int) -> None:
        """
        Set the number of discarded samples at the beginning of the chain for each state.
        """
        for state in self.states:
            state.n_discard_per_chain = value

    def sample(
        self,
        *,
        chain_length: Optional[int] = None,
        n_samples: Optional[int] = None,
        n_discard_per_chain: Optional[int] = None,
    ):
        """
        Sample a certain number of configurations for each state.

        If one among chain_length or n_samples is defined, that number of samples
        are generated. Otherwise the value set internally is used.

        :param chain_length: the length of the Markov chain used for sampling.
        :param n_samples: the total number of samples across all MPI ranks for each state.
        :param n_discard_per_chain: number of discarded samples at the beginning of the chain for each state.
        """
        for state in self.states:
            state.sample(
                chain_length=chain_length,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
            )

    @property
    def samples(self) -> list[jnp.ndarray]:
        """
        Return the set of cached samples for each state.
        """
        return [state.samples for state in self.states]

    def log_value(self, σ: jnp.ndarray) -> list[jnp.ndarray]:
        r"""
        Evaluate the each variational state for a batch of states and returns
        the logarithm of the amplitude of the quantum state.

        For pure states,
        this is :math:`\log(\langle\sigma|\psi\rangle)`, whereas for mixed states
        this is :math:`\log(\langle\sigma_r|\rho|\sigma_c\rangle)`, where
        :math:`\psi` and :math:`\rho` are respectively a pure state
        (wavefunction) and a mixed state (density matrix).
        For the density matrix, the left and right-acting states (row and column)
        are obtained as :code:`σr=σ[::,0:N]` and :code:`σc=σ[::,N:]`.

        Given a batch of inputs :code:`(Nb, N)`, returns a batch of outputs
        :code:`(Nb,)`.
        """
        return [
            jit_evaluate(state._apply_fun, state.variables, σ) for state in self.states
        ]

    def local_estimators(
        self, op: AbstractOperator, *, chunk_size: Optional[int] = None
    ):
        r"""
        Compute the local estimators for the operator :code:`op` (also known as local energies
        when :code:`op` is the Hamiltonian) at the current configuration samples :code:`self.samples` for
        each state in the :code:`self.states`.

        .. math::

            O_\mathrm{loc}(s) = \frac{\langle s | \mathtt{op} | \psi \rangle}{\langle s | \psi \rangle}

        .. warning::

            The samples differ between MPI processes, so returned the local estimators will
            also take different values on each process. To compute sample averages and similar
            quantities, you will need to perform explicit operations over all MPI ranks.
            (Use functions like :code:`self.expect` to get process-independent quantities without
            manual reductions.)

        Args:
            op: The operator.
            chunk_size: Suggested maximum size of the chunks used in forward and backward evaluations
                of the model. (Default: :code:`self.chunk_size`)
        """
        return [
            local_estimators(state, op, chunk_size=chunk_size) for state in self.states
        ]

    def expect_and_grad(
        self,
        O: AbstractOperator,
        *,
        mutable: Optional[CollectionFilter] = None,
        **kwargs,
    ) -> list[tuple[Stats, PyTree]]:
        r"""Estimates the quantum expectation value and its gradient
        for a given operator :math:`O` for each state in the :code:`self.states`.

        Args:
            O: The operator :math:`O` for which expectation value and
                gradient are computed.
            mutable: Can be bool, str, or list. Specifies which collections in the
                     model_state should be treated as  mutable: bool: all/no collections
                     are mutable. str: The name of a single mutable  collection. list: A
                     list of names of mutable collections. This is used to mutate the state
                     of the model while you train it (for example to implement BatchNorm. Consult
                     `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                     for a more in-depth explanation).
            use_covariance: whether to use the covariance formula, usually reserved for
                hermitian operators,
                :math:`\textrm{Cov}[\partial\log\psi, O_{\textrm{loc}}\rangle]`

        Returns: a list of
            - estimates of the quantum expectation value <O>.
            - estimates of the gradient of the quantum expectation value <O>.
        """
        return [
            state.expect_and_grad(O, mutable=mutable, **kwargs) for state in self.states
        ]

    def expect_and_forces(
        self,
        O: AbstractOperator,
        *,
        mutable: Optional[CollectionFilter] = None,
    ) -> list[tuple[Stats, PyTree]]:
        r"""Estimates the quantum expectation value and the corresponding force
        vector for a given operator O for each state in the :code:`self.states`.

        The force vector :math:`F_j` is defined as the covariance of log-derivative
        of the trial wave function and the local estimators of the operator. For complex
        holomorphic states, this is equivalent to the expectation gradient
        :math:`\frac{\partial\langle O\rangle}{\partial(\theta_j)^\star} = F_j`. For real-parameter states,
        the gradient is given by :math:`\frac{\partial\partial_j\langle O\rangle}{\partial\partial_j\theta_j} = 2 \textrm{Re}[F_j]`.

        Args:
            O: The operator O for which expectation value and force are computed.
            mutable: Can be bool, str, or list. Specifies which collections in
                the model_state should be treated as  mutable: bool: all/no
                collections are mutable. str: The name of a single mutable
                collection. list: A list of names of mutable collections. This is
                used to mutate the state of the model while you train it (for
                example to implement BatchNorm. Consult
                `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                for a more in-depth explanation).

        Returns: a list of
            - estimates of the quantum expectation value <O>.
            - estimates of the force vector
            :math:`F_j = \textrm{Cov}[\partial_j\log\psi, O_{\textrm{loc}}]`.
        """
        return [state.expect_and_forces(O, mutable=mutable) for state in self.states]

    def quantum_geometric_tensor(
        self, qgt_T: Optional[LinearOperator] = None
    ) -> list[LinearOperator]:
        r"""Computes an estimate of the quantum geometric tensor G_ij for each state in the :code:`self.states`.
        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.
        """
        return [state.quantum_geometric_tensor(qgt_T) for state in self.states]

    def to_array(
        self, normalize: bool = True, *, state: Optional[int] = None
    ) -> Union[list[jnp.ndarray], jnp.ndarray]:
        if state is None:
            return [state.to_array(normalize=normalize) for state in self.states]

        if state > self.n_states:
            raise ValueError(f"There are only {self.n_states} states available.")
        return self.states[state].to_array(normalize=normalize)

    def project(
        self,
        wrap_model: Callable[..., Any],
        *,
        reuse_cached_samples: bool = True,
        **wrapper_kwargs,
    ) -> "MultiMCState":
        """
        Generic model-projection method.

        Wraps the current ansatz with an arbitrary `wrap_model(self.model, **kwargs)` and returns a
        new MultiMCState whose variables contain the trained parameters under the wrapper's `"base"`
        scope.

        This wraps each state in the states saved in this container.

        :param wrap_model: A function that projects the model in any way desired, i.e.
          wrap_model(model, **kwargs) -> wrapped_model.

        :param reuse_cached_samples: If True and cached samples are available, reuse them. This is
          safe for symmetry projections that do not alter sampling distributions.

        :param wrapper_kwargs: Extra keyword arguments forwarded to wrap_model().

        :returns: A new MCState with the wrapped model and transplanted parameters.
        """
        raise NotImplementedError(
            "This feature is still under development. If you are interested, please get in touch!"
        )

    def to_group_averaged(
        self,
        *,
        symmetries=None,
        graph=None,
        index_perms=None,
        characters=None,
        reuse_cached_samples=True,
    ) -> "MultiMCState":
        """
        Return a NEW MultiMCState that evaluates the group-projected wavefunction built on top of
        the trained vanilla model parameters, reusing the SAME sampler, SamplerState,
        n_samples, n_discard_per_chain, chunk_size, and mutables policy.
        """
        raise NotImplementedError(
            "This feature is still under development. If you are interested, please get in touch!"
        )

    @property
    def hilbert(self):
        """
        Shared Hilbert space.

        All contained states are required to have this same Hilbert space.

        :return: The common Hilbert space object.
        """

        return self._hilbert

    @property
    def parameters(self) -> list[Any]:
        """
        Parameters of all contained states.

        :return: A list of parameter pytrees, one per state, in the same order as :attr:`states`.
        """

        return [s.parameters for s in self.states]

    @parameters.setter
    def parameters(self, pars) -> None:
        """
        Set the parameters of all contained states.

        The input is zipped with the current list of states, if the sequence is shorter than the
        number of states, remaining states are unchanged. For best practice, pass a sequence of
        length :attr:`n_states`.

        :param pars: Sequence of parameter pytrees matching the number and structure of states.
        """

        for s, p in zip(self.states, pars):
            s.parameters = p

    def reset(self) -> None:
        """
        Reset all contained states.

        This typically discards cached samples and statistics so that subsequent estimators use
        fresh Monte Carlo data generated by each state's sampler.
        """

        for s in self.states:
            s.reset()

    def expect(self, O) -> list[Stats]:
        """
        Compute expectation values of an operator/observable on all states.

        This is a convenience wrapper around calling ``state.expect(O)`` on each contained state.

        :param O: Operator or observable to evaluate.
        :return: List of statistics objects, one per state, as returned by each state's ``expect``.
        """

        return [s.expect(O) for s in self.states]

    #
    #
    #   overlap / orthogonality diagnostics

    def fidelity_matrix(
        self,
        *,
        resample: bool = False,
        return_sigma: bool = False,
        assume_machine_pow_2: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate the pairwise fidelity matrix between all contained states.

        The returned matrix ``F`` has entries ``F[i, j]`` estimating the magnitude-squared overlap
        between state *i* and state *j*. The diagonal is set to ``1`` by construction.

        For ``machine_pow == 2`` in both samplers, the estimator corresponds to the normalised
        pure-state fidelity

        .. math::

           F_{ij} = \\frac{|\\langle \\psi_i | \\psi_j \\rangle|^2}
                         {\\langle \\psi_i | \\psi_i \\rangle\\,\\langle \\psi_j | \\psi_j \\rangle}.

        Uncertainty handling
        --------------------
        When available, the returned sigma values are taken from the NetKet statistics produced by
        :func:`netket.jax.expect`. These are only as meaningful as the chain structure information
        provided to the estimator.

        .. warning::
           The standard quantum fidelity interpretation is exact only for ``machine_pow == 2``.
           If ``assume_machine_pow_2`` is ``True`` and any state uses a different ``machine_pow``,
           this method raises ``ValueError``.

        :param resample: If ``True``, call :meth:`reset` before computing the matrix.
        :param return_sigma: If ``True``, also return an estimated standard error matrix ``dF``.
        :param assume_machine_pow_2: Enforce ``machine_pow == 2`` for all states.

        :return: If ``return_sigma`` is ``False``, returns ``F`` with shape ``(n_states, n_states)``.
                 If ``return_sigma`` is ``True``, returns ``(F, dF)`` where ``dF`` is an error estimate
                 with the same shape.

        :raises ValueError: If ``assume_machine_pow_2`` is ``True`` and any state uses ``machine_pow != 2``.
        """

        if resample:
            self.reset()

        n = self.n_states
        F = np.eye(n, dtype=float)
        dF = np.zeros((n, n), dtype=float)

        # sanity: machine_pow
        if assume_machine_pow_2 and _is_master():
            bad = []
            for i, st in enumerate(self.states):
                mp = getattr(getattr(st, "sampler", None), "machine_pow", 2)
                if int(mp) != 2:
                    bad.append((i, int(mp)))
            if bad:
                raise ValueError(
                    "[MultiMCState] fidelity_matrix() is the standard quantum fidelity "
                    "when machine_pow=2. Found machine_pow != 2 for states: "
                    + ", ".join([f"{i}:{mp}" for i, mp in bad])
                )

        for i in range(n):
            si = self.states[i]
            for j in range(i + 1, n):
                sj = self.states[j]

                mp_i = jnp.asarray(
                    int(getattr(si.sampler, "machine_pow", 2)), dtype=jnp.float32
                )
                mp_j = jnp.asarray(
                    int(getattr(sj.sampler, "machine_pow", 2)), dtype=jnp.float32
                )

                fid_val, fid_stats = _fidelity_expect_joint(
                    si._apply_fun,
                    sj._apply_fun,
                    mp_i,
                    mp_j,
                    si.parameters,
                    si.model_state,
                    sj.parameters,
                    sj.model_state,
                    si.samples,
                    sj.samples,
                )

                # in some NetKet versions fid_val and fid_stats.mean agree, stats also gives sigma.
                m, s = _get_stats_mean_sigma(fid_stats)
                if not np.isfinite(m):
                    # fallback to fid_val if stats extraction failed
                    m = float(np.asarray(fid_val))
                    s = 0.0

                # clamp numerically (fidelity should be in [0,1], small MC noise may violate)
                m = float(np.clip(m, 0.0, 1.0))

                F[i, j] = m
                F[j, i] = m
                dF[i, j] = s
                dF[j, i] = s

        return (F, dF) if return_sigma else F

    def overlap_matrix(
        self,
        *,
        kind: str = "fidelity",
        resample: bool = False,
        return_sigma: bool = False,
        assume_machine_pow_2: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute an overlap-style matrix derived from the fidelity matrix.

        This is a lightweight post-processing wrapper around :meth:`fidelity_matrix`.

        Supported values of ``kind`` are:

        - ``"fidelity"``: returns ``F[i, j]``.
        - ``"overlap"``: returns ``sqrt(F[i, j])`` (magnitude of overlap).
        - ``"orthogonality"``: returns ``1 - F[i, j]``.

        If ``return_sigma`` is ``True`` and ``kind == "overlap"``, a simple error propagation is used:

        .. math::

           \\sigma_{\\sqrt{F}} \\approx \\frac{\\sigma_F}{2\\sqrt{F}}.

        :param kind: Which matrix to return: ``"fidelity"``, ``"overlap"``, or ``"orthogonality"``.
        :param resample: If ``True``, call :meth:`reset` before computing.
        :param return_sigma: If ``True``, also return an estimated standard error matrix.
        :param assume_machine_pow_2: Forwarded to :meth:`fidelity_matrix`.

        :return: The requested matrix ``M`` (and optionally its uncertainty ``dM``).

        :raises ValueError: If ``kind`` is not one of the supported options.
        """

        kind = str(kind).lower().strip()
        out = self.fidelity_matrix(
            resample=resample,
            return_sigma=return_sigma,
            assume_machine_pow_2=assume_machine_pow_2,
        )

        if return_sigma:
            F, dF = out  # type: ignore[misc]
        else:
            F = out  # type: ignore[assignment]

        if kind == "fidelity":
            return (F, dF) if return_sigma else F

        if kind == "orthogonality":
            G = 1.0 - F
            return (G, dF) if return_sigma else G

        if kind == "overlap":
            G = np.sqrt(np.clip(F, 0.0, 1.0))
            if not return_sigma:
                return G
            # crude error propagation: d(sqrt(F)) ≈ dF / (2*sqrt(F))
            denom = 2.0 * np.maximum(G, 1e-12)
            dG = dF / denom
            return G, dG

        raise ValueError(
            f"Unknown kind='{kind}'. Use 'fidelity', 'overlap', or 'orthogonality'."
        )

    def print_overlap_matrix(
        self,
        *,
        kind: str = "fidelity",
        resample: bool = False,
        digits: int = 3,
        show_sigma: bool = False,
        assume_machine_pow_2: bool = True,
    ) -> None:
        """
        Pretty-print an overlap-style matrix on the master process.

        The matrix is computed on all ranks (to keep any potential collectives consistent),
        but printing is performed only on the global master as determined by :func:`_is_master`.

        :param kind: Which matrix to print: ``"fidelity"``, ``"overlap"``, or ``"orthogonality"``.
        :param resample: If ``True``, call :meth:`reset` before computing.
        :param digits: Number of decimal digits used for formatting.
        :param show_sigma: If ``True``, also print uncertainty estimates when available.
        :param assume_machine_pow_2: Forwarded to :meth:`overlap_matrix`.
        """

        if not _is_master():
            # still compute (MPI collectives may be inside), but do not print
            _ = self.overlap_matrix(
                kind=kind,
                resample=resample,
                return_sigma=show_sigma,
                assume_machine_pow_2=assume_machine_pow_2,
            )
            return

        if show_sigma:
            M, dM = self.overlap_matrix(
                kind=kind,
                resample=resample,
                return_sigma=True,
                assume_machine_pow_2=assume_machine_pow_2,
            )
        else:
            M = self.overlap_matrix(
                kind=kind,
                resample=resample,
                return_sigma=False,
                assume_machine_pow_2=assume_machine_pow_2,
            )
            dM = None

        n = self.n_states
        title = f"{kind.strip().upper()} matrix (n_states={n})"
        print("\n" + title)
        print("-" * len(title))

        # header
        hdr = "      " + " ".join([f"{j:>10d}" for j in range(n)])
        print(hdr)

        fmt = f"{{:>{digits+6}.{digits}f}}"
        for i in range(n):
            row = [f"{i:>3d} | "]
            for j in range(n):
                if dM is None:
                    row.append(fmt.format(M[i, j]))
                else:
                    row.append(
                        (fmt.format(M[i, j]) + "±" + f"{dM[i,j]:.{digits}f}").rjust(
                            10 + 1 + digits + 1
                        )
                    )
            print("".join(row))

        print("")

    def __repr__(self):
        return (
            "MultiMCState("
            + f"\n  states = {self.states},"
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  sampler = {self.sampler},"
            + f"\n  n_samples = {self.n_samples},"
            + f"\n  n_discard_per_chain = {self.n_discard_per_chain},"
            + f"\n  sampler_state = {self.sampler_state},"
            + f"\n  n_parameters = {self.n_parameters},"
            + f"\n  is_group_averaged = {self._is_group_averaged})"
        )
