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


"""Abstract variational-state interfaces."""

from __future__ import annotations

import abc
from typing import Any

import jax
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator import AbstractOperator
from neuralqx.utils.frameworks import AbstractModelFramework


class AbstractVariationalState(abc.ABC):
    r"""Interface between a log-amplitude model and discrete-state estimators.

    A state supplies the Hilbert basis, trainable parameters, non-trainable
    model collections, and configurations used for Monte Carlo estimation.
    Model evaluation returns :math:`\log\psi_\theta(\sigma)`, with one scalar
    per flat Hilbert configuration. Sampling, caching, and variable storage
    belong to the concrete state.

    The methods defined here delegate to the estimator and optimizer APIs.
    They provide a common entry point for local values, expectations,
    gradients, and the quantum geometric tensor. Operator-specific estimator
    rules remain in the VQS dispatch layer, so a subclass does not need to
    implement each operator family separately.

    Subclasses using these methods must also supply the evaluation kernels
    and sampling metadata consumed by those APIs, including
    ``apply_variables_kernel``, ``merge_variables_kernel``, and ``machine_pow``.
    See ``VariationalState`` for the concrete sampled implementation and its
    cache-invalidation behavior.
    """

    @property
    @abc.abstractmethod
    def hilbert(self) -> DiscreteHilbertSpace:
        """Hilbert space defining the basis and trailing configuration dimension."""

    @property
    @abc.abstractmethod
    def framework(self) -> AbstractModelFramework:
        """Adapter for model initialization, evaluation, and variable management."""

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        """Trainable parameter pytree used by differentiation and optimization."""

    @parameters.setter
    @abc.abstractmethod
    def parameters(self, value: Any) -> None:
        """Update trainable model parameters."""

    @property
    @abc.abstractmethod
    def model_state(self) -> Any:
        """Non-trainable model collections held fixed during differentiation."""

    @model_state.setter
    @abc.abstractmethod
    def model_state(self, value: Any) -> None:
        """Update non-trainable model state."""

    @property
    @abc.abstractmethod
    def variables(self) -> Any:
        """Complete framework variables combining parameters and model state."""

    @variables.setter
    @abc.abstractmethod
    def variables(self, value: Any) -> None:
        """Update full framework variables."""

    @property
    @abc.abstractmethod
    def samples(self) -> jax.Array:
        """Configurations used by estimators, generating them if necessary.

        The trailing dimension is ``hilbert.size``. Sampled states normally
        retain separate leading chain and time axes for statistics.
        """

    @property
    @abc.abstractmethod
    def chunk_size(self) -> int | None:
        """Default model-evaluation batch limit, or ``None`` for no chunking."""

    def local_estimators(self, operator: Any, **kwargs: Any):
        r"""Evaluate an operator's local estimator on the sampled configurations.

        For a discrete operator :math:`A`, the ordinary row estimator is

        .. math::

            A_{\mathrm{loc}}(\sigma)=
            \sum_{\sigma'} A_{\sigma,\sigma'}
            \frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.

        ``Squared(A)`` instead uses :math:`|A_{\mathrm{loc}}|^2`, whose Born
        average gives :math:`\langle A^\dagger A\rangle`. An operator sequence
        contributes the sum of its estimators for each sample.

        Args:
            operator: Supported operator, wrapper, or sequence on this state's
                Hilbert space.
            **kwargs: Options forwarded to ``vqs.local_estimators``. These
                include ``samples`` to replace the cached batch for this call,
                ``chunk_size`` for model evaluation, and operator streaming,
                sharding, and connection-deduplication settings.

        Returns:
            A ``LocalEstimators`` object. Its ``data`` follows the leading
            shape of the sample batch and ``to_stats()`` summarizes the values.
        """
        from neuralqx.vqs.estimators import local_estimators

        return local_estimators(self, operator, **kwargs)

    def expect(self, operator: Any, **kwargs: Any):
        r"""Estimate an operator expectation using the current sample batch.

        With Born sampling, the mean estimates
        :math:`\langle\psi|A|\psi\rangle/\langle\psi|\psi\rangle`.
        Calling this method does not advance chains when samples are already
        cached. Use ``sample()`` on a sampled state to collect a fresh batch.

        Args:
            operator: Supported operator, wrapper, or operator sequence.
            **kwargs: Local-estimator options, including an explicit
                ``samples`` batch or a model ``chunk_size`` override.

        Returns:
            ``Stats`` containing the estimated mean, error of the mean,
            variance, and chain diagnostics. For wrappers such as ``Squared``,
            the reported variance describes the chosen local estimator.
        """
        from neuralqx.vqs.expectation import expect

        return expect(self, operator, **kwargs)

    def expect_and_forces(self, operator: AbstractOperator, **kwargs: Any):
        r"""Estimate a Hermitian expectation and its covariance forces.

        For real parameter coordinates, let
        :math:`O_i=\partial_{\theta_i}\log\psi_\theta`. The force is

        .. math::

            F_i=\mathbb{E}_p\!\left[
                O_i^*\left(A_{\mathrm{loc}}
                -\mathbb{E}_p[A_{\mathrm{loc}}]\right)\right].

        For Born sampling and a Hermitian operator, the real-parameter
        gradient is :math:`2\operatorname{Re}F_i`. Use
        :meth:`expect_and_grad` when an optimizer-ready gradient is needed.
        Positive estimators from ``Squared`` and ``SumOfSquares`` require
        additional parameter derivatives and are rejected by this method.

        Args:
            operator: Supported Hermitian operator or Hermitian sequence with
                a valid covariance estimator.
            **kwargs: Estimator options, including ``samples`` and ``chunk_size``.

        Returns:
            ``(stats, forces)``, with a force pytree following the trainable
            parameter structure. Force leaves may be complex even when the
            corresponding parameters are real.

        Raises:
            ValueError: The operator does not support the covariance-force path.
        """
        from neuralqx.vqs.expectation import expect_and_forces

        return expect_and_forces(self, operator, **kwargs)

    def expect_and_grad(self, operator: AbstractOperator, **kwargs: Any):
        r"""Estimate an expectation and its gradient for optimization.

        The estimator chooses the covariance formula when valid and otherwise
        includes the local estimator's explicit parameter dependence. In
        particular, ``Squared`` and ``SumOfSquares`` keep that direct
        derivative. Their Hermiticity alone does not justify using only the
        covariance of their positive local values.

        Args:
            operator: Supported operator, wrapper, or operator sequence.
            **kwargs: Estimator options such as ``samples`` and ``chunk_size``.
                ``use_covariance=None`` selects the gradient path automatically.
                ``preconditioner`` optionally transforms the resulting gradient
                and receives the optional ``step`` value.

        Returns:
            ``(stats, gradient)`` with the parameter pytree structure. For a
            real objective and a complex parameter :math:`z=x+iy`, the
            optimizer convention is :math:`\partial_x E+i\partial_y E`.
            If a preconditioner is supplied, the second result is the
            preconditioned update direction.
        """
        from neuralqx.vqs.expectation import expect_and_grad

        return expect_and_grad(self, operator, **kwargs)

    def quantum_geometric_tensor(self, **kwargs: Any):
        r"""Construct the matrix-free QGT from current parameters and samples.

        Writing :math:`O_i=\partial_{\theta_i}\log\psi_\theta`, the complex
        covariance is

        .. math::

            S_{ij}=\mathbb{E}_p[O_i^*O_j]
                -\mathbb{E}_p[O_i^*]\mathbb{E}_p[O_j].

        The selected differentiation mode determines whether the operator
        uses complex holomorphic coordinates or the real metric on real
        parameter coordinates. No dense Jacobian is stored. Rebuild the QGT
        after changing parameters or samples to use the updated state.

        Args:
            **kwargs: Arguments forwarded to ``optimizer.qgt_onthefly``, such
                as ``diag_shift``, ``holomorphic``, ``mode``, ``samples``,
                ``sample_weights``, and ``chunk_size``.

        Returns:
            A ``QGTOnTheFly`` linear operator supporting matrix-vector products
            and linear solves for stochastic reconfiguration.
        """
        from neuralqx.optimizer import qgt_onthefly

        return qgt_onthefly(self, **kwargs)

    def qgt(self, **kwargs: Any):
        """Construct the default QGT through :meth:`quantum_geometric_tensor`.

        Keyword arguments and the returned linear operator are identical to
        those of the full method name.
        """
        return self.quantum_geometric_tensor(**kwargs)

    @abc.abstractmethod
    def apply_variables(self, variables: Any, states: Any) -> Any:
        """Evaluate supplied framework variables without replacing stored variables.

        ``states`` is a model input batch with a trailing Hilbert dimension.
        The result contains log amplitudes, rather than amplitudes or
        probabilities.
        """

    @abc.abstractmethod
    def log_value(self, states: Any, **kwargs: Any) -> jax.Array:
        """Evaluate log amplitudes while preserving leading configuration axes.

        A single configuration has shape ``(hilbert.size,)``. A batch has shape
        ``(..., hilbert.size)`` and produces one scalar per configuration.
        """

    @abc.abstractmethod
    def to_array(self, **kwargs: Any) -> jax.Array:
        """Evaluate amplitudes in Hilbert basis order through exact enumeration.

        The concrete state controls normalization, enumeration limits, and
        model batching. This operation requires an indexable Hilbert space.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Invalidate samples and synchronize sampling state with model variables.

        Concrete sampler settings determine whether chain configurations are
        retained or redrawn.
        """


__all__ = ["AbstractVariationalState"]
