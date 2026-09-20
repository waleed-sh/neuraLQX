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
    """Base API for variational states."""

    @property
    @abc.abstractmethod
    def hilbert(self) -> DiscreteHilbertSpace:
        """Hilbert space supporting the variational state."""

    @property
    @abc.abstractmethod
    def framework(self) -> AbstractModelFramework:
        """Framework adapter used to evaluate the model."""

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        """Trainable model parameters."""

    @parameters.setter
    @abc.abstractmethod
    def parameters(self, value: Any) -> None:
        """Update trainable model parameters."""

    @property
    @abc.abstractmethod
    def model_state(self) -> Any:
        """Non-trainable model state."""

    @model_state.setter
    @abc.abstractmethod
    def model_state(self, value: Any) -> None:
        """Update non-trainable model state."""

    @property
    @abc.abstractmethod
    def variables(self) -> Any:
        """Full framework variables."""

    @variables.setter
    @abc.abstractmethod
    def variables(self, value: Any) -> None:
        """Update full framework variables."""

    @property
    @abc.abstractmethod
    def samples(self) -> jax.Array:
        """Current cached samples, generating them if necessary."""

    @property
    @abc.abstractmethod
    def chunk_size(self) -> int | None:
        """Model-evaluation chunk size."""

    def local_estimators(self, operator: Any, **kwargs: Any):
        """Return per-sample local estimators."""
        from neuralqx.vqs.estimators import local_estimators

        return local_estimators(self, operator, **kwargs)

    def expect(self, operator: Any, **kwargs: Any):
        """Return expectation statistics for ``operator``."""
        from neuralqx.vqs.expectation import expect

        return expect(self, operator, **kwargs)

    def expect_and_forces(self, operator: AbstractOperator, **kwargs: Any):
        """Return expectation statistics and covariance forces."""
        from neuralqx.vqs.expectation import expect_and_forces

        return expect_and_forces(self, operator, **kwargs)

    def expect_and_grad(self, operator: AbstractOperator, **kwargs: Any):
        """Return expectation statistics and optimizer gradient."""
        from neuralqx.vqs.expectation import expect_and_grad

        return expect_and_grad(self, operator, **kwargs)

    def quantum_geometric_tensor(self, **kwargs: Any):
        """Return the default matrix-free quantum geometric tensor."""
        from neuralqx.optimizer import qgt_onthefly

        return qgt_onthefly(self, **kwargs)

    def qgt(self, **kwargs: Any):
        """Alias for :meth:`quantum_geometric_tensor`."""
        return self.quantum_geometric_tensor(**kwargs)

    @abc.abstractmethod
    def apply_variables(self, variables: Any, states: Any) -> Any:
        """Evaluate arbitrary framework variables on a batch of states."""

    @abc.abstractmethod
    def log_value(self, states: Any, **kwargs: Any) -> jax.Array:
        """Evaluate log-amplitudes on one or more basis states."""

    @abc.abstractmethod
    def to_array(self, **kwargs: Any) -> jax.Array:
        """Materialize the full wavefunction on an indexable Hilbert space."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset cached samples and sampler state."""


__all__ = ["AbstractVariationalState"]
