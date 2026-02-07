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

# Copyright 2021 The NetKet Authors - All rights reserved.
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

"""
Variational Monte Carlo driver adapted from NetKet.

This module provides a VMC driver that closely follows NetKet's implementation, with
a small behavioural difference around operator handling. In particular, it avoids
explicitly unwrapping a Squared lazy wrapper operator type and instead forwards the
Hamiltonian object directly to the variational state's expectation and gradient routines.

The driver supports passing either a single operator or a list of operators. It checks
that all operators act on the same Hilbert space as the provided variational state.
It then performs gradient based optimisation of the minimized quantity, labelled as
"Constraint" in the driver.

NOTE: parts of this file are derived from NetKet source code, and the original NetKet
copyright and license apply in addition to neuraLQX licensing.
"""

from textwrap import dedent

from netket.utils.types import PyTree
from netket.utils.types import Optimizer
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.optimizer import identity_preconditioner
from netket.optimizer import PreconditionerT
from netket.vqs import VariationalState

from neuralqx.driver.abstract_variational_driver import AbstractVariationalDriver


class VMC(AbstractVariationalDriver):
    """
    Variational Monte Carlo optimisation driver.

    This driver minimises the expectation value of a Hamiltonian like objective using
    samples produced by the provided variational state. It computes the loss statistics
    and gradient via `variational_state.expect_and_grad`, optionally applies a
    preconditioner, and forwards the resulting update direction to the underlying
    `AbstractVariationalDriver` optimisation loop.

    :param hamiltonian: The operator to minimise, or a list of operators to be passed
        directly to `variational_state.expect_and_grad`.
    :param optimizer: Optimiser defining how parameter updates are applied given a gradient.
    :param variational_state: Variational state that supplies sampling, expectation values,
        and gradients.
    :param preconditioner: Callable that transforms the raw gradient before optimisation.
        If None, the identity preconditioner is used.
    :raises TypeError: If any operator Hilbert space differs from `variational_state.hilbert`.
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator | list,
        optimizer: Optimizer,
        *,
        variational_state: VariationalState,
        preconditioner: PreconditionerT = identity_preconditioner,
    ):
        """
        Initialise the VMC driver.

        Ensures the Hamiltonian is stored as a list of operators, validates that all operators
        share the same Hilbert space as the variational state, and configures the driver to
        minimise a quantity named "Constraint".

        :param hamiltonian: The operator to minimise, or a list of operators.
        :param optimizer: Optimiser used to update parameters from a (preconditioned) gradient.
        :param variational_state: Variational state providing sampling and gradient evaluation.
        :param preconditioner: Preconditioner used to modify the raw loss gradient before it is
            passed to the optimiser. If None, the identity preconditioner is used.
        :return: None.
        :raises TypeError: If `variational_state.hilbert` does not match any operator Hilbert space.
        """

        # ensure that we have at least one Hamiltonian wrapped in a list
        if isinstance(hamiltonian, AbstractOperator):
            hamiltonian = [hamiltonian]

        # check the hilbert spaces per Hamiltonian
        for _h in hamiltonian:
            if variational_state.hilbert != _h.hilbert:
                raise TypeError(
                    dedent(
                        f"""the variational_state has hilbert space {variational_state.hilbert}
                        (this is normally defined by the hilbert space in the sampler), but
                        the hamiltonian has hilbert space {_h.hilbert}.
                        The two should match.
                        """
                    )
                )

        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Constraint"
        )

        self._ham = hamiltonian

        self.preconditioner = preconditioner

        self._dp: PyTree = None
        self._S = None

    @property
    def preconditioner(self):
        """
        Return the preconditioner used to modify the raw gradient.

        The preconditioner is a callable with signature

            preconditioner(vstate, grad, step)

        where `vstate` is the variational state, `grad` is the PyTree gradient, and `step` is
        the current optimisation step count. If the preconditioner was set to None, the
        identity preconditioner is used.

        :return: The current preconditioner callable.
        """

        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: PreconditionerT | None):
        """
        Set the preconditioner used to modify the raw gradient.

        If `val` is None, the identity preconditioner is used.

        :param val: Preconditioner callable, or None to select the identity preconditioner.
        :return: None.
        """

        if val is None:
            val = identity_preconditioner

        self._preconditioner = val

    @property
    def energy(self) -> Stats:
        """
        Return the most recently computed loss statistics.

        The returned value corresponds to the statistics produced by the last call to the
        forward and backward evaluation, and represents the expectation value being minimised
        by this driver.

        :return: A `netket.stats.Stats` instance for the current objective estimate.
        """

        return self._loss_stats

    def __repr__(self):
        return (
            "Vmc("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
