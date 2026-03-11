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


import abc
import logging
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.operator import LocalOperator

from humanize import scientific

from .abstract_lqx_model import AbstractLqxModel
from neuralqx.debug import timeit, event
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.operators.types import ComputationalOperator
from neuralqx.operators.types import ComputationalJaxOperator
from neuralqx.utils.io.printing import NQXPrinter

# convenience type helper
ArrayT = Union[jnp.ndarray, np.ndarray]


class AbstractLqxInterface(abc.ABC):
    """
    An interface for arbitrary LQX models. This interface should provide access to the constraint
    of the model, the gauge group, Hilbert space and graph.
    """

    _graph: AbstractGraph
    """The graph the LQX model will be defined on."""

    _hilbert: AbstractHilbertInterface
    """The Hilbert space interface the model will be defined on."""

    _gauge_group: AbstractGaugeGroup
    """The gauge group to be used in the LQX model"""

    _model: AbstractLqxModel
    """A class implementing the actual LQG model."""

    printer: NQXPrinter = NQXPrinter()
    """A printer used to output formatted messages to the user."""

    _model_name: str
    """A name for the model being implemented used for logging."""

    _ground_state: Optional[ArrayT] = None
    """The ground state obtained from exact diagonalisation methods, if possible."""

    _ground_energy: Optional[Union[int, float, complex]] = None
    """The minimum eigenvalue of the constraint computed from exact diagonalisation methods."""

    _user_given_ground_energy: Optional[Union[int, float, complex]] = None
    """A ground energy supplied from the user."""

    _constraint: Union[
        LocalOperator, ComputationalOperator, ComputationalJaxOperator
    ] = None
    """The constraint implementation of the model."""

    def __init__(
        self,
        H: AbstractHilbertInterface,
        gauge_group: AbstractGaugeGroup,
        *,
        lazy_load: bool = True,
        model_name: str = "",
    ):
        event(
            msg="INITIALIZING LQX INTERFACE",
            tag="LQXINTERFACE:INIT",
            level=logging.INFO,
            lazy_load=lazy_load,
            model_name=model_name,
            gauge_group=gauge_group,
            H=H,
        )

        # local copies
        self._hilbert = H
        self._graph = H.graph
        self._gauge_group = gauge_group
        self._model_name = model_name

        # NOTE:
        #   any subclass should implement the model before the lazy loading is prompted as the
        #   loading would require an implemented model (the constraint of the interface is created
        #   from the constraint of the model)

        # construct constraints if not lazy
        if not lazy_load:
            self.initialize_constraint()

    @property
    def hilbert(self):
        return self._hilbert

    @property
    def graph(self):
        return self._graph

    @property
    def gauge_group(self):
        return self._gauge_group

    @property
    def constraint(
        self,
    ) -> Union[LocalOperator, ComputationalOperator, ComputationalJaxOperator]:
        """
        Any subclass of the AbstractLqxInterface should implement this property which should yield
        back the constraint of the model. The constraint encoded here will be the one minimised.
        """
        if self._constraint is None:
            self.initialize_constraint()
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        """
        This should be a setter for the constraint of the model, allowing easy runtime adjustment
        of the constraint attempted to be minimised.
        """
        self._constraint = value

    @property
    def hilbert_dimensions(self):
        return self.hilbert.dimensions

    @property
    def gauge_dimensions(self):
        return self.hilbert.gauge_dimensions

    @property
    def spacetime_dimensions(self):
        return self.model.spacetime_dimensions

    @property
    def model_name(self):
        return self._model_name

    @property
    def model(self):
        return self._model

    @property
    def ground_state(self):
        # check if we already computed this
        if self._ground_state is None:
            # check if the user previously supplied a ground energy value
            if self._ground_energy is not None:
                raise ValueError(
                    f"The current ground energy was explicitly supplied previously. In that case,"
                    f"the corresponding ground state remains undefined."
                    f"\n"
                    f"If you wish to obtain a ground state, please use `exact_diagonalization()`"
                    f"without supplying any parameters to it to run the Lanczos algorithm."
                    f"\n"
                    f"Note that in this case, your previously supplied ground energy will be "
                    f"overwritten."
                )
            # if not, try to compute if we do not require too much RAM
            if self._maybe_diagonalised():
                self.exact_diagonalization()
        return self._ground_state

    @property
    def ground_energy(self):
        # check if we already computed this
        if self._ground_energy is None:
            # if not, try to compute if we do not require too much RAM
            if self._maybe_diagonalised():
                self.exact_diagonalization()
        return self._ground_energy

    @property
    def has_ground_energy(self):
        """
        Returns True if a ground energy is already available (either user-supplied or computed),
        without triggering exact diagonalisation.
        """
        return self._ground_energy is not None

    @property
    def has_ground_state(self) -> bool:
        return self._ground_state is not None

    def initialize_constraint(self):
        """
        An intermediate method which initialises the constraint of the model and promote it to the
        interface.
        """
        if self.hilbert.is_gauge_invariant:
            self.constraint = self.model.constraint
        else:
            self.constraint = self.model.constraint + self.gauge_group.constraint

    def exact_diagonalization(
        self,
        ground_energy: Optional[Union[int, float, complex]] = None,
        *,
        recompute: bool = False,
    ) -> None:
        """
        This method attempts to perform exact diagonalisation of the constraint to obtain a true
        ground state and the smallest eigenvalue. This method uses the iterative Lanczos algorithm
        to obtain these values. If the Hilbert space is too large (as in, not indexable), this
        method will not run.
        """

        # highest priority: user provided ground energy
        if ground_energy is not None:
            self._user_given_ground_energy = ground_energy
            self._ground_energy = ground_energy
            self._ground_state = None

            self.printer.print(
                f"User-determined minimum eigenvalue {ground_energy} has been set.\n"
                f"Note: the ground state remains undefined."
            )
            return

        # explicit re-computation requested
        if recompute:
            self._ground_energy = None
            self._ground_state = None
            self._user_given_ground_energy = None

            self._lanczos()
            return

        # lazy evaluation
        if self._ground_energy is not None:
            return

        self._lanczos()

    @timeit(tag="LANCZOS", warn_ms=10000)
    def _lanczos(self):
        """
        Computes the lowest eigenvalue and eigenvector of the model's constraint using Lanczos'
        iterative method.
        """
        if self.constraint is None:
            # concrete constraint first
            self.initialize_constraint()

        evals = nk.exact.lanczos_ed(
            self.constraint,
            compute_eigenvectors=True,
            matrix_free=True,
            k=1,
        )

        self._ground_state = evals[1]
        self._ground_energy = evals[0]

        self.printer.print(
            f"Minimum eigenvalue using Lanczos algorithm: {self.ground_energy[0]}"
        )

    def _maybe_diagonalised(self):
        """
        Helper function to check whether the constraint can be solved using iterative exact methods
        or will it (probably) require too much memory. This is done by checking if the Hilbert space
        of the model is indexable or not. When it is not, then its dimensions are more than
        which means that fitting the constraint as a sparse/dense matrix into memory would probably
        fail.

        Note that this is a soft flag. In principle, even if the space is not indexable, the
        constraint can be exactly diagonalised given enough memory.
        """
        if self.hilbert.hilbert_netket.is_indexable:
            return True

        # warning
        self.printer.print(
            "The Hilbert space is too large to be indexed. Attempting to find the lowest eigenvalue"
            "and eigenstate could result in an out-of-memory error and therefore has not been "
            "attempted."
        )
        return False

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"gauge_invariant_model={self.hilbert.is_gauge_invariant}, "
            f"hilbert_dimensions={scientific(self.hilbert_dimensions)}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"model_name={self.model_name}"
            f")"
        )
