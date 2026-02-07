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

from neuralqx.debug import event
from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.graph.core import AbstractGraph
from neuralqx.gauge_groups import AbstractGaugeGroup


class AbstractLqxModel(abc.ABC):
    """
    An abstract implementation of a LQG model. Any subclass must implement a constraint for the
    model. It should accept a valid gauge group and Hilbert space implementation.
    """

    _hilbert: AbstractHilbertInterface
    """The Hilbert space interface the model will be defined on."""

    _gauge_group: AbstractGaugeGroup
    """The graph the LQX model will be defined on."""

    _graph: AbstractGraph
    """The graph the LQX model will be defined on."""

    _model_name: str
    """A name for the implemented LQG model used for logging."""

    _spacetime_dimensions: int
    """The spacetime dimensions of the model being implemented."""

    _is_computational: bool
    """A flag to indicate whether this model's operators are of computational type."""

    _constraint: object = None
    """The constraint implementation of the model."""

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: AbstractGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: bool = True,
        spacetime_dimensions: int = 4,
        model_name: str = "AbstractLqxModel",
    ):
        # local copies
        self._hilbert = hilbert
        self._graph = graph
        self._gauge_group = gauge_group

        self._is_computational = computational
        self._spacetime_dimensions = spacetime_dimensions
        self._model_name = model_name

        event(
            msg="INITIALIZING LQX CORE",
            tag="LQX:CORE:INIT",
            level=logging.INFO,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=self.model_name,
        )

    @property
    @abc.abstractmethod
    def constraint(self):
        """
        This property should return the constraint of the model to be considered in the optimisation
        process
        """

    @constraint.setter
    @abc.abstractmethod
    def constraint(self, val):
        """
        A setter for the constraint of the model. This should be used to help initialise the
        constraint.
        """

    @property
    def is_computational(self):
        return self._is_computational

    @property
    def hilbert(self):
        return self._hilbert

    @property
    def gauge_group(self):
        return self._gauge_group

    @property
    def graph(self):
        return self._graph

    @property
    def model_name(self):
        return self._model_name

    @property
    def gauge_dimensions(self):
        return self.gauge_group.dimensions

    @property
    def hilbert_dimensions(self):
        return self.hilbert.dimensions

    @property
    def spacetime_dimensions(self):
        return self._spacetime_dimensions

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"model_name={self.model_name}, "
            f"spacetime_dimensions={self.spacetime_dimensions}, "
            f"hilbert_dimensions={self.hilbert_dimensions}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"hilbert={self.hilbert.core.hilbert}"
            f")"
        )
