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
from typing import Tuple

from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph import HalfLadderGraph
from neuralqx.hilbert import AbstractHilbertInterface
from ..abstract_lqx_model import AbstractLqxModel


class AbstractLqxSphericalBoSw(AbstractLqxModel):

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: HalfLadderGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: bool = True,
        spacetime_dimensions: int = 3,
        model_name: str = "LqxSphericalBoSw",
        Immirzi: float = 1.0,
        delta: float = 2.0,
    ):
        if not isinstance(graph, HalfLadderGraph):
            raise ValueError(
                f"This model is only compatible with a graph of the type ``HalfLadderGraph``, got"
                f"{type(graph).__name__} instead."
            )

        # local copies
        self._immirzi = Immirzi
        self._delta = delta
        self._vertices_k = graph.vertices_k
        self._inner_vertices = graph.vertices_k[2:-2]

        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
        )

    @property
    def Immirzi(self):
        return self._immirzi

    @Immirzi.setter
    def Immirzi(self, new_val):
        self._immirzi = new_val

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, new_val):
        self._delta = new_val

    @abc.abstractmethod
    def spherical_volume(
        self,
        vertex: int,
        *,
        shift: float = 1.0,
        jax: bool = True,
    ):
        """
        Subclasses should implement the volume operator according to the Bojowald-Swiderski model.

        :return:
        """

    @abc.abstractmethod
    def spherical_Ex(
        self,
        vertex: int,
        *,
        jax: bool = True,
    ):
        """
        Subclasses should implement the spherical flux according to the Bojowald-Swiderski model.

        :return:
        """

    @abc.abstractmethod
    def holonomy(
        self,
        edge: Tuple,
        adjoint: bool = False,
        *,
        jax: bool = True,
    ):
        """
        Subclasses should implement the spherical holonomy operator acting on the specified edge
        according to the Bojowald-Swiderski model.

        :return:
        """

    @abc.abstractmethod
    def number(
        self,
        edge: Tuple,
        *,
        jax: bool = True,
    ):
        """
        Subclasses should implement the number operator acting on the specified edge according to
        the Bojowald-Swiderski model.

        :return:
        """

    @property
    def vertices_k(self):
        return self._vertices_k

    @property
    def inner_vertices(self):
        return self._inner_vertices
