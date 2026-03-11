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

from typing import Tuple

from neuralqx.lqx.spherical.core import LqxSphericalBoSw
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.hilbert import AbstractHilbertInterface
from ..abstract_lqx_interface import AbstractLqxInterface
from neuralqx.utils.io.printing import NQXPrinter
from neuralqx.utils.errors import HilbertSpaceGaugeGroupDimensionsMismatchError


class LqxBojowaldSwiderski(AbstractLqxInterface):

    def __init__(
        self,
        H: AbstractHilbertInterface,
        gauge_group: AbstractGaugeGroup,
        *,
        lazy_load: bool = True,
        spacetime_dimensions: int = 3,
        immirzi: float = 1.0,
        delta: int = 2,
    ):
        # make sure both objects have the same gauge dimensions
        if not gauge_group.dimensions == H.hilbert.gauge_dimensions:
            raise HilbertSpaceGaugeGroupDimensionsMismatchError

        self._p = NQXPrinter()

        if spacetime_dimensions != 3:
            raise ValueError(
                f"`spacetime_dims` must be 3, got {spacetime_dimensions} instead."
            )

        if gauge_group.dimensions != 1:
            raise ValueError(
                f"`gauge_dimensions` must be 1, got {gauge_group.dimensions} instead."
            )

        self._model = LqxSphericalBoSw(
            H,
            H.graph,
            gauge_group,
            Immirzi=immirzi,
            delta=delta,
            model_name="Spherical Model (Bojowald-Swiderski)",
        )

        super().__init__(
            H,
            gauge_group,
            lazy_load=lazy_load,
            model_name="Spherical Model (Bojowald-Swiderski)",
        )

    def volume(
        self,
        vertex: int,
        *,
        shift: float = 1.0,
        jax: bool = True,
    ):
        """
        Returns the spherical volume operator according to the Bojowald-Swiderski model.

        :return:
        """
        return self.model.spherical_volume(vertex=vertex, shift=shift, jax=jax)

    def flux(
        self,
        vertex: int,
        *,
        jax: bool = True,
    ):
        """
        Returns the spherical flux operator according to the Bojowald-Swiderski model.

        :return:
        """
        return self.model.spherical_Ex(vertex=vertex, jax=jax)

    def holonomy(
        self,
        edge: Tuple,
        adjoint: bool = False,
        *,
        jax: bool = True,
    ):
        """
        Returns the holonomy operator acting on an edge in the graph according to the
        Bojowald-Swiderski model.

        :return:
        """
        return self.model.holonomy(edge=edge, adjoint=adjoint, jax=jax)

    def number(
        self,
        edge: Tuple,
        *,
        jax: bool = True,
    ):
        """
        Returns a number-like operator which returns the quantum number of the edge it acts on
        according to the Bojowald-Swiderski model.

        :return:
        """
        return self.model.number(edge=edge, jax=jax)

    @property
    def vertices_k(self):
        return self.model.vertices_k

    @property
    def inner_vertices(self):
        return self.model.inner_vertices

    @property
    def Immirzi(self):
        return self.model.Immirzi

    @Immirzi.setter
    def Immirzi(self, new_val):
        self.model.Immirzi = new_val

    @property
    def delta(self):
        return self.model.delta

    @delta.setter
    def delta(self, new_val):
        self.model.delta = new_val

    def initialize_constraint(self):
        """
        An intermediate method which initialises the constraint of the model and promote it to the
        interface.
        """
        self.constraint = self.model.constraint
