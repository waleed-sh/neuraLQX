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

from netket.operator import Squared

from neuralqx.lqx.spherical.abstract_lqx_bosw_model import AbstractLqxSphericalBoSw
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph import HalfLadderGraph
from neuralqx.hilbert import AbstractHilbertInterface

from neuralqx.operators import InverseExpectationCost
from neuralqx.operators.computational.misc import NumberOperator
from neuralqx.operators.computational.misc import U1HolonomyOperator
from neuralqx.operators.computational.spherical import SphericalExOperator
from neuralqx.operators.computational.spherical import (
    SphericalVertexConstraintBojowaldSwiderskiOperator,
)
from neuralqx.operators.computational.spherical import SphericalVolumeOperator


class LqxSphericalBoSw(AbstractLqxSphericalBoSw):

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

        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
            Immirzi=Immirzi,
            delta=delta,
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

    def spherical_hamiltonian_on_vertex(
        self,
        vertex,
        *,
        jax: bool = True,
        fast: bool = False,
    ):
        # return H_L(v) + H_C(v) + H_R(v)
        return SphericalVertexConstraintBojowaldSwiderskiOperator(
            self.hilbert,
            vertex,
            include_gamma_terms=True,
            delta=self.delta,
            outer_km_constant=0,
            outer_kp_constant=0,
            immirzi=self.Immirzi,
            jax=jax,
            fast=fast,
        )

    def hamiltonian_list(self):
        # return sum_v of (H_L(v) + H_C(v) + H_R(v))^2
        return [
            Squared(self.spherical_hamiltonian_on_vertex(v))
            for v in self.inner_vertices
        ]

    def hamiltonian_boundaries(self):
        # right now, just an IEC for the volume, change as you please
        alpha = 0.1

        return [
            InverseExpectationCost(self.spherical_volume(vertex=2), 1, alpha),
        ]

    def _init_constraint(self):
        # return [*self.hamiltonian_list(), *self.hamiltonian_boundaries()]
        return self.hamiltonian_list()

    @property
    def constraint(self):
        if self._constraint is None:
            self._constraint = self._init_constraint()
        return self._constraint

    def spherical_volume(
        self,
        vertex: int,
        *,
        shift: float = 1.0,
        jax: bool = True,
    ):
        """
        Returns the volume operator according to the Bojowald-Swiderski model.

        :return:
        """
        return SphericalVolumeOperator(self.hilbert, vertex, shift=shift, jax=jax)

    def spherical_Ex(
        self,
        vertex: int,
        *,
        jax: bool = True,
    ):
        """
        Returns the spherical flux according to the Bojowald-Swiderski model.

        :return:
        """
        return SphericalExOperator(self.hilbert, vertex, gamma=self.Immirzi, jax=jax)

    def holonomy(
        self,
        edge: Tuple,
        adjoint: bool = False,
        *,
        jax: bool = True,
    ):
        """
        Returns the spherical holonomy operator acting on the specified edge according to the
        Bojowald-Swiderski model.

        :return:
        """

        return U1HolonomyOperator(
            self.hilbert, self.graph.edge_to_index(edge), adjoint=adjoint, jax=jax
        )

    def number(
        self,
        edge: Tuple,
        *,
        jax: bool = True,
    ):
        """
        Returns the number operator acting on the specified edge according to
        the Bojowald-Swiderski model.

        :return:
        """

        return NumberOperator(self.hilbert, self.graph.edge_to_index(edge), jax=jax)

    @property
    def vertices_k(self):
        return self._vertices_k

    @property
    def inner_vertices(self):
        return self._inner_vertices
