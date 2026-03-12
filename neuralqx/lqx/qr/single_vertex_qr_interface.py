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

from neuralqx.lqx.qr.core import LqxSVQRLG
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.hilbert import AbstractHilbertInterface
from ..abstract_lqx_interface import AbstractLqxInterface
from neuralqx.utils.errors import ComputationalModelConcretizationWarning


class LqxSingleVertexQR(AbstractLqxInterface):

    def __init__(
        self,
        H: AbstractHilbertInterface,
        gauge_group: AbstractGaugeGroup,
        *,
        lazy_load: bool = True,
        computational: bool = True,
        jax: bool = True,
        immirzi: float = 1.0,
    ):
        if not computational:
            ComputationalModelConcretizationWarning()
            computational = True

        self._model = LqxSVQRLG(
            H,
            H.graph,
            gauge_group,
            model_name="Single Vertex QRLG",
            computational=computational,
            jax=jax,
            immirzi=immirzi,
        )

        super().__init__(
            H,
            gauge_group,
            lazy_load=lazy_load,
            model_name="Single Vertex QRLG",
        )

    def initialize_constraint(self):
        """
        An intermediate method which initialises the constraint of the model and promote it to the
        interface.
        """
        self.constraint = self.model.constraint

    @property
    def euclidean_constraint(self):
        return self.model.euclidean_constraint

    @property
    def lorentzian_constraint(self):
        return self.model.lorentzian_constraint

    def creation(
        self,
        edge: int,
        n: int,
        *,
        computational: bool = False,
        jax: bool = False,
    ):
        return self.model.creation(edge=edge, n=n, computational=computational, jax=jax)

    def annihilation(
        self,
        edge: int,
        n: int,
        *,
        computational: bool = False,
        jax: bool = False,
    ):
        return self.model.annihilation(
            edge=edge, n=n, computational=computational, jax=jax
        )

    def s(
        self,
        edge: int,
        no_i: bool = False,
        *,
        computational: bool = False,
        jax: bool = True,
    ):
        """
        The symmetric holonomy operator s which increments or decrements the quantum number of the
        specified edge symmetrically by a value of 1.
        """
        return self.model.s(edge=edge, no_i=no_i, computational=computational, jax=jax)

    def c(
        self,
        edge: int,
        *,
        computational: bool = False,
        jax: bool = True,
    ):
        """
        The symmetric holonomy operator c which increments or decrements the quantum number of the
        specified edge symmetrically by a value of 1.
        """
        return self.model.c(edge=edge, computational=computational, jax=jax)

    def E(
        self,
        edge: int,
        power: float = 1.0,
        *,
        computational: bool = False,
        jax: bool = True,
    ):
        """
        The flux operator which acts as a number operator but can additionally output eigenvalues
        raised to some power.
        """

        return self.model.E(
            edge=edge, power=power, computational=computational, jax=jax
        )

    def E_inv(
        self,
        edge: int,
        power: float = 1.0,
        *,
        computational: bool = False,
        jax: bool = True,
    ):
        """
        The flux operator which acts as an 'inverse' number operator in the sense that it outputs
        1/spin instead of spin as the eigenvalue. Additionally, it can output eigenvalues
        raised to some power.
        """
        return self.model.E_inv(
            edge=edge, power=power, computational=computational, jax=jax
        )
