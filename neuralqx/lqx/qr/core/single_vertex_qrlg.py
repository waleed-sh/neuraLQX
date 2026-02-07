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

from typing import Optional

from neuralqx.lqx.qr.abstract_single_vertex_qr_model import AbstractLqxSVQRLGModel
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface

from neuralqx.operators.computational.qr import EuclideanConstraintOperator
from neuralqx.operators.computational.qr import QRAnnihilationOperator
from neuralqx.operators.computational.qr import QRCreationOperator
from neuralqx.operators.computational.qr import QRFluxOperator


class LqxSVQRLG(AbstractLqxSVQRLGModel):

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: AbstractGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
        spacetime_dimensions: int = 4,
        model_name: str = "LqxSVQRLGModel",
    ):
        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            jax=jax,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
        )

    @property
    def constraint(self):
        return self.euclidean_constraint

    def _init_euclidean_constraint(
        self,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        A method to implement the Euclidean part of the constraint only with the lapse as defined
        upon instantiation
        """

        # here, we follow the convention of labeling the edges as the following:
        #   edge x -> 0
        #   edge y -> 1
        #   edge z -> 2

        # an empty FunctionalLocalOperator object to hold the actual constraint
        # we force the operator to be Hermitian (known from small cutoff tests) as this then
        # avoids the hermiticity check in the expect_and_grad which turns the entire operator
        # into a dense matrix: bottleneck for high cutoffs

        # check the computational flag
        if computational:
            return EuclideanConstraintOperator(
                self.hilbert, lapse=self.lapse, power=0.25, jax=jax
            )

        elif self.is_computational and computational is None:
            return EuclideanConstraintOperator(
                self.hilbert, lapse=self.lapse, power=0.25, jax=jax
            )

        else:
            raise NotImplementedError(
                "This model only supports computational operators as a backend."
            )

    @property
    def euclidean_constraint(self):
        """
        Subclasses should implement the Euclidean constraint according to the single-vertex model
        of QRLG.

        :return:
        """
        if self._euclidean_constraint is None:
            self._euclidean_constraint = self._init_euclidean_constraint(
                computational=self._is_computational, jax=self._is_jax
            )

        return self._euclidean_constraint

    @property
    def lorentzian_constraint(self):
        """
        Implements the Lorentzian constraint according to the single-vertex model
        of QRLG.

        :return:
        """
        raise NotImplementedError

    def creation(
        self,
        edge: int,
        n: int,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        Subclasses should implement a creation operator acting on the specified edge according to
        the single-vertex model of QRLG.

        :param edge:
        :param n:
        :param computational:
        :param jax:
        :return:
        """

        # check the computational flag
        if computational:
            return QRCreationOperator(self.hilbert, site=edge, n=n, jax=jax)

        elif self.is_computational and computational is None:
            return QRCreationOperator(self.hilbert, site=edge, n=n, jax=jax)

        else:
            raise NotImplementedError(
                "This model only supports computational operators as a backend."
            )

    def annihilation(
        self,
        edge: int,
        n: int,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        Subclasses should implement an annihilation operator acting on the specified edge according
        to the single-vertex model of QRLG.

        :param edge:
        :param n:
        :param computational:
        :param jax:
        :return:
        """

        # check the computational flag
        if computational:
            return QRAnnihilationOperator(self.hilbert, site=edge, n=n, jax=jax)

        elif self.is_computational and computational is None:
            return QRAnnihilationOperator(self.hilbert, site=edge, n=n, jax=jax)

        else:
            raise NotImplementedError(
                "This model only supports computational operators as a backend."
            )

    def s(
        self,
        edge: int,
        no_i: bool = False,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        The symmetric holonomy operator s which increments or decrements the quantum number of the
        specified edge symmetrically by a value of 1.
        """

        factor = 1 / 2 if no_i else 1 / 2j

        return factor * (
            self.creation(edge, 1, computational=computational, jax=jax)
            - self.annihilation(edge, 1, computational=computational, jax=jax)
        )

    def c(
        self,
        edge: int,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        The symmetric holonomy operator c which increments or decrements the quantum number of the
        specified edge symmetrically by a value of 1.
        """

        return (1 / 2) * (
            self.creation(edge, 1, computational=computational, jax=jax)
            + self.annihilation(edge, 1, computational=computational, jax=jax)
        )

    def E(
        self,
        edge: int,
        power: float = 1.0,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        The flux operator which acts as a number operator but can additionally output eigenvalues
        raised to some power.

        :param edge: the site the flux operator should act on
        :param power: the power the eigenvalues should be raised to
        """

        # check the computational flag
        if computational:
            return QRFluxOperator(
                self.hilbert, site=edge, power=power, inverse=False, jax=jax
            )

        elif self.is_computational and computational is None:
            return QRFluxOperator(
                self.hilbert, site=edge, power=power, inverse=False, jax=jax
            )

        else:
            raise NotImplementedError(
                "This model only supports computational operators as a backend."
            )

    def E_inv(
        self,
        edge: int,
        power: float = 1.0,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        The flux operator which acts as an 'inverse' number operator in the sense that it outputs
        1/spin instead of spin as the eigenvalue. Additionally, it can output eigenvalues
        raised to some power.

        :param edge: the site the inverted flux operator should act on
        :param power: the power the eigenvalues should be raised to
        """

        # check the computational flag
        if computational:
            return QRFluxOperator(
                self.hilbert, site=edge, power=power, inverse=True, jax=jax
            )

        elif self.is_computational and computational is None:
            return QRFluxOperator(
                self.hilbert, site=edge, power=power, inverse=True, jax=jax
            )

        else:
            raise NotImplementedError(
                "This model only supports computational operators as a backend."
            )
