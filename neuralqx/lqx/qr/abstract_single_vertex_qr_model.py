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

from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface
from ..abstract_lqx_model import AbstractLqxModel


class AbstractLqxSVQRLGModel(AbstractLqxModel):

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: AbstractGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: bool = True,
        jax: bool = True,
        spacetime_dimensions: int = 4,
        model_name: str = "AbstractLqxSVQRLGModel",
    ):
        self._is_jax = jax

        self._euclidean_constraint = None

        self._lorentzian_constraint = None

        self.lapse = 1

        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
        )

    @property
    def is_jax(self):
        return self._is_jax

    @property
    @abc.abstractmethod
    def euclidean_constraint(self):
        """
        Subclasses should implement the Euclidean constraint according to the single-vertex model
        of QRLG.

        :return:
        """

    @property
    @abc.abstractmethod
    def lorentzian_constraint(self):
        """
        Subclasses should implement the Lorentzian constraint according to the single-vertex model
        of QRLG.

        :return:
        """

    @abc.abstractmethod
    def creation(
        self,
        edge: int,
        n: int,
        *,
        computational: bool = False,
        jax: bool = False,
    ):
        """
        Subclasses should implement a creation operator acting on the specified edge according to
        the single-vertex model of QRLG.

        :param edge: the edge the creation operator is to act on.
        :param n: how many creation operators should be applied, alternatively how much of an
          ìncrement should the operator apply on the specified edge.
        :param computational: if True, the operator is returned as a computational type
        :param jax: if True, and computational is True, the operator is returned as a JAX compatible
          computational type.
        :return:
        """

    @abc.abstractmethod
    def annihilation(
        self,
        edge: int,
        n: int,
        *,
        computational: bool = False,
        jax: bool = False,
    ):
        """
        Subclasses should implement an annihilation operator acting on the specified edge according
        to the single-vertex model of QRLG.

        :param edge: the edge the annihilation operator is to act on.
        :param n: how many annihilation operators should be applied, alternatively how much of a
          decrement should the operator apply on the specified edge.
        :param computational: if True, the operator is returned as a computational type
        :param jax: if True, and computational is True, the operator is returned as a JAX compatible
          computational type.
        :return:
        """

    @abc.abstractmethod
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

        :param edge: the edge the operator should act on
        :param no_i: if True, the imaginary pre-factor is not applied
        :param computational: if True, the operator is returned as a computational type
        :param jax: if True, and computational is True, the operator is returned as a JAX compatible
          computational type.
        """

    @abc.abstractmethod
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

        :param edge: the edge the operator acts on
        :param computational: if True, the operator is returned as a computational type
        :param jax: if True, and computational is True, the operator is returned as a JAX compatible
          computational type.
        """

    @abc.abstractmethod
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

        :param edge: the site the flux operator should act on
        :param power: the power the eigenvalues should be raised to
        :param computational: if True, the operator is returned as a computational type
        :param jax: if True, and computational is True, the operator is returned as a JAX compatible
          computational type.
        """

    @abc.abstractmethod
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

        :param edge: the site the inverted flux operator should act on
        :param power: the power the eigenvalues should be raised to
        :param computational: if True, the operator is returned as a computational type
        :param jax: if True, and computational is True, the operator is returned as a JAX compatible
          computational type.
        """
