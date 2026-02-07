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
from typing import Optional, Union

from ..abstract_lqx_model import AbstractLqxModel


class LqxWclModel(AbstractLqxModel):

    @abc.abstractmethod
    def volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        """
        Returns the volume operator of the underlying wrapped model.

        :param vertex: the vertex the volume operator should act on
        :param computational: if True, the volume operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """

    @abc.abstractmethod
    def area(
        self,
        surface: list,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        """
        Returns the area operator of the underlying wrapped model.

        :param surface: the surface the area operator acts on. This should be a list composed of
          edges in their raw representation as provided from the AbstractGraph.edges property
        :param computational: if True, the area operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """

    @abc.abstractmethod
    def minimal_loop_holonomy(
        self,
        minimal_loop: list,
        vertex_index: int = None,
        adjoint: int = 0,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
        dimension: Optional[int] = 1,
    ):
        """
        Returns the minimal loop holonomy operator of the underlying wrapped model.

        :param minimal_loop: the loop the operator should act on. This should be one of the loops provided
          by the AbstractGraph.dressed_minimal_loops()
        :param vertex_index: the current vertex index in te minimal loop (used for recursion)
        :param adjoint: a flag which identifies if the adjoint is requested (1 or 0)
        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :param dimension: This controls the dimension on which the minimal loop holonomy is
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator)
        :return:
        """

    @abc.abstractmethod
    def holonomy(
        self,
        edge: Union[list, tuple],
        adjoint: bool = False,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
        dimension: Optional[int] = 1,
    ):
        """
        Returns the minimal loop holonomy operator of the underlying wrapped model.

        :param edge: the edge the operator should act on. This should be one of the loops provided
          by the AbstractGraph.edges
        :param computational: if True, the operator is implemented as a computational
          operator
        :param adjoint: if True, the operator acts as a lowering operator, otherwise a raising
          operator on the U(1) DOFs
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :param dimension: This controls the dimension on which the holonomy operator is
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator)
        :return:
        """

    @abc.abstractmethod
    def number(
        self,
        edge: tuple,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
        dimension: Optional[int] = 1,
        kwargs: Optional[dict] = None,
    ):
        """
        Returns the number operator acting on the specified edge

        :param edge: the edge the operator should act on. This should be one of the loops provided
          by the AbstractGraph.edges
        :param computational: if True, the operator is implemented as a computational
          operator
        :param adjoint: if True, the operator acts as a lowering operator, otherwise a raising
          operator on the U(1) DOFs
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :param dimension: This controls the dimension on which the holonomy operator is
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator)
        :param kwargs: and additional information to be passed to the LocalOperator implementation
        :return:
        """

    @abc.abstractmethod
    def curvature_constraint(
        self,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
    ):
        r"""
        Implements the curvature constraint on the graph by imposing flatness
        over the minimal loops. The constraint takes the form

        .. math::
            \hat{F} = \sum_{a \in L({\gamma})} ( \hat{h}_{a} - 1 ) ( \hat{h}_{a}^\dagger - 1 )

        for every minimal loop :math:`a` in the graph. Here :math:`\hat{h}_{a}` is a minimal loop
        holonomy.
        """

    @abc.abstractmethod
    def thiemann_quadratic_constraint(
        self,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
        **kwargs,
    ):
        r"""
        For appropriate models, any subclass should implement a quadratic constraint implemented
        using the Thiemann regularised quantum Hamilton constraint
        """
