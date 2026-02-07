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
from typing import Union

import netket as nk

from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface

from ..abstract_lqx_wcl_model import LqxWclModel

from neuralqx.operators import get_quantum_number
from neuralqx.operators import holonomy
from neuralqx.operators.computational.misc import MinimalLoopHolonomyOperator
from neuralqx.operators.computational.misc import U1HolonomyOperator
from neuralqx.operators.computational.misc import NumberOperator


class LqxWCL1D(LqxWclModel):

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: AbstractGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: bool = True,
        spacetime_dimensions: int = 3,
        model_name: str = "U(1) BF Model",
        is_4d: bool = False,
    ):
        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
        )

        if is_4d:
            # from neuralqx.experimental.operators.types import MarkedLocalOperator

            self._lop_type = nk.operator.LocalOperator
            # self._marked_type = MarkedLocalOperator
        else:
            self._lop_type = nk.operator.LocalOperator

        # necessary for higher dimensional U(1)^N model which subclass this implementation
        self.all_graphs = []

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
        Retuens a U(1) holonomy operator acting on the specified edge.

        :param edge: the edge the minimal loop holonomy operator should act on. This should be one
          specified from the
        :param adjoint: if True, the operator acts as a lowering operator, otherwise a raising
          operator on the U(1) DOFs
        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :param adjoint: if True, the operator acts as a lowering operator, otherwise a raising
          operator on the U(1) DOFs. Note that this is a dummy parameter in this model.
        :param dimension: This controls the dimension on which the holonomy operator is
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator). Note
          that this is a dummy parameter in this model.
        :return:
        """

        # check the computational flag
        if computational:
            return U1HolonomyOperator(
                self.hilbert,
                self.graph.edge_to_index(edge),
                adjoint=bool(adjoint),
                jax=jax,
            )
        elif self.is_computational and computational is None:
            return U1HolonomyOperator(
                self.hilbert,
                self.graph.edge_to_index(edge),
                adjoint=bool(adjoint),
                jax=jax,
            )
        else:
            return holonomy(
                self.hilbert, self.graph.edge_to_index(edge), adjoint=adjoint
            )

    def minimal_loop_holonomy(
        self,
        minimal_loop: list,
        vertex_index: int = None,
        adjoint: int = 0,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
        dimension: Optional[int] = 1,
    ):
        r"""
        Implements the minimal loop holonomy operator recursively for a given
        minimal loop. It can either be the adjoint or the non-adjoint version of the operator.

        Usage:
            - if you want the non-adjoint version: only specify the minimal loop
            - if you want the adjoint version: set the adjoint flag to 1

        Additionally, this will provide computational or local operator as specified, with the
        default choice being the flag the class has been instantiated with.

        :param minimal_loop: the minimal loop the operator should act on
        :param vertex_index: the current vertex index in te minimal loop (used for recursion)
        :param adjoint: a flag which identifies if the adjoint is requested (1 or 0)
        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :param dimension: This controls the dimension on which the minimal loop holonomy is
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator). This is
          not operational in this model.
        """

        # check the computational flag
        if computational:
            return MinimalLoopHolonomyOperator(
                self.hilbert,
                minimal_loop,
                copy=0,
                adjoint=bool(adjoint),
                jax=jax,
            )
        elif self.is_computational and computational is None:
            return MinimalLoopHolonomyOperator(
                self.hilbert,
                minimal_loop,
                copy=0,
                adjoint=bool(adjoint),
                jax=jax,
            )
        else:
            if adjoint == 1 and vertex_index is None:
                vertex_index = len(minimal_loop) - 1

            # we check if we want the adjoint of the minimal loop holonomy
            if adjoint == 1:
                # if so, then we apply the minimal loop holonomy in backwards ordering
                # check if we have reached the first vertex in the minimal loop
                if vertex_index == 0:
                    # if so, return the holonomy operator on that vertex
                    # dev: add the key and correct the "type" to get from the data at idx 1 now
                    return holonomy(
                        self.hilbert,
                        self.graph.edge_to_index(minimal_loop[vertex_index][0]),
                        adjoint,
                        minimal_loop[vertex_index][1]["type"],
                    )

                # if not, return the holonomy operator on that vertex times the holonomy operator
                # on the previous vertex by calling this method again with decremented vertex number
                # dev: add the key and correct the "type" to get from the data at idx 1 now
                return holonomy(
                    self.hilbert,
                    self.graph.edge_to_index(minimal_loop[vertex_index][0]),
                    adjoint,
                    minimal_loop[vertex_index][1]["type"],
                ) * LqxWCL1D.minimal_loop_holonomy(
                    self,
                    minimal_loop,
                    vertex_index=vertex_index - 1,
                    adjoint=adjoint,
                    computational=False,
                    jax=jax,
                    dimension=1,
                )

            # if not, we apply the holonomies in forward order of operations
            if vertex_index is None:
                vertex_index = 0

            # check if we reached the last edge
            if vertex_index == len(minimal_loop) - 1:
                # if so, return the holonomy operator on that dual vertex
                # dev: add the key and correct the "type" to get from the data at idx 1 now
                return holonomy(
                    self.hilbert,
                    self.graph.edge_to_index(minimal_loop[vertex_index][0]),
                    adjoint,
                    minimal_loop[vertex_index][1]["type"],
                )

            # if not, return the holonomy operator on that dual vertex times the holonomy
            # operator on the next dual vertex by calling this method again with incremented vertex
            # number
            # dev: add the key and correct the "type" to get from the data at idx 1 now
            return holonomy(
                self.hilbert,
                self.graph.edge_to_index(minimal_loop[vertex_index][0]),
                adjoint,
                minimal_loop[vertex_index][1]["type"],
            ) * LqxWCL1D.minimal_loop_holonomy(
                self,
                minimal_loop,
                vertex_index=vertex_index + 1,
                adjoint=adjoint,
                jax=jax,
                dimension=1,
                computational=False,
            )

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

        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        """

        # create an empty operator which we will populate with terms for every minimal loop
        # override based on the computational flag
        if computational:
            _flatness_constraint = 0
        # default to the class flag if the function flag is None
        elif self.is_computational and computational is None:
            _flatness_constraint = 0
        # default to LOPs if both are False/None
        else:
            _flatness_constraint = self._lop_type(self.hilbert.core.hilbert)

        # loop through every minimal loop
        for minimal_loop in self.graph.dressed_minimal_loops():
            # for every minimal loop \alpha, we add a term in constraint which corresponds to
            # (h_\alpha - I)((h_\alpha)^\dagger - I) = - h_\alpha - h_\alpha^\dagger + 2I

            # add the - h_\alpha - h_\alpha^\dagger term
            _flatness_constraint += -self.minimal_loop_holonomy(
                minimal_loop,
                computational=computational,
                jax=jax,
            ) - self.minimal_loop_holonomy(
                minimal_loop,
                adjoint=1,
                vertex_index=len(minimal_loop) - 1,
                computational=computational,
                jax=jax,
            )

        # add the +2I term, for every minimal loop
        # (= + 2*(# of minimal loops)*(# dim of gauge group)*I)
        # override based on the function flag
        if computational:
            _flatness_constraint += (
                2 * len(self.graph.minimal_loops()) * self.gauge_dimensions
            )
        # default to the class flag if the function flag is None
        elif self.is_computational and computational is None:
            _flatness_constraint += (
                2 * len(self.graph.minimal_loops()) * self.gauge_dimensions
            )
        # default to LOPs otherwise
        else:
            _flatness_constraint += self._lop_type(
                self.hilbert.core.hilbert,
                operators=None,
                constant=2 * len(self.graph.minimal_loops()) * self.gauge_dimensions,
            )

        return _flatness_constraint

    @property
    def constraint(self):
        """
        This property should return the constraint of the model to be considered in the optimisation
        process
        """

        if self._constraint is None:
            self._constraint = self.curvature_constraint()
        return self._constraint

    @constraint.setter
    def constraint(self, val):
        """
        A setter for the constraint of the model. This should be used to help initialise the
        constraint.
        """

        self._constraint = val

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

        if kwargs is None:
            kwargs = {}

        # check the computational flag
        if computational:
            return NumberOperator(
                self.hilbert,
                self.graph.edge_to_index(edge),
                jax=jax,
            )
        elif self.is_computational and computational is None:
            return NumberOperator(
                self.hilbert,
                self.graph.edge_to_index(edge),
                jax=jax,
            )
        else:
            return get_quantum_number(
                self.hilbert, self.graph.edge_to_index(edge), **kwargs
            )

    def volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        raise NotImplementedError(
            "This model does not have a Volume operator implementation."
        )

    def area(
        self,
        surface: list,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        raise NotImplementedError(
            "This model does not have an Area operator implementation."
        )

    def thiemann_quadratic_constraint(
        self,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
        **kwargs,
    ):
        raise NotImplementedError(
            "This model does not have a Thiemann quadratic constraint implementation."
        )
