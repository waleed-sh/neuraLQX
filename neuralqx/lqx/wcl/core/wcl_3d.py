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

import itertools
import logging
from functools import reduce
from typing import Optional
from typing import Union
from typing import List

from cffi.backend_ctypes import long

import numpy as np

import netket as nk
from netket.operator import Squared

from neuralqx.debug import event

from .wcl_1d import LqxWCL1D
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph import Graph
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface

from neuralqx.operators.types import FunctionalLocalOperator
from neuralqx.operators.computational.misc import NDMinimalLoopHolonomyOperator
from neuralqx.operators.computational.misc import NDU1HolonomyOperator
from neuralqx.operators.computational.Euclidean3d import VolumeOperator
from neuralqx.operators.computational.Euclidean3d import SqrtVolumeOperator
from neuralqx.operators.computational.Euclidean3d import SquaredVolumeOperator
from neuralqx.operators.computational.Euclidean3d import (
    ThiemannRegularisedVertexConstraintOperator,
)
from neuralqx.operators.computational.wrappers import Product
from neuralqx.operators.computational.wrappers import ProductJax

from neuralqx.utils.errors import CrossProductInHigherDimensionsError
from neuralqx.utils.errors import IncompatibleNonPlanarGraphModel
from neuralqx.utils.errors import ComputationalModelConcretizationWarning
from neuralqx.utils.misc.arithmetic import factorial
from neuralqx.utils.misc.graph import get_common_base_vertex
from neuralqx.utils.misc.graph import get_edges_at_base_vertex
from neuralqx.utils.misc.arithmetic import plus_key


def _update_edges(edges_list: list, factor: int = 1) -> list:
    """
    A function that creates a list of edges identical to the one used to create the main
    graph but with vertex labels shifted according to the graph number

    :param edges_list (list): the list of directed edges which compose the graph
    :param factor (int): a multiplicative factor used to raise the order of vertices between
    every new graph
    """

    # find the number of vertices in the graph
    # we here take care of keyed edges so we exclude it from the computation
    max_num = factor * (max(max(sublist[:-1]) for sublist in edges_list) + 1)

    # return the same list of edges with each element in the list shifted by the number of
    # vertices in the graph
    # once again _plus_key to exclude keys from being changde
    return [plus_key(sublist, max_num) for sublist in edges_list]


def _An(
    order: int,
    double_sqrt: bool,
):
    r"""
    A function to return the :math:`A_{n}` coefficient for the n-th derivative of the function
    :math:`f(x) = \sqrt{x}` or :math:`f(x) = \sqrt{\sqrt{x}}` as requested

    :param double_sqrt: (bool) a flag to identify if we want the coefficient
        for :math:`\sqrt{x}` or :math:`\sqrt{\sqrt{x}}`
    :param order: (int) the requested order
    :return: (float) the coefficient at that order
    """

    # check if the order is 0 or 1, then return the appropriate values
    if order == 0:
        return 1

    if order == 1:
        return 1 / 4 if double_sqrt else 1 / 2

    # get the term for the double factorial
    _dfact_term = 4 * order - 5 if double_sqrt else 2 * order - 3

    # get the term for the denominator
    _denom_term = 4 if double_sqrt else 2

    # get the term to which we decrement the double factorial
    _decr = -4 if double_sqrt else -2

    # return the A_order coefficient
    return ((-1) ** (order + 1)) * (
        (reduce(long.__mul__, range(_dfact_term, 0, _decr))) / (_denom_term**order)
    )


def commutator(a, b):
    """
    A function that returns the commutator [A, B] = AB - BA
    """
    return a * b - b * a


def _non_identical_triangulations(two_triangulations) -> bool:
    """
    A helper function that returns true of two pairs of edges are not identical, irrespective
    of their order
    """
    return (
        two_triangulations[0] != two_triangulations[1]
        and two_triangulations[0] != two_triangulations[1][::-1]
    )


def shift_loop(loop, k, per_copy_nodes):
    """Helper function to shift a minimal loop from level 0 to k, while keeping data."""
    return [
        ((u + k * per_copy_nodes, v + k * per_copy_nodes, key), data)
        for ((u, v, key), data) in loop
    ]


class LqxWCL3D(LqxWCL1D):

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: AbstractGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: bool = True,
        spacetime_dimensions: int = 3,
        model_name: str = "(2+1)-Euclidean LQG",
        is_4d: bool = False,
    ):
        # check for planarity, this model only supports planar graphs
        if not is_4d:
            if not graph.is_planar:
                raise IncompatibleNonPlanarGraphModel()

        if not computational:
            ComputationalModelConcretizationWarning()
            computational = True

        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
            is_4d=is_4d,
        )

        # save the 4d flag
        self._is4d = is_4d

        self._setup()

    def _setup(self) -> None:
        """
        This method is used to internally set up the structure of the model. This includes lifting
        the one level graph to 3 graphs, with correct minimal loops, as well as collecting all the
        terms relevant to the Thiemann regularised Hamilton constraint.
        """

        # set up the graphs
        self._init_graphs()

        # lift the minimal loops
        self._lift_minimal_loops()

        # collect the constraint terms
        if not self._is4d:
            self._init_triangulation_edges()

    def _init_graphs(self) -> None:
        """
        An internal helper function that creates `gauge_dimensions` many dictionaries, each
        containing the edges of the graphs for every dimension, the graphs themselves and the
        mappings from each graph. It also alters the mapping for the underlying graph of 1-d model to
        extend it to lifted 3-copy graph.
        """

        # create the dicts which will hold the information about the graphs
        self._all_graphs: dict = {}
        self._all_mappings: dict = {}
        self._all_edges: dict = {}

        # get the original set of edges
        edges = self.graph.edges

        for i in range(self.gauge_dimensions):

            # keyed version update
            self._all_edges[i] = _update_edges(edges, i)

            # new graphs based on the updated keyed edges
            # dev: what happens if the user used a non-generic graph?
            self._all_graphs[i] = Graph(self._all_edges[i], plot=False)

            # no keyed business needed
            self._all_mappings[i] = {
                key: value + i * len(self._all_graphs[0].dual_nx_graph.nodes)
                for key, value in self._all_graphs[i].mapping.items()
            }

        # extend the mapping of the graph on which netket operators on to handle the new edges
        # here we have to access the handler as graph.mapping is a property
        self.graph.handler.mapping = {
            key: value
            for inner_dict in self._all_mappings.values()
            for key, value in inner_dict.items()
        }

    def _lift_minimal_loops(self) -> None:
        """
        An internal helper function which promotes the minimal loops from the 1-d graph to the full
        graph of the model.
        """

        # get the 1-copy minimal loops
        _1d_minimal_loops = self.graph.minimal_loops()

        # create the list of minimal loops in the full graph
        _promoted_minimal_loops = [_1d_minimal_loops]

        # loop over every gauge dimension and promote
        for i in range(1, self.hilbert.gauge_dimensions):
            # for every dimension we get the minimal loops from the 1-d case, and we add to them
            # the appropriate number of vertices such that the minimal loop in the second graph
            # would have the same structure but with vertex labels shifted by N. Here, N denotes
            # just the number of vertices in the graph

            # a temp graph level minimal loops list
            _temp_gl_loops = []

            # loop through every minimal loop and shift the vertices
            for min_loop in _1d_minimal_loops:

                _temp = []
                shift = self.graph.n_vertices
                for edge in min_loop:
                    _temp.append(plus_key(edge, i * shift))

                _temp_gl_loops.append(_temp)

            _promoted_minimal_loops.append(_temp_gl_loops)

        # create a list of lists, of as many minimal loops as there are
        _nd_loops = [[] for _ in range(len(_promoted_minimal_loops[0]))]

        # for every minimal loop, we want to add all of them into a sublist
        for graph_minimal_loop in _promoted_minimal_loops:
            for i in range(len(graph_minimal_loop)):
                _nd_loops[i].append(graph_minimal_loop[i])

        # set the class attribute
        self._all_minimal_loops = np.array(_nd_loops)

    def _init_triangulation_edges(self) -> None:
        """
        An internal helper function to prepare all triangulation edge pairs as well as triangulation
        edge pairs per vertex of the graph
        """

        # initialise triangulations by creating pairs of minimal loops
        # here, the minimal loops are going to be lists of keyed edges
        init_triangulations = list(
            itertools.combinations(self.graph.minimal_loops(), 2)
        )

        # create all permuted pairs of minimal loops
        all_triangulations = init_triangulations + [
            tuple(reversed(item)) for item in init_triangulations
        ]

        # a dict to hold all the terms which will contribute to the TRC
        self._sum_terms = {}

        # the epsilon contribution for every edge pair
        self._epsilon = {}

        # loop through every triangulation pair
        # here, a triangulation pair is a pair of two minimal loops in the keyed edge representation
        for triangulation_pair in all_triangulations:

            # get the base vertex
            base_v = get_common_base_vertex(triangulation_pair)

            # create the sum terms for the given triangulation pair
            self._sum_terms[str(triangulation_pair)] = {"Base Vertices": base_v}
            self._sum_terms[str(triangulation_pair)]["Terms"] = {}

            # create the minimal loops for every triangulation in the edge representation
            # these are just each of the pairs in the triangulation pair as they are already
            # minimal loops in the keyed edges representation
            min_l1_edge_rep = triangulation_pair[0]
            min_l2_edge_rep = triangulation_pair[1]

            # sum over every common base vertex and add a contribution
            for vertex in base_v:
                # create appropriate sum_terms dict entries
                self._sum_terms[str(triangulation_pair)]["Terms"][str(vertex)] = {}
                self._sum_terms[str(triangulation_pair)]["Terms"][str(vertex)][
                    "ij"
                ] = []
                self._sum_terms[str(triangulation_pair)]["Terms"][str(vertex)][
                    "kl"
                ] = []

                # append to the sum_terms
                self._sum_terms[str(triangulation_pair)]["Terms"][str(vertex)][
                    "ij"
                ].append(get_edges_at_base_vertex(min_l1_edge_rep, vertex))
                self._sum_terms[str(triangulation_pair)]["Terms"][str(vertex)][
                    "kl"
                ].append(get_edges_at_base_vertex(min_l2_edge_rep, vertex))

                # append to the epsilons
                self._epsilon[
                    str(get_edges_at_base_vertex(min_l1_edge_rep, vertex))
                ] = 1
                self._epsilon[
                    str(get_edges_at_base_vertex(min_l1_edge_rep, vertex)[::-1])
                ] = -1

                self._epsilon[
                    str(get_edges_at_base_vertex(min_l2_edge_rep, vertex))
                ] = 1
                self._epsilon[
                    str(get_edges_at_base_vertex(min_l2_edge_rep, vertex)[::-1])
                ] = -1

    def minimal_loop_holonomy(
        self,
        minimal_loop: list,
        vertex_index: int = None,
        adjoint: int = 0,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
        dimension: Optional[int] = 3,
    ):
        """
        Returns the 3-D minimal loop holonomy. Here, the minimal loop holonomy over
        the entire tensor product space is h_T = h_1 ⊗ I_2 ⊗ I_3 ⊗ ... ⊗ I_N +  I_1 ⊗ h_2 ⊗ I_3 ⊗
        ... I_N +  I_1 ⊗ I_2 ⊗ ... ⊗ h_N

        :param minimal_loop: the minimal loop currently being requested to compute the minimal loop
          holonomy for
        :param adjoint: a flag to determine whether the adjoint operator is requested
        :param vertex_index: this is a dummy parameter in this model
        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :param dimension: This controls the dimension on which the minimal loop holonomy is
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator).
        :return: the N-D minimal loop holonomy
        """

        # call the 1-d operator
        if dimension == 1:
            return super(LqxWCL3D, self).minimal_loop_holonomy(
                minimal_loop,
                adjoint=adjoint,
                vertex_index=vertex_index,
                computational=computational,
                jax=jax,
            )

        # check the computational flag
        if computational:
            return NDMinimalLoopHolonomyOperator(
                self.hilbert,
                minimal_loop,
                adjoint=bool(adjoint),
                jax=jax,
            )
        elif self.is_computational and computational is None:
            return NDMinimalLoopHolonomyOperator(
                self.hilbert,
                minimal_loop,
                adjoint=bool(adjoint),
                jax=jax,
            )
        else:
            # make sure all graphs are alive
            if len(self._all_graphs) == 0:
                self._init_graphs()

            # get the shift per graph level, aka the number of nodex in the level-0 graph
            per_copy_nodes = len(self._all_graphs[0].nx_graph.nodes)

            # create the empty operator
            ret = self._lop_type(self.hilbert.hilbert_netket)

            # loop through all gauge dimensions/graph levels
            for k in range(self.gauge_dimensions):

                # create the loop in the specific gauge level, respect orientation
                loop_k = shift_loop(minimal_loop, k, per_copy_nodes)

                # handle the adjoint case
                if adjoint == 1:
                    ret += self.minimal_loop_holonomy(
                        loop_k,
                        computational=False,
                        adjoint=1,
                        vertex_index=len(loop_k) - 1,
                        dimension=1,
                    )
                else:
                    # non-adjoint case
                    ret += self.minimal_loop_holonomy(
                        loop_k, dimension=1, computational=False
                    )

            # return the minimal loop
            return ret

    def holonomy(
        self,
        edge: Union[list, tuple],
        adjoint: bool = False,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
        dimension: Optional[int] = 3,
    ):
        r"""
        Returns the holonomy operator acting on an edge by acting on each gauge
        dimension individually

        .. math::
            \hat{h}_{e} = \sum_{i}^{gauge_dims} \hat{h}_{e_{i}}

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
          implemented in (e.g. call dimensions = 1 in a 3-d model to get the U(1) operator)
        :return:
        """

        # call the 1-d operator
        if dimension == 1:
            return super(LqxWCL3D, self).holonomy(
                edge,
                adjoint=adjoint,
                computational=computational,
                jax=jax,
            )

        # check the computational flag
        if computational:
            return NDU1HolonomyOperator(
                self.hilbert, self.graph.edge_to_index(edge), adjoint=adjoint, jax=jax
            )

        elif self.is_computational and computational is None:
            return NDU1HolonomyOperator(
                self.hilbert, self.graph.edge_to_index(edge), adjoint=adjoint, jax=jax
            )
        else:
            # create an empty operator to hold the holonomy
            _return_op = self._lop_type(self.hilbert.hilbert_netket)

            # shift factor
            _shift_factor = self.graph.n_vertices

            # insert a term for every graph level
            for _ in range(self.gauge_dimensions):
                _return_op += self.holonomy(
                    plus_key(edge, _ * _shift_factor),
                    adjoint,
                    computational=False,
                    dimension=1,
                )

            return _return_op

    def _cross_product(
        self,
        first_edge: int,
        second_edge: int,
        comp: int,
    ) -> tuple:
        """
        Returns the operators corresponding to the three components of the cross
        product on of the vectors a and b containing the spin numbers on the given edges

        .. math::
            a x b = c = (c_0, c_1, c_2), where

                c_0 = a_1 * b_2 - a_2 * b_1,

                c_1 = a_2 * b_0 - a_0 * b_2,

                c_2 = a_0 * b_1 - a_1 * b_0

        :param first_edge: the label of the first edge being acted on in the cross product
        :param second_edge: the label of the second edge being acted on in the cross product
        :return: a tuple of FunctionalLocalOperator objects for every component in the
         cross product
        """
        if self.gauge_dimensions != 3:
            raise CrossProductInHigherDimensionsError

        # get the number of edges in one graph
        _fact = self.graph.n_edges

        # get the indices of the edges through the different graphs
        _e1_indices = [
            self.graph.index_to_edge(first_edge + _ * _fact)
            for _ in range(self.gauge_dimensions)
        ]
        _e2_indices = [
            self.graph.index_to_edge(second_edge + _ * _fact)
            for _ in range(self.gauge_dimensions)
        ]

        # compute the requested component
        if comp == 0:
            c = self.number(_e1_indices[1]) * self.number(_e2_indices[2]) - self.number(
                _e1_indices[2]
            ) * self.number(_e2_indices[1])

        elif comp == 1:
            c = self.number(_e1_indices[2]) * self.number(_e2_indices[0]) - self.number(
                _e1_indices[0]
            ) * self.number(_e2_indices[2])

        elif comp == 2:
            c = self.number(_e1_indices[0]) * self.number(_e2_indices[1]) - self.number(
                _e1_indices[1]
            ) * self.number(_e2_indices[0])

        else:
            # handling an incorrect component selection
            c = None

        # return the requested component
        return c

    def _volume_component(
        self, vertex: int, component: int, is_taylor: bool = False, switcher: int = 0
    ):
        """
        A function to create the chosen component of the resulting cross product of the volume
        operator at a given vertex

        :param vertex: the vertex the volume operator has been applied to
        :param component: an identifier to decide whether we want the 0, 1 or 2 component of
            the operator
        :param is_taylor: a flag to determine if the volume operator is being Taylor expanded
        """

        # define the vertex we are acting on, change to str type to check the dicts
        vertex = str(vertex)

        # define an empty operator
        if self._is4d:
            comp_op = FunctionalLocalOperator(
                self.hilbert.hilbert_netket, specialised=True
            )
        elif is_taylor:
            comp_op = nk.operator.LocalOperator(self.hilbert.hilbert_netket)
        else:
            if switcher == 0:
                comp_op = nk.operator.LocalOperator(self.hilbert.hilbert_netket)
            elif switcher == 1:
                comp_op = FunctionalLocalOperator(
                    self.hilbert.hilbert_netket,
                    mels_func=lambda x: np.sqrt(x),
                )
            else:
                comp_op = FunctionalLocalOperator(
                    self.hilbert.hilbert_netket,
                    mels_func=lambda x: np.sqrt(np.sqrt(x)),
                )

        # iterate through all the pairs of edges connected at the vertex and in all orders
        # of pairings
        for edge_pair in self.graph.handler.graph_edges_data["graph"]["connectivities"][
            vertex
        ]["edges"]:
            # map the edges to the dual (edge) representation
            e1 = self.graph.edge_to_index(edge_pair[0])
            e2 = self.graph.edge_to_index(edge_pair[1])

            # get the sign of this edge pair in that order
            sign = int(self.graph.signs[vertex][str(edge_pair)])

            # get the `component`-th component of the cross product between these two edges
            op = sign * self._cross_product(e1, e2, component)

            # add that to the component operator, and then repeat for all connected edges
            comp_op += op

        # return the square of the sum
        return comp_op * comp_op

    def volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        """
        Returns the 3-d spatial volume operator without any Taylor expansion

        :param vertex: the vertex the operator acts on
        :param computational: (non-functional)
        :param jax: (non-functional)
        :return:
        """

        # check the computational flag
        if computational:
            return VolumeOperator(self.hilbert, vertex, jax=jax)

        elif self.is_computational and computational is None:
            return VolumeOperator(self.hilbert, vertex, jax=jax)

        else:

            # return the sum of all the components of the volume operator
            return (
                self._volume_component(vertex, 0, is_taylor=False, switcher=1)
                + self._volume_component(vertex, 1, is_taylor=False, switcher=1)
                + self._volume_component(vertex, 2, is_taylor=False, switcher=1)
            )

    def squared_volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        """
        Returns the squared 3-d spatial volume operator without any Taylor expansion

        :param vertex: the vertex the operator acts on
        :param computational: (non-functional)
        :param jax: (non-functional)
        :return:
        """

        # check the computational flag
        if computational:
            return SquaredVolumeOperator(self.hilbert, vertex, jax=jax)

        elif self.is_computational and computational is None:
            return SquaredVolumeOperator(self.hilbert, vertex, jax=jax)

        else:

            # return the sum of all the components of the volume operator
            return (
                self._volume_component(vertex, 0, is_taylor=False, switcher=0)
                + self._volume_component(vertex, 1, is_taylor=False, switcher=0)
                + self._volume_component(vertex, 2, is_taylor=False, switcher=0)
            )

    def sqrt_volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        """
        Returns the squared 3-d spatial volume operator without any Taylor expansion

        :param vertex: the vertex the operator acts on
        :param computational: (non-functional)
        :param jax: (non-functional)
        :return:
        """

        # check the computational flag
        if computational:
            return SqrtVolumeOperator(self.hilbert, vertex, jax=jax)

        elif self.is_computational and computational is None:
            return SqrtVolumeOperator(self.hilbert, vertex, jax=jax)

        else:

            # return the sum of all the components of the volume operator
            return (
                self._volume_component(vertex, 0, is_taylor=False, switcher=2)
                + self._volume_component(vertex, 1, is_taylor=False, switcher=2)
                + self._volume_component(vertex, 2, is_taylor=False, switcher=2)
            )

    def _rec_mul(
        self,
        n: int,
        a: float,
        vertex: int,
    ):
        r"""
        A function to return the :math:`(x - a)^{n}` term in the Taylor series expansion

        :param n: (int) the order at the Taylor series expansion
        :param a: (float) the point to epand around
        :param vertex: (int) the vertex the volume operator would be acting on
        :return: :math:`(x - a)^{n}` where x is volumeOperatorAtVertex
        """

        if n == 0:
            return 1

        volume_term = (
            self._volume_component(vertex, 0, is_taylor=True)
            + self._volume_component(vertex, 1, is_taylor=True)
            + self._volume_component(vertex, 2, is_taylor=True)
        )

        return (volume_term - a) * self._rec_mul(n - 1, a, vertex)

    def taylor_expanded_volume(
        self,
        order: int,
        a: float,
        vertex: int,
        square_root: bool,
    ):
        r"""
        A function the return the Taylor expansion of
        :math:`f(x) = \sqrt{x}` or :math:`f(x) = \sqrt{\sqrt{x}}` where
        x = volume operator around a point a and up to given order

        Essentially, x is :math:`V^{2}` where :math:`V` is the (2+1)-d volume operator. Therefore,
        sqrtVol = False then returns an
        approximation of the volume up to the given order, and sqrtVol = True returns an
        approximation of :math:`\sqrt{V}` up to the given order

        :param order: the order of the expansion
        :param a: the point to expand around
        :param vertex: the vertex the volume operator acts on
        :param square_root: a flag to see if we request the volume or its square root
        """

        # define the power of the x^(factor - n) term in the f^(n) term in the expansion
        factor = 1 / 4 if square_root else 1 / 2

        # see if the trivial zeroth order is requested and if so, return appropriate values
        # without recursion
        if order == 0:
            # return either sqrt(a) or sqrt(sqrt(a))
            return a**factor

        return (1 / factorial(order)) * (a ** (factor - order)) * _An(
            order, square_root
        ) * self._rec_mul(order, a, vertex) + self.taylor_expanded_volume(
            order - 1, a, vertex, square_root
        )

    def thiemann_regularized_constraint(
        self,
        vertex: str,
        *,
        jax: Optional[bool] = False,
        fast: Optional[bool] = True,
        adjoint: Optional[bool] = False,
        apply_lapse: Optional[bool] = False,
    ) -> List:
        """
        A function that creates the Thiemann regularized constraint. This will return the TRC on
        the specified vertex.

        :param vertex: the vertex on which the constraint acts on
        :param adjoint: a flag to determine if the adjoint operator is requested
        :param jax: a flag to determine whether the returned ComputationalOperator should be of
          ``ComputationalOperator`` or ``ComputationalJaxOperator`` type
        :param fast: if True, a faster Numba based ``ComputationalOperator`` implementation is
          returned
        :param apply_lapse: if True, the operator will use the lapse saved in self.lapses
        :return: the Thiemann regularized constraint
        """

        return ThiemannRegularisedVertexConstraintOperator(
            self,
            vertex,
            adjoint=adjoint,
            apply_lapse=apply_lapse,
            jax=jax,
        )

    def thiemann_quadratic_constraint(
        self,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
        **kwargs,
    ):
        """
        Return a quadratic constraint object Σᵥ Ĥᵥ Ĥᵥᵀ where Ĥᵥᵀ is the
        adjoint of the Thiemann regularised constraint at the vertex v.

        When ``kwargs['adjoint'] = True`` (default is False), this method returns Σᵥ Ĥᵥᵀ Ĥᵥ instead
        of Σᵥ Ĥᵥ Ĥᵥᵀ.

        Note: for the optional ``kwargs``, you can specify:
            - kwargs['adjoint'] (bool): if set to False, the Σᵥ Ĥᵥ Ĥᵥᵀ version is returned, otherwise Σᵥ Ĥᵥᵀ Ĥᵥ
            - kwargs['apply_lapse'] (bool): if set to True, lapses will be applied when constructing
                the individual constraints.
            - kwargs['lazy'] (bool): if set to True, the quadratic constraint is implemented using
                NetKet's ``Squared`` lazy wrapper.
            - kwargs['fast'] (bool): if True, and ``jax`` is False, a faster Numba based
                ``ComputationalOperator`` is returned

        :param computational: (not functional)
        :param kwargs: see the docstring
        :param jax: a flag to determine whether the returned ComputationalOperator should be of
          ``ComputationalOperator`` or ``ComputationalJaxOperator`` type
        """

        if kwargs is None:
            kwargs = {}

        lazy = kwargs.get("lazy", True)
        adjoint = kwargs.get("adjoint", False)
        apply_lapse = kwargs.get("apply_lapse", False)

        event(
            msg="CALLING_THIEMANN_QUADRATIC_3D_WCL",
            tag="LQXWCL4D",
            level=logging.INFO,
            lazy=lazy,
            adjoint=adjoint,
            apply_lapse=apply_lapse,
        )

        if lazy:
            # return the Master-like constraint for every vertex constraint in a list
            return [
                Squared(
                    ThiemannRegularisedVertexConstraintOperator(
                        self,
                        v,
                        adjoint=adjoint,
                        apply_lapse=apply_lapse,
                        jax=jax,
                    )
                )
                for v in self.graph.vertices
            ]
        product_wrapper = Product if not jax else ProductJax
        return [
            product_wrapper(
                ThiemannRegularisedVertexConstraintOperator(
                    self,
                    v,
                    adjoint=False if adjoint else True,
                    apply_lapse=apply_lapse,
                    jax=jax,
                ),
                ThiemannRegularisedVertexConstraintOperator(
                    self,
                    v,
                    adjoint=True if adjoint else False,
                    apply_lapse=apply_lapse,
                    jax=jax,
                ),
                is_hermitian=True,
            )
            for v in self.graph.vertices
        ]
