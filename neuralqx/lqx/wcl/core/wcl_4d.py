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

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from collections import Counter

import numpy as np

import jax.numpy as jnp

from netket.operator import Squared
from netket.operator import LocalOperator

from neuralqx.debug import event
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface

from neuralqx.operators.types import FunctionalLocalOperator
from neuralqx.operators.computational.Euclidean4d import AreaDifferenceSquaredOperator
from neuralqx.operators.computational.Euclidean4d import (
    AreaDifferenceSquaredSurfacesOperator,
)
from neuralqx.operators.computational.Euclidean4d import AreaOperator
from neuralqx.operators.computational.Euclidean4d import (
    ThiemannRegularisedVertexConstraintOperator,
)
from neuralqx.operators.computational.Euclidean4d import VolumeOperator
from neuralqx.operators.computational.wrappers import ProductJax
from neuralqx.operators.computational.wrappers import Product

from neuralqx.utils.errors import AreaDifferenceEdgesError
from neuralqx.utils.errors import CrossProductInHigherDimensionsError
from neuralqx.utils.errors import IncompatiblePlanarGraphModel
from neuralqx.utils.errors import InvalidSurfaceError
from neuralqx.utils.misc.levi_civita import compute_levi_civita_key
from neuralqx.utils.misc.graph import count_contributing_triplets_key
from neuralqx.utils.misc.graph import create_dressed_minimal_loop_key
from neuralqx.utils.misc.graph import find_minimal_loops_with_edges_key
from neuralqx.utils.misc.graph import get_triplets_key
from neuralqx.utils.misc.graph import reorder_edge_triplet_key
from neuralqx.utils.misc.graph import reorder_minimal_loop_key
from neuralqx.utils.misc.graph import get_true_edge_triplet
from neuralqx.utils.misc.graph import reduce_permutations
from neuralqx.utils.misc.arithmetic import plus_key
from neuralqx.utils.misc.triangulation import Triangulation

from .wcl_3d import LqxWCL3D


class LqxWCL4D(LqxWCL3D):

    def __init__(
        self,
        hilbert: AbstractHilbertInterface,
        graph: AbstractGraph,
        gauge_group: AbstractGaugeGroup,
        *,
        computational: bool = True,
        spacetime_dimensions: int = 3,
        model_name: str = "(3+1)-Euclidean LQG",
        is_4d: bool = True,
    ):
        # check for planarity, this model only supports planar graphs
        if graph.is_planar:
            raise IncompatiblePlanarGraphModel()

        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
            is_4d=is_4d,
        )

        self._setup()

    @property
    def H(self):
        # TODO: remove this when the models are no longer experimental
        return self.hilbert

    def _setup(self) -> None:
        """
        This method is used to internally set up the structure of the model. This includes lifting
        the one level graph to 3 graphs, with correct minimal loops, as well as collecting all the
        terms relevant to the Thiemann regularised Hamilton constraint.
        """

        super(LqxWCL4D, self)._setup()

        # a list to hold the vertices the TRMC is allowed to act on (e.g. no 2-valent vertices)
        self._admissible_vertices = []

        # create the empty dict of ε(e₁, e₂, e₃) to be populated during the Thiemann constraint init
        self.epsilons = {}

        # a temp var to hold the maximum number of terms in the constraint based on the vertices
        self._max_terms = 0

        # a dict to hold all the terms in the constraint
        self._constraint_terms = {
            "0": self.init_constraint_terms(0),
            "1": self.init_constraint_terms(1),
            "2": self.init_constraint_terms(2),
        }

        # define the lapse function in the constraint
        self.lapses = [1 for _ in range(len(self.graph.nx_graph.nodes))]

        # get the set, because calling init_constraint_terms thrice above will produce thrice as
        # many of actual admissible vertices
        self._admissible_vertices = set(self._admissible_vertices)

        self._master_constraint = None

    def init_constraint_terms(
        self,
        graph_level: int,
    ) -> dict[str, dict[str, Union[list[Any], None]]]:
        """
        A function which collects all the terms the constraint will act on for a given component
        in the constraint. Specifically, it finds all possible edge triplets (e₁, e₂, e₃)
        for every vertex and for every (e₁, e₂), it finds all contributing minimal loops. It
        also finds all ε(e₁, e₂, e₃) for all edge triplets.

        :param graph_level: the graph level to construct the terms for
        :return: a dict of all the contributing terms at the graph level
        """

        # which graph level do we act on
        graph_shift = len(self.graph.nx_graph.nodes) * graph_level

        # empty local dict to hold the terms
        _constraint_terms = {}

        # a counter to determine the max number of terms per vertex
        _max_terms = 0

        # a list to hold the number of terms per vertex
        _max_terms_per_vertex = []

        # loop through every vertex
        for vertex in self.graph.nx_graph.nodes:

            # a registry to hold the minimal loop edges at the current vertex
            _vertex_minimal_loop_edges_registry = []

            # a registry to hold the segment edges at the current vertex
            _vertex_segment_edges_registry = []

            # add the current vertex to the local dict
            _constraint_terms[str(vertex)] = {"triplets": [], "T": None}

            # get all edge triplets at the vertex
            all_triplet_edges = get_triplets_key(self.graph.edges, vertex)

            # filter out permutations in the first two edges
            all_triplet_edges = reduce_permutations(all_triplet_edges)

            # check if we even have edge triplets, for 2-valent nodes, we skip
            if not all_triplet_edges:
                # delete anything about this vertex and skip it
                _constraint_terms.pop(str(vertex))
                continue

            # if the vertex is going to be considered, add it to the list of vertices admissible
            # by the constraint
            self._admissible_vertices.append(vertex)

            # loop through every edge triplet
            for edge_triplet in all_triplet_edges:

                # compute the ε(e₁, e₂, e₃) for all the permutations of the current triplet
                compute_levi_civita_key(
                    list(itertools.permutations(edge_triplet, 3)), self.epsilons
                )

                # record the ε(e₁, e₂, e₃) contribution before shifting the edges to the chosen
                # graph level
                ε = self.epsilons[str(tuple(edge_triplet))]

                # get the reordered edges such that they all emanate from the vertex
                reordered_edge_triplet, reordering_sign = reorder_edge_triplet_key(
                    edge_triplet, vertex
                )

                # set the two edges which will should fall in the minimal loop, they are e₁ and e₂,
                # so we take the first two
                minimal_loop_edge1 = reordered_edge_triplet[0]
                minimal_loop_edge2 = reordered_edge_triplet[1]

                # set the segment edge
                segment_edge = reordered_edge_triplet[2]

                # get all the minimal loops which these two edge pairs form, if any
                all_minimal_loops = find_minimal_loops_with_edges_key(
                    self.graph.minimal_loops(),
                    [minimal_loop_edge1, minimal_loop_edge2],
                )

                # get the numerical factor L
                L = len(all_minimal_loops)

                # shift all edges to the current graph level
                segment_edge = list(plus_key(segment_edge, graph_shift))
                minimal_loop_edge1 = list(plus_key(minimal_loop_edge1, graph_shift))
                minimal_loop_edge2 = list(plus_key(minimal_loop_edge2, graph_shift))

                # only proceed to add terms if there are minimal loops for the edge pair (e₁, e₂)
                if L > 0:

                    _constraint_terms[str(vertex)]["triplets"].append(
                        {
                            "triplet": [
                                minimal_loop_edge1,
                                minimal_loop_edge2,
                                segment_edge,
                            ],
                            "ε": ε,
                            "L": L,
                            "terms": [],
                        }
                    )

                    # loop through every minimal loop which includes the edge pair and add a term
                    # in the constraint
                    for minimal_loop in all_minimal_loops:
                        # shift the current minimal loop to the correct graph level
                        minimal_loop = [
                            list(plus_key(edge, graph_shift)) for edge in minimal_loop
                        ]

                        # now reorder the minimal loop such that it starts from e₁ and ends
                        # at (e₂)⁻¹
                        reordered_minimal_loop = reorder_minimal_loop_key(
                            minimal_loop, [minimal_loop_edge1, minimal_loop_edge2]
                        )

                        # and now dress the reordered minimal loop
                        dressed_minimal_loop = create_dressed_minimal_loop_key(
                            self.graph, reordered_minimal_loop, graph_shift
                        )

                        # create a triangulation object and add it to the local dict
                        _constraint_terms[str(vertex)]["triplets"][-1]["terms"].append(
                            Triangulation(
                                **{
                                    "reordered_edge_triplet": [
                                        minimal_loop_edge1,
                                        minimal_loop_edge2,
                                        segment_edge,
                                    ],
                                    "reordering_sign": reordering_sign,
                                    "ε": ε,
                                    "L": L,
                                    "minimal_loop_edges": [
                                        minimal_loop_edge1,
                                        minimal_loop_edge2,
                                    ],
                                    "segment_edge": segment_edge,
                                    "reordered_minimal_loop": reordered_minimal_loop,
                                    "dressed_minimal_loop": dressed_minimal_loop,
                                }
                            )
                        )
                        _max_terms += 1

            # add the factor T in the constraint
            T = count_contributing_triplets_key(self.epsilons, all_triplet_edges)
            _constraint_terms[str(vertex)]["T"] = T

            # append the number of maximum terms for this vertex and reset the counter
            _max_terms_per_vertex.append(_max_terms)
            _max_terms = 0

        # set the number of max terms
        self._max_terms = max(_max_terms_per_vertex)

        return _constraint_terms

    def area(
        self,
        surface: Union[list, tuple],
        standalone: bool = True,
        *,
        squared: bool = False,
        computational: bool = False,
        jax: Optional[bool] = False,
    ) -> Union[List, AreaOperator]:
        r"""
        Returns the area operator for a surface through which edges pass through. The specified
        ``surface`` should be a list of edges of the graph which are assumed to pucnture through
        the desired surface.

        The area for a given graph is given by

        .. math::

            \hat{A}(S) |\vec{m}\rangle
            =
            \sum_{e \cap S\neq\varnothing}
            \sqrt{ \big(m_e^{(1)}\big)^2 + \big(m_e^{(2)}\big)^2 + \big(m_e^{(3)}\big)^2 } |\vec{m}\rangle

        i.e. for each edge intersecting \(S\) take its three U(1) charges, square them, sum, and
        take the square root. (In the generic normalization one would multiply by 8πℓₚ² here that
        prefactor is set to 1 by convention, see Notes.)

        Orientation of an edge does not affect the result since the operator uses the charge
        **norm**. If an edge punctures the surface multiple times, it should be included with
        several times.

        For non-planar graphs, these edges should be supplied in accordance to the edges appearing
        from the ``neuralqx.graph.Graph.edges`` (that is, not in the non-planar representation).

        Notes: here it is assumed the following
            - unit Planck length
            - Immirzi parameter of value 1.0
            - Hence the 8πℓₚ² factor is absorbed into units

        :param surface: a list of edges passing through the assumed surface
        :param standalone: a flag to determine if the operator will be multiplied with other
          ``LocalOperator`` types or not
        :param squared: a flag to determine if the area or area squared is requested
        :param computational: a flag to determine whether a ``LocalOperator`` or
          ``ComputationalOperator`` based implementation is requested
        :param jax: a flag to determine whether the returned ComputationalOperator should be of
          ``ComputationalOperator`` or ``ComputationalJaxOperator`` type
        :return: a FunctionalLocalOperator denoting the area operator for the given surface
        """

        # ensure surface type
        if not isinstance(surface, List) and not isinstance(surface, Tuple):
            raise InvalidSurfaceError(type(surface).__name__)

        # check the computational flag
        if computational:
            return AreaOperator(self.hilbert, surface, squared=squared, jax=jax)

        elif self.is_computational and computational is None:
            return AreaOperator(self.hilbert, surface, squared=squared, jax=jax)

        else:
            # a list to hold the area operators for each edge
            areas_ops = []

            # check dtype based on standalone flag
            dtype = jnp.float64 if standalone else jnp.complex128

            # local function for brevity
            n̂ = self.number

            # edges to indices
            surface_idx = [self.graph.edge_to_index(e) for e in surface]

            # edge multiplicities to avoid creating redundant operator objects
            # here, the keys are the edge indices and the vals are the multiplicities
            surface_idx_mult = Counter(surface_idx)

            # construct the lambda function for the FunctionalLocalOperator based on the flag
            if squared:
                lfunc = lambda x: x.real.astype(np.float64)
            else:
                lfunc = lambda x: np.sqrt(x.real.astype(np.float64))

            # loop through each edge and create an area operator
            for edge, mult in surface_idx_mult.items():

                # here the multiplicity is squared because with NetKet's algebra, this will
                # actually jump inside the square root and beside each \big(m_e^{(i)}\big)^2
                # of course, this is not the case if we are requesting the squared area operators
                mult = mult**2 if not squared else mult

                # an empty area operator with the mels_func being a square root
                # here we do not care about specialised multiplication unless specified
                Âₑ = FunctionalLocalOperator(
                    self.hilbert.hilbert_netket,
                    specialised=False if standalone else True,
                    mels_func=lfunc,
                    dtype=dtype,
                )

                # get the charges of the charge vector on the edge using number operators

                # get the number of edges in one graph
                _fact = self.hilbert.tiny_size

                charge_vector_idx = [
                    edge + i * _fact for i in range(self.gauge_dimensions)
                ]

                # for each charge in this charge vector, add a corresponding number operator squared
                for m in charge_vector_idx:

                    # this constructs \big(m_e^{(i)}\big)^2
                    # and appends to the current area operator on that edge

                    Âₑ += n̂(
                        self.graph.index_to_edge(m),
                        computational=False,
                        jax=False,
                        kwargs=dict(is_complex=not standalone, is_4d=not standalone),
                    ) * n̂(
                        self.graph.index_to_edge(m),
                        computational=False,
                        jax=jax,
                        kwargs=dict(is_complex=not standalone, is_4d=not standalone),
                    )

                # now we are done with that edge, append that area operator to the list
                areas_ops.append(mult * Âₑ)

            # return the list of area operators for edges puncturing the surface
            return areas_ops

    def area_difference_squared_operator(
        self,
        edges: List,
        standalone: bool = True,
        *,
        computational: bool = False,
        jax: Optional[bool] = False,
    ) -> Union[List, AreaDifferenceSquaredOperator]:
        r"""
        Returns an area difference squared operator between two edges such that

        .. math::

            \langle(\hat{A}_{e_1} - \hat{A}_{e_2})^2\rangle =
            \langle \hat{A}_{e_1}^2 \rangle + \langle \hat{A}_{e_2}^2 \rangle
            - 2 \langle \hat{A}_{e_1} \hat{A}_{e_2} \rangle

        where the area operator is constructed according to ``EuclideanWCL4D.area_operator()``
        (see its documentation for details).

        For non-planar graphs, these edges should be supplied in accordance to the edges appearing
        from the ``neuralqx.graph.Graph.edges`` (that is, not in the non-planar representation).

        Notes: here it is assumed when computing the area the following
            - unit Planck length
            - Immirzi parameter of value 1.0
            - Hence the 8πℓₚ² factor in a given area operator is absorbed into units

        :param edges: a list of two edges to compute their area difference from
        :param standalone: a flag to determine whether this operator is going to be multiplied with
          other ``LocalOperator`` types
        :param computational: a flag to determine whether a ``LocalOperator`` or
          ``ComputationalOperator`` based implementation is requested
        :param jax: a flag to determine whether the returned ComputationalOperator should be of
          ``ComputationalOperator`` or ``ComputationalJaxOperator`` type
        :return: a list of operators representing the right hand side of the area difference
          operator described above
        """

        # ensure that we have only two edges
        if len(edges) > 2:
            raise AreaDifferenceEdgesError(len(edges))

        # fastpath with the computational implementation
        if computational:
            return AreaDifferenceSquaredOperator(self.hilbert, edges, jax=jax)

        # convert the edges to concrete indices
        edges_idx = [self.graph.edge_to_index(e) for e in edges]

        # construct the first two parts of the RHS of the area difference operator
        # TODO: are we double edge to indexing? Here and in self.area?
        A_e1_sq = self.area(surface=[edges[0]], standalone=standalone, squared=True)[0]

        A_e2_sq = self.area(surface=[edges[1]], standalone=standalone, squared=True)[0]

        # check dtype based on standalone flag
        dtype = jnp.float64 if standalone else jnp.complex128

        # local function for brevity
        n̂ = self.number

        # now the third piece
        # here we need to construct the product of two area operators
        # we need a double sum over the three components of each charge vector

        # get the number of edges in one graph
        _fact = self.hilbert.tiny_size

        # get the indices of the components of the charge vector associated with the first edge
        cv_e1_idxs = [
            edges_idx[0] + i * self.hilbert.tiny_size
            for i in range(self.gauge_dimensions)
        ]

        # get the indices of the components of the charge vector associated with the second edge
        cv_e2_idxs = [
            edges_idx[1] + i * self.hilbert.tiny_size
            for i in range(self.gauge_dimensions)
        ]

        def squared_number_op(i):
            r"""Helper function that returns the number operator squared on the given edge index."""
            return n̂(
                self.graph.index_to_edge(i),
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                self.graph.index_to_edge(i),
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            )

        # note, here we will multiply later with a -2, so if we naively do the mels_func as done
        # in the area operator, we will get NaNs as answer, because the -2 will be inside the
        # square root
        Ae1_Ae2 = FunctionalLocalOperator(
            self.hilbert.hilbert_netket,
            specialised=False if standalone else True,
            mels_func=lambda x: np.sign(x.real)
            * np.sqrt(np.abs(x.real.astype(np.float64))),
            dtype=dtype,
        )

        # loop through the components of the first edge
        for cv_e1_i in cv_e1_idxs:

            # now we need to construct the square of the number operator for this component for
            # the first edge
            N_e1_i = squared_number_op(cv_e1_i)

            # loop through the components of the second edge
            for cv_e2_i in cv_e2_idxs:

                # now construct the squared number operator for this component for the second edge
                N_e2_i = squared_number_op(cv_e2_i)

                # now we multiply these two squared components
                Ae1_Ae2 += N_e1_i * N_e2_i

        # now, the Ae1_Ae2 operator should be populated with the product of squared number operators
        # of all components, we are safe to put everything in a list and return
        # NOTE: here we multiply by -4, not -2, because it will (according to the LocalOperator
        # algebra) jumpy inside the square root
        return [A_e1_sq, A_e2_sq, -4 * Ae1_Ae2]

    def area_difference_squared_surfaces_operator(
        self,
        surfaces: List,
        jax: bool = False,
    ) -> Union[AreaDifferenceSquaredSurfacesOperator]:
        r"""
        Returns the squared area difference operator between two surfaces \(S_1, S_2\),
        each defined as a list of edges, such that

        .. math::

            \langle(\hat{A}(S_1) - \hat{A}(S_2))^2\rangle
            =
            \langle \hat{A}(S_1)^2 \rangle + \langle \hat{A}(S_2)^2 \rangle
            - 2\,\langle \hat{A}(S_1)\hat{A}(S_2) \rangle.

        Args:
        :param surfaces: A list of two surfaces, each surface being a list of edges.
          Example: [[(0,1,0),(0,2,0)], [(0,4,0),(0,3,0)]].
        :param jax: a flag to determine whether the returned ComputationalOperator should be of
          ``ComputationalOperator`` or ``ComputationalJaxOperator`` type

        Note: this operator only has a ComputationalOperator implementation.
        """

        return AreaDifferenceSquaredSurfacesOperator(self.hilbert, surfaces, jax=jax)

    def _triple_scalar_product_comp(
        self,
        edge_triple: tuple,
        comp: int,
        standalone: bool = False,
    ) -> LocalOperator:
        """
        A function that returns the operators corresponding to the three components of the triple
        scalar product on of the vectors a, b and c in the edge_triple tuple which contain the
        spin numbers on the given edges

        Here,
        .. math::
            a x b = c = (c_0, c_1, c_2) where

             c_0 = a_1 * b_2 - a_2 * b_1,

              c_1 = a_2 * b_0 - a_0 * b_2,

               c_2 = a_0 * b_1 - a_1 * b_0

        And the scalar triple product is therefore

        .. math::
            d • (a x b) = d • c = d_0 * c_0 + d_1 * c_1 + d_2 * c_2

        The function returns the following based on the value of the comp param:

        - comp = 0   ->   d_0 * c_0
        - comp = 1   ->   d_1 * c_1
        - comp = 2   ->   d_2 * c_2

        :param edge_triple: the labels of each of the three edges being acted on by the volume
            operator
        :param standalone: a flag to determine whether the operator is going to be multiplied with
          ``LocalOperator`` types or not
        :return: a LocalOperator objects for every component in the product
        """

        # local function for brevity
        n̂ = self.number

        if self.gauge_dimensions != 3:
            raise CrossProductInHigherDimensionsError

        # get the number of edges in one graph
        _fact = self.hilbert.tiny_size

        # create a list of lists holding the component indices of the vectors in the product
        # example: [[1, 6, 11], [2, 7, 12], [3, 8, 13]] where [1, 6, 11] are the edges which when
        # we get their quantum numbers, form the first charge vector. So _idx[0] means we are acting
        # on the edge 0, and _idx[0][1] means edge 0 and the graph level 1
        _idx = []
        for _ in range(len(edge_triple)):
            _idx.append(
                [
                    self.graph.index_to_edge(edge_triple[_] + i * _fact)
                    for i in range(self.gauge_dimensions)
                ]
            )

        """
        Changes:
            Removed is_4d checks

        Reasons:
            The number operator used in the volume should be created as a 
            netket.operator.LocalOperator as it will be constructed as a netket operator first and 
            then appended to a MarkedLocalOperator
        """
        # compute the requested component
        if comp == 0:

            # c_0 = a_1 * b_2 - a_2 * b_1
            _c = n̂(
                _idx[1][1],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                _idx[2][2],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) - n̂(
                _idx[1][2],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                _idx[2][1],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            )

            # d_0 * c_0
            c = (
                n̂(
                    _idx[0][0],
                    computational=False,
                    jax=False,
                    kwargs=dict(is_complex=not standalone, is_4d=not standalone),
                )
                * _c
            )

        elif comp == 1:

            # c_1 = a_2 * b_0 - a_0 * b_2
            _c = n̂(
                _idx[1][2],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                _idx[2][0],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) - n̂(
                _idx[1][0],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                _idx[2][2],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            )

            # d_1 * c_1
            c = (
                n̂(
                    _idx[0][1],
                    computational=False,
                    jax=False,
                    kwargs=dict(is_complex=not standalone, is_4d=not standalone),
                )
                * _c
            )

        elif comp == 2:

            # c_2 = a_0 * b_1 - a_1 * b_0
            _c = n̂(
                _idx[1][0],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                _idx[2][1],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) - n̂(
                _idx[1][1],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            ) * n̂(
                _idx[2][0],
                computational=False,
                jax=False,
                kwargs=dict(is_complex=not standalone, is_4d=not standalone),
            )

            # d_2 * c_2
            c = (
                n̂(
                    _idx[0][2],
                    computational=False,
                    jax=False,
                    kwargs=dict(is_complex=not standalone, is_4d=not standalone),
                )
                * _c
            )

        else:
            c = None

        # return the component
        return c

    def _volume_operator_component(
        self,
        vertex: int,
        component: int,
        standalone: bool = False,
        *args,
    ) -> FunctionalLocalOperator:
        """
        A method that returns the requested component of the volume operator. Here, component
        means the component with respect to the triple scalar product.

        :param vertex: the vertex being acted on
        :param component: the requested component
        :param standalone: a flag to determine whether the operator is going to be multiplied with
          ``LocalOperator`` types or not
        :return: returns a LocalSqrtAbsOperator for the volume operator component
        """

        # define the vertex we are acting on, change to str type to check the dicts
        vertex = str(vertex)

        """
        Changes: 
            Here, the volume will be computed as a netket.operator.LocalOperator and then
            in the TRC, it will be appended in a MarkedLocalOperator

        Reasons:
            This ensures that the volume is created normally, but once in the TRC, it is treated 
            as a marked operator.

        Comments:
            self._lop_type here is netket.operator.LocalOperator
        """
        # define an empty operator which will hold the component operator at the end
        # using _lop_type will cause mismatch in the sign of the real and imaginary parts
        comp_op = FunctionalLocalOperator(
            self.hilbert.hilbert_netket,
            dtype=jnp.float64 if standalone else jnp.complex128,
            mels_func=lambda x: np.sqrt(np.abs(np.real(x))),
            specialised=True,
        )

        for edge_triple in self.graph.handler.graph_edges_data["graph"][
            "connectivities"
        ][vertex]["edges"]:
            e1 = self.graph.edge_to_index(edge_triple[0])
            e2 = self.graph.edge_to_index(edge_triple[1])
            e3 = self.graph.edge_to_index(edge_triple[2])

            sign = int(
                self.graph.signs[vertex][get_true_edge_triplet(edge_triple, self.graph)]
            )

            # get the `component`-th component of the scalar triple product between these
            # three edges
            op = sign * self._triple_scalar_product_comp(
                (e1, e2, e3), component, standalone
            )

            # add that to the component operator, and then repeat for all connected edge triples
            comp_op += op

        # return the sum
        return comp_op

    def volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
        standalone: bool = False,
    ) -> Union[FunctionalLocalOperator, VolumeOperator]:
        """
        A method to return the volume operator at a given vertex

        :param standalone: a flag to determine whether the operator is going to be multiplied with
          ``LocalOperator`` types or not
        :param computational: a flag to determine whether a ``LocalOperator`` or
          ``ComputationalOperator`` based implementation is requested
        :param vertex: the vertex on which the volume operator acts on
        :param jax: a flag to determine whether the returned ComputationalOperator should be of
          ``ComputationalOperator`` or ``ComputationalJaxOperator`` type
        :return: the volume operator on that vertex as a LocalSqrtAbsOperator
        """

        # check the computational flag
        if computational:
            return VolumeOperator(self.hilbert, vertex, jax=jax)

        elif self.is_computational and computational is None:
            return VolumeOperator(self.hilbert, vertex, jax=jax)

        else:

            volume = (
                self._volume_operator_component(vertex, 0, standalone)
                + self._volume_operator_component(vertex, 1, standalone)
                + self._volume_operator_component(vertex, 2, standalone)
            )

            # return the sum of all the components of the volume operator
            return volume

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
            fast=fast,
            jax=jax,
        )

    def thiemann_quadratic_constraint(
        self,
        *,
        computational: Optional[bool] = True,
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

        # some defaults
        if kwargs is None:
            kwargs = {}

        lazy = kwargs.get("lazy", True)
        adjoint = kwargs.get("adjoint", False)
        apply_lapse = kwargs.get("apply_lapse", False)
        fast = kwargs.get("fast", True)

        event(
            msg="CALLING_THIEMANN_QUADRATIC_4D_WCL",
            tag="LQXWCL4D",
            level=logging.INFO,
            lazy=lazy,
            adjoint=adjoint,
            apply_lapse=apply_lapse,
            fast=fast,
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
                        fast=fast,
                    )
                )
                for v in self._admissible_vertices
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
                    fast=fast,
                ),
                ThiemannRegularisedVertexConstraintOperator(
                    self,
                    v,
                    adjoint=True if adjoint else False,
                    apply_lapse=apply_lapse,
                    jax=jax,
                    fast=fast,
                ),
                is_hermitian=True,
            )
            for v in self._admissible_vertices
        ]

    def _cross_product(
        self,
        first_edge: int,
        second_edge: int,
        comp: int,
    ) -> tuple:
        raise NotImplementedError

    def _volume_component(
        self, vertex: int, component: int, is_taylor: bool = False, switcher: int = 0
    ):
        raise NotImplementedError

    def squared_volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        raise NotImplementedError

    def taylor_expanded_volume(
        self,
        order: int,
        a: float,
        vertex: int,
        square_root: bool,
    ):
        raise NotImplementedError

    def sqrt_volume(
        self,
        vertex: int,
        *,
        computational: Optional[bool] = None,
        jax: Optional[bool] = None,
    ):
        raise NotImplementedError

    def _thiemann_regularized_constraint_comp(
        self,
        comp: int,
        order: int = 1,
        a: float = 4,
        is_adjoint: bool = False,
    ):
        raise NotImplementedError
