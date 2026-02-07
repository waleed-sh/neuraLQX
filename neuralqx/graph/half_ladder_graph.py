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

import jax.numpy as jnp

from .graph import Graph
from .core.utils.build import relabel_edges_to_nonplanar


class HalfLadderGraph(Graph):
    """
    This Graph is one which represents half a ladder. That is, it looks like this:

            5   6   7   8   9
            |   |   |   |   |
            0---1---2---3---4

    where the integers represent vertices and the lines represent edges (oriented in incrementing
    order).

    If you specify `connected = True`, then it will create a loop graph, connected in the example
    above the vertex 4 to 0.

    In this context, the vertices lying on the main graph (0, 1, 2, 3, 4) are called k-vertices
    and the edges connecting them are called k-edges. The vertices (5, 6, 7, 8, 9) which branch out
    from the k-vertices are called μ-vertices. In similar fashion, edges connecting the k- and μ-
    vertices are called μ-edges.

    NOTE:
        This half ladder implementation contains extra information such as the k- and μ-
        vertices/edges because it was initially developed for a certain spherically symmetric
        model.
    """

    def __init__(
        self,
        length: int,
        connected: bool = False,
        *,
        plot: bool = False,
        non_planar: bool = False,
    ):

        # dev: the implementation is broken, so until fixed...
        if non_planar:
            raise NotImplementedError(
                f"Currently, the {type(self).__name__} is only available as a planar graph."
            )

        # create the k-vertices
        self._vertices_k = list(range(length + 1))

        # connect all of them with edges to one another incrementally
        # dev: include 0 keys, no parallel edges assumed
        edges = [
            [self._vertices_k[i], self._vertices_k[i + 1], 0] for i in range(length)
        ]

        # check if we want a looped graph
        if connected:
            edges.append([self._vertices_k[-1], self._vertices_k[0], 0])

        self._is_connected = connected

        # save the k-edges
        self._edges_k = edges.copy()

        # now we create the mu vertices
        # first get the int label of the last k-vertex
        max_v_k = self._vertices_k[-1]

        # add as many μ-vertices as there are k-vertices, but now with
        # integer labels starting from the last k-vertex label
        # if we wish to omit the μ-vertices on the initial and ending k-vertices, then the
        # range should be range(len(self._vertices_k) - 2), with -2 accounting for the 2 k-vertices
        # we are going to ignore
        self._vertices_mu = [max_v_k + i + 1 for i in range(len(self._vertices_k))]

        # append new edges between the k- and μ-vertices
        # if we wish to omit μ-vertices on the initial and ending k-vertices, then the loop should
        # be over zip(self._vertices_k[1:-1], self._vertices_mu)
        # note: you must then also alter the _vertices_mu as described in the comment above for the
        # two to have the same length and stay consistent in the following loop
        # dev: we assume no parallel edges here, all edges have the same key
        self._edges_mu = [
            [v, mu, 0] for v, mu in zip(self._vertices_k, self._vertices_mu)
        ]

        # we now have the complete list of edges to pass to the handler
        edges += self._edges_mu.copy()

        # if we have a non-planar graph, we need to do more stuff
        if non_planar:
            # first relabel all the edges to a non-planar 3-tuple representation
            edges, vertex_mapping = relabel_edges_to_nonplanar(edges)

            # relable the k- and μ-edges as well
            self._edges_k = [
                [vertex_mapping[v] for v in edge] for edge in self._edges_k
            ]
            self._edges_mu = [
                [vertex_mapping[v] for v in edge] for edge in self._edges_mu
            ]

            # and the k- and μ-vertices too
            self._vertices_mu = [vertex_mapping[v] for v in self._vertices_mu]
            self._vertices_k = [vertex_mapping[v] for v in self._vertices_k]

        # we need to init before the next step, because the next chunk of code uses self.handler
        # which only exists after the AbstractGraph is initialised
        super().__init__(edges, plot)

        # we continue again...
        if non_planar:
            # now relabel everything according to the non-planar vertex mapping
            # we basically went from user based integer [a, b] representation, to a 3-tuple
            # ((a, b, c), (x, y, z)) representation, which we then mapped to an integer based
            # [A, B] representation which we can map to the NetKet integer based labeling... fun
            self._edges_k = [
                [self.nonplanar_vertex_mapping.get(v) for v in edge]
                for edge in self._edges_k
            ]
            self._edges_mu = [
                [self.nonplanar_vertex_mapping.get(v) for v in edge]
                for edge in self._edges_mu
            ]
            self._vertices_mu = [
                self.nonplanar_vertex_mapping.get(v) for v in self._vertices_mu
            ]
            self._vertices_k = [
                self.nonplanar_vertex_mapping.get(v) for v in self._vertices_k
            ]

        # save the list of indices of the μ-edges
        self.edges_mu_idx = jnp.array(
            [self.edge_to_index(edge) for edge in self.edges_mu],
            dtype=jnp.int32,
        )

    @property
    def edges_k(self):
        return self._edges_k

    @property
    def edges_mu(self):
        return self._edges_mu

    @property
    def vertices_k(self):
        return self._vertices_k

    @property
    def vertices_mu(self):
        return self._vertices_mu

    @property
    def is_connected(self):
        return self._is_connected

    def get_edges_at_k_vertex(
        self,
        vertex: int,
        dual_rep: bool = False,
    ) -> tuple[list, list]:
        """
        This function returns the μ-edge and the two k-edges attached to the requested `vertex`. The
        requested vertex needs to be a k-vertex.

        :return: a tuple where the first entry is the μ-edge and the second entry is a list of the
                 two k-edges, or the dual vertex representation if `dual_rep = True`
        """
        if vertex not in self.vertices_k:
            raise ValueError("\n" f"The requested vertex `{vertex}` is not a k-vertex.")

        # NOTE TO ROBIN: here, I excluded the keys being considered as actual vertices of the edge
        if dual_rep == False:
            k_edges = [k_edge for k_edge in self.edges_k if vertex in k_edge[:-1]]
            mu_edge = [mu_edge for mu_edge in self.edges_mu if vertex in mu_edge[:-1]]
        else:
            k_edges = [
                self.edge_to_index(k_edge)
                for k_edge in self.edges_k
                if vertex in k_edge[:-1]
            ]
            mu_edge = [
                self.edge_to_index(mu_edge)
                for mu_edge in self.edges_mu
                if vertex in mu_edge[:-1]
            ]

        return mu_edge, k_edges

    def has_k_vertex_neighbours(self, vertex: int, direction: str) -> bool:
        """
        A helper to check if a given k-vertex has neighbours in either direction based on the
        specified `direction` argument (`right` for right neighbour, `left` for left neighbour).
        Returns True if found, otherwise False.

        :param vertex: a k-vertex to check neighbours for
        :param direction: `left` or `right` to check for left or right k-vertex neighbours
        :return: True if the requested neighbour is found, otherwise False
        """

        # set the delta to shift the current vertex
        delta = -1 if direction == "left" else 1

        # check if the requested vertex in the requested direction exists as a k-vertex
        if vertex + delta in self.vertices_k:
            return True
        return False

    def get_kminus_kplus(
        self,
        vertex: int,
        dual_rep: bool = False,
    ):
        """
        A helper function to return the two neighbouring k-edges for a given k-vertex. This returns
        a list of two lists, whereby the first list contains the left k-edge and the second contains
        the right k-edge. If either is not found, the respective list is empty.

        :param vertex: a k-vertex to get the attached k-edges
        :param dual_rep: return the k-edges in the dual representation
        :return:
        """

        # the returned k-edges list
        _ret = [[], []]

        # get the k-edges
        _, k_edges = self.get_edges_at_k_vertex(vertex, dual_rep)

        # check if the first edge is a left neighbour
        is_left_neighbour = k_edges[0][1] == vertex

        # append based on left or right
        if is_left_neighbour:
            # is left, append to the left sublist
            _ret[0].append(k_edges[0])

            # append the other edge if found
            if len(k_edges) > 1:
                _ret[1].append(k_edges[1])

        else:
            # is right, append to the right sublist
            _ret[1].append(k_edges[0])

            # append the other edge if found
            if len(k_edges) > 1:
                _ret[0].append(k_edges[1])

        return _ret

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"n_edges={self.n_edges}, "
            f"n_vertices={self.n_vertices}, "
            f"is_planar={self.is_planar}, "
            f"n_edges_k={len(self.edges_k)}, "
            f"n_edges_mu={len(self.edges_mu)}, "
            f"n_vertices_k={len(self.vertices_k)}, "
            f"n_vertices_mu={len(self.vertices_mu)}, "
            f"is_connected={self.is_connected}"
            f")"
        )
