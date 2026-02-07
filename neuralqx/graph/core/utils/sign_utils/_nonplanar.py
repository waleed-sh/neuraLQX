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

r"""
Non planar sign tables for embedded graphs.

This module constructs the local sign function :math:`\varepsilon` at each vertex of an
embedded graph, using the orientations of edge tangents in :math:`\mathbb{R}^3`. For any
ordered triple of incident edges :math:`(e_I, e_J, e_K)` at a vertex :math:`v`, define

.. math::

   \varepsilon_v(e_I, e_J, e_K) = \operatorname{sgn}\!\bigl(\det[\hat t_I\ \hat t_J\ \hat t_K]\bigr)

where :math:`\hat t_I` is the unit tangent of edge :math:`e_I` at :math:`v`, and the
tangents are placed as columns of a :math:`3 \times 3` matrix. If the determinant is
numerically close to zero, the sign is set to zero according to a configurable tolerance.

To ensure a generic configuration when multiple edges share the same local tangent
direction at a vertex, a small deterministic perturbation is applied. For a group of
parallel edges, each edge tangent is rotated by a small angle around a private axis
orthogonal to the base tangent, producing distinct directions on a small circle on
the unit sphere.
"""

import math
from dataclasses import dataclass

from itertools import combinations
from itertools import permutations

from typing import List
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

# aliases
Vec = Tuple[float, float, float]
EdgeKey = Tuple[Union[int, float], Union[int, float], Union[int, float]]

# a small number for collinearity/zero‑norm tests
_EPS = 1e-12

# roughly 0.57 degrees, big enough to break coplanarity
# TODO: this should be larger?
_SMALL_ANGLE = 1.0e-2


# edge geometry dataclass
@dataclass
class EdgeGeom:
    r"""
    Geometry container for an embedded edge in :math:`\mathbb{R}^3`.

    An edge is represented by a start point and an end point, with an optional polyline
    that provides intermediate points along the embedding. If a polyline is present, it is
    used to infer the first direction away from a chosen endpoint.

    :param start: Cartesian coordinates of the vertex the edge emanates from.
    :param end: Cartesian coordinates of the vertex the edge is incident at.
    :param polyline: Optional sequence of intermediate points along the embedded edge.
    """

    start: Vec
    """The Cartesian coordinates of the vertex the edge emanates from"""

    end: Vec
    """The Cartesian coordinates of the vertex the edge is incident at"""

    polyline: List[Vec] | None = None

    def first_point_after(self, vertex: Vec) -> np.ndarray:
        """
        Return the first point along the embedded edge that differs from the given vertex.

        For a straight edge with no polyline, the returned point is typically the opposite
        endpoint. For a polyline, the first point in the polyline that is not equal to the
        given vertex is returned.

        :param vertex: Vertex coordinates from which to step along the edge.
        :return: The next point along the edge away from `vertex` as a NumPy array.
        :raises NotImplementedError: If the edge degenerates to a point at the given vertex.
        """

        # create the edge
        seq = self.polyline if self.polyline else [self.start, self.end]

        # cast to arrays
        v = np.asarray(vertex, dtype=float)

        # get the next point
        for p in seq:
            if not np.allclose(p, v, atol=_EPS):
                return np.asarray(p, dtype=float)

        # raise an error for self loops
        raise NotImplementedError("Edge degenerates to a point at the given vertex")


def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    r"""
    Rotate a vector about an axis by a given angle using Rodrigues' rotation formula.

    Given a unit axis :math:`\hat a` and a vector :math:`v`, the rotated vector is

    .. math::

       v' = v\cos\theta + (\hat a \times v)\sin\theta + \hat a(\hat a \cdot v)(1-\cos\theta)

    The caller is expected to provide a unit axis.

    :param v: Vector to rotate.
    :param axis: Rotation axis, expected to be a unit vector.
    :param angle: Rotation angle in radians.
    :return: Rotated vector as a NumPy array.
    """

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    return v * cos_a + np.cross(axis, v) * sin_a + axis * (axis.dot(v)) * (1.0 - cos_a)


# NonplanarSigns
class NonplanarSigns:
    r"""
    Compute :math:`\varepsilon` sign tables at all vertices of an embedded multi graph.

    For each vertex :math:`v`, each incident edge is assigned a unit tangent direction
    :math:`\hat t_e(v)` in :math:`\mathbb{R}^3`. For any ordered triple of distinct incident
    edges :math:`(e_I, e_J, e_K)`, the sign is defined by

    .. math::

       \varepsilon_v(e_I, e_J, e_K) =
       \begin{cases}
         0, & |\det[\hat t_I\ \hat t_J\ \hat t_K]| \le \text{tol} \\
         +1, & \det[\hat t_I\ \hat t_J\ \hat t_K] > \text{tol} \\
         -1, & \det[\hat t_I\ \hat t_J\ \hat t_K] < -\text{tol}
       \end{cases}

    Parallel or duplicated edges can yield identical tangents and therefore non generic
    tables. To avoid this, edges that share the same endpoint pair are deterministically
    perturbed at the vertex by rotating the tangent around a randomly chosen axis that is
    orthogonal to the base tangent. The randomness is deterministic, derived from the edge
    key and a global seed, so repeated runs are reproducible.

    :param handler: Graph handler providing node coordinates and edge keys for the embedding.
    :param tol: Determinant magnitude tolerance below which the sign is set to zero.
    :param seed: Optional seed controlling deterministic perturbations, may be an int, a
      string, or None.
    """

    def __init__(
        self,
        handler: "GraphHandler",
        *,
        tol: float = 1e-14,
        seed: int | str | None = None,
    ) -> None:
        """
        Initialise geometry, adjacency, and duplicate edge groups for sign computation.

        This constructs edge geometry objects from the handler coordinate map, builds a per
        vertex adjacency list of incident edges, and groups multi edges by their unordered
        endpoint pair so that deterministic perturbations can be applied when needed.

        :param handler: Graph handler providing the embedding coordinates and the edge list.
        :param tol: Determinant magnitude tolerance below which the sign is set to zero.
        :param seed: Optional seed controlling deterministic perturbations, may be an int, a
          string, or None.
        :return: None.
        """

        # save the coordinates locally
        self.coords = handler._nonplanar_int2orig_map

        # generate the edge geometries
        self.edge_geom: Dict[EdgeKey, EdgeGeom] = {
            (u, v, k): EdgeGeom(start=self.coords[u], end=self.coords[v])
            for u, v, k in handler.edges
        }

        # save the tolerance
        self.tol = tol

        # save the global seed
        self._global_seed = 0 if seed is None else hash(seed) & 0xFFFF_FFFF

        # build adjacency list and duplicate groups once, this runs in O(|E|)
        self._adjacency: Dict[int, List[EdgeKey]] = {v: [] for v in self.coords}

        # a temp dict to store duplicates
        dup: Dict[Tuple[int, int], List[EdgeKey]] = {}

        # create the adjacency
        for u, v, k in self.edge_geom:
            ek = (u, v, k)
            self._adjacency[u].append(ek)
            self._adjacency[v].append(ek)
            pair = (u, v) if u <= v else (v, u)
            dup.setdefault(pair, []).append(ek)

        # this is a list of EdgeKey entries which are the unordered endpoint pair
        self._duplicates = dup

    def compute_all(self) -> Dict[int, Dict[Tuple[EdgeKey, EdgeKey, EdgeKey], int]]:
        """
        Compute :math:`\varepsilon` tables for every vertex in the graph.

        :return: Mapping from vertex index to an :math:`\varepsilon` table. Each table maps an
          ordered triple of incident edge keys to an integer in {-1, 0, 1}.
        """

        return {v: self.vertex_signs(v) for v in self._adjacency}

    def vertex_signs(self, vertex: int) -> Dict[Tuple[EdgeKey, EdgeKey, EdgeKey], int]:
        r"""
        Compute the :math:`\varepsilon` table for a single vertex.

        For each incident edge, a unit tangent :math:`\hat t_e(v)` is computed. For every
        unordered triple of edges, all orderings are considered and assigned a sign via

        .. math::

           \varepsilon_v(e_I, e_J, e_K) = \operatorname{sgn}\!\bigl(\det[\hat t_I\ \hat t_J\ \hat t_K]\bigr)

        with tolerance based zeroing.

        :param vertex: Vertex index at which to compute the sign table.
        :return: Mapping from ordered edge triples to signs in {-1, 0, 1}.
        """

        # get the edges from the adjacency list
        inc = self._adjacency[vertex]

        # get the Cartesian coordinates of the requested vertex
        v_xyz = self.coords[vertex]

        # get the tangents at that vertex for every edge
        tangents = {e: self._tangent(e, v_xyz) for e in inc}

        # the eventual ε-table to be populated below
        table: Dict[Tuple[EdgeKey, EdgeKey, EdgeKey], int] = {}

        # loop through triplet of edges
        for eI, eJ, eK in combinations(inc, 3):

            # loop through all possible permutations for the current edge triplet
            for perm in permutations((eI, eJ, eK), 3):

                # compute the sgn(det(M)) where M is the 3x3 matrix formed by stacking the
                # tangents of the three edges as columns
                det = np.linalg.det(
                    np.column_stack(
                        (tangents[perm[0]], tangents[perm[1]], tangents[perm[2]])
                    )
                )

                # check if the determinant is less than the allowed tolerance, in that case, the
                # ε is zero
                if abs(det) <= self.tol:
                    table[perm] = 0
                else:
                    # if not, append the correct sign
                    table[perm] = 1 if det > 0 else -1

        # return the ε-table
        return table

    def _tangent(self, edge: EdgeKey, vertex_xyz: Vec) -> np.ndarray:
        r"""
        Compute the unit tangent of an edge at a given vertex, with deterministic perturbation.

        The base tangent is defined by the vector from the vertex position to the first point
        along the edge away from that vertex, normalised to unit length. If the edge belongs
        to a duplicate group that shares the same unordered endpoint pair, the base tangent is
        rotated by a small angle to break degeneracies.

        For a duplicate group of size :math:`m`, the perturbation angles are chosen symmetrically
        about zero so that the set of tangents remains balanced. The rotation is performed around
        an axis orthogonal to the base tangent.

        :param edge: Edge key identifying the embedded edge.
        :param vertex_xyz: Cartesian coordinates of the vertex at which the tangent is evaluated.
        :return: Unit tangent vector at the vertex as a NumPy array.
        :raises ValueError: If the tangent vector has near zero length.
        """

        # get the geometry for the edge
        geom = self.edge_geom[edge]

        # get the first point after the current vertex in the edge and subtract it from the
        # current vertex
        base = geom.first_point_after(vertex_xyz) - np.asarray(vertex_xyz)

        # compute the norm
        n = np.linalg.norm(base)
        if n < _EPS:
            raise ValueError(f"Zero‑length tangent for edge {edge}")

        # normalise, this is not a unit vector
        base /= n

        # now we auto‑perturb if this edge is in a multi‑edge group

        # reorient
        pair = (edge[0], edge[1]) if edge[0] <= edge[1] else (edge[1], edge[0])

        # get all duplicates
        siblings = self._duplicates[pair]

        # if duplicates exist, go ahead and perturb
        if len(siblings) > 1:

            # deterministic local RNG using edge key and global seed
            local_seed = (hash(edge) ^ self._global_seed) & 0xFFFF_FFFF
            rng = np.random.default_rng(local_seed)

            # create/choose a random perpendicular axis
            # we also reject if accidentally collinear
            # 10 tries should be enough, but we can increase this
            for _ in range(10):

                # draw a standard normal
                rand_vec = rng.standard_normal(3)

                # project perpendicular to the base
                axis = rand_vec - rand_vec.dot(base) * base

                # normalise (if possible)
                norm_axis = np.linalg.norm(axis)
                if norm_axis > _EPS:
                    axis /= norm_axis
                    break
            else:
                # extremely unlikely case that we keep having collinear tangents:
                # fall back to arbitrary perpendicular axis
                axis = np.cross(base, [0.0, 1.0, 0.0])

                # normalise
                axis /= np.linalg.norm(axis)

            # now spread siblings symmetrically using a small deterministic angle
            idx = sorted(siblings).index(edge)
            m = len(siblings)
            angle = _SMALL_ANGLE * (idx - (m - 1) / 2.0)
            base = _rotate_about_axis(base, axis, angle)
        return base
