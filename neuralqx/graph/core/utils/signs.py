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
Edge sign utilities used by GraphHandler.

This module computes orientation dependent sign data associated with ordered edge pairs
in planar embeddings and ordered edge triples in non planar embeddings.

Planar case
For a vertex :math:`v` and two outgoing directed edges :math:`e_1=(v\to v_1)` and
:math:`e_2=(v\to v_2)`, let :math:`t_1,t_2\in\mathbb R^2` be the corresponding tangent
vectors in the embedding. The sign is determined by the relative orientation of the two
directions, which can be understood via the oriented area

.. math::

   \operatorname{sgn}\bigl(t_1 \times t_2\bigr)
   =
   \operatorname{sgn}\bigl(t_{1x} t_{2y} - t_{1y} t_{2x}\bigr)

In practice this module uses angular ordering derived from :func:`numpy.arctan2` and a
deterministic convention for clockwise versus anticlockwise comparison, combined with a
handler supplied factor that accounts for edge direction conventions.

Non planar case
For a vertex :math:`v` and three incident directed edges, a sign is computed from the
determinant of the matrix whose columns are the three tangent vectors
:math:`c_1,c_2,c_3\in\mathbb R^3`

.. math::

   \varepsilon(e_1,e_2,e_3)
   =
   \operatorname{sgn}\bigl(\det[c_1\ c_2\ c_3]\bigr)

This yields values in :math:`\{-1,0,+1\}` where zero indicates coplanarity or numerical
degeneracy. For non planar multigraphs, the higher level :class:`NonplanarSigns` helper is
used to ensure a generic configuration by applying small deterministic perturbations to
coincident tangents before evaluating determinants.
"""

import numpy as np
import itertools

from .sign_utils import NonplanarSigns


def _compute_angles(
    first_edge: tuple[int, int, int],
    second_edge: tuple[int, int, int],
    node_pos: dict,
) -> float:
    r"""
    Compute the geometric angle between two directed keyed edges in a planar embedding.

    Each edge is provided as a triple ``(u, v, key)`` but the key is ignored for the angle
    calculation. The embedding is given by `node_pos` mapping vertex labels to 2D coordinates.

    Let

    .. math::

       t_1 = p(v_1) - p(u_1),
       \qquad
       t_2 = p(v_2) - p(u_2)

    where :math:`p(\cdot)` returns the 2D position. The returned angle is

    .. math::

       \theta = \arccos\!\Bigl(\frac{t_1\cdot t_2}{\lVert t_1\rVert\lVert t_2\rVert}\Bigr)

    expressed in degrees.

    :param first_edge: First keyed edge as ``(u, v, key)``.
    :param second_edge: Second keyed edge as ``(u, v, key)``.
    :param node_pos: Mapping from vertex label to a 2D coordinate, compatible with NumPy array conversion.
    :return: Angle in degrees in the interval :math:`[0, 180]`.
    :raises ValueError: If either edge has zero length in the provided embedding.
    """

    # vectors from origin to endpoints
    v1 = np.array(node_pos[first_edge[1]]) - np.array(node_pos[first_edge[0]])
    v2 = np.array(node_pos[second_edge[1]]) - np.array(node_pos[second_edge[0]])

    # cosine rule
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return np.degrees(np.arccos(cos_theta))


def compute_sgn_det(
    handler: "GraphHandler", edge_set: list[tuple[int, int, int]]
) -> float:
    r"""
    Compute the sign of the determinant formed by three incident edge tangents in 3D.

    Given a list of three keyed edges ``(u, v, key)``, this constructs three 3D direction
    vectors :math:`c_1,c_2,c_3` from the handler stored non planar coordinates and computes

    .. math::

       s = \operatorname{sgn}\bigl(\det[c_1\ c_2\ c_3]\bigr)

    where the columns are the three vectors. The sign takes values in :math:`\{-1,0,+1\}`.

    The orientation of each tangent is chosen consistently with the handler's original edge
    orientation. If a reversed directed edge exists in ``handler._original_edges``, the
    corresponding tangent is flipped so that the local direction at the incident vertex is
    used consistently.

    :param handler: GraphHandler instance providing ``_nonplanar_int2orig_map`` and ``_original_edges``.
    :param edge_set: List of exactly three keyed edges as ``(u, v, key)``.
    :return: Sign of the determinant, as a float that is numerically one of -1.0, 0.0, +1.0.
    :raises KeyError: If a vertex label in `edge_set` is missing from the handler coordinate map.
    :raises ValueError: If `edge_set` does not contain exactly three edges.
    """

    incident = []
    for u, v, _ in edge_set:
        p1 = handler._nonplanar_int2orig_map[u]
        p2 = handler._nonplanar_int2orig_map[v]
        # if any reverse (v,u) exists, flip orientation
        if (v, u, _) in handler._original_edges:
            c = np.array(p1) - np.array(p2)
        else:
            c = np.array(p2) - np.array(p1)
        incident.append(c)

    M = np.column_stack(incident)
    return np.sign(np.linalg.det(M))


def compute_sign_at_vertex(
    handler: "GraphHandler",
    vertex,
) -> dict[str, float]:
    r"""
    Compute determinant signs for all ordered triples of keyed edges incident to a vertex.

    This enumerates all ordered triples of incident keyed edges at `vertex` and stores, for
    each triple, the sign returned by :func:`compute_sgn_det`.

    If :math:`E(v)` is the set of incident keyed edges at vertex :math:`v`, then this
    evaluates

    .. math::

       \varepsilon(e_i,e_j,e_k) = \operatorname{sgn}\bigl(\det[c_i\ c_j\ c_k]\bigr)

    for all ordered triples :math:`(e_i,e_j,e_k)\in E(v)^3` with distinct edges.

    The returned dictionary uses stringified triples as keys, suitable for JSON style logging.

    :param handler: GraphHandler instance providing access to the NetworkX multigraph and coordinate map.
    :param vertex: Vertex identifier as used by the handler NetworkX graph.
    :return: Mapping from stringified ordered edge triples to determinant sign values.
    """

    signs = {}

    # get all keyed edges at the vertex
    evs = list(handler.nx_graph.edges(vertex, keys=True))
    for triple in itertools.permutations(evs, 3):
        signs[str(triple)] = compute_sgn_det(handler, list(triple))
    return signs


def _rotation_matrix(theta: float) -> np.ndarray:
    r"""
    Construct the 2D rotation matrix for a given angle.

    The returned matrix represents a counter clockwise rotation by `theta` in the plane

    .. math::

       R(\theta) =
       \begin{pmatrix}
       \cos\theta & -\sin\theta \\
       \sin\theta & \cos\theta
       \end{pmatrix}

    :param theta: Rotation angle in radians.
    :return: A 2 by 2 NumPy array implementing the rotation.
    """

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _find_sign(
    edge1: tuple[float, float], edge2: tuple[float, float], show_plots: bool = False
) -> int:
    r"""
    Determine an ordering sign for two planar direction vectors using angular comparison.

    Given two 2D vectors `edge1` and `edge2`, this computes their polar angles using
    :func:`numpy.arctan2` and compares the angles after applying a fixed rotation direction
    convention. The output is a sign in :math:`\{-1,+1\}` indicating which direction comes
    first in the chosen angular ordering.

    This sign is a purely planar notion of relative orientation. In higher level use, it is
    typically multiplied by a handler supplied factor that accounts for edge direction
    conventions at the chosen vertex.

    :param edge1: First 2D vector as a pair ``(x, y)``.
    :param edge2: Second 2D vector as a pair ``(x, y)``.
    :param show_plots: If True, the function is expected to produce a diagnostic plot. The current
      implementation raises an error because plotting is not implemented here.
    :return: -1 if the first vector precedes the second under the chosen angular convention, otherwise +1.
    :raises NotImplementedError: If `show_plots` is True.
    """

    dir_rot = -1
    t1 = np.arctan2(edge1[1], edge1[0]) * dir_rot
    t2 = np.arctan2(edge2[1], edge2[0]) * dir_rot

    a1 = np.degrees(t1) % 360
    a2 = np.degrees(t2) % 360

    if show_plots:
        # dev: plotting omitted
        raise NotImplementedError

        # import matplotlib.pyplot as plt

        # print(-1 if a1 < a2 else 1)
        # fig, ax = plt.subplots()
        # plt.show()

    return -1 if a1 < a2 else 1


def compute_non_planar_graph_signs(
    handler: "GraphHandler",
    graph_type: str = "graph",
) -> None:
    r"""
    Compute and store non planar sign tables for every vertex of the selected graph.

    This routine delegates to :class:`NonplanarSigns`, which computes per vertex
    :math:`\varepsilon` tables based on determinant signs of triplets of tangents on
    :math:`\mathbb R^3`. For a vertex :math:`v` and three incident edges, the sign is

    .. math::

       \varepsilon(e_1,e_2,e_3) = \operatorname{sgn}\bigl(\det[c_1\ c_2\ c_3]\bigr)

    where each :math:`c_i` is a unit tangent at :math:`v`. For multiedges that would produce
    coincident tangents, :class:`NonplanarSigns` applies a small deterministic perturbation to
    produce a generic configuration before taking determinants.

    Depending on `graph_type`, the results are stored into either ``handler.graph_signs`` or
    ``handler.dual_graph_signs`` using stringified vertex identifiers as top level keys.

    :param handler: GraphHandler instance providing graphs and storage dictionaries.
    :param graph_type: Either ``"graph"`` for the primal graph or any other value for the dual graph.
    :return: None.
    """

    if graph_type == "graph":
        G = handler.nx_graph
        store = handler.graph_signs
    else:
        G = handler.dual_nx_graph
        store = handler.dual_graph_signs

    nps = NonplanarSigns(handler)

    for v in G.nodes:
        store[str(v)] = nps.vertex_signs(vertex=v)


def compute_signs(
    handler: "GraphHandler", graph_type: str = "graph", show_plots: bool = False
) -> None:
    r"""
    Compute and store edge sign data for planar or non planar graphs, primal or dual.

    This is the main entry point used by GraphHandler. It chooses between planar and non
    planar logic based on the handler flag.

    Non planar graphs
    For non planar graphs, this calls :func:`compute_non_planar_graph_signs` and returns.

    Planar graphs
    For planar graphs, this computes signs for ordered pairs of edges incident at each
    vertex using :func:`_find_sign`. For a chosen vertex :math:`u` and two edges
    :math:`(u\to v_1)` and :math:`(u\to v_2)`, it constructs direction vectors

    .. math::

       t_1 = p(v_1) - p(u),
       \qquad
       t_2 = p(v_2) - p(u)

    and assigns a sign based on angular ordering of :math:`t_1` and :math:`t_2`. The raw sign
    is then multiplied by ``handler._get_edge_direction_factor`` to enforce a consistent
    convention under edge orientation changes.

    Results are stored into the appropriate handler dictionaries

    - ``handler.graph_signs`` and ``handler.graph_positions`` for `graph_type` equal to ``"graph"``
    - ``handler.dual_graph_signs`` and ``handler.dual_graph_positions`` otherwise

    The stored positions capture the origin shifted vectors used to compute the sign, which is
    useful for debugging and optional plotting.

    :param handler: GraphHandler instance providing layout positions, connectivity information, and storage.
    :param graph_type: Either ``"graph"`` for the primal graph or any other value for the dual graph.
    :param show_plots: If True, enable optional plotting in :func:`_find_sign`. Currently this raises an
      error because plotting is not implemented in the low level helper.
    :return: None.
    """

    # non-planar case
    if not handler.is_planar:
        compute_non_planar_graph_signs(handler, graph_type)
        return

    # planar case: use pairwise _find_sign
    if graph_type == "graph":
        pos = handler.positions
        out_signs = handler.graph_signs
        out_pos = handler.graph_positions
    else:
        pos = handler.dual_positions
        out_signs = handler.dual_graph_signs
        out_pos = handler.dual_graph_positions

    # for each vertex (string key) in connectivity
    for v_key, info in handler.graph_edges_data[graph_type]["connectivities"].items():
        out_signs[v_key] = {}
        out_pos[v_key] = {}
        for first_edge, second_edge in info["edges"]:
            # both edges are (u, v, key)
            u = first_edge[0]
            # shift origin to u
            origin = pos[u]
            f1 = pos[first_edge[1]] - origin
            f2 = pos[second_edge[1]] - origin

            s = _find_sign(f1, f2, show_plots)
            s *= handler._get_edge_direction_factor((first_edge, second_edge), u)

            pair_key = str((first_edge, second_edge))
            out_signs[v_key][pair_key] = str(s)
            out_pos[v_key][pair_key] = (
                (pos[first_edge[0]] - origin, f1),
                (pos[second_edge[0]] - origin, f2),
            )
