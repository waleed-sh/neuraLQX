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
Graph construction helpers for the GraphHandler.

This module builds derived graph objects used by neuraLQX, in particular the dual graph
defined as the line graph. For a graph :math:`G=(V,E)`, the line graph :math:`L(G)` has
vertex set :math:`E`, and two vertices in :math:`L(G)` are connected when the
corresponding edges in :math:`G` share an endpoint.

The helpers also provide relabelling utilities for NetworkX graphs and a labelling
scheme that maps integer vertices to fixed length 3 tuples for non planar embeddings.
"""

from typing import Any
from typing import List
from typing import Tuple

import netket as nk
import networkx as nx


def map_distinct_vertices(connections_list: list) -> tuple[list[tuple], dict]:
    """
    Build a stable index mapping for the distinct vertices of a dual graph edge list.

    The dual graph is represented by edges whose endpoints are original graph edges, for
    example edge keys like ``(u, v, k)``. This function collects all distinct such endpoints,
    sorts them deterministically, and assigns an integer index to each.

    :param connections_list: List of dual graph edges, where each element contains two
      endpoints that represent original graph edges.
    :return: A pair ``(sorted_vertices, distinct_vertices_map)`` where ``sorted_vertices``
      is the sorted list of distinct endpoints and ``distinct_vertices_map`` maps each
      endpoint to its integer index.
    """

    set_of_distinct_vertices = set()

    for edge_pair in connections_list:
        for edge in edge_pair:
            # (u, v, k) if keys exist
            set_of_distinct_vertices.add(edge)

    sorted_vertices = sorted(set_of_distinct_vertices)
    distinct_vertices_map = {vertex: idx for idx, vertex in enumerate(sorted_vertices)}

    return sorted_vertices, distinct_vertices_map


def map_edges(
    connections_list: list,
    mapping: dict,
) -> list[list]:
    """
    Map a list of dual graph edges onto integer vertex indices.

    Given a list of edges expressed in terms of endpoint keys and a mapping from those keys
    to integers, this function returns the same edge list but with endpoints replaced by
    their mapped indices.

    :param connections_list: List of edges as pairs of endpoint keys.
    :param mapping: Mapping from endpoint key to integer index.
    :return: List of mapped edges as pairs of integers.
    """

    # return the set of mapped edges based on the mapping provided
    return [[mapping[key1], mapping[key2]] for key1, key2 in connections_list]


def create_dual_nk_graph(
    handler: "GraphHandler",
    g: int = 0,
    *,
    is_nonplanar: bool = False,
) -> tuple[nk.graph.Graph, dict]:
    r"""
    Create the dual NetKet graph and an associated mapping.

    The dual graph is constructed as the line graph :math:`L(G)` of a NetworkX graph :math:`G`.
    Vertices of :math:`L(G)` correspond to edges of :math:`G`, and adjacency corresponds to
    sharing an endpoint in :math:`G`.

    By default, the input graph is taken from the handler. If `is_nonplanar` is True, the
    returned mapping is inverted so it maps dual vertex indices back to the original edge
    keys, which is convenient for non planar workflows.

    :param handler: GraphHandler instance providing the base NetworkX graph.
    :param g: Optional NetworkX graph to dualise. If set to 0, the handler graph is used.
    :param is_nonplanar: If True, invert the mapping to return index to key.
    :return: A pair ``(dual_nk_graph, mapping)`` where ``dual_nk_graph`` is a
      :class:`netket.graph.Graph` for the line graph and ``mapping`` converts between the
      original edge key representation and the dual vertex indices, possibly inverted.
    """

    # check if we are requesting to create the dual of a specific graph
    if g == 0:
        # if not, then the graph at hand will be the same graph instructed at initialization
        g = handler.nx_graph

    # get the NetworkX dual graph
    dual_nx_graph = nx.line_graph(g)

    # get the distinct vertices and the mapping for the dual NetworkX graph
    distinct_vertices, mapping = map_distinct_vertices(
        list(dual_nx_graph.edges())
    )  # pylint: disable=W0612

    # from those, we can now create dual Netket edges
    dual_nk_edges = map_edges(list(dual_nx_graph.edges()), mapping)

    if is_nonplanar:
        mapping = {value: key for key, value in mapping.items()}

    # and now we create a dual Netket graph
    return nk.graph.Graph(edges=dual_nk_edges), mapping


def create_dressed_nx_graph(handler) -> None:
    """
    Add orientation metadata to the handler NetworkX multigraph edges.

    Each edge is dressed with two attributes

    - ``s`` the start vertex of the directed edge
    - ``e`` the end vertex of the directed edge

    These attributes are used later to classify edges along oriented cycles.

    :param handler: GraphHandler instance providing ``nx_graph`` and the edge list.
    :return: None.
    """

    # loop over every edge in the list of edges
    for edge in handler.edges:
        # add the attributes
        # here we also add the keys from the keyed edge accordingly
        handler.nx_graph.add_edge(
            edge[0], edge[1], edge[2], **{"s": edge[0], "e": edge[1]}
        )


def create_dual_nx_edges(handler) -> None:
    """
    Populate a handler side list of dual NetworkX edges expressed in dual vertex indices.

    This walks the edges of ``handler.dual_nx_graph`` and maps each endpoint, which is an
    original graph edge key, to an integer dual vertex label using the handler mapping.

    :param handler: GraphHandler instance providing ``dual_nx_graph`` and a mapping method.
    :return: None.
    """

    handler.dual_nx_edges = []
    for edge in handler.dual_nx_graph.edges():
        handler.dual_nx_edges.append(
            [
                handler.map_edge_to_dual_vertex(edge[0]),
                handler.map_edge_to_dual_vertex(edge[1]),
            ]
        )


def find_connected_edges(
    handler: "GraphHandler",
    graph_vertex: int,
) -> dict[str, list[tuple[int, int, int]]]:
    """
    Collect incoming and outgoing edges incident to a given vertex.

    Edges are considered directed according to the dressed attributes ``s`` and ``e`` stored
    on the handler NetworkX multigraph. An edge is outgoing if its ``s`` equals the queried
    vertex, and incoming if its ``e`` equals the queried vertex.

    :param handler: GraphHandler instance providing the dressed NetworkX multigraph.
    :param graph_vertex: Vertex index for which to collect incident edges.
    :return: Dictionary with keys ``"outgoing"`` and ``"incoming"`` each containing a list
      of edges represented as ``(u, v, key)`` tuples.
    """

    # initialise empty lists to save the outgoing and incoming edges
    outgoing = []
    incoming = []

    # get the list of all the edges in the graph with their attributes
    edge_list = list(handler.nx_graph.edges(data=True, keys=True))

    # loop over every edge in the graph
    for edge in edge_list:
        # get the list of attributes
        attributes = edge[3]

        # get the key
        key = edge[2]

        # check if the edge's attribute 's' coincides with the current graph_vertex
        if attributes.get("s") == graph_vertex:
            # if so, then this edge starts at the graph_vertex, so we append it to the
            # list of outgoing edges
            outgoing.append((edge[0], edge[1], key))

        # check if the edge's attribute 'e' coincides with the current graph_vertex
        if attributes.get("e") == graph_vertex:
            # if so, then this edge ended at the graph_vertex, so we append it to the
            # list of incoming edges
            incoming.append((edge[0], edge[1], key))

    # return the outgoing and incoming edge lists as a dictionary
    return {"outgoing": outgoing, "incoming": incoming}


def init_node_connectivity(handler) -> None:
    """
    Initialise per vertex connectivity lists for the handler.

    For each vertex in the handler NetworkX graph, this computes the incoming and outgoing
    incident edges and stores them under ``handler.list_of_node_connectivity``.

    :param handler: GraphHandler instance providing ``nx_graph`` and storage fields.
    :return: None.
    """

    # we initialize the list of connectivites
    handler.list_of_node_connectivity = {}

    # for every node in the graph, get the connected edges
    for node in handler.nx_graph.nodes():
        handler.list_of_node_connectivity[node] = find_connected_edges(handler, node)


def dress_minimal_cycles_edges(handler: "GraphHandler"):
    """
    Dress edges of minimal cycles with creation and annihilation types.

    For each minimal cycle, each directed edge is compared against the dressed NetworkX
    edge attributes to decide whether it matches the stored start direction. If it does,
    the edge is labelled as ``"creation"``, otherwise it is labelled as ``"annihilation"``.
    If the edge cannot be matched, it is labelled as ``"unknown"``.

    The dressed cycles are appended to ``handler.minimal_cycles``.

    :param handler: GraphHandler instance providing minimal cycles and the dressed NetworkX graph.
    :return: None.
    """

    if len(handler.minimal_cycles) > 1:
        pass

    for minimal_cycle in handler.get_smallest_loops():

        cycle_edges = []

        for edge in minimal_cycle:

            key = edge[-1]
            reversed_edge = (*edge[:-1][::-1], key)

            if edge in handler.nx_graph.edges(
                keys=True
            ) or reversed_edge in handler.nx_graph.edges(keys=True):
                edge_data = handler.nx_graph.get_edge_data(
                    edge[0], edge[1]
                ) or handler.nx_graph.get_edge_data(reversed_edge[0], reversed_edge[1])
                edge_data = edge_data[key]

                edge_type = "creation" if edge_data["s"] == edge[0] else "annihilation"

                cycle_edges.append((edge, {"type": edge_type, "key": edge[-1]}))
            else:
                cycle_edges.append((edge, {"type": "unknown", "key": edge[-1]}))

        handler.minimal_cycles.append(cycle_edges)


def _dress_minimal_cycles_edges(handler):
    """
    Dress minimal cycles using a key based orientation heuristic.

    This variant first builds a lookup of directed edges per key. For each cycle, it chooses
    an orientation that maximises the number of edges that match the stored directed edges
    for that key, and then labels each edge as ``"creation"`` when it matches the stored
    direction and ``"annihilation"`` when only the reverse direction exists.

    The dressed cycles are appended to ``handler.minimal_cycles``.

    :param handler: GraphHandler instance providing edges and minimal loops.
    :return: None.
    """

    if len(handler.minimal_cycles) > 1:
        pass

    # build a quick lookup: for each key, which directed (u,v) exist?
    edges_by_key = {}
    for u, v, k in handler.edges:
        edges_by_key.setdefault(k, set()).add((u, v))

    dressed = []
    for key, cycle in handler.get_smallest_loops():
        # pick an orientation (cycle vs reversed) that maximizes "creation" edges
        def creation_count(seq):
            cnt = 0
            n = len(seq)
            for i in range(n):
                u, v = seq[i], seq[(i + 1) % n]
                if (u, v) in edges_by_key[key]:
                    cnt += 1
            return cnt

        seq1 = cycle
        seq2 = list(reversed(cycle))
        oriented = seq1 if creation_count(seq1) >= creation_count(seq2) else seq2

        # now actually dress the edges in that orientation
        n = len(oriented)
        cycle_edges = []
        for i in range(n):
            u, v = oriented[i], oriented[(i + 1) % n]
            if (u, v) in edges_by_key[key]:
                etype = "creation"
            elif (v, u) in edges_by_key[key]:
                etype = "annihilation"
            else:
                # should not happen if our loops are valid
                etype = "unknown"
            cycle_edges.append((u, v, {"type": etype, "key": key}))

        handler.minimal_cycles.append(cycle_edges)


def index_to_label(i: int) -> tuple[int, int, int]:
    r"""
    Map an integer vertex index to a fixed length 3 tuple label.

    The mapping is defined by splitting the integer into a group index and a two bit suffix.
    Let :math:`i` be the input, then

    .. math::

       g = \lfloor i / 4 \rfloor,
       \qquad
       r = i \bmod 4,
       \qquad
       r = 2 b_1 + b_2

    The output label is :math:`(g, b_1, b_2)`.

    :param i: Integer vertex label.
    :return: Tuple ``(group, y, z)`` where ``group`` is the quotient and ``(y, z)`` are the
      two bits of the remainder in binary.
    """

    # determine the group: increments by one every 4 vertices
    group = i // 4

    # calculate the remainder, which will be converted to binary
    remainder = i % 4

    # convert the remainder to a 2-digit binary string
    binary_str = format(remainder, "02b")

    # return the 3-tuple label
    return group, int(binary_str[0]), int(binary_str[1])


def relabel_edges_to_nonplanar(
    edges: list[tuple[int, int, int]],
) -> tuple[
    list[tuple[tuple[int, int, int], tuple[int, int, int], int]],
    dict[int, tuple[int, int, int]],
]:
    """
    Relabel MultiGraph style edges to fixed length 3 tuple vertex labels.

    This maps the set of unique endpoints to a contiguous index range and then applies
    `index_to_label` to obtain 3 tuple labels. Edge keys are preserved.

    :param edges: List of edges in the form ``(u, v, key)``.
    :return: A pair ``(new_edges, vertex_mapping)`` where ``new_edges`` is a list of
      ``((x1, y1, z1), (x2, y2, z2), key)`` edges and ``vertex_mapping`` maps each original
      vertex to its new 3 tuple label.
    """

    # collect all the endpoints (ignore keys for the mapping step)
    unique_verts = sorted({u for u, v, k in edges} | {v for u, v, k in edges})

    # give each original vertex a 0...n−1 index, then map that to a 3‐tuple
    seq = {v: i for i, v in enumerate(unique_verts)}
    vertex_mapping = {v: index_to_label(seq[v]) for v in unique_verts}

    # rebuild each edge, replacing u -> tuple, v -> tuple, but preserve key
    new_edges = [(vertex_mapping[u], vertex_mapping[v], k) for u, v, k in edges]

    return new_edges, vertex_mapping


def relabel_nx_edges(g: nx.Graph) -> list:
    """
    Relabel NetworkX edges to a contiguous integer labelling scheme.

    Vertices are mapped to integers in the order provided by ``g.nodes``. The returned edge
    list contains the relabelled endpoints.

    :param g: NetworkX graph whose edges are to be relabelled.
    :return: List of edges with integer endpoints.
    """

    # create a mapping for its vertices
    v_mapping = {v: i for i, v in enumerate(list(g.nodes))}

    # relabel all edges
    edges = [[v_mapping.get(v) for v in edge] for edge in list(g.edges)]

    return edges


def canonical_rotate(loop: List[Tuple[Any, Any, int]]) -> List[Tuple[Any, Any, int]]:
    r"""
    Choose a canonical cyclic rotation of an oriented loop.

    A loop is represented as a list of edges ``(tail, head, key)``. A rotation is considered
    valid if it forms a closed oriented chain, meaning

    .. math::

       \mathrm{head}(e_j) = \mathrm{tail}(e_{j+1})

    with indices taken modulo the loop length. Among all valid cyclic rotations, this
    function returns the one whose first tail vertex is minimal under the default ordering.

    :param loop: List of directed edges representing a loop.
    :return: Canonically rotated loop as a list of directed edges. If no valid rotation
      exists, the input loop is returned unchanged.
    """

    n = len(loop)
    valid = []
    for i in range(n):
        cand = loop[i:] + loop[:i]
        if all(cand[j][1] == cand[(j + 1) % n][0] for j in range(n)):
            valid.append(cand)
    if not valid:
        return loop
    # pick rotation with smallest starting tail
    return min(valid, key=lambda c: c[0][0])
