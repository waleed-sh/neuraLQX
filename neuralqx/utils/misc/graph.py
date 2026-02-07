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
from typing import List, Tuple, Union, Any

from neuralqx.utils.misc.arithmetic import minus_key


def find_corner_vertex_keyed(
    edge_pair: Union[tuple, list],
) -> tuple[bool, Union[bool, Any]]:
    """
    Find the common (corner) vertex where two keyed edges meet.

    Each edge is assumed to be in keyed form ``(a, b, k)``, where ``a`` and ``b`` are the endpoint
    vertices and ``k`` is an edge key (e.g. to distinguish parallel edges). The key component is
    ignored when determining adjacency.

    The function returns ``(True, v)`` if the two edges share exactly one common vertex ``v``.
    Otherwise it returns ``(True, False)``.

    :param edge_pair: Pair of keyed edges, each a 3-tuple/list ``(a, b, k)``.
    :returns: Tuple ``(True, corner_vertex)`` if exactly one corner exists, else ``(True, False)``.
    """

    # extract only the node parts from each edge, ignore keys
    edge1_nodes = edge_pair[0][:2]
    edge2_nodes = edge_pair[1][:2]

    common_vertices = set(edge1_nodes).intersection(edge2_nodes)

    return True, (list(common_vertices)[0] if len(common_vertices) == 1 else False)


def get_vertices_from_minimal_loop(minimal_loops: Union[list[Any], tuple[Any]]) -> list:
    """
    Extract the unique vertices traversed by one or more minimal loops (keyed-edge format).

    Minimal loops are expected to be sequences of keyed edges ``(a, b, k)``. The key ``k`` is
    ignored and only endpoints ``(a, b)`` are used to collect vertices. For each provided loop, the
    returned vertex list is sorted and duplicates are removed.

    :param minimal_loops: One or more minimal loops in keyed-edge representation.
    :returns: List of vertex-lists, one per input loop, containing the sorted unique vertices.
    """

    # unkey the edges in the minimal loop
    unkeyed_loops = [[l[:-1] for l in loop] for loop in minimal_loops]

    # a list to hold the vertices of each provided minimal loop
    vertices = []

    # loop through the unkeyed loops
    for ul in unkeyed_loops:
        # get and sort the vertices
        sorted_vertices = sorted(set(i for pair in ul for i in pair))

        # append to the vertices list
        vertices.append(sorted_vertices)

    return vertices


def get_common_base_vertex(triangulation_pair: tuple[Any, ...]) -> list:
    """
    Return the common base vertices of a pair of triangulations/minimal loops.

    The input is a pair of minimal loops in keyed-edge representation. The function extracts the
    vertex sets of each loop and returns the intersection (the vertices common to both).

    :param triangulation_pair: Pair of minimal loops (keyed-edge representation).
    :returns: List of vertices common to both loops.
    """

    # here, the triangulation pair are two minimal loops which are in the keyed edge representation
    # so we will just get the vertices these loops traverse
    triangulation_pair = get_vertices_from_minimal_loop(triangulation_pair)

    # return the common base vertices for the triangulation pair
    return list(set(triangulation_pair[0]).intersection(*triangulation_pair))


def get_edges_at_base_vertex(
    minimal_loop: list, base_vertex: int
) -> Union[tuple, None]:
    """
    Return the pair of edges in a minimal loop that meet at a given base vertex.

    The minimal loop is provided in keyed-edge representation. The function iterates over all
    unordered pairs of edges and returns the first pair whose corner vertex (shared endpoint,
    ignoring keys) equals ``base_vertex``.

    :param minimal_loop: Minimal loop (triangulation) as a list of keyed edges ``(a, b, k)``.
    :param base_vertex: Vertex at which to find the incident edge pair.
    :returns: A 2-tuple of keyed edges that meet at ``base_vertex``, or ``None`` if no such pair is
              found.
    """

    # loop through every possible edge pair (do not permute)
    for edge_pair in itertools.combinations(minimal_loop, 2):
        # check if the corner vertex of the edge pair is the same as the base vertex
        if find_corner_vertex_keyed(edge_pair)[1] == base_vertex:
            # if so, return the edge pair
            return edge_pair
        # dev: indent this to outside the loop?
        # return None


def get_reversed_keyed_edge(
    edge: Union[tuple, list],
    as_list: bool = False,
) -> tuple:
    """
    Reverse a keyed edge while keeping the key in the last position.

    For an edge ``(a, b, k)``, the reversed keyed edge is ``(b, a, k)``. If ``as_list`` is True,
    the result is returned as a list instead of a tuple.

    :param edge: Keyed edge ``(a, b, k)``.
    :param as_list: If True, return the reversed edge as a list. Defaults to False.
    :returns: Reversed keyed edge with the key preserved as the last element.
    """

    re = (*edge[:-1][::-1], edge[-1])

    return re if not as_list else list(re)


def get_triplets_key(
    edges: list[list],
    vertex: int,
) -> list[list]:
    """
    Enumerate all ordered edge triplets incident to a given vertex (keyed-edge format).

    The function filters the provided keyed edges to those connected to ``vertex`` (membership
    checked against the endpoint portion of the edge, i.e. ``edge[:-1]``), then returns all
    length-3 permutations of those incident edges.

    :param edges: List of keyed edges of the graph, each of the form ``(a, b, k)``.
    :param vertex: Vertex at which to collect incident edge triplets.
    :returns: List of edge triplets (as lists), containing all permutations of three incident edges.
    """

    # filter edges that are connected to the given vertex
    connected_edges = [edge for edge in edges if vertex in edge[:-1]]

    # generate all combinations of 3 edges from the connected edges
    edge_combinations = list(itertools.permutations(connected_edges, 3))

    # convert each combination from tuple to list
    edge_combinations = [list(comb) for comb in edge_combinations]

    return edge_combinations


def reorder_edge_triplet_key(
    edge_triplet: list[list],
    base_vertex: int,
) -> tuple[list[Union[int, list[int], list]], int]:
    """
    Orient a keyed edge triplet so each edge has the desired direction relative to ``base_vertex``.

    The triplet is interpreted positionally:

    - ``edge_triplet[0]`` is treated as an outgoing edge and is oriented to start at ``base_vertex``.
    - ``edge_triplet[1]`` is treated as an incoming edge and is oriented to end at ``base_vertex``.
    - ``edge_triplet[2]`` is treated as a segment edge and is oriented to start at ``base_vertex``.

    Whenever an edge must be reversed to satisfy the orientation constraint, a factor of ``-1`` is
    accumulated. The returned sign is the product of these factors.

    :param edge_triplet: List of three keyed edges ``(a, b, k)``.
    :param base_vertex: Vertex with respect to which orientations are enforced.
    :returns: Tuple ``(reordered_triplet, sign)`` where ``reordered_triplet`` is the reoriented
              triplet and ``sign`` is ``+1`` or ``-1``.
    """

    _commutative_sign = 1

    # dev: these are now keyed edges
    _reordered_triplet = [edge_triplet[0], edge_triplet[1], edge_triplet[2]]

    # this will be the outgoing edge of the minimal loop later, so it needs to start at base_vertex
    if edge_triplet[0][0] != base_vertex:
        _reordered_triplet[0] = get_reversed_keyed_edge(edge_triplet[0])
        _commutative_sign *= -1

    # this will be the incoming edge of the minimal loop later, so it needs to end at base_vertex
    if edge_triplet[1][1] != base_vertex:
        _reordered_triplet[1] = get_reversed_keyed_edge(edge_triplet[1])
        _commutative_sign *= -1

    # this will be a segment edge later, so it needs to start a base_vertex
    if edge_triplet[2][0] != base_vertex:
        _reordered_triplet[2] = get_reversed_keyed_edge(edge_triplet[2])
        _commutative_sign *= -1

    return _reordered_triplet, _commutative_sign


def find_minimal_loops_with_edges_key(
    minimal_loop_list: list,
    edge_pair: list,
) -> list[list]:
    """
    Find all minimal loops that contain a specified pair of keyed edges (up to orientation).

    The function checks each minimal loop in ``minimal_loop_list`` and includes it if both edges in
    ``edge_pair`` appear in the loop either in the same orientation or reversed (endpoint-flipped,
    key preserved).

    :param minimal_loop_list: List of minimal loops, each a list of keyed edges ``(a, b, k)``.
    :param edge_pair: Pair of keyed edges (as lists/tuples) to search for within each loop.
    :returns: List of minimal loops that contain both edges in the pair (considering both
              orientations).
    :raises ValueError: If the two edges in ``edge_pair`` refer to the same edge (possibly reversed).
    """

    # first convert the edge pair from list of lists to list of tuples for consistency
    edge_pair = [tuple(edge) for edge in edge_pair]

    # check if the pair contains the same edge, they should be two distinct edges
    if edge_pair[0] == edge_pair[1] or edge_pair[0] == get_reversed_keyed_edge(
        edge_pair[1]
    ):
        raise ValueError(
            "The specified edge pair contains the same edge in both entries."
        )

    # a list which will contain all minimal loops which contain the edge pair
    inclusive_minimal_loops = []

    # loop through every minimal loop
    for minimal_loop in minimal_loop_list:

        # check if all edges in the pair (considering both orientations) are present in the current
        # minimal loop
        if all(
            any(
                edge == item or get_reversed_keyed_edge(edge) == item
                for item in minimal_loop
            )
            for edge in edge_pair
        ):
            # if so, add the loop to the list
            inclusive_minimal_loops.append(minimal_loop)

    return inclusive_minimal_loops


def reorder_minimal_loop_key(
    minimal_loop: list[list[int]],
    edge_pair: list[list[int]],
) -> list[list[int]]:
    """
    Reorder and reorient a keyed minimal loop to start/end with a specified edge pair.

    Given a minimal loop (a cyclic sequence of keyed edges) and a pair of keyed edges
    ``[start, end]``, this function constructs a new path that:

    - begins with ``start``,
    - follows the loop in consistent orientation by matching consecutive endpoints, and
    - ends with ``end``,

    while reversing intermediate edges as needed so that each edge starts at the current vertex.
    The function ensures the chosen ``end`` edge is oriented to share the base vertex with the
    start edge (i.e. the start's first vertex matches the end's second vertex), if not, ``end`` is
    reversed before path construction.

    :param minimal_loop: Minimal loop as a list of keyed edges ``(a, b, k)``.
    :param edge_pair: Two keyed edges ``[start, end]`` specifying the first and last edges of the
                      reordered loop.
    :returns: Reordered minimal loop (list of keyed edges) starting with ``start`` and ending with
              ``end``.
    :raises ValueError: If the remaining edges cannot be connected into a continuous path.
    """

    def remove_edge(L: Union[Tuple, List], e: List[int]) -> None:
        """
        A function which removes an edge e (or its inverse) from a given list of edges L
        """

        # ensure list type
        e = list(e)
        try:
            L.remove(e)
        except ValueError:
            L.remove(get_reversed_keyed_edge(e, True))

    # make sure that the requested starting edge [m, n] and ending edge [o, p] are oriented such
    # that m == p (they share the same vertex) as starting and ending points
    # if not, we reverse the last edge
    if edge_pair[0][0] != edge_pair[1][1]:
        edge_pair[1] = get_reversed_keyed_edge(edge_pair[1])

    # cast all edges in minimal_loop to lists for consistency
    minimal_loop = [list(e) for e in minimal_loop]

    # define start (the first edge in edge_pair) and end (the second edge)
    start = edge_pair[0]
    end = edge_pair[1]

    # remove these two edges from minimal_loop
    remove_edge(minimal_loop, start)
    remove_edge(minimal_loop, end)

    # our path will begin with start
    path = [start]

    # we will keep track of the current_vertex we need to match
    current_vertex = start[1]

    # now, we will keep looping until we've used up all the edges in minimal_loop
    # in each iteration, we find the edge that emanates from current_vertex
    # and append it in the correct orientation.
    while minimal_loop:
        # find the next edge whose endpoints contain current_vertex
        # (there should be exactly one in a proper minimal loop)
        next_edge = None
        for e in minimal_loop:
            if current_vertex in e[:-1]:
                next_edge = e
                break

        if next_edge is None:
            # this would be unusual for a proper loop, but we will guard anyway
            raise ValueError(
                "No connected edge found, invalid loop or disconnected edges"
            )

        # orient next_edge so that it starts at current_vertex
        if next_edge[0] != current_vertex:
            next_edge = get_reversed_keyed_edge(next_edge)

        # append to path
        path.append(next_edge)

        # remove from minimal_loop
        remove_edge(minimal_loop, next_edge)

        # advance current_vertex
        current_vertex = next_edge[1]

    # lastly, we append end, which should start at current_vertex
    # if not, flip it
    if end[0] != current_vertex:
        end = get_reversed_keyed_edge(end)
    path.append(end)

    return path


def detect_edge_orientation_key(
    graph,
    edge: list,
    graph_shift: int,
) -> str:
    """
    Determine holonomy operator type for an edge relative to an underlying graph orientation.

    The edge is shifted by ``graph_shift`` (using :func:`neuralqx.utils.misc.arithmetic.minus_key`)
    and checked for membership in ``graph.edges``. If present, the edge is classified as
    ``"creation"``, otherwise it is classified as ``"annihilation"``.

    :param graph: Graph object providing an ``edges`` container for membership testing.
    :param edge: Keyed edge ``(a, b, k)`` (as a list/tuple) whose orientation is to be classified.
    :param graph_shift: Constant shift applied to the edge endpoints prior to membership testing.
    :returns: Either ``"creation"`` or ``"annihilation"``.
    """

    if tuple(minus_key(edge, graph_shift)) in graph.edges:
        return "creation"
    return "annihilation"


def create_dressed_minimal_loop_key(
    graph,
    minimal_loop: list[list],
    graph_shift: int,
) -> list[tuple[tuple[Any, Any, Any], dict[str, str | Any]]]:
    """
    Attach holonomy-operator metadata to each edge in a keyed minimal loop.

    For each edge in ``minimal_loop``, the function determines the operator type using
    :func:`detect_edge_orientation_key` and returns a new "dressed" representation where each entry
    is a pair:

    - the keyed edge as a tuple ``(a, b, k)``, and
    - a metadata dict containing ``{"type": <creation|annihilation>, "key": k}``.

    :param graph: Graph object providing an ``edges`` container for membership testing.
    :param minimal_loop: Minimal loop as a list of keyed edges ``(a, b, k)``.
    :param graph_shift: Constant shift applied when classifying each edge's operator type.
    :returns: Dressed minimal loop as a list of ``((a, b, k), metadata)`` entries.
    """

    # the dressed minimal loop, now empty
    dressed_minimal_loop = []

    # populate a list of operator types for every edge in the loop
    o_type_list = [
        detect_edge_orientation_key(graph, edge, graph_shift) for edge in minimal_loop
    ]

    # dress the loop
    # dev: try enumerate
    for _ in range(len(minimal_loop)):  # pylint: disable=C0200
        dressed_minimal_loop.append(
            (
                (minimal_loop[_][0], minimal_loop[_][1], minimal_loop[_][2]),
                {"type": o_type_list[_], "key": minimal_loop[_][2]},
            )
        )
    return dressed_minimal_loop


def count_contributing_triplets_key(
    levi_civita_dict: dict,
    triplets: list,
) -> int:
    """
    Count edge triplets with positive Levi-Civita contribution.

    The function counts how many edge triplets in ``triplets`` have Levi-Civita value
    ``ε(e₁, e₂, e₃) = +1`` according to ``levi_civita_dict``. Triplets are stringified as
    ``str(tuple(triplet))`` for lookup.

    This count is used as the contributing-triplets factor (often denoted ``T``) in regularized
    Euclidean Thiemann-type constructions.

    :param levi_civita_dict: Dictionary mapping stringified edge triplets to Levi-Civita values.
    :param triplets: List of edge triplets to test.
    :returns: Number of triplets with Levi-Civita value equal to ``+1``.
    """

    count = 0
    for triplet in triplets:
        key = str(tuple(triplet))
        if key in levi_civita_dict and levi_civita_dict[key] == 1:
            count += 1
    return count


def get_ml_graph_shift_key(
    minimal_loop: list[tuple[int, int, dict]],
    mapping: callable,
    adjoint: bool = False,
) -> dict:
    """
    Split a dressed minimal loop into holonomy and adjoint-holonomy edge lists.

    The input ``minimal_loop`` is expected to be in "dressed" form where each element contains a
    keyed edge and a metadata dict with a ``"type"`` field. Edges tagged as ``"creation"`` are
    collected under ``"h"`` and all others under ``"ha"`` after applying ``mapping`` to the keyed
    edge.

    If ``adjoint`` is True, the roles of ``"h"`` and ``"ha"`` are swapped in the returned dict.

    :param minimal_loop: Dressed minimal loop entries of the form ``(edge, meta)`` where
                         ``meta["type"]`` indicates operator type.
    :param mapping: Callable applied to each keyed edge prior to insertion into the output lists.
    :param adjoint: If True, swap the returned lists under keys ``"h"`` and ``"ha"``.
    :returns: Dict with keys ``"h"`` and ``"ha"`` containing the mapped edge lists.
    """

    h = []
    ha = []
    for edge in minimal_loop:
        if edge[1]["type"] == "creation":
            h.append(mapping((edge[0])))
        else:
            ha.append(mapping((edge[0])))

    return {"h": h, "ha": ha} if not adjoint else {"h": ha, "ha": h}


def get_true_edge_triplet(edge_list: Union[List, Tuple], G: "AbstractGraph") -> Tuple:
    """
    Reorient an edge triplet so every edge is present in ``G.edges``.

    For each edge in ``edge_list``, the function checks membership in ``G.edges``. If the edge is
    not present, it is replaced by its reversed keyed form (endpoint-flipped, key preserved).

    :param edge_list: Iterable of keyed edges to be checked/reoriented.
    :param G: Graph providing an ``edges`` container for membership testing.
    :returns: Tuple of keyed edges where each edge is guaranteed to be in ``G.edges``.
    """

    return tuple(
        [
            edge if edge in G.edges else get_reversed_keyed_edge(edge)
            for edge in edge_list
        ]
    )


def reduce_permutations(perms: List):
    """
    Reduce a list of 3-edge permutations by identifying swap-equivalent entries.

    Given permutations of triplets ``(a, b, c)``, this function removes duplicates where the last
    edge ``c`` is the same and the first two edges are swapped, i.e. it keeps only one of
    ``(a, b, c)`` and ``(b, a, c)`` for each pair.

    :param perms: List of ordered triplets (permutations) of three edges.
    :returns: Filtered list where swap-equivalent triplets differing only by the order of the first
              two edges have been removed.
    """

    seen = set()
    reduced = []

    for triplet in perms:
        a, b, c = triplet

        # create a canonical form: (a, b, c) or (b, a, c), whichever appears first
        key1 = (a, b, c)
        key2 = (b, a, c)

        if key1 not in seen and key2 not in seen:
            seen.add(key1)
            reduced.append(triplet)

    return reduced
