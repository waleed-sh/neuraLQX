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
Validation and normalisation helpers for GraphHandler edge data.

These utilities enforce a consistent internal representation for graph edges and
vertex labels before higher level graph construction begins.

Two vertex labelling conventions are supported.

- Planar graphs use integer vertex labels, typically :math:`v \in \{0,\dots,N-1\}`.
- Non planar graphs use tuple labels, typically coordinates such as :math:`(x,y,z)`.

Multi edges are supported through an explicit integer key :math:`k`, so an edge is
represented as :math:`(u, v, k)`. The key disambiguates distinct edges that share the
same endpoints, which is required for multigraphs and for constructing the line graph
dual where original edges become dual vertices.

The functions in this module perform the following tasks.

- Validate edge tuple lengths and key types.
- Ensure that all edges are either keyed or unkeyed, never a mixture.
- Insert a default key of 0 when keys are omitted and there is no ambiguity.
- Detect inconsistent vertex label types and classify the edge set as planar or non planar.
- Relabel non planar vertex objects to contiguous integer labels while preserving keys.
"""

from collections import defaultdict

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

from neuralqx.utils.errors import IncorrectEdgeFormatError


def validate_and_insert_keys(edges: List[Union[List, Tuple]]) -> List[Tuple]:
    """
    Validate an edge list and ensure an explicit integer key is present on every edge.

    Edges may be provided as either 2 tuples ``(u, v)`` or 3 tuples ``(u, v, key)``.

    Rules enforced.

    - If any edge is provided with a key, then every edge must be provided with a key.
    - Keys must be integers.
    - If keys are omitted, a default key of 0 is inserted.
    - If keys are omitted and the same endpoint pair occurs more than once, the input is
      ambiguous and an error is raised. In that case, keys must be provided explicitly to
      distinguish multi edges.

    This function does not reorder or canonicalise endpoints. Duplicate detection is based on
    the ordered pair ``(u, v)`` exactly as provided.

    :param edges: List of edges as ``(u, v)`` or ``(u, v, key)`` entries.
    :return: List of normalised edges as 3 tuples ``(u, v, key)`` with integer keys.
    :raises TypeError: If any key is not an integer.
    :raises TypeError: If keyed and unkeyed edges are mixed in the same input list.
    :raises ValueError: If an edge has an unsupported length.
    :raises ValueError: If an unkeyed duplicate endpoint pair ``(u, v)`` is detected.
    """

    edge_counter = defaultdict(int)
    output_edges = []

    has_keyed = False
    has_unkeyed = False

    for edge in edges:
        if len(edge) == 2:
            # no key present
            u, v = edge
            key = None
            has_unkeyed = True
        elif len(edge) == 3:
            # check that keys are integers
            if not isinstance(edge[2], int):
                raise TypeError(
                    f"Keys are only allowed to be integers. "
                    f"Got type `{type(edge[2]).__name__}` instead."
                )

            u, v, key = edge
            has_keyed = True
        else:
            raise ValueError(f"Edge format not recognized: {edge}")

        edge_id = tuple((u, v))

        if key is None:
            edge_counter[edge_id] += 1
            if edge_counter[edge_id] > 1:
                raise ValueError(
                    f"Duplicate edge {edge_id} found without a key. Use explicit keys."
                )

        output_edges.append((u, v, key))

    if has_keyed and has_unkeyed:
        raise TypeError(
            "You cannot specify a key for only some edges. "
            "Either specify keys for all edges or do not specify keys at all."
        )

    # assign default key = 0 where missing
    result = []
    for u, v, key in output_edges:
        if key is None:
            key = 0
        result.append((u, v, key))

    return result


def check_vertex_type_consistency(edges):
    """
    Classify an edge list as planar or non planar by inspecting vertex label types.

    This function inspects the endpoint objects ``u`` and ``v`` appearing in each edge.

    - If every vertex label is an ``int``, the graph is classified as ``"planar"``.
    - If every vertex label is a ``tuple``, the graph is classified as ``"non-planar"``.
    - If both types appear, the input is considered inconsistent and an error is raised.

    Only the first two entries of each edge are used. A key, if present, is ignored.

    :param edges: Iterable of edges where each edge has at least two items.
    :return: The string ``"planar"`` or ``"non-planar"``.
    :raises ValueError: If any edge has fewer than two entries.
    :raises ValueError: If both integer and tuple vertex types are present.
    """

    vertex_types = set()

    for edge in edges:
        if len(edge) < 2:
            raise ValueError(f"Invalid edge (too short): {edge}")

        u, v = edge[0], edge[1]
        vertex_types.add(type(u))
        vertex_types.add(type(v))

    if vertex_types == {int}:
        return "planar"
    elif vertex_types == {tuple}:
        return "non-planar"
    else:
        raise ValueError(
            f"Inconsistent vertex types detected: {vertex_types}. "
            "All vertices must be either int (for planar) or tuple (for non-planar)."
        )


def get_vertices_from_edges(edges):
    """
    Extract the set of unique vertices from an edge list.

    Each edge is expected to have at least two entries, interpreted as the endpoints ``u``
    and ``v``. Any additional entries, such as a key, are ignored. The returned list is
    sorted using Python's default ordering for the vertex objects.

    This is intended for two common cases.

    - Planar graphs where vertices are integers and sorting is well defined.
    - Non planar graphs where vertices are tuples and lexicographic sorting provides a
      stable ordering.

    :param edges: Iterable of edges with endpoints in positions 0 and 1.
    :return: Sorted list of distinct vertex labels.
    :raises ValueError: If any edge has fewer than two entries.
    """

    vertices = set()

    for edge in edges:
        if len(edge) < 2:
            raise ValueError(f"Invalid edge: {edge}")
        u, v = edge[0], edge[1]
        vertices.add(u)
        vertices.add(v)

    return sorted(vertices)


def _check_planarity(
    edges: list,
) -> str:
    """
    Determine whether an edge list represents a planar or non planar vertex labelling scheme.

    This is a lightweight classifier that inspects the type of the first vertex label
    encountered and returns a string label used elsewhere in the graph handling pipeline.

    - Returns ``"planar"`` if the first vertex label is an ``int``.
    - Returns ``"non-planar"`` if the first vertex label is a ``tuple``.

    The function requires the edge list to be non empty. If the first vertex label is neither
    an ``int`` nor a ``tuple``, an error specific to neuraLQX edge formatting is raised.

    :param edges: Edge list with at least one edge and at least one endpoint per edge.
    :return: The string ``"planar"`` or ``"non-planar"``.
    :raises ValueError: If `edges` is empty or the first edge is empty.
    :raises IncorrectEdgeFormatError: If the first vertex label is not an int or a tuple.
    """

    # check if the list of edges is empty
    if not edges or not edges[0]:
        raise ValueError("You cannot create a graph which has no edges")

    # check if planar
    if isinstance(edges[0][0], int):
        return "planar"
    if isinstance(edges[0][0], tuple):
        return "non-planar"

    raise IncorrectEdgeFormatError


def _relabel_non_planar_edges(
    edges: Iterable[Tuple[Any, Any, Any]],
) -> Tuple[List[int], List[Tuple[int, int, Any]], Dict[int, Any], Dict[Any, int]]:
    r"""
    Relabel non planar vertex objects to contiguous integer labels while preserving edge keys.

    This helper is used when a graph uses non integer vertex labels, for example coordinate
    tuples :math:`(x,y,z)`. It constructs a bijection between the original vertex objects and
    integers :math:`0,\dots,N-1`, then rewrites every edge ``(u, v, key)`` as
    ``(u_int, v_int, key)``.

    The returned mappings allow round trip conversion.

    - ``orig_to_int`` maps original labels to integers.
    - ``int_to_orig`` maps integers back to original labels.

    Note on determinism.
    The integer assignment order is derived from a set of vertex objects. Unless the caller
    pre sorts the vertex objects before calling this function, the resulting mapping may vary
    between Python runs. If deterministic labelling is required, sort the vertices explicitly
    before constructing the mapping.

    :param edges: Iterable of edges as ``(u, v, key)`` where ``u`` and ``v`` are hashable objects.
    :return: A tuple ``(vertices, relabeled_edges, int_to_orig, orig_to_int)`` where
      ``vertices`` is a list of integer vertex labels, ``relabeled_edges`` is the rewritten
      edge list, and the final two entries are the conversion mappings.
    """

    # flatten all original vertex‐objects
    all_vertices = {u for u, _, _ in edges} | {v for _, v, _ in edges}

    # build original -> integer map
    orig2int: Dict[Any, int] = {orig: idx for idx, orig in enumerate(all_vertices)}

    # invert to integer -> original
    int2orig: Dict[int, Any] = {idx: orig for orig, idx in orig2int.items()}

    # relabel edges
    relabeled_edges: List[Tuple[int, int, Any]] = [
        (orig2int[u], orig2int[v], key) for u, v, key in edges
    ]

    # return
    return list(int2orig.keys()), relabeled_edges, int2orig, orig2int
