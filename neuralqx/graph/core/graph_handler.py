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
Internal construction utilities for neuraLQX graphs.

This module provides :class:`~neuralqx.graph._graph_handler.GraphHandler`, the implementation behind
the public :class:`~neuralqx.graph.core.AbstractGraph` interface. The handler owns the concrete graph
data structures and is responsible for turning a user supplied edge list into

- a NetKet graph :math:`G` used by samplers and Hilbert spaces
- a NetworkX multigraph used for connectivity queries and cycle computations
- the dual graph, defined as the line graph :math:`L(G)`
- a bijection between oriented edges and integer indices, used to index edge degrees of freedom in
  array based configurations
- local sign tables used by volume like and grasping constructions

Planar versus non planar
A planar graph uses integer vertex labels. A non planar graph uses vertices that are coordinate
tuples :math:`(x,y,z)\in\mathbb R^3`. For non planar input, the handler relabels the coordinate
tuples to contiguous integers for efficient indexing, while preserving an inverse map to recover
the original coordinates.

Dual graph
Let :math:`G=(V,E)` be the (multi)graph. The dual graph used here is the line graph :math:`L(G)`,
whose vertex set is :math:`V(L(G))=E(G)` and where two dual vertices are adjacent whenever their
corresponding primal edges share an endpoint. The handler stores a mapping

.. math::

   \mu : E(G) \to \{0,1,\dots,|E(G)|-1\}

which identifies each (keyed) primal edge with an integer label for the corresponding dual vertex.

Local sign data
Depending on planarity, sign data is computed using different geometric conventions

- Non planar, triple edge sign at a vertex :math:`v` using tangents :math:`t_i\in\mathbb R^3`

  .. math::

     \varepsilon(e_1,e_2,e_3) = \operatorname{sgn}\bigl(\det[t_1\ t_2\ t_3]\bigr)

- Planar, ordered edge pair sign at a vertex using a 2D orientation convention, for tangents
  :math:`t_1,t_2\in\mathbb R^2`

  .. math::

     \operatorname{sgn}(e_1,e_2) = \operatorname{sgn}(t_{1x}t_{2y}-t_{1y}t_{2x})

The handler computes and stores these signs for both the primal graph and, when needed, the dual
graph.
"""

import itertools

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

from itertools import combinations
from collections import defaultdict, deque

import netket as nk
import networkx as nx
from matplotlib import pyplot as plt

from .utils.plotting import _draw_graph_and_dual
from .utils.plotting import draw_nonplanar_graph
from .utils.signs import compute_signs
from .utils.build import canonical_rotate
from .utils.build import create_dressed_nx_graph
from .utils.build import create_dual_nk_graph
from .utils.build import dress_minimal_cycles_edges
from .utils.build import init_node_connectivity
from .utils.checks import _relabel_non_planar_edges
from .utils.checks import check_vertex_type_consistency
from .utils.checks import get_vertices_from_edges
from .utils.checks import validate_and_insert_keys
from neuralqx.debug import errors_only


def _pick_key(seg2keys: dict, x: Any, y: Any) -> int:
    keys = seg2keys.get(frozenset((x, y)), [])
    if not keys:
        raise ValueError(f"Undirected segment {{{x},{y}}} not found in keyed edges.")
    return min(keys)


def _girth_and_witness_edges(G: nx.Graph) -> tuple[int, set[tuple[Any, Any]]]:
    """
    Returns (girth, witness_edges) where witness_edges is a set of undirected edges (u,v) that
    participate as the closing edge in at least one shortest cycle found by BFS detection.
    """
    INF = 10**9
    best = INF
    witness: set[tuple[Any, Any]] = set()

    nodes = list(G.nodes())
    for s in nodes:
        dist = {s: 0}
        parent = {s: None}
        q = deque([s])

        while q:
            v = q.popleft()
            dv = dist[v]
            # no need to explore beyond best-1 (can't improve)
            if dv + 1 >= best:
                continue

            for w in G.neighbors(v):
                if w not in dist:
                    dist[w] = dv + 1
                    parent[w] = v
                    q.append(w)
                elif parent[v] != w:
                    # found an undirected cycle via edge (v,w):
                    # length = dist[v] + dist[w] + 1
                    cycle_len = dist[v] + dist[w] + 1
                    if cycle_len < best:
                        best = cycle_len
                        witness = {(v, w) if v <= w else (w, v)}
                    elif cycle_len == best:
                        witness.add(((v, w) if v <= w else (w, v)))

    if best == INF:
        # acyclic graph
        return 0, set()
    return best, witness


def _all_shortest_paths_avoid_edge(
    G: nx.Graph,
    s: Any,
    t: Any,
    avoid: tuple[Any, Any],
    target_len: int,
    max_paths: int = 100000,
) -> list[list[Any]]:
    """
    Return all shortest simple paths from s to t of exact length target_len, while treating
    'avoid' (undirected) as removed. Uses BFS distance + predecessor backtracking.
    """
    a, b = avoid
    avoid_set = {a, b}

    # BFS distances and predecessors
    dist = {s: 0}
    preds: dict[Any, list[Any]] = defaultdict(list)
    q = deque([s])

    while q:
        v = q.popleft()
        dv = dist[v]
        if dv >= target_len:
            continue

        for w in G.neighbors(v):
            # skip the avoided undirected edge
            if (v in avoid_set and w in avoid_set) and ({v, w} == set(avoid)):
                continue

            nd = dv + 1
            if w not in dist:
                dist[w] = nd
                preds[w].append(v)
                q.append(w)
            else:
                if nd == dist[w]:
                    preds[w].append(v)

    if t not in dist or dist[t] != target_len:
        return []

    # backtrack all paths of length target_len
    paths: list[list[Any]] = []
    stack = [(t, [t])]
    while stack:
        node, rev_path = stack.pop()
        if len(paths) >= max_paths:
            break
        if node == s:
            paths.append(list(reversed(rev_path)))
            continue
        for p in preds.get(node, []):
            stack.append((p, rev_path + [p]))

    # ensure simple paths (should be, but backtracking can create repeats in some graphs)
    out = []
    for p in paths:
        if len(p) == len(set(p)):
            out.append(p)
    return out


class GraphHandler:
    r"""
    Concrete graph construction and bookkeeping.

    The :class:`GraphHandler` class is the workhorse that builds all graph representations needed by
    neuraLQX. It is created with a list of oriented edges and optionally a coordinate map for non planar
    graphs. After initialisation it provides

    - :attr:`edges` and :attr:`vertices` in internal integer labelling
    - :attr:`nk_graph` as a :class:`netket.graph.Graph`
    - :attr:`nx_graph` as a :class:`networkx.MultiGraph` with dressed edge attributes
    - :attr:`dual_nk_graph` and :attr:`dual_nx_graph` as dual representations
    - :attr:`mapping`, a bijection between (keyed) edges and integer indices
    - :attr:`graph_signs` and :attr:`dual_graph_signs`, local sign tables
    - :attr:`minimal_cycles`, a collection of dressed minimal loops

    Input edge format
    --------------------
    Edges may be given as ``(u, v)`` or ``(u, v, key)``. If any parallel edges exist between the same
    endpoints, keys must be provided for all edges to disambiguate them. If no keys are provided and no
    duplicates exist, the handler inserts the default key ``0`` for every edge.

    Non planar relabelling
    -------------------------
    For non planar input, vertex labels are coordinate tuples. The handler constructs a relabelling map

    .. math::

       \pi : V_{\mathrm{orig}} \to \{0,1,\dots,|V|-1\}

    and rewrites all edges in terms of these integer labels while preserving

    - ``_nonplanar_int2orig_map`` mapping integers back to coordinates
    - ``_nonplanar_orig2int_map`` mapping coordinates to integers

    Minimal loops
    ---------------
    Minimal loops are extracted from a minimum cycle basis of the underlying undirected simple graph,
    with special handling for two edge cycles arising from parallel edges. Loop edges are returned as
    keyed triples ``(u, v, key)`` and then canonically rotated to a stable representative.

    :param edges: Oriented edge list. Each edge is ``(u, v)`` or ``(u, v, key)``.
    :param coordinates: Optional mapping used for random embeddings, typically keyed by integer labels.
    :param plot: If True, compute layout positions and display plots for diagnostics.
    :raises TypeError: If keyed and unkeyed edges are mixed.
    :raises ValueError: If duplicate unkeyed edges are supplied.
    """

    def __init__(
        self,
        edges: List[List[int]],
        coordinates: Optional[Dict] = None,
        plot: bool = False,
    ):
        r"""
        Build all graph representations and precompute auxiliary data.

        This initialiser performs the full construction pipeline

        1. Determine planarity from vertex label types
        2. Validate edges, enforce consistent key usage, and insert default keys where appropriate
        3. For non planar input, relabel coordinate tuple vertices to integers and store forward and inverse
           maps
        4. Build the NetKet graph used by samplers and Hilbert spaces
        5. Build the dressed NetworkX multigraph and its line graph dual
        6. Build the dual NetKet graph and an edge to index mapping
        7. Extract and dress minimal cycles
        8. Compute node connectivity tables for later sign computations
        9. Compute layout positions for planar graphs or register coordinates for non planar graphs
        10. Compute and store local sign tables

        The resulting object is intended to be stable and query friendly. After initialisation, most
        methods are pure lookups and do not mutate the combinatorial structure.

        :param edges: Oriented edges defining the graph.
        :param coordinates: Optional coordinate mapping for non planar embeddings.
        :param plot: If True, show the graph in a diagnostic plot.
        :return: None.
        """

        self.edges: Union[List[List[int]], List[tuple]]
        """The edges which define the graph"""

        self._random_embedding_map: Dict = coordinates
        """The mapping between the user coordinates and the chosen random embedding."""

        self.dual_edges: List
        """The edges of the dual graph"""

        self.vertices: List
        """The vertices of the graph"""

        self.nx_graph: nx.MultiGraph = nx.MultiGraph()
        """The unoriented NetworkX graph created from the specified edges"""

        self.dual_nx_graph: nx.Graph
        """The NetworkX graph dual to the original graph"""

        self.nk_graph: nk.graph.Graph
        """A NetKet copy of the NetworkX graph"""

        self.dual_nk_graph: nk.graph.Graph
        """A NetKet copy of the dual NetworkX graph"""

        self.is_planar: bool
        """A flag to determine whether the graph is planar or not"""

        self.mapping: Dict
        """A mapping between the edges and a sequential integer representation"""

        self.graph_signs: Dict = {}
        """The signs of edges in the graph, depending on the planarity"""

        self.dual_graph_signs: Dict = {}
        """The signs of edges in the dual graph, depending on the planarity"""

        self.graph_positions: Dict = {}
        """The coordinate positions of edges and vertices in the graph"""

        self.dual_graph_positions: Dict = {}
        """The coordinate positions of edges and vertices in the dual graph"""

        self.graph_edges_data: Dict = {
            "graph": {"connectivities": {}},
            "dualGraph": {"connectivities": {}},
        }
        """A dict of connectivities for both the graph and the dual graph"""

        self.minimal_cycles: List[List[List[int]]] = []
        """The list of dressed minimal loops of the graph"""

        self._nonplanar_vertex_map: Dict
        """A mapping of non-planar vertices from tuples to sequential integer labeling"""

        self._original_vertices: List
        """The original non-planar vertices"""

        self._original_edges: List[List[int]]
        """The original list of non-planar edges"""

        self._n_vertices: int
        """The number of vertices in the graph"""

        # first check planarity based on the type of edges specified
        if check_vertex_type_consistency(edges) == "planar":
            # set the planarity flag
            self.is_planar = True

            # now we do the following:
            #   1. check if there are any duplicate keyless edges
            #   2. if all clear, check if there are keys
            #   3. if not, insert 0 keys to all edges
            self.edges = validate_and_insert_keys(edges)
            # store the vertices
            self.vertices = get_vertices_from_edges(self.edges)

            # store the original edges and vertices (redundant for planar case)
            self._original_edges = edges
            self._original_vertices = self.vertices
        else:
            # non-planar case
            # here we need to convert everything back to planar notation

            # set the planarity flag to False
            self.is_planar = False

            # now we will relabel prior to obtaining changing the representation back to planar
            edges = validate_and_insert_keys(edges)

            # get the vertices, edges and the non-planar vertex map in planar notation
            # this function will reinsert keys correctly for each mapped edge
            (
                self.vertices,
                self.edges,
                self._nonplanar_int2orig_map,
                self._nonplanar_orig2int_map,
            ) = _relabel_non_planar_edges(edges)

            # readjust the non-planar vertex map
            self._nonplanar_vertex_map = {
                value: key for key, value in self._nonplanar_int2orig_map.items()
            }

            # store the original non-planar edges and vertices
            self._original_edges = edges
            self._original_vertices = get_vertices_from_edges(edges)

        # find the number of vertices in the graph
        self._n_vertices = len(self.vertices)

        # store keyed and keyless version of the final edges
        self._edges_keyed = set(self.edges)
        self._edges_keyless = {(u, v) for u, v, k in self._edges_keyed}

        # create the NetKet graph
        # here, we have to ignore the keys
        self.nk_graph = nk.graph.Graph(self.edges)

        # create the NetworkX graph
        create_dressed_nx_graph(self)

        # create the dual NetKet graph
        self.dual_nk_graph, self.mapping = create_dual_nk_graph(
            self
        )  # , is_nonplanar = not self.is_planar)

        # create the dressed minimal cycles for the given graph
        dress_minimal_cycles_edges(self)

        # create the dual NetworkX graph
        self.dual_nx_graph = nx.line_graph(self.nx_graph)

        # determine the incoming and outgoing edges of every vertex in the graph
        init_node_connectivity(self)

        # create the positions of the graph and dual graph to be used in computing the signs
        if self.is_planar:
            _draw_graph_and_dual(self, plot)
        else:
            _draw_graph_and_dual(self, False)
            if plot:
                draw_nonplanar_graph(self)

        # compute the signs of the graph and its dual
        compute_signs(self)
        # compute_signs(self, "dualGraph")

        self.__name__ = "Graph"

    def map_edge_to_dual_vertex(
        self,
        edge: tuple[int, int],
    ) -> Optional[int]:
        r"""
        Map a keyed primal edge to the corresponding dual vertex label.

        The dual graph is the line graph, so every primal edge becomes a dual vertex. The handler stores a
        mapping :math:`\mu` from keyed edges ``(u, v, key)`` to an integer label. If the exact oriented edge
        is not found, the reversed orientation is also tried while keeping the key fixed.

        :param edge: A keyed edge triple ``(u, v, key)``.
        :return: The integer label of the corresponding dual vertex, or None if not found.
        """

        # return the mapped dual vertex label for the given edge (a, b) (or (b, a))
        # note that when we search for the reversed edge, we keep the key but reverse the rest
        return self.mapping.get(
            edge, self.mapping.get((*edge[:-1][::-1], edge[-1]), None)
        )

    def inverse_mapping(
        self,
        vertex: int,
    ) -> Optional[tuple[int, int]]:
        r"""
        Invert the edge to index mapping for one dual vertex.

        Given a dual vertex label :math:`w\in V(L(G))`, this returns the primal edge ``(u, v, key)``
        satisfying :math:`\mu(u,v,key)=w` if it exists.

        This is a convenience method for diagnostics and for algorithms that traverse the dual graph and
        need to recover the associated primal edge.

        :param vertex: Integer label of the dual vertex.
        :return: The corresponding keyed edge triple, or None if not found.
        """

        # return the original label of the dual vertex which was mapped to the current vertex label
        return next(
            (key for key, value in self.mapping.items() if value == vertex), None
        )

    def convert_vertex_to_int(
        self,
        v: tuple[int, int, int],
    ) -> Union[int, None]:
        r"""
        Convert a non planar coordinate vertex to its internal integer label.

        For non planar graphs, vertices are originally supplied as coordinate tuples
        :math:`(x,y,z)\in\mathbb R^3`. The handler relabels them to integers for indexing. This method
        performs the inverse lookup, returning the integer label associated with the given coordinate
        triple.

        :param v: Vertex coordinate tuple ``(x, y, z)``.
        :return: Integer label if present, otherwise None.
        :raises NotImplementedError: If called on a planar graph.
        """

        if self.is_planar:
            raise NotImplementedError(
                "This function is only available for non-planar graphs."
            )

        for key, val in self._nonplanar_int2orig_map.items():
            if val == v:
                return key
        return None

    def _get_edge_direction_factor(
        self,
        edge_pair: tuple,
        vertex: int,  # pylint: disable=W0613
    ) -> int:
        r"""
        Compute a direction consistency factor for planar sign computations.

        For planar graphs, local sign conventions are typically defined for ordered pairs of edges incident
        at a vertex. In practice, the stored edge orientation may differ from the orientation implied by
        the local pair being considered. This helper returns a factor :math:`\pm 1` that corrects for this
        difference.

        Current behaviour
        The method checks whether both edges in the requested pair exist in the stored edge list, either in
        their given orientation or both reversed with their keys preserved. If the pair is consistent with
        the stored representation it returns ``+1``. Otherwise it returns ``-1``.

        :param edge_pair: Tuple containing two keyed edges.
        :param vertex: Vertex at which the sign is being evaluated.
        :return: ``+1`` if orientations are consistent, otherwise ``-1``.
        :raises ValueError: If the edge list contains mixed container types.
        """

        found_flag = -1

        # check if all edges are tuples
        is_tuple = all([isinstance(edge, tuple) for edge in self.edges])

        # check if all edges are lists
        is_list = all([isinstance(edge, list) for edge in self.edges])

        if not is_list and not is_tuple:
            raise ValueError("Encountered a non-uniform edge list in the graph.")

        etype = tuple if is_tuple else list

        first_edge = etype(edge_pair[0])
        second_edge = etype(edge_pair[1])

        if (first_edge in self.edges and second_edge in self.edges) or (
            (*first_edge[:-1][::-1], first_edge[-1]) in self.edges
            and (*second_edge[:-1][::-1], second_edge[-1]) in self.edges
        ):
            found_flag = 1

        return found_flag

    def _register_graph_data(
        self,
        graph: nx.MultiGraph,
        graph_type: str,
    ) -> None:
        r"""
        Register per vertex connectivity data needed by sign computation.

        This method constructs a data structure that, for every vertex, enumerates the local combinations of
        incident edges to be signed later.

        - Planar graphs store ordered pairs of incident edges. For a vertex with incident edge set
          :math:`\{e_1,\dots,e_d\}`, this corresponds to all ordered pairs
          :math:`(e_i,e_j)` with :math:`i\neq j`
        - Non planar graphs store ordered triples of incident edges, corresponding to the determinant sign
          definition

        The data are stored under :attr:`graph_edges_data[graph_type]["connectivities"]`, keyed by the
        string form of the vertex label.

        :param graph: NetworkX graph for which connectivity data should be registered.
        :param graph_type: Either ``"graph"`` or ``"dualGraph"``.
        :return: None.
        """

        # create the appropriate dict
        self.graph_edges_data[graph_type] = {}

        # and the approriate sub-dict
        self.graph_edges_data[graph_type]["connectivities"] = {}

        # loop through every node in the graph
        for vertex in graph.nodes():
            # create a sub-sub-dict for the pairs of edges connected at that vertex
            self.graph_edges_data[graph_type]["connectivities"][str(vertex)] = {}

            # get all edges connected at the vertex
            connected_edges = list(graph.edges(vertex, keys=True))

            # create all combinations of edge pairs
            if self.is_planar:
                edge_pairs = list(itertools.combinations(connected_edges, 2)) + [
                    (edge_tuple[1], edge_tuple[0])
                    for edge_tuple in itertools.combinations(connected_edges, 2)
                ]
            else:
                edge_pairs = list(itertools.permutations(connected_edges, 3))

            # add to the dict
            self.graph_edges_data[graph_type]["connectivities"][str(vertex)][
                "edges"
            ] = edge_pairs

    def get_smallest_loops(self) -> List[List[Tuple[Any, Any, int]]]:
        r"""
        Return all smallest-length loops (shortest simple cycles) as keyed oriented edges (u, v, key).

        What "smallest-length" means here:
          - If the multigraph has parallel edges between the same unordered endpoints {u,v},
            then there exist 2-edge cycles. Those are the shortest possible, so we return all
            such 2-edge cycles and stop.
          - Otherwise, we compute the girth (length of the shortest cycle) of the underlying
            simple undirected graph (keys + orientations ignored), and return all simple cycles
            of that length, converted back to keyed oriented edges using a deterministic key choice
            (the minimum key on each undirected segment).

        Notes:
          - This avoids nx.minimum_cycle_basis() because that returns only a basis, not all shortest cycles.
          - We deduplicate cycles via canonical rotation.
        """

        loops: List[List[Tuple[Any, Any, int]]] = []

        # two-edge loops from parallel edges (shortest possible cycles)
        by_pair: dict[tuple[Any, Any], list[tuple[Any, Any, int]]] = defaultdict(list)
        for u, v, k in self.edges:
            a, b = (u, v) if u <= v else (v, u)
            by_pair[(a, b)].append((u, v, k))

        two_cycles: list[list[tuple[Any, Any, int]]] = []
        for (a, b), lst in by_pair.items():
            # if there are >=2 distinct keyed edges between same endpoints, we have 2-cycles
            if len(lst) >= 2:
                # take all combinations of two distinct keyed edges
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        u1, v1, k1 = lst[i]
                        u2, v2, k2 = lst[j]

                        # e1 as-is
                        e1 = (u1, v1, k1)

                        # e2 oriented to return to the tail of e1
                        # e1: u1 -> v1, so we want e2: v1 -> u1
                        if u2 == v1 and v2 == u1:
                            e2 = (u2, v2, k2)
                        elif v2 == v1 and u2 == u1:
                            e2 = (v2, u2, k2)
                        elif u2 == v1 and v2 != u1:
                            # same endpoint-pair but stored orientation doesn't match, flip
                            e2 = (v2, u2, k2)
                        else:
                            e2 = (u2, v2, k2)

                        two_cycles.append(canonical_rotate([e1, e2]))

        if two_cycles:
            # if 2-cycles exist, they are the smallest possible loops. Return all of them.
            # Deduplicate just in case
            seen = set()
            for cyc in two_cycles:
                key = tuple(cyc)
                if key not in seen:
                    seen.add(key)
                    loops.append(cyc)
            return loops

        # build underlying simple graph SG (ignore keys + orientation)
        SG = nx.Graph()

        # ignore self-loops if any (shouldn't be present, but be safe)
        undirected_edges = {
            (u, v) if u <= v else (v, u) for u, v, _ in self.edges if u != v
        }
        SG.add_edges_from(undirected_edges)

        # precompute: undirected segment -> available keys
        seg2keys: dict[frozenset, list[int]] = defaultdict(list)
        for x, y, k in self.edges:
            if x == y:
                continue
            seg2keys[frozenset((x, y))].append(k)

        # compute girth (shortest cycle length) using BFS from each node
        g, witness_edges = _girth_and_witness_edges(SG)
        if g == 0:
            # no cycles at all
            return []

        # enumerate all cycles of length g.
        #    Approach: for each undirected edge (u,v), find all shortest paths of length g-1 from u to v
        #    in G with that edge "blocked". Each such path + the blocked edge gives a cycle of length g.
        # deduplicate cycles by canonicalised edge list
        seen_cycles: set[tuple[tuple[Any, Any, int], ...]] = set()

        # iterate undirected edges (u,v) as potential closing edges
        for u, v in SG.edges():
            und = (u, v) if u <= v else (v, u)
            if und not in witness_edges:
                # this edge never showed up as a closing edge in a shortest cycle during girth detection.
                # skipping it keeps enumeration lighter, correctness is still OK because witness_edges covers
                # at least one closing edge for every shortest-cycle instance discovered.
                continue

            # find all paths of length g-1 from u to v avoiding edge (u,v)
            paths = _all_shortest_paths_avoid_edge(SG, u, v, und, target_len=g - 1)
            for path in paths:
                # path is [u, ..., v], build directed edges along the path plus closing edge v->u
                raw_edges: list[tuple[Any, Any, int]] = []
                for a, b in zip(path[:-1], path[1:]):
                    raw_edges.append((a, b, _pick_key(seg2keys, a, b)))
                raw_edges.append((v, u, _pick_key(seg2keys, u, v)))

                cyc = canonical_rotate(raw_edges)

                # also consider the reversed orientation as the same cycle
                rev = canonical_rotate([(b, a, k) for (a, b, k) in reversed(cyc)])

                # lexicographic compare of lists of tuples
                key = tuple(min(cyc, rev))

                if key not in seen_cycles:
                    seen_cycles.add(key)
                    loops.append(cyc)

        # final safety filter: keep only cycles of length g (should already be true)
        loops = [c for c in loops if len(c) == g]

        return loops

    def _get_smallest_loops(
        self, edge_list: list | None
    ) -> List[List[Tuple[Any, Any, int]]]:
        r"""
        Return a collection of minimal loops as keyed oriented edges.

        This method returns a list of minimal loops, where each loop is a list of keyed oriented edges
        ``(u, v, key)`` that form a closed walk.

        Two sources of loops are considered

        - Parallel edge two cycles
        If there are parallel edges between the same two vertices, a two edge loop exists. For two distinct
        keys :math:`k_1\neq k_2` connecting the same unordered endpoint pair :math:`\{u,v\}`, the loop is

        .. math::

           (u \to v, k_1),\ (v \to u, k_2)

        with the second edge oriented so that the walk closes.

        - Minimum cycle basis loops
        The method constructs a simple undirected graph by ignoring keys and orientations, then uses
        NetworkX to compute a minimum cycle basis. Each basis cycle is then converted back into keyed edges
        by selecting a representative key for each undirected segment.

        - Canonical rotation
        Each loop is canonicalised by cyclic rotation so that consecutive edges satisfy head to tail
        continuity and the first tail vertex is minimal, improving reproducibility in logs and tests.

        :return: List of minimal loops, each a list of keyed oriented edges.
        """

        # an empty list to hold all the loops we are about to collect
        loops: List[List[Tuple[Any, Any, int]]] = []

        # first case: two-edge loops from parallels

        # an empty list to hold the parallels
        by_pair = defaultdict(list)

        if edge_list is None:
            edges = self.edges
        else:
            edges = edge_list

        # loop through all keyed edges
        for u, v, k in edges:

            # sort the actual edges irrespective of the key
            a, b = sorted((u, v))

            # append to the dict at the key corresponding to the edge (a, b) the keyed egde
            by_pair[(a, b)].append((u, v, k))

        # loop through every list of parallel edge pairs we have collected
        for lst in by_pair.values():

            # we loop through all possible combinations of length 2 for the edge pairs
            for (u1, v1, k1), (u2, v2, k2) in combinations(lst, 2):

                # take the first edge as-is
                e1 = (u1, v1, k1)

                # the second must return to v1
                if u2 == v1:
                    e2 = (u2, v2, k2)
                else:
                    e2 = (v2, u2, k2)

                # rotate the nodes of the edges to unsure that the loop closes correctly and start
                # at the smallest tail so we are in ascending order
                loops.append(canonical_rotate([e1, e2]))

        # second case: longer minimal loops handled via NetowrkX's minimum_cycle_basis

        # create the simple graph
        SG = nx.Graph()

        # append edges to it from the list of stored edges
        SG.add_edges_from({tuple(sorted((u, v))) for u, v, _ in edges})

        # loop through every minimal cycle
        for cycle in nx.minimum_cycle_basis(SG):

            # a list of raw minimal loops
            raw: List[Tuple[Any, Any, int]] = []

            # get the cycle length
            m = len(cycle)

            # loop through as many edges as there are in this cycle
            for i in range(m):

                # get the starting and ending node
                u, v = cycle[i], cycle[(i + 1) % m]

                # pick any key on this endpoint-pair
                seg_keys = [k for x, y, k in edges if {x, y} == {u, v}]
                k = min(seg_keys)

                # always emit (u -> v, k), that's the undirected cycle order
                raw.append((u, v, k))

            # now raw is guaranteed continuous/closed just rotate to start at smallest tail
            loops.append(canonical_rotate(raw))

        # dev: quick fix for having non-minimal length loops in the set of minimal loops, note that
        #      mathematically those can be in the cycle basis indeed!
        if len(loops) > 1:
            min_len = min([len(loop) for loop in loops])

            for i, loop in enumerate(loops):
                if len(loop) > min_len:
                    loops.pop(i)

        return loops

    def get_smallest_dual_loops(
        self,
        edge_rep: bool = False,
    ) -> Union[list[list[tuple[int, int, int]]], list[list]]:
        r"""
        Return all smallest-length loops (shortest simple cycles) as keyed oriented edges (u, v, key) in the
        dual loop. By default, the vertices in the dual graph are labelled by the primal edges. If the ``edge_rep`` is
        truthy, the minimal loops are returned such that the vertices are labelled as dual vertices and not primal
        edges.

        What "smallest-length" means here:
          - If the multigraph has parallel edges between the same unordered endpoints {u,v},
            then there exist 2-edge cycles. Those are the shortest possible, so we return all
            such 2-edge cycles and stop.
          - Otherwise, we compute the girth (length of the shortest cycle) of the underlying
            simple undirected graph (keys + orientations ignored), and return all simple cycles
            of that length, converted back to keyed oriented edges using a deterministic key choice
            (the minimum key on each undirected segment).

        Notes:
          - This avoids nx.minimum_cycle_basis() because that returns only a basis, not all shortest cycles.
          - We deduplicate cycles via canonical rotation.
        """

        # get the minimal loops in the dual graph
        # this returns them in the native representation, where vertices are labelled by primal edges
        loops = self._get_smallest_loops(
            validate_and_insert_keys(list(self.dual_nx_graph.edges()))
        )

        if edge_rep:
            edge_rep_loops = []
            for loop in loops:
                new_loop = []
                for edge in loop:
                    new_edge = [self.map_edge_to_dual_vertex(e) for e in edge[:-1]]
                    new_edge.append(edge[-1])
                    new_edge = tuple(new_edge)
                    new_loop.append(new_edge)
                edge_rep_loops.append(new_loop)

        return loops if not edge_rep else edge_rep_loops

    def print_signs(
        self,
        dual_signs: bool = False,
    ) -> None:
        r"""
        Pretty print stored sign tables.

        This is a diagnostic helper that prints the sign tables stored in :attr:`graph_signs` or
        :attr:`dual_graph_signs`. For planar graphs the printed entries correspond to ordered edge pairs,
        while for non planar graphs they correspond to ordered edge triples.

        :param dual_signs: If True, print signs for the dual graph, otherwise print primal signs.
        :return: None.
        """

        # check if we need the dual graph
        if dual_signs:
            # if so, load the correct dict
            signs = self.dual_graph_signs
        else:
            # we deal with the graph and not its dual, load the correct dict
            signs = self.graph_signs

        # loop through the entire dict and print the vertex and the sign of all the combinations of
        # the edges connected to it
        for item, value in signs.items():
            print(f"Vertex: {item}")
            for keyword, sign in value.items():
                print(f"sgn{keyword} = {sign}")
            print()

    @errors_only(tag="GRAPH_EXPORT")
    def export(
        self,
        filename: str,
        *,
        dpi: int = 300,
    ) -> None:
        r"""
        Export a static visualisation of the graph to an image file.

        This method creates a figure without requiring interactive plotting and writes it to `filename`.
        The renderer depends on planarity

        - Planar graphs use the 2D directed multigraph drawing routine, optionally with dual data
        - Non planar graphs use the 3D renderer based on stored coordinate maps

        :param filename: Output image path.
        :param dpi: Resolution used when saving the figure.
        :return: None.
        :raises RuntimeError: If no figure could be produced for export.
        """

        fig = None

        if self.is_planar:
            # planar graph, use the standard primal/dual drawing
            fig = _draw_graph_and_dual(
                self,
                plot=False,
                return_fig=True,
            )
        else:
            # Non-planar graph, use the 3D renderer
            fig = draw_nonplanar_graph(
                self,
                plot=False,
                return_fig=True,
            )

        if fig is None:
            raise RuntimeError("Graph export failed: no figure was produced.")

        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    def __repr__(self):
        print(
            f"A graph composed of {self.nk_graph.n_edges} edges and {self.nk_graph.n_nodes} "
            f"vertices with the following edges:"
            f"{self.edges}. Its dual graph is composed of {self.dual_nk_graph.n_edges} edges and "
            f"{self.dual_nk_graph.n_nodes} vertices."
            f" The graph contains {len(self.get_smallest_loops())} minimal loops"
        )
        return ""

    def __str__(self):
        return (
            f"A graph composed of {self.nk_graph.n_edges} edges and {self.nk_graph.n_nodes} "
            f"vertices with the following edges: "
            f"{self.edges}. Its dual graph is composed of {self.dual_nk_graph.n_edges} edges and "
            f"{self.dual_nk_graph.n_nodes} vertices. "
            f"The graph contains {len(self.get_smallest_loops())} minimal loops"
        )
