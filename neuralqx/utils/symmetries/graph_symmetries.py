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

from typing import Iterable, List, Dict, Optional, Tuple, Sequence
from collections import defaultdict, Counter
import itertools

from .types import Edge, Automorphism, Cycle, Permutation, _canon_edge
from .utils import _vmap_name, _product_of_bucket_permutations


class GraphSymmetries:
    """
    Compute permutations, cycles, and automorphisms of an oriented multigraph with parallel edges.

    The input edges are *oriented* and may include *parallel edges* distinguished by an integer
    key. For structural computations (permutations / cycles / automorphisms), edge orientation is
    ignored by converting edges to a canonical, orientation-agnostic form (via ``_canon_edge``).
    However, the original oriented domain is preserved for indexing and for the concrete
    permutation objects returned.

    Key permutations:
        If ``allow_key_permutations`` is enabled, automorphisms may additionally permute the keys
        within each unordered vertex pair, i.e. within each multiedge bucket connecting the same
        two vertices. This can greatly increase the number of automorphisms, so a cap
        ``max_total_key_permutations`` is used to avoid combinatorial blow-up.

    The class lazily computes and caches:

    - ``permutations``: edge permutations (up to a size cap)
    - ``cycles``: cyclic shifts of the canonical edge ordering
    - ``automorphisms``: structure-preserving vertex maps (plus optional within-pair key
      permutations)
    - ``graph_symmetries``: concatenation of the above

    :param edges: Iterable of oriented keyed edges. Orientation is ignored for structure but
                  preserved in the returned symmetry objects' domain.
    :param max_permutation_edges: If the number of unique canonical edges exceeds this value,
                                  ``permutations`` returns only the identity (avoids factorial
                                  growth).
    :param allow_key_permutations: If True, automorphisms include permutations of keys within each
                                   unordered vertex pair when there are multiple parallel edges.
    :param max_total_key_permutations: Upper bound on the total number of combinations of per-pair
                                       key permutations for a given vertex automorphism. If this
                                       cap is exceeded, key permutations fall back to identity per
                                       pair.
    :raises ValueError: If no edges are provided, or if duplicate keys occur within an unordered
                        vertex pair.
    """

    def __init__(
        self,
        edges: Iterable[Edge],
        *,
        max_permutation_edges: int = 10,
        allow_key_permutations: bool = True,
        max_total_key_permutations: int = 16384,
    ):
        """
        Construct a :class:`GraphSymmetries` object from an oriented keyed multigraph.

        This initializer:

        - deduplicates the oriented input edges while preserving order,
        - builds the canonical (orientation-agnostic) edge set for structural computations,
        - indexes canonical edges deterministically,
        - extracts and sorts the vertex set,
        - validates that keys are unique within each unordered vertex pair,
        - precomputes permutation-invariant vertex invariants for pruning the automorphism search,
        - initializes lazy caches for permutations, cycles, automorphisms, and their concatenation.

        :param edges: Oriented keyed edges. Orientation is ignored for structure but preserved in the
                      domain of returned symmetry objects.
        :param max_permutation_edges: Maximum number of canonical edges for which full ``n!`` edge
                                      permutations are enumerated. If exceeded, only identity is
                                      provided.
        :param allow_key_permutations: If True, allow key permutations within each unordered vertex
                                       pair when enumerating automorphisms.
        :param max_total_key_permutations: Cap on total per-vertex-map key-permutation combinations.
        :raises ValueError: If ``edges`` is empty, or if duplicate keys are found within a vertex pair.
        """

        # preserve oriented edges as provided, deduping while preserving order
        raw = list(edges)

        # check for empty edges
        if not raw:
            raise ValueError("No edges provided.")

        # add to the domain, dedup but preserve order
        seen = set()
        self._domain_edges: List[Edge] = []
        for e in raw:
            if e not in seen:
                self._domain_edges.append(e)
                seen.add(e)

        # canonical unique edges (structure only)
        canon_edges = {_canon_edge(e) for e in self._domain_edges}

        # edges sorted in deterministic order
        self._edges: List[Edge] = sorted(canon_edges)

        # indexing
        self._edge_index: Dict[Edge, int] = {e: i for i, e in enumerate(self._edges)}

        # local copies of the variables
        self._max_perm_edges = max_permutation_edges
        self._allow_key_permutations = allow_key_permutations
        self._max_total_key_permutations = max_total_key_permutations

        # now deal with vertices
        vs = set()
        for u, v, _ in self._edges:
            vs.add(u)
            vs.add(v)

        # again, sort in deterministic order
        self._vertices: List[int] = sorted(vs)

        # unordered pair -> sorted keys (validate uniqueness per pair)
        self._pair_keys: Dict[frozenset, List[int]] = defaultdict(list)
        for u, v, k in self._edges:
            self._pair_keys[frozenset({u, v})].append(k)
        for pair, keys in self._pair_keys.items():
            keys.sort()
            if len(keys) != len(set(keys)):
                raise ValueError(
                    f"Duplicate keys found for pair {tuple(pair)}. "
                    f"Keys must be unique within each unordered vertex pair."
                )

        # fast lookup per pair: key -> index
        self._pair_key_index: Dict[frozenset, Dict[int, int]] = {
            pair: {k: i for i, k in enumerate(keys)}
            for pair, keys in self._pair_keys.items()
        }

        # invariants for automorphism pruning
        self._deg_multiset_by_vertex: Dict[int, Counter] = (
            self._compute_vertex_invariants()
        )

        # lazy caches
        self._permutations: Optional[List[Permutation]] = None
        self._cycles: Optional[List[Cycle]] = None
        self._automorphisms: Optional[List[Automorphism]] = None
        self._graph_symmetries: Optional[List[object]] = None

    @property
    def permutations(self) -> List[Permutation]:
        """
        Lazily compute and return edge permutations on the canonical edge set.

        The returned list always includes the identity permutation. If the number of canonical edges
        ``n`` is at most ``max_permutation_edges``, all ``n!`` permutations are generated (excluding the
        identity which is already included). If ``n`` exceeds the cap, only the identity permutation is
        returned to avoid combinatorial blow-up.

        :returns: List of :class:`~neuralqx.utils.symmetries.types.Permutation` objects acting on the stored domain.
        """

        # if already computed and cached, use that
        if self._permutations is not None:
            return self._permutations

        # if not, then compute and cache
        n = len(self._edges)
        res: List[Permutation] = []

        # identity permutation
        id_map = {e: e for e in self._edges}
        res.append(
            Permutation(mapping_canon=id_map, domain=self._domain_edges, name="id")
        )

        # here we only permute less than ``max_perm_edges`` many edges
        if n <= self._max_perm_edges:

            # all n! permutations (skip identity already added)
            for p_idx, perm in enumerate(
                itertools.permutations(self._edges, n), start=1
            ):
                # skip the identity
                if list(perm) == self._edges:
                    continue

                mapping = {self._edges[i]: perm[i] for i in range(n)}
                res.append(
                    Permutation(
                        mapping_canon=mapping,
                        domain=self._domain_edges,
                        name=f"perm_{p_idx}",
                    )
                )

        # cache
        self._permutations = res

        return self._permutations

    @property
    def cycles(self) -> List[Cycle]:
        """
        Lazily compute and return cyclic shifts of the canonical edge ordering.

        For ``n`` canonical edges, this produces ``n`` cycles, where the ``k``-th cycle maps edge
        position ``i`` to position ``(i + k) mod n`` in the canonical ordering.

        :returns: List of :class:`~neuralqx.utils.symmetries.types.Cycle` objects.
        """

        # if cycles are computed and cached, use those
        if self._cycles is not None:
            return self._cycles

        # if not, compute and cache
        n = len(self._edges)

        cyc: List[Cycle] = []

        # loop through as many edges, and cycle them each time
        for k in range(n):
            shifted = self._edges[k:] + self._edges[:k]
            mapping = {self._edges[i]: shifted[i] for i in range(n)}
            cyc.append(Cycle(mapping_canon=mapping, domain=self._domain_edges, shift=k))

        # cache
        self._cycles = cyc

        return self._cycles

    @property
    def automorphisms(self) -> List[Automorphism]:
        """
        Lazily compute and return automorphisms of the multigraph.

        Automorphisms are computed by enumerating candidate vertex permutations subject to pruning:

        - Vertices are bucketed by a permutation-invariant signature derived from incident multiedge
          structure (see :meth:`_compute_vertex_invariants`).
        - Only within-bucket vertex permutations are combined (cartesian product across buckets).
        - Each candidate vertex map is checked for structure preservation (including key multisets per
          unordered pair).

        If ``allow_key_permutations`` is True, then for each vertex automorphism, all admissible
        permutations of parallel-edge keys within each mapped unordered vertex pair are also enumerated.
        The total number of key-permutation combinations is capped by ``max_total_key_permutations``,
        if exceeded, key permutations fall back to the identity per pair.

        The identity automorphism is guaranteed to be included.

        :returns: List of :class:`~neuralqx.utils.symmetries.types.Automorphism` objects.
        """

        # if already cached, use that
        if self._automorphisms is not None:
            return self._automorphisms

        # if not, compute them
        autos: List[Automorphism] = []

        # bucket vertices by permutation-invariant signatures to prune search
        buckets: Dict[Tuple, List[int]] = defaultdict(list)

        for v in self._vertices:
            buckets[self._vertex_signature(v)].append(v)

        bucket_lists = [sorted(b) for b in buckets.values()]

        # for each product of within-bucket permutations
        for within_maps in _product_of_bucket_permutations(bucket_lists):
            # note, this is not related to anything that has to do with jax.vmap!
            vmap = {}
            for src_bucket, permuted in zip(bucket_lists, within_maps):
                for s, t in zip(src_bucket, permuted):
                    vmap[s] = t
            if not self._is_automorphism(vmap):
                continue

            # unordered pairs in stable order and their images
            pairs: List[frozenset] = list(self._pair_keys.keys())

            # deterministic order
            pairs.sort(key=lambda p: tuple(sorted(p)))
            pair_to_idx = {p: i for i, p in enumerate(pairs)}

            mapped_pairs = [frozenset({vmap[a], vmap[b]}) for a, b in map(tuple, pairs)]

            # source/target key lists (sorted)
            src_keys_lists = [self._pair_keys[p] for p in pairs]
            tgt_keys_lists = [self._pair_keys[mp] for mp in mapped_pairs]

            # per-pair permutations of keys (or identity only)
            per_pair_perms: List[List[List[int]]] = []
            total = 1
            for keys in tgt_keys_lists:
                m = len(keys)
                if self._allow_key_permutations and m > 1:
                    perms = list(itertools.permutations(keys))
                else:
                    perms = [tuple(keys)]
                plist = [list(p) for p in perms]
                per_pair_perms.append(plist)
                total *= len(plist)

            # cap explosion
            if total > self._max_total_key_permutations:
                per_pair_perms = [
                    [tgt_keys_lists[i]] for i in range(len(tgt_keys_lists))
                ]

            # precompute key -> index for source pairs
            src_key_index = [self._pair_key_index[p] for p in pairs]

            # for each combination of per-pair key permutations, emit an automorphism
            for combo in itertools.product(*per_pair_perms):

                # build canonical edge map
                edge_map_canon: Dict[Edge, Edge] = {}

                for u, v, k in self._edges:
                    pair = frozenset({u, v})
                    i = pair_to_idx[pair]

                    # position in sorted src keys
                    idx_in_pair = src_key_index[i][k]

                    # permuted key at that position
                    k_prime = combo[i][idx_in_pair]

                    uu, vv = vmap[u], vmap[v]
                    edge_map_canon[(u, v, k)] = _canon_edge((uu, vv, k_prime))

                autos.append(
                    Automorphism(
                        vertex_map=vmap,
                        edge_map_canon=edge_map_canon,
                        domain=self._domain_edges,
                        name=_vmap_name(vmap),
                    )
                )

        # ensure identity present
        if not any(all(v == a.vertex_map[v] for v in self._vertices) for a in autos):
            id_vmap = {v: v for v in self._vertices}
            edge_map_canon = {e: e for e in self._edges}
            autos.append(
                Automorphism(
                    vertex_map=id_vmap,
                    edge_map_canon=edge_map_canon,
                    domain=self._domain_edges,
                    name="auto_id",
                )
            )

        autos.sort(key=lambda a: (a.name, tuple(sorted(a.edge_map_canon.items()))))

        # cache
        self._automorphisms = autos
        return self._automorphisms

    @property
    def graph_symmetries(self) -> List[object]:
        """
        Return all computed graph symmetries as a single list.

        This is the concatenation of ``permutations``, ``cycles``, and ``automorphisms`` in that order.
        The result is cached after first construction.

        :returns: List containing permutations, cycles, and automorphisms.
        """

        # if not cached, compute
        if self._graph_symmetries is None:

            # this is essentially concatenating all other symmetries
            self._graph_symmetries = (
                list(self.permutations) + list(self.cycles) + list(self.automorphisms)
            )

        return self._graph_symmetries

    def _compute_vertex_invariants(self) -> Dict[int, Counter]:
        """
        Compute permutation-invariant incident-structure summaries for each vertex.

        For each vertex ``v``, this builds a multiset signature of incident neighbour-types by
        considering, for each other vertex ``w``, whether there is an unordered pair ``{v, w}`` and, if
        so, collecting the sorted key tuple of edges between them. Each neighbour contributes a record
        ``(multiplicity, keys_tuple)``, where:

        - ``multiplicity`` is the number of parallel edges between ``v`` and ``w``
        - ``keys_tuple`` is the sorted tuple of their keys

        These records are accumulated into a :class:`collections.Counter`, yielding a signature that is
        invariant under vertex relabeling and useful for bucketing/pruning automorphism search.

        :returns: Dictionary mapping each vertex to a Counter over neighbour-type signatures.
        """

        per_pair = defaultdict(list)

        for u, v, k in self._edges:
            per_pair[frozenset({u, v})].append(k)

        for pair in per_pair:
            per_pair[pair].sort()

        inv: Dict[int, Counter] = {}

        for v in self._vertices:
            sigs = []
            for w in self._vertices:
                if w == v:
                    continue
                pair = frozenset({v, w})
                if pair in per_pair:
                    keys = per_pair[pair]
                    sigs.append((len(keys), tuple(keys)))
            inv[v] = Counter(sigs)

        return inv

    def _vertex_signature(self, v: int) -> Tuple:
        """
        Return a stable, hashable signature for a vertex based on precomputed invariants.

        The signature is derived from ``self._deg_multiset_by_vertex[v]`` and expanded into a sorted
        tuple with multiplicities, making it suitable for grouping vertices into permutation buckets.

        :param v: Vertex id.
        :returns: Hashable tuple signature used for bucketing vertices.
        """

        c = self._deg_multiset_by_vertex[v]
        return tuple(sorted((m, ks) for (m, ks), cnt in c.items() for _ in range(cnt)))

    def _is_automorphism(self, vmap: Dict[int, int]) -> bool:
        """
        Check whether a proposed vertex map preserves multigraph structure.

        The map must be a permutation of the current vertex set. Additionally, for every unordered
        vertex pair in the source graph, the mapped unordered pair must exist and have exactly the
        same sorted key list (i.e. the same multiedge multiplicity and key multiset).

        :param vmap: Candidate vertex mapping ``{src_vertex: dst_vertex}``.
        :returns: True if ``vmap`` is an automorphism of the keyed multigraph, otherwise False.
        """

        # check vmap is a permutation on current vertex set
        if set(vmap.keys()) != set(self._vertices) or set(vmap.values()) != set(
            self._vertices
        ):
            return False

        # preserve multigraph structure with key multisets
        for pair, keys in self._pair_keys.items():
            u, v = tuple(pair)
            uu, vv = vmap[u], vmap[v]
            tgt = frozenset({uu, vv})
            tgt_keys = self._pair_keys.get(tgt)
            if tgt_keys is None or tgt_keys != keys:
                return False
        return True

    @property
    def edges(self) -> List[Edge]:
        """
        Return canonical, orientation-agnostic unique edges in deterministic order.

        These edges represent the structural edge set used for symmetry computations. To iterate edges
        in a way that respects the original oriented input, iterate over the emitted
        Permutation/Cycle/Automorphism objects (which carry ``domain`` built from the oriented input).

        :returns: List of canonical edges.
        """

        return list(self._edges)

    @property
    def vertices(self) -> List[int]:
        """
        Return the sorted list of vertices present in the canonical edge set.

        :returns: List of vertex ids in deterministic order.
        """

        return list(self._vertices)

    def make_permutation_from_order(
        self, new_order: Sequence[Edge], name: str = "custom"
    ) -> Permutation:
        """
        Build a :class:`Permutation` from a provided ordering of the canonical edge set.

        The provided ``new_order`` must be a permutation of :attr:`edges` up to canonicalization.
        Oriented versions are allowed, only their canonical forms are used for validity and mapping.

        :param new_order: Sequence of edges representing the desired order, must match the current
                          canonical edge set up to permutation.
        :param name: Name assigned to the resulting permutation. Defaults to ``"custom"``.
        :returns: A :class:`~neuralqx.utils.symmetries.types.Permutation` mapping the current canonical ordering to
                  ``new_order``.
        :raises ValueError: If ``new_order`` is not a permutation of the canonical edge set.
        """

        canon_new = [_canon_edge(e) for e in new_order]

        if sorted(canon_new) != self._edges:
            raise ValueError(
                "new_order must be a permutation of GraphSymmetries.edges (canonical)."
            )

        mapping = {self._edges[i]: canon_new[i] for i in range(len(self._edges))}
        return Permutation(mapping_canon=mapping, domain=self._domain_edges, name=name)

    def __repr__(self):
        return (
            f"GraphSymmetries("
            f"edges={self._domain_edges}, "
            f"#Permutations = {len(self.permutations)}, "
            f"#Cycles = {len(self.cycles)}, "
            f"#Automorphisms = {len(self.automorphisms)}"
            f")"
        )
