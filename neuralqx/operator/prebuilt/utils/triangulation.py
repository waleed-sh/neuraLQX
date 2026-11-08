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


"""WCL Thiemann triangulation term collection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from itertools import permutations
from typing import Any

from neuralqx.hilbert.u1.spaces.abstract import AbstractU1Hilbert
from neuralqx.utils.typing import Vertex

from .graph import edge_emanates_from
from .graph import graph_vertex
from .graph import reduce_first_pair_permutations
from .loops import loop_edges_at_vertex
from .loops import loops_containing_edges
from .loops import parse_loop
from .u1 import copy_sites
from .u1 import edge_index
from .u1 import site_index


@dataclass(frozen=True, slots=True)
class Thiemann2p1Term:
    """One ordered loop-pair contribution to the planar Thiemann constraint.

    The record stores the two segment shifts used in square-root-volume
    differences, the loop action in each gauge copy, and the permutation
    weight. It contains only static metadata and can be reused by JAX kernels.
    """

    weight: float
    """Product of the two loop-pair permutation signs."""

    seg_k_sites: tuple[int, ...]
    """Flat indices of the first segment in every gauge copy."""

    seg_l_sites: tuple[int, ...]
    """Flat indices of the second segment in every gauge copy."""

    seg_k_shift: int
    """Signed inverse-holonomy step along the first segment."""

    seg_l_shift: int
    """Signed inverse-holonomy step along the second segment."""

    loop_sites_by_copy: tuple[tuple[int, ...], ...]
    """Flat loop-edge indices for each gauge copy."""

    loop_directions: tuple[int, ...]
    """Signed steps along the loop relative to stored graph edges."""


@dataclass(frozen=True, slots=True)
class Thiemann3p1Term:
    """One loop-and-segment contribution to the 3+1 Thiemann constraint.

    The coefficient already includes the permutation parity, edge-reorientation
    sign, loop average, and vertex normalization. Each gauge copy applies the
    same oriented loop and its inverse with opposite signs. `segment_shift`
    retains the inverse outward-segment direction after the segment has been
    converted to a canonical edge index. It is independent of the coefficient
    sign and is unchanged when the constraint is adjointed.
    """

    coefficient: float
    """Permutation and reorientation sign divided by the vertex and loop counts."""

    segment_sites: tuple[int, ...]
    """Flat indices of the volume-commutator segment in every copy."""

    segment_shift: int
    """Inverse outward-segment shift, negative for stored outgoing edges and positive for incoming edges."""

    loop_sites_by_copy: tuple[tuple[int, ...], ...]
    """Flat loop-edge indices for each gauge copy."""

    loop_directions: tuple[int, ...]
    """Signed steps along the loop relative to stored graph edges."""


def collect_thiemann_2p1_terms(
    hilbert: AbstractU1Hilbert, vertex: Any
) -> tuple[Thiemann2p1Term, ...]:
    r"""Resolve the loop pairs entering the planar Thiemann constraint.

    For each ordered pair of distinct minimal loops through $v$, the first
    loop supplies a holonomy and the second supplies two segments. A record
    describes the ket contribution

    $$
    w_t U_{\alpha_t,c}D_{k_t,c}D_{l_t,c},\qquad
    D_{s,c}(q)=\sqrt{V_v(q)}-\sqrt{V_v(q-d_s h e_{s,c})}.
    $$

    Here $d_s=+1$ for a stored edge leaving $v$ and $-1$ for an edge entering
    it. The stored segment shifts are $-d_s$. The loop dressing specifies
    charge changes on kets. A row kernel must invert those changes and
    evaluate the diagonal factors at its connected configuration.

    The two-edge permutation signs follow upstream's loop ordering. They
    are independent of the geometric orientation signs used in the volume.
    Each encountered pair is assigned $+1$ and its reverse $-1$. If a later
    loop supplies the reversed pair, its assignment replaces the earlier
    one before any records are emitted. Both orders of each loop pair are
    retained, and no loop-pair average is taken.

    Args:
        hilbert: U(1) space whose graph and copy-major layout define the sites.
            The consuming planar constraint requires three gauge copies.
        vertex: Graph vertex or its index in `graph.vertices`.

    Returns:
        result (tuple[Thiemann2p1Term, ...]): Static records in loop-pair order.
            Empty when fewer than two minimal loops meet the vertex in exactly
            two edges each. Lapse and gauge-component sums belong to the kernel.
    """
    # Resolve index input once so all incidence checks compare Vertex objects.
    vertex = graph_vertex(hilbert.graph, vertex)
    # Dressing retains the ket traversal direction relative to each stored edge.
    loops = tuple(hilbert.graph.dressed_minimal_loops())
    # Accumulate immutable records only after the sign table is complete.
    terms: list[Thiemann2p1Term] = []

    # Choose each pair of distinct loops once in the graph's loop order.
    ordered_loop_pairs = list(combinations(loops, 2))
    # Include the reverse assignment of holonomy loop and commutator loop.
    ordered_loop_pairs += [(right, left) for left, right in ordered_loop_pairs]

    # Keep only pairs that define two incident segments on both loops.
    local_pairs = []
    # This is a combinatorial sign table, not the volume's tangent-sign table.
    epsilons = {}
    # Visit pairs in the same order used by the upstream planar triangulation.
    for loop_i, loop_j in ordered_loop_pairs:
        # The first loop's incident pair fixes its orientation convention.
        pair_i = loop_edges_at_vertex(hilbert, loop_i, vertex)
        # The second loop's incident pair supplies the two commutator segments.
        pair_j = loop_edges_at_vertex(hilbert, loop_j, vertex)
        # A loop missing the vertex cannot contribute to its local constraint.
        if pair_i is None or pair_j is None:
            # Do not enter an incomplete pair in either the records or sign table.
            continue
        # Save the loop and both ordered pairs for the second construction pass.
        local_pairs.append((loop_i, pair_i, pair_j))
        # Assign a reference orientation to each pair encountered in this order.
        for pair in (pair_i, pair_j):
            # The encountered ordering defines the positive permutation.
            epsilons[pair] = +1
            # Exchanging the two entries changes the permutation sign.
            epsilons[pair[::-1]] = -1

    # All sign assignments are now fixed, including shared incident pairs.
    for loop_i, pair_i, pair_j in local_pairs:
        # The weight contains one permutation sign from each loop pair.
        eps_i, eps_j = epsilons[pair_i], epsilons[pair_j]
        # Resolve the first loop to base-edge indices and ket shift directions.
        loop_edges, loop_directions = parse_loop(hilbert, loop_i)
        # Reverse the whole loop if its incident ordering has negative parity.
        loop_directions = tuple(
            int(eps_i) * int(direction) for direction in loop_directions
        )
        # Copy-major offsets give a separate loop action in every U(1) component.
        loop_sites_by_copy = tuple(
            tuple(site_index(hilbert, edge, copy) for edge in loop_edges)
            for copy in range(int(hilbert.gauge_dimensions))
        )
        # The other loop supplies the two diagonal volume differences.
        seg_k, seg_l = pair_j
        # Package the weight and directions without evaluating a charge state.
        terms.append(
            Thiemann2p1Term(
                # Both permutation signs multiply the ket coefficient.
                weight=float(eps_i * eps_j),
                # Each component shifts the same geometric first segment.
                seg_k_sites=tuple(copy_sites(hilbert, seg_k)),
                # The second segment is resolved in the same copy-major layout.
                seg_l_sites=tuple(copy_sites(hilbert, seg_l)),
                # Invert the first segment oriented away from the vertex.
                seg_k_shift=-1 if edge_emanates_from(seg_k, vertex) else +1,
                # An incoming stored edge needs a positive inverse-outward shift.
                seg_l_shift=-1 if edge_emanates_from(seg_l, vertex) else +1,
                # The kernel chooses one component at a time from these sites.
                loop_sites_by_copy=loop_sites_by_copy,
                # Directions describe ket action, independently of row conversion.
                loop_directions=loop_directions,
            )
        )

    # Tuples can be cached as static metadata by the JAX operator.
    return tuple(terms)


def collect_thiemann_3p1_terms(
    hilbert: AbstractU1Hilbert, vertex: Any
) -> tuple[Thiemann3p1Term, ...]:
    r"""Assemble the oriented triples, loop averages, and segment shifts at a vertex.

    List incident edges in `graph.edges` order. For every three-edge subset,
    keep one representative under exchange of the first two entries. With
    subset labels $a<b<c$, the retained triples are $(a,b,c)$, $(a,c,b)$,
    and $(b,c,a)$, with parities $+1,-1,+1$. The third entry always remains
    the commutator segment. It must not be discarded when reducing the pair.

    For each retained triple $(i,j,k)$, orient a minimal loop to leave $v$
    along $i$ and return along $j$. Write $d_e=+1$ for a stored outgoing edge
    and $-1$ for an incoming edge. Reorienting the triple to outgoing,
    incoming, outgoing contributes

    $$
    r_{ijk}=d_i(-d_j)d_k,\qquad
    c_{ijk,\alpha}=\frac{\epsilon_{ijk}r_{ijk}}{T_v L_{ij}}.
    $$

    $L_{ij}$ is the number of minimal loops containing the pair. $T_v$
    counts positive-parity retained triples, including triples whose pair
    has no loop. Thus $T_v=2\binom{n_v}{3}$ for $n_v\geq3$. This is the
    upstream normalization, not the number of emitted records. A pair
    without a loop emits no record and never causes division by zero.
    These permutation signs must not be replaced by the tangent determinant
    signs used by the volume operator.

    The resulting ket contribution is
    $c_{ijk,\alpha}(U_{\alpha,c}-U_{\alpha,c}^\dagger)D_{k,c}$, where
    $D_{k,c}(q)=V_v(q)-V_v(q-d_k h e_{k,c})$. The record retains $-d_k$
    explicitly. An edge index alone loses this orientation for incoming
    segments. The consuming kernel converts this ket action into matrix rows.

    Args:
        hilbert: U(1) space supplying the graph and copy-major site layout.
            The consuming 3+1 constraint requires three gauge copies.
        vertex: Graph vertex or its index in `graph.vertices`.

    Returns:
        result (tuple[Thiemann3p1Term, ...]): One static record per retained
            triple and matching minimal loop. Empty for valence below three
            or when none of the incident pairs belongs to a minimal loop.
    """
    # Resolve a positional vertex argument before comparing graph endpoints.
    vertex = graph_vertex(hilbert.graph, vertex)
    # Use dressed loops so canonical edge indexing does not erase traversal signs.
    loops = tuple(hilbert.graph.dressed_minimal_loops())
    # Global edge order fixes the reference permutation for every three-edge subset.
    incident = tuple(edge for edge in hilbert.graph.edges if vertex in edge.vertices)
    # Pair exchange reverses the loop, but a different third edge is a distinct term.
    triples = reduce_first_pair_permutations(tuple(permutations(incident, 3)))
    # Records will already include both normalization factors needed by the kernel.
    terms: list[Thiemann3p1Term] = []

    # Keep parities aligned with triples rather than indexing by stringified edges.
    parities = []
    # Compute each parity relative to increasing incident-edge order.
    for triple in triples:
        # Gaps in these indices do not affect the parity within the chosen subset.
        indices = tuple(incident.index(edge) for edge in triple)
        # Every inversion contributes one transposition of the reference ordering.
        inversions = sum(
            indices[i] > indices[j] for i in range(3) for j in range(i + 1, 3)
        )
        # Even permutations have sign +1, odd permutations have sign -1.
        parities.append(-1 if inversions % 2 else +1)
    # A lower-valence vertex has no terms, so use a harmless nonzero denominator.
    normalization = max(1, parities.count(+1))

    # Preserve the association between each triple and its permutation sign.
    for (edge_i, edge_j, segment), eps in zip(triples, parities, strict=True):
        # Average only over minimal loops containing this particular incident pair.
        candidate_loops = loops_containing_edges(hilbert, loops, edge_i, edge_j)
        # No loop means no curvature contribution for this triple.
        if not candidate_loops:
            # Skip before dividing by the loop count.
            continue
        # The first loop edge must leave the selected vertex.
        direction_i = +1 if edge_emanates_from(edge_i, vertex) else -1
        # The second loop edge must enter it, giving the extra minus sign below.
        direction_j = +1 if edge_emanates_from(edge_j, vertex) else -1
        # The commutator segment is always oriented away from the vertex.
        direction_k = +1 if edge_emanates_from(segment, vertex) else -1
        # Multiply the signs needed for the outgoing, incoming, outgoing ordering.
        reordering_sign = -direction_i * direction_j * direction_k
        # Fold the parity, reorientation, vertex average, and loop average into one weight.
        coefficient = float(eps * reordering_sign) / (
            normalization * len(candidate_loops)
        )
        # Each matching loop contributes separately with the same averaged weight.
        for loop in candidate_loops:
            # Read the graph's loop dressing before enforcing the chosen local direction.
            loop_edges, loop_directions = parse_loop(hilbert, loop)
            # Locate the first incident edge even if the stored loop starts elsewhere.
            first_position = loop_edges.index(edge_index(hilbert, edge_i))
            # Multiplying every direction by this sign makes the first edge outgoing.
            loop_flip = direction_i * loop_directions[first_position]
            # In U(1), reversing all traversal signs represents the inverse loop.
            loop_directions = tuple(
                loop_flip * direction for direction in loop_directions
            )
            # Translate the same geometric loop into each independent gauge copy.
            loop_sites_by_copy = tuple(
                tuple(site_index(hilbert, edge, copy) for edge in loop_edges)
                for copy in range(int(hilbert.gauge_dimensions))
            )
            # Store directions separately from weights because they affect different factors.
            terms.append(
                Thiemann3p1Term(
                    # The kernel must not divide by the vertex or loop counts again.
                    coefficient=coefficient,
                    # Resolve the commutator segment independently in each component.
                    segment_sites=tuple(copy_sites(hilbert, segment)),
                    # The inverse outward holonomy raises an incoming stored edge.
                    segment_shift=-direction_k,
                    # Every component receives the same oriented loop geometry.
                    loop_sites_by_copy=loop_sites_by_copy,
                    # Plain integers keep the record independent of traced JAX values.
                    loop_directions=tuple(
                        int(direction) for direction in loop_directions
                    ),
                )
            )

    # The immutable tuple is shared by forward and mathematical-adjoint kernels.
    return tuple(terms)


def admissible_thiemann_3p1_vertices(hilbert: AbstractU1Hilbert) -> tuple[Vertex, ...]:
    """Select vertices with an actual loop-and-segment contribution.

    Valence alone is insufficient. A vertex with three or more incident edges
    still gives a zero constraint if no incident pair occurs in a minimal
    loop. This selection uses the same construction as the vertex operator,
    so the model's quadratic list contains exactly its contributing vertices.

    Args:
        hilbert: U(1) space on the graph being triangulated.

    Returns:
        result (tuple[Vertex, ...]): Contributing vertices in graph vertex order.
    """
    # Preserve graph order when filtering the model's vertex list.
    out = []
    # Test every vertex with the actual triangulation rule, not a valence shortcut.
    for vertex in hilbert.graph.vertices:
        # At least one emitted record is enough to establish a local contribution.
        if collect_thiemann_3p1_terms(hilbert, vertex):
            # Keep the graph's Vertex object for later operator construction.
            out.append(vertex)
    # Return immutable metadata suitable for caching in the model facade.
    return tuple(out)


__all__ = [
    "Thiemann2p1Term",
    "Thiemann3p1Term",
    "admissible_thiemann_3p1_vertices",
    "collect_thiemann_2p1_terms",
    "collect_thiemann_3p1_terms",
]
