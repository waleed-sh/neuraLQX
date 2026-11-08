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


"""Half-ladder stencil resolution for Bojowald-Swiderski operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralqx.graph import HalfLadderGraph
from neuralqx.hilbert.u1.spaces.abstract import AbstractU1Hilbert
from neuralqx.operator.prebuilt.utils import k_vertex
from neuralqx.operator.prebuilt.utils import other_endpoint
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_half_ladder_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.utils.typing import Vertex


@dataclass(frozen=True, slots=True)
class SphericalStencil:
    """Flat-site indices for the neighborhood of a radial half-ladder vertex.

    The central vertex has a transverse edge and two adjacent radial edges.
    The wider neighborhood is needed for the spin connection and its forward
    difference. Missing outer entries are encoded as `-1`. Callers must mask
    those entries rather than interpreting them as Python negative indices.
    """

    vertex: Vertex
    """Radial vertex or its index in `graph.vertices_k`."""

    vertex_index: int
    """Position of the central radial vertex in `graph.vertices_k`."""

    idx_mu: int
    """Flat index of the central transverse edge, or `-1` if absent."""

    idx_mu_m: int
    """Flat index of the left-neighbor transverse edge, or `-1` if absent."""

    idx_mu_p: int
    """Flat index of the right-neighbor transverse edge, or `-1` if absent."""

    idx_mu_mm: int
    """Flat index of the second transverse edge to the left, or `-1` if absent."""

    idx_mu_pp: int
    """Flat index of the second transverse edge to the right, or `-1` if absent."""

    idx_km: int
    """Flat index of the adjacent left radial edge, or `-1` if absent."""

    idx_kp: int
    """Flat index of the adjacent right radial edge, or `-1` if absent."""

    idx_kmm: int
    """Flat index of the second radial edge to the left, or `-1` if absent."""

    idx_kpp: int
    """Flat index of the second radial edge to the right, or `-1` if absent."""

    idx_kppp: int
    """Flat index of the third radial edge to the right, or `-1` if absent."""

    idx_kmmm: int
    """Flat index of the third radial edge to the left, or `-1` if absent."""


def require_spherical_hilbert(hilbert: Any, *, operator: str) -> AbstractU1Hilbert:
    r"""Require the single-copy U(1) space used by the spherical difference equation.

    The radial and transverse labels share one base-edge block. The stencil
    therefore needs a HalfLadderGraph and exactly one gauge copy, in addition
    to the charge-range interface supplied by AbstractU1Hilbert.

    Args:
        hilbert: Candidate Hilbert-space object.
        operator: Name of the requesting spherical operator.

    Returns:
        result (AbstractU1Hilbert): The validated original Hilbert space.

    Raises:
        TypeError: If the object is not a U(1) Hilbert space.
        ValueError: If it has multiple gauge copies or another graph type.
    """
    # Establish the charge-range and state-layout interface first.
    hilbert = require_u1_hilbert(hilbert, operator=operator)
    # Spherical graph indices address a single gauge-copy block.
    require_gauge_dimensions(hilbert, 1, operator=operator)
    # Require the radial-chain accessors used by stencil construction.
    require_half_ladder_graph(hilbert.graph, operator=operator)
    # Retain the validated object shared by the model and its operators.
    return hilbert


def spherical_stencil(graph: HalfLadderGraph, vertex: Any) -> SphericalStencil:
    r"""Resolve the radial neighborhood needed by a spherical vertex constraint.

    At radial position $i$, the transverse label is $\mu_i$ and the two
    adjacent radial labels are $k_{i-1}$ and $k_i$. The stencil records
    transverse sites from $i-2$ through $i+2$ and radial links from $k_{i-3}$
    through $k_{i+2}$. The wider range supports the spin connection and its
    difference without assuming that those outer links exist at a boundary.

    The central transverse edge and the two adjacent radial edges are required.
    Unavailable outer entries use -1. This value is a sentinel, not a Python
    index into the last state component. Kernels must substitute their configured
    boundary data before gathering a missing site. Since spherical models have
    one gauge copy, graph edge indices are also their flat state sites.

    Args:
        graph: Half-ladder graph with an ordered radial chain and transverse edges.
        vertex: Radial vertex or its positional index in `graph.vertices_k`.

    Returns:
        result (SphericalStencil): Immutable site metadata for the central vertex,
            its immediate neighbors, and the available outer neighborhood.

    Raises:
        ValueError: If the selected vertex is not radial or lacks an immediate
            left or right neighbor.
        RuntimeError: If a central transverse or adjacent radial edge cannot be
            resolved uniquely.
    """
    # Resolve both the radial Vertex object and its position along the chain.
    k_v, idx = k_vertex(graph, vertex)
    # The Hamiltonian needs one immediate radial neighbor on either side.
    if idx < 1 or idx > len(graph.vertices_k) - 2:
        # Reject an endpoint before looking up its missing central radial link.
        raise ValueError(
            "Spherical vertex operators require a k-vertex with left and right neighbours."
        )

    # Separate the central transverse edge from the incident radial links.
    mu_edges, _k_edges = graph.get_edges_at_k_vertex(k_v)
    # Use radial-chain orientation to distinguish left and right links.
    left_edges, right_edges = graph.get_kminus_kplus(k_v)
    # Each required central label must correspond to exactly one edge.
    if len(mu_edges) != 1 or len(left_edges) != 1 or len(right_edges) != 1:
        # Fail on ambiguous central incidence instead of choosing an arbitrary edge.
        raise RuntimeError(f"Could not resolve spherical stencil at vertex {k_v!r}.")

    # Follow the adjacent left link regardless of its stored edge direction.
    left_v = other_endpoint(left_edges[0], k_v)
    # Resolve the immediate right radial neighbor in the same way.
    right_v = other_endpoint(right_edges[0], k_v)
    # The second left vertex is optional near the chain boundary.
    left2_v = graph.vertices_k[idx - 2] if idx >= 2 else None
    # Guard the second right vertex before indexing the radial sequence.
    right2_v = graph.vertices_k[idx + 2] if idx + 2 < len(graph.vertices_k) else None
    # A third left vertex determines whether the outermost left link exists.
    left3_v = graph.vertices_k[idx - 3] if idx >= 3 else None
    # A third right vertex similarly determines the outermost right link.
    right3_v = graph.vertices_k[idx + 3] if idx + 3 < len(graph.vertices_k) else None

    # Read the second left radial link only when both of its endpoints exist.
    left_left = graph.get_kminus_kplus(left_v)[0] if left2_v is not None else []
    # Read the second right radial link with the same boundary guard.
    right_right = graph.get_kminus_kplus(right_v)[1] if right2_v is not None else []
    # The third left link is optional spin-connection boundary metadata.
    left_left_left = graph.get_kminus_kplus(left2_v)[0] if left3_v is not None else []
    # The third right link completes the wider finite-difference neighborhood.
    right_right_right = (
        graph.get_kminus_kplus(right2_v)[1] if right3_v is not None else []
    )

    # Package all sites once so kernels need no graph traversal during tracing.
    return SphericalStencil(
        # Keep the resolved central vertex for diagnostics and model bookkeeping.
        vertex=k_v,
        # Record its radial position, independent of overall graph vertex order.
        vertex_index=idx,
        # The central transverse label is required and has already been checked.
        idx_mu=int(graph.edge_to_index(mu_edges[0])),
        # Resolve the transverse label one vertex to the left.
        idx_mu_m=_mu_index_or_missing(graph, left_v),
        # Resolve the transverse label one vertex to the right.
        idx_mu_p=_mu_index_or_missing(graph, right_v),
        # Use the missing sentinel for an unavailable second left transverse label.
        idx_mu_mm=_mu_index_or_missing(graph, left2_v),
        # Use the same sentinel convention on the second right transverse label.
        idx_mu_pp=_mu_index_or_missing(graph, right2_v),
        # The adjacent left radial link is part of the required central stencil.
        idx_km=int(graph.edge_to_index(left_edges[0])),
        # The adjacent right radial link is likewise always present here.
        idx_kp=int(graph.edge_to_index(right_edges[0])),
        # Store the second left radial link or its boundary sentinel.
        idx_kmm=_edge_index_or_missing(graph, left_left),
        # Store the second right radial link or its boundary sentinel.
        idx_kpp=_edge_index_or_missing(graph, right_right),
        # Store the third right radial link needed by the wider difference.
        idx_kppp=_edge_index_or_missing(graph, right_right_right),
        # Store the third left radial link with the same missing-site convention.
        idx_kmmm=_edge_index_or_missing(graph, left_left_left),
    )


def _edge_index_or_missing(graph: HalfLadderGraph, edges: list[Any]) -> int:
    r"""Resolve an optional outer radial link to a site or the missing sentinel.

    Both an empty list and a non-unique edge list are treated as unavailable.
    Required central links are checked separately by `spherical_stencil`.

    Args:
        graph: Half-ladder graph defining edge indices.
        edges: Candidate links for a single outer stencil position.

    Returns:
        result (int): Unique base-edge index, or -1 when the link is unavailable.
    """
    # Only a unique candidate can identify an outer radial site.
    if len(edges) != 1:
        # Mark the unavailable site for explicit boundary handling by the kernel.
        return -1
    # Convert the unique optional edge to its flat single-copy site.
    return int(graph.edge_to_index(edges[0]))


def _mu_index_or_missing(graph: HalfLadderGraph, vertex: Vertex | None) -> int:
    r"""Resolve the transverse label at an optional radial vertex.

    A missing vertex or an ambiguous transverse edge gives the sentinel -1.
    This allows the same stencil structure to describe interior vertices and
    vertices near the outer boundary.

    Args:
        graph: Half-ladder graph supplying transverse-edge incidence.
        vertex: Radial vertex to inspect, or None beyond the radial chain.

    Returns:
        result (int): The unique transverse-edge site, or -1 if unavailable.
    """
    # A vertex beyond the radial chain has no transverse label.
    if vertex is None:
        # Mark the unavailable site for explicit boundary handling by the kernel.
        return -1
    # Inspect transverse incidence only after confirming that the vertex exists.
    mu_edges, _k_edges = graph.get_edges_at_k_vertex(vertex)
    # An absent or non-unique transverse edge cannot define a single label.
    if len(mu_edges) != 1:
        # Mark the unavailable site for explicit boundary handling by the kernel.
        return -1
    # Return the unique transverse site in graph edge order.
    return int(graph.edge_to_index(mu_edges[0]))


__all__ = ["SphericalStencil", "require_spherical_hilbert", "spherical_stencil"]
