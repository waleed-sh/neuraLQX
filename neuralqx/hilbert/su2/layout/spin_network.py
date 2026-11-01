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


"""Flat spin-network layout for gauge-invariant SU(2) states.

The layout maps graph edges and vertex recoupling channels into flat state
sites. Edge doubled-spin labels are stored first, followed by left-comb channel
labels grouped by vertex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def channel_max_label(two_j_max: int, valence: int, channel_index: int) -> int:
    """Returns a conservative doubled-spin upper bound for one channel.

    Args:
        two_j_max: Maximum doubled-spin label allowed on graph edges.
        valence: Number of incident edges at the vertex.
        channel_index: Left-comb channel index.

    Returns:
        Maximum doubled-spin label for the requested channel.
    """
    cutoff = int(two_j_max)
    degree = int(valence)
    slot = int(channel_index)
    if cutoff < 0:
        raise ValueError("two_j_max must be non-negative.")
    if degree < 0:
        raise ValueError("valence must be non-negative.")
    if slot < 0 or slot >= max(degree - 3, 0):
        raise IndexError(f"channel_index={slot} out of range for valence {degree}.")
    left_legs = slot + 2
    right_legs = degree - slot - 2
    return min(left_legs, right_legs) * cutoff


@dataclass(frozen=True)
class SpinNetworkCoord:
    """Semantic coordinate of one flat SU(2) state site.

    Attributes:
        kind: Site kind, either ``"edge"`` or ``"channel"``.
        edge_index: Edge index when ``kind`` is ``"edge"``.
        vertex_index: Vertex position when ``kind`` is ``"channel"``.
        vertex_id: Original graph vertex id when ``kind`` is ``"channel"``.
        channel_index: Channel index when ``kind`` is ``"channel"``.
    """

    kind: str
    """Site kind, either ``"edge"`` or ``"channel"``."""

    edge_index: int | None = None
    """Edge index when ``kind`` is ``"edge"``."""

    vertex_index: int | None = None
    """Vertex position when ``kind`` is ``"channel"``."""

    vertex_id: Any | None = None
    """Original graph vertex id when ``kind`` is ``"channel"``."""

    channel_index: int | None = None
    """Channel index when ``kind`` is ``"channel"``."""


@dataclass(frozen=True)
class SU2SiteInfo:
    """Rich metadata for one flat SU(2) spin-network site.

    Attributes:
        site: Flat site index.
        kind: Site kind, either ``"edge"`` or ``"channel"``.
        max_label: Maximum allowed integer label at the site.
        edge_index: Edge index when ``kind`` is ``"edge"``.
        vertex_index: Vertex position when ``kind`` is ``"channel"``.
        vertex_id: Original graph vertex id when ``kind`` is ``"channel"``.
        channel_index: Channel index when ``kind`` is ``"channel"``.
    """

    site: int
    """Flat site index."""

    kind: str
    """Site kind, either ``"edge"`` or ``"channel"``."""

    max_label: int
    """Maximum allowed integer label at the site."""

    edge_index: int | None = None
    """Edge index when ``kind`` is ``"edge"``."""

    vertex_index: int | None = None
    """Vertex position when ``kind`` is ``"channel"``."""

    vertex_id: Any | None = None
    """Original graph vertex id when ``kind`` is ``"channel"``."""

    channel_index: int | None = None
    """Channel index when ``kind`` is ``"channel"``."""


@dataclass(frozen=True)
class SpinNetworkView:
    """Semantic view of a flat SU(2) state array.

    The view splits the flat spin-network layout into the two blocks that users
    usually reason about directly. ``edge_spins`` contains doubled-spin labels
    for graph edges, while ``vertex_channels`` contains the concatenated
    left-comb recoupling-channel labels allocated to graph vertices.

    The class is a lightweight frozen dataclass and does not validate
    admissibility. It is returned by ``SU2GaugeInvariantHilbert.view`` and can
    be passed back to ``flatten`` when callers want to rebuild flat state
    arrays after working with semantic blocks.

    Attributes:
        edge_spins: Edge doubled-spin labels.
        vertex_channels: Vertex recoupling-channel labels.
    """

    edge_spins: Any
    """Edge doubled-spin labels."""

    vertex_channels: Any
    """Vertex recoupling-channel labels."""


class SpinNetworkLayout:
    """Flat layout ``[edge two_j labels | vertex left-comb channels]``.

    Attributes:
        n_edges: Number of graph edge spin sites.
        vertex_ids: Graph vertex identifiers in layout order.
        valence_by_vertex: Incident-edge count for each vertex.
        two_j_max: Maximum doubled-spin label allowed on graph edges.
        n_vertices: Number of graph vertices in the layout.
        edge_sites: Flat sites containing edge doubled-spin labels.
        vertex_num_channels: Number of channel sites allocated to each vertex.
        n_channel_sites: Total number of vertex channel sites.
        channel_sites: Flat sites containing vertex channel labels.
        vertex_channel_sites: Channel-site tuples for every vertex.
        vertex_channel_max_labels: Maximum channel labels for every vertex.
        size: Total flat state size.
        site_max_label_tuple: Maximum label at every flat site.
        site_kind_tuple: Integer kind tag at every flat site.
    """

    n_edges: int
    """Number of graph edge spin sites."""

    vertex_ids: tuple[Any, ...]
    """Graph vertex identifiers in layout order."""

    valence_by_vertex: tuple[int, ...]
    """Incident-edge count for each vertex."""

    two_j_max: int
    """Maximum doubled-spin label allowed on graph edges."""

    n_vertices: int
    """Number of graph vertices in the layout."""

    edge_sites: tuple[int, ...]
    """Flat sites containing edge doubled-spin labels."""

    vertex_num_channels: tuple[int, ...]
    """Number of channel sites allocated to each vertex."""

    n_channel_sites: int
    """Total number of vertex channel sites."""

    channel_sites: tuple[int, ...]
    """Flat sites containing vertex channel labels."""

    vertex_channel_sites: tuple[tuple[int, ...], ...]
    """Channel-site tuples for every vertex."""

    vertex_channel_max_labels: tuple[tuple[int, ...], ...]
    """Maximum channel labels for every vertex."""

    size: int
    """Total flat state size."""

    site_max_label_tuple: tuple[int, ...]
    """Maximum label at every flat site."""

    site_kind_tuple: tuple[int, ...]
    """Integer kind tag at every flat site."""

    def __init__(
        self,
        *,
        n_edges: int,
        vertex_ids: tuple[Any, ...],
        valence_by_vertex: tuple[int, ...],
        two_j_max: int,
    ) -> None:
        """Builds the flat spin-network layout from graph metadata.

        Args:
            n_edges: Number of edge spin sites to allocate first in the flat
                state.
            vertex_ids: Graph vertex identifiers in layout order.
            valence_by_vertex: Incident-edge count for each vertex.
            two_j_max: Maximum doubled-spin label allowed on graph edges.

        Raises:
            ValueError: If edge count, cutoff, or vertex metadata are invalid.
        """
        if isinstance(n_edges, bool) or int(n_edges) < 0:
            raise ValueError("n_edges must be a non-negative integer.")
        if isinstance(two_j_max, bool) or int(two_j_max) < 0:
            raise ValueError("two_j_max must be a non-negative integer.")
        if len(vertex_ids) != len(valence_by_vertex):
            raise ValueError(
                "vertex_ids and valence_by_vertex must have matching length."
            )

        self.n_edges = int(n_edges)
        self.vertex_ids = tuple(vertex_ids)
        self.valence_by_vertex = tuple(int(v) for v in valence_by_vertex)
        self.two_j_max = int(two_j_max)
        self.n_vertices = len(self.vertex_ids)
        self.edge_sites = tuple(range(self.n_edges))
        self.vertex_num_channels = tuple(max(v - 3, 0) for v in self.valence_by_vertex)

        channel_sites: list[int] = []
        vertex_channel_sites: list[tuple[int, ...]] = []
        vertex_channel_max_labels: list[tuple[int, ...]] = []
        coords: list[SpinNetworkCoord] = [
            SpinNetworkCoord(kind="edge", edge_index=edge)
            for edge in range(self.n_edges)
        ]
        maxima: list[int] = [self.two_j_max] * self.n_edges
        cursor = self.n_edges
        for vertex_index, (vertex_id, valence, n_channels) in enumerate(
            zip(
                self.vertex_ids,
                self.valence_by_vertex,
                self.vertex_num_channels,
                strict=True,
            )
        ):
            sites: list[int] = []
            max_labels: list[int] = []
            for channel_index in range(n_channels):
                max_label = channel_max_label(self.two_j_max, valence, channel_index)
                sites.append(cursor)
                channel_sites.append(cursor)
                max_labels.append(max_label)
                maxima.append(max_label)
                coords.append(
                    SpinNetworkCoord(
                        kind="channel",
                        vertex_index=vertex_index,
                        vertex_id=vertex_id,
                        channel_index=channel_index,
                    )
                )
                cursor += 1
            vertex_channel_sites.append(tuple(sites))
            vertex_channel_max_labels.append(tuple(max_labels))

        self.n_channel_sites = len(channel_sites)
        self.channel_sites = tuple(channel_sites)
        self.vertex_channel_sites = tuple(vertex_channel_sites)
        self.vertex_channel_max_labels = tuple(vertex_channel_max_labels)
        self.size = self.n_edges + self.n_channel_sites
        self.site_max_label_tuple = tuple(maxima)
        self.site_kind_tuple = tuple(
            0 if coord.kind == "edge" else 1 for coord in coords
        )
        self._coords = tuple(coords)
        self._vertex_position_by_id = {
            vertex_id: idx for idx, vertex_id in enumerate(vertex_ids)
        }

    def __hash__(self) -> int:
        """Returns a structural hash for the spin-network layout."""
        return hash(
            (
                type(self),
                self.n_edges,
                self.vertex_ids,
                self.valence_by_vertex,
                self.two_j_max,
            )
        )

    @property
    def site_max_label(self) -> jax.Array:
        """Per-site maximum doubled-spin label."""
        return jnp.asarray(self.site_max_label_tuple, dtype=jnp.int32)

    @property
    def site_kind(self) -> jax.Array:
        """Per-site kind: ``0`` edge spin, ``1`` recoupling channel."""
        return jnp.asarray(self.site_kind_tuple, dtype=jnp.int32)

    def validate_site(self, site: int) -> None:
        """Validates that a flat site index is inside this layout.

        Args:
            site: Candidate flat site index.

        Raises:
            IndexError: If ``site`` is negative or greater than or equal to the
            flat layout size.
        """
        site_i = int(site)
        if site_i < 0 or site_i >= self.size:
            raise IndexError(f"site={site_i} out of range [0,{self.size}).")

    def coord_of(self, site: int) -> SpinNetworkCoord:
        """Decodes a flat site index into a semantic coordinate.

        Args:
            site: Flat site index in the spin-network state.

        Returns:
            Coordinate describing whether the site stores an edge spin or a
            vertex recoupling channel, including the associated graph entity.
        """
        self.validate_site(site)
        return self._coords[int(site)]

    def edge_site(self, edge_index: int) -> int:
        """Returns the flat site storing one edge's doubled-spin label.

        Args:
            edge_index: Zero-based edge index in the graph's edge order.

        Returns:
            Flat site index of that edge label.

        Raises:
            IndexError: If ``edge_index`` is outside the allocated edge range.
        """
        edge_i = int(edge_index)
        if edge_i < 0 or edge_i >= self.n_edges:
            raise IndexError(f"edge_index={edge_i} out of range [0,{self.n_edges}).")
        return edge_i

    def channel_site(self, vertex: int | Any, channel_index: int) -> int:
        """Returns the flat site for a vertex recoupling channel.

        Args:
            vertex: Vertex id or integer vertex position.
            channel_index: Zero-based left-comb channel index at that vertex.

        Returns:
            Flat site index of the requested channel.

        Raises:
            IndexError: If the vertex cannot be resolved or the channel index
            is outside the channel count for that vertex.
        """
        vertex_index = self.vertex_position(vertex)
        slot = int(channel_index)
        channels = self.vertex_channel_sites[vertex_index]
        if slot < 0 or slot >= len(channels):
            raise IndexError(
                f"channel_index={slot} out of range [0,{len(channels)}) "
                f"for vertex {self.vertex_ids[vertex_index]!r}."
            )
        return channels[slot]

    def vertex_position(self, vertex: int | Any) -> int:
        """Resolves a graph vertex id or integer layout position.

        Args:
            vertex: Original graph vertex id or zero-based vertex position.

        Returns:
            Integer vertex position used by this spin-network layout.

        Raises:
            IndexError: If ``vertex`` is neither a known id nor a valid
            position.
        """
        if vertex in self._vertex_position_by_id:
            return self._vertex_position_by_id[vertex]
        if isinstance(vertex, int) and not isinstance(vertex, bool):
            vertex_i = int(vertex)
            if 0 <= vertex_i < self.n_vertices:
                return vertex_i
        raise IndexError(f"Unknown SU(2) vertex {vertex!r}.")

    def site_info(self, site: int) -> SU2SiteInfo:
        """Returns rich semantic metadata for one flat site.

        Args:
            site: Flat site index in the spin-network state.

        Returns:
            :class:`SU2SiteInfo` describing the site kind, maximum label, and
            associated edge or vertex-channel coordinates.
        """
        coord = self.coord_of(site)
        return SU2SiteInfo(
            site=int(site),
            kind=coord.kind,
            max_label=self.site_max_label_tuple[int(site)],
            edge_index=coord.edge_index,
            vertex_index=coord.vertex_index,
            vertex_id=coord.vertex_id,
            channel_index=coord.channel_index,
        )


__all__ = [
    "SU2SiteInfo",
    "SpinNetworkCoord",
    "SpinNetworkLayout",
    "SpinNetworkView",
    "channel_max_label",
]
