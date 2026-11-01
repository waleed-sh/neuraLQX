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


"""Flat-state helpers for SU(2) spin-network Hilbert spaces.

SU(2) states are already integer local indices: edge doubled-spin labels are
stored first, followed by vertex channel labels. This mixin provides conversion
methods and semantic views over that flat representation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.hilbert.su2.layout import SpinNetworkView
from neuralqx.hilbert.utils import ensure_2d_trailing


class SU2StateMixin:
    """State conversion and semantic views for SU(2) spin networks.

    In this representation the stored physical labels are already the local
    integer indices. Edge entries are doubled-spin labels, and channel entries
    are left-comb recoupling labels. The mixin therefore implements identity
    index conversion with dtype normalization, per-site bound checks, and
    helper views that split a flat state into edge and vertex-channel blocks.

    Concrete spaces must provide ``size``, ``dtype``, ``n_edges``, ``layout``,
    ``site_max_label``, ``graph``, and ``graph_metadata``. The methods are part
    of the public SU(2) Hilbert API through ``SU2GaugeInvariantHilbert``.
    """

    def states_to_local_indices(self, states: Any) -> jax.Array:
        """Returns SU(2) state values as local integer indices.

        Args:
            states: Flat spin-network state array.

        Returns:
            Integer array with the same shape as ``states``. SU(2) doubled-spin
            and channel labels are already local indices, so this method only
            validates shape and normalizes dtype.
        """
        arr = jnp.asarray(states, dtype=jnp.int32)
        ensure_2d_trailing(arr, self.size)
        return arr

    def local_indices_to_states(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array:
        """Converts local integer indices to SU(2) state values.

        Args:
            indices: Local index array whose trailing dimension equals
                ``self.size``.
            dtype: Optional dtype for the returned state values.

        Returns:
            State-value array with the same shape as ``indices``. The values are
            numerically identical to the indices because the spin-network basis
            stores integer labels directly.
        """
        idx = jnp.asarray(indices, dtype=jnp.int32)
        ensure_2d_trailing(idx, self.size)
        out_dtype = self.dtype if dtype is None else dtype
        return idx.astype(out_dtype)

    def local_states_valid(self, states: Any) -> jax.Array:
        """Checks per-site integer bounds without vertex admissibility.

        Args:
            states: Candidate flat spin-network state array.

        Returns:
            Boolean mask over the leading batch axes. The result confirms only
            that every site label is between zero and that site's maximum
            label. Vertex coupling rules are checked by the SU(2) constraint.
        """
        arr, single, batch = ensure_2d_trailing(states, self.size)
        arr = arr.astype(jnp.int32)
        site_max = self.site_max_label[None, :]
        valid = jnp.all((arr >= 0) & (arr <= site_max), axis=-1)
        if single:
            return valid[0]
        return valid.reshape(batch)

    def view(self, states: Any) -> SpinNetworkView:
        """Views flat SU(2) states as edge spins and vertex channels.

        Args:
            states: Flat spin-network state array.

        Returns:
            :class:`SpinNetworkView` whose ``edge_spins`` block contains the
            first ``n_edges`` sites and whose ``vertex_channels`` block contains
            the remaining channel sites.
        """
        arr = jnp.asarray(states)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]}, expected {self.size}."
            )
        return SpinNetworkView(
            edge_spins=arr[..., : self.n_edges],
            vertex_channels=arr[..., self.n_edges :],
        )

    def flatten(self, view: SpinNetworkView | None = None, **blocks: Any) -> jax.Array:
        """Flattens edge-spin and vertex-channel blocks into state arrays.

        Args:
            view: Optional :class:`SpinNetworkView` to flatten.
            **blocks: ``edge_spins`` and ``vertex_channels`` arrays when
                ``view`` is not supplied.

        Returns:
            Flat spin-network state array with edge labels followed by channel
            labels.
        """
        if view is not None:
            edge_spins = view.edge_spins
            vertex_channels = view.vertex_channels
        else:
            edge_spins = blocks.get("edge_spins")
            vertex_channels = blocks.get("vertex_channels")
        if edge_spins is None or vertex_channels is None:
            raise ValueError("edge_spins and vertex_channels are required.")
        return jnp.concatenate(
            [jnp.asarray(edge_spins), jnp.asarray(vertex_channels)], axis=-1
        )

    def edge_spins(self, states: Any) -> jax.Array:
        """Returns the edge doubled-spin block from flat states.

        Args:
            states: Flat spin-network state array.

        Returns:
            Array containing the first ``n_edges`` sites, preserving leading
            batch axes.
        """
        arr = jnp.asarray(states)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]}, expected {self.size}."
            )
        return arr[..., : self.n_edges]

    def vertex_channels(self, states: Any, vertex: Any | None = None) -> jax.Array:
        """Returns all vertex channels or the channels for one vertex.

        Args:
            states: Flat spin-network state array.
            vertex: Optional vertex id or layout position.

        Returns:
            Concatenated channel block when ``vertex`` is ``None``. Otherwise,
            only the channel sites associated with the requested vertex are
            returned.
        """
        arr = jnp.asarray(states)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]}, expected {self.size}."
            )
        if vertex is None:
            return arr[..., self.n_edges :]
        vertex_index = self.layout.vertex_position(vertex)
        sites = self.layout.vertex_channel_sites[vertex_index]
        if not sites:
            return jnp.empty((*arr.shape[:-1], 0), dtype=arr.dtype)
        return jnp.take(arr, jnp.asarray(sites, dtype=jnp.int32), axis=-1)

    def edge_values(self, states: Any, edge: int | Any) -> jax.Array:
        """Returns the doubled-spin label for one graph edge.

        Args:
            states: Flat spin-network state array.
            edge: Edge index or graph edge object accepted by the graph.

        Returns:
            Doubled-spin label for the requested edge, preserving leading batch
            axes.
        """
        edge_index = (
            int(edge) if isinstance(edge, int) else self.graph.edge_to_index(edge)
        )
        return jnp.asarray(states)[..., self.layout.edge_site(edge_index)]

    def local_vertex_state(
        self, states: Any, vertex: Any
    ) -> tuple[jax.Array, jax.Array]:
        """Returns incident edge spins and channel labels for one vertex.

        Args:
            states: Flat spin-network state array.
            vertex: Vertex id or layout position.

        Returns:
            Pair ``(edge_spins, channels)`` for the local vertex environment,
            preserving leading batch axes.
        """
        vertex_index = self.layout.vertex_position(vertex)
        edge_sites = jnp.asarray(
            self.graph_metadata.incident_edges_by_vertex[vertex_index],
            dtype=jnp.int32,
        )
        return jnp.take(jnp.asarray(states), edge_sites, axis=-1), self.vertex_channels(
            states,
            vertex_index,
        )


__all__ = ["SU2StateMixin"]
