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


"""Flat-state encoding and semantic views for U(1) Hilbert spaces.

U(1) spaces store charges in flat component-major arrays. This mixin provides
conversion between charge values and local indices, graph layout views, and a
gauge-component view shaped by ``gauge_dimensions`` and graph edge count.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.hilbert.layout import HilbertStateView


class U1StateMixin:
    """State conversion and semantic views for U(1) flat arrays.

    U(1) state arrays store scalar charge values in the flat layout created by
    ``GraphHilbertLayout``. This mixin converts those charge values to local
    integer indices, converts indices back to charge values, validates local
    membership, and exposes graph-aware views over edges and gauge components.

    Concrete spaces must provide ``size``, ``dtype``, ``local_space``,
    ``layout``, ``gauge_dimensions``, and ``tiny_size``. The methods are
    inherited by both unconstrained and gauge-fixed U(1) spaces, so they form a
    public part of the U(1) Hilbert API.
    """

    def states_to_local_indices(self, states: Any) -> jax.Array:
        """Maps U(1) charge values to site-local integer indices.

        Args:
            states: Flat U(1) state array whose trailing dimension equals
                ``self.size``.

        Returns:
            Integer array with the same shape as ``states``. Each entry is the
            zero-based position of the corresponding charge in the shared local
            charge range.
        """
        arr = jnp.asarray(states)
        _validate_trailing_size(arr, self.size, "State")
        return self.local_space.values_to_indices(arr)

    def local_indices_to_states(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array:
        """Maps site-local integer indices back to U(1) charge values.

        Args:
            indices: Integer array whose trailing dimension equals
                ``self.size``.
            dtype: Optional dtype for the returned charge values.

        Returns:
            Charge-value array with the same shape as ``indices`` and values
            drawn from the shared U(1) local range.
        """
        idx = jnp.asarray(indices)
        _validate_trailing_size(idx, self.size, "Index")
        out_dtype = self.dtype if dtype is None else dtype
        return self.local_space.indices_to_values(idx, dtype=out_dtype)

    def local_states_valid(self, states: Any) -> jax.Array:
        """Checks local U(1) charge membership without global constraints.

        Args:
            states: Candidate flat U(1) state array.

        Returns:
            Boolean mask over the leading batch axes. The result only checks
            the scalar charge range at each flat site. Gauge-fixing relations,
            when present, are checked by the space constraint.
        """
        arr = jnp.asarray(states, dtype=self.dtype)
        _validate_trailing_size(arr, self.size, "State")
        return jnp.all(self.local_space.contains(arr), axis=-1)

    def view(self, states: Any) -> HilbertStateView:
        """Splits flat U(1) states into graph edge and vertex blocks.

        Args:
            states: Flat state array in the layout owned by this U(1) space.

        Returns:
            :class:`HilbertStateView` with edge blocks populated from the flat
            U(1) state. Vertex blocks are present only when the graph layout
            allocates vertex degrees of freedom.
        """
        return self.layout.view(states)

    def flatten(self, view: HilbertStateView | None = None, **blocks: Any) -> jax.Array:
        """Flattens graph edge and vertex blocks into U(1) state arrays.

        Args:
            view: Optional graph-layout view to flatten.
            **blocks: Edge or vertex blocks accepted by the underlying graph
                layout when ``view`` is not supplied.

        Returns:
            Flat U(1) state array with trailing dimension ``self.size``.
        """
        return self.layout.flatten(view, **blocks)

    def gauge_view(self, states: Any) -> jax.Array:
        """Reshapes flat states into gauge-component and edge axes.

        Args:
            states: Flat U(1) state array whose trailing dimension equals
                ``self.size``.

        Returns:
            Array with trailing shape ``(gauge_dimensions, n_edges)``. This
            representation is useful when algorithms operate per gauge copy
            rather than on the fully flattened state.
        """
        arr = jnp.asarray(states)
        _validate_trailing_size(arr, self.size, "State")
        return arr.reshape(
            (*arr.shape[:-1], int(self.gauge_dimensions), int(self.tiny_size))
        )

    def flatten_gauge_view(self, values: Any) -> jax.Array:
        """Converts a gauge-component view back to the flat U(1) layout.

        Args:
            values: Array whose final two axes are
                ``(gauge_dimensions, n_edges)``.

        Returns:
            Flat state array with trailing dimension ``self.size``.

        Raises:
            ValueError: If the trailing gauge-component shape does not match
            this Hilbert space.
        """
        arr = jnp.asarray(values)
        expected = (int(self.gauge_dimensions), int(self.tiny_size))
        if arr.shape[-2:] != expected:
            raise ValueError(
                "Gauge view trailing shape does not match (gauge_dimensions, n_edges)."
            )
        return arr.reshape((*arr.shape[:-2], self.size))

    def edge_values(self, states: Any, edge: int | Any) -> jax.Array:
        """Returns the U(1) charge block attached to one graph edge.

        Args:
            states: Flat U(1) state array.
            edge: Edge index or graph edge label accepted by the layout.

        Returns:
            Charge values for the requested edge, preserving any leading batch
            dimensions from ``states``.
        """
        return self.layout.edge_values(states, edge)


def _validate_trailing_size(arr: jax.Array, size: int, name: str) -> None:
    """Validates the trailing dimension of a U(1) state-like array."""
    if arr.shape[-1] != size:
        raise ValueError(f"{name} trailing dimension {arr.shape[-1]}, expected {size}.")
