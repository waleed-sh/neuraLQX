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


"""Gauge-invariant SU(2) spin-network constraints.

The SU(2) gauge constraint checks local bounds at every flat site and left-comb
admissibility at each graph vertex.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.su2.recoupling import left_comb_valid
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.utils.struct import field


class SU2GaugeConstraint(AbstractDiscreteConstraint):
    """Local SU(2) admissibility at every graph vertex.

    Args:
        incident_edges_by_vertex: Incident edge sites for every vertex.
        vertex_channel_sites: Channel sites allocated to every vertex.
        site_max_label: Maximum integer label for every flat site.
        dimension_value: Optional exact constrained dimension.
        recoupling: Recoupling scheme used by channel admissibility.

    Attributes:
        incident_edges_by_vertex: Incident edge sites for every vertex.
        vertex_channel_sites: Channel sites allocated to every vertex.
        site_max_label: Maximum integer label for every flat site.
        dimension_value: Optional exact constrained dimension.
        recoupling: Recoupling scheme used by channel admissibility.
    """

    incident_edges_by_vertex: tuple[tuple[int, ...], ...] = field(static=True)
    """Incident edge sites for every vertex."""

    vertex_channel_sites: tuple[tuple[int, ...], ...] = field(static=True)
    """Channel sites allocated to every vertex."""

    site_max_label: tuple[int, ...] = field(static=True)
    """Maximum integer label for every flat site."""

    dimension_value: int | None = field(static=True, default=None)
    """Optional exact constrained dimension."""

    recoupling: str = field(static=True, default="left_comb")
    """Recoupling scheme used by channel admissibility."""

    def __post_init__(self) -> None:
        """Normalizes static site metadata and validates recoupling scheme."""
        if self.recoupling != "left_comb":
            raise ValueError(
                "SU2GaugeConstraint currently supports recoupling='left_comb'."
            )
        object.__setattr__(
            self,
            "incident_edges_by_vertex",
            tuple(
                tuple(int(site) for site in row)
                for row in self.incident_edges_by_vertex
            ),
        )
        object.__setattr__(
            self,
            "vertex_channel_sites",
            tuple(
                tuple(int(site) for site in row) for row in self.vertex_channel_sites
            ),
        )
        object.__setattr__(
            self,
            "site_max_label",
            tuple(int(label) for label in self.site_max_label),
        )

    def __hash__(self) -> int:
        """Returns a structural hash for the SU(2) gauge constraint."""
        return hash(
            (
                type(self),
                self.incident_edges_by_vertex,
                self.vertex_channel_sites,
                self.site_max_label,
                self.dimension_value,
                self.recoupling,
            )
        )

    def validate_hilbert(self, hilbert: Any) -> None:
        """Checks that constraint metadata is compatible with a Hilbert space.

        Args:
            hilbert: SU(2) Hilbert space about to use this constraint.

        Raises:
            ValueError: If the per-site maximum-label table has the wrong
            length for the target space.
            IndexError: If any incident edge or channel site references a flat
            site outside the target layout.
        """
        size = int(hilbert.size)
        if len(self.site_max_label) != size:
            raise ValueError(
                "SU2GaugeConstraint site_max_label length does not match Hilbert size."
            )
        for edge_sites, channel_sites in zip(
            self.incident_edges_by_vertex,
            self.vertex_channel_sites,
            strict=True,
        ):
            for site in (*edge_sites, *channel_sites):
                if site < 0 or site >= size:
                    raise IndexError(
                        f"SU2GaugeConstraint site {site} out of range [0,{size})."
                    )

    def dimension(self, hilbert: Any) -> int | None:
        """Returns the cached exact constrained dimension when available.

        Args:
            hilbert: Hilbert space requesting the dimension. It is accepted for
                the shared constraint protocol and is not otherwise needed.

        Returns:
            Exact number of valid spin-network states when the space was built
            with exact enumeration, otherwise ``None``.
        """
        del hilbert
        return self.dimension_value

    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates SU(2) site bounds and vertex admissibility.

        Args:
            states: Flat spin-network state array. The final axis must match
                the layout size represented by ``site_max_label``.

        Returns:
            Boolean mask over the leading batch axes. A state is accepted when
            every site label is in range and every vertex satisfies the
            left-comb SU(2) admissibility rules.
        """
        arr, single, batch = ensure_2d_trailing(states, len(self.site_max_label))
        arr = arr.astype(jnp.int32)
        site_max = jnp.asarray(self.site_max_label, dtype=jnp.int32)
        ok = jnp.all((arr >= 0) & (arr <= site_max[None, :]), axis=-1)

        for edge_sites, channel_sites in zip(
            self.incident_edges_by_vertex,
            self.vertex_channel_sites,
            strict=True,
        ):
            edge_idx = jnp.asarray(edge_sites, dtype=jnp.int32)
            edge_vals = jnp.take(arr, edge_idx, axis=-1)
            if channel_sites:
                channel_idx = jnp.asarray(channel_sites, dtype=jnp.int32)
                channel_vals = jnp.take(arr, channel_idx, axis=-1)
            else:
                channel_vals = jnp.zeros((arr.shape[0], 0), dtype=arr.dtype)
            ok = ok & jax.vmap(left_comb_valid, in_axes=(0, 0))(edge_vals, channel_vals)

        if single:
            return ok[0]
        return ok.reshape(batch)


GaugeInvariantSU2Constraint = SU2GaugeConstraint


__all__ = ["GaugeInvariantSU2Constraint", "SU2GaugeConstraint"]
