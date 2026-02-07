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

"""
Strided gauge-copy layout utilities.

This module defines :class:`~neuralqx.hilbert.utils.layout.GaugeLayout`, a small helper that
formalises how a configuration with multiple gauge copies is flattened into a single
one-dimensional array.

If each gauge copy contains :math:`E` edge degrees of freedom and there are :math:`G` copies, the
flattened configuration has length :math:`N = E G` and is stored as contiguous blocks,

.. math::

    \\sigma
    =
    (\\sigma^{(0)}_0, \\ldots, \\sigma^{(0)}_{E-1},
     \\sigma^{(1)}_0, \\ldots, \\sigma^{(1)}_{E-1},
     \\ldots,
     \\sigma^{(G-1)}_0, \\ldots, \\sigma^{(G-1)}_{E-1}).

The layout is the bijection between:

- a pair ``(g, e)`` with gauge-copy index :math:`g \\in \\{0,\\ldots,G-1\\}` and within-copy edge
  index :math:`e \\in \\{0,\\ldots,E-1\\}`, and
- the flattened site index :math:`s \\in \\{0,\\ldots,EG-1\\}`,

given by

.. math::

    s = gE + e, \\qquad g = \\left\\lfloor \\frac{s}{E} \\right\\rfloor, \\qquad e = s \\bmod E.

Centralising this convention avoids indexing drift across random-state generation, move proposals,
constraint lifting, and any logic that needs to interpret or manipulate the flattened state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GaugeLayout:
    """
    Gauge-dimension strided layout: [copy0 (E sites) | copy1 (E sites) | ...].

    A tiny helper that formalizes how a composite (multi-gauge-copy) configuration is flattened into
    a single 1D array of length `edges_per_copy * gauge_dimensions`.

    Conceptually, the state is arranged as `gauge_dimensions` consecutive blocks, each block
    containing `edges_per_copy` edge degrees of freedom:

        copy 0: sites [0 .. E-1]
        copy 1: sites [E .. 2E-1]
        ...
        copy g: sites [g*E .. (g+1)*E - 1]

    This class provides two inverse mappings:
    - `site(edge_index, gauge_copy)` maps a (copy, edge) pair to the flattened site index.
    - `decode(site)` maps a flattened site index back to (gauge_copy, edge_index).

    These utilities reduce off-by-one mistakes and keep indexing logic consistent across random-
    state generation, move proposals, constraint lifting, etc.
    """

    edges_per_copy: int
    """Number of edge sites in a single gauge copy (E)."""

    gauge_dimensions: int
    """Number of gauge copies (G)."""

    def site(self, edge_index: int, gauge_copy: int = 0) -> int:
        """
        Return the flattened site index for a given base edge and gauge copy.

        Maps the pair `(edge_index, gauge_copy)` to the corresponding position in the
        flattened 1D state vector following the block layout:

            site = gauge_copy * edges_per_copy + edge_index

        This method validates both indices to catch errors early (negative indices or indices
        exceeding the configured layout).

        :param edge_index: Base edge index inside one gauge copy. Must satisfy
            `0 <= edge_index < edges_per_copy`.
        :param gauge_copy: Gauge copy index. Must satisfy `0 <= gauge_copy < gauge_dimensions`.

        :return: Flattened site index in `[0, edges_per_copy * gauge_dimensions)`.

        :raises ValueError: If `edge_index` or `gauge_copy` is out of range.
        """

        if gauge_copy < 0 or gauge_copy >= self.gauge_dimensions:
            raise ValueError(
                f"gauge_copy={gauge_copy} out of range [0,{self.gauge_dimensions})."
            )
        if edge_index < 0 or edge_index >= self.edges_per_copy:
            raise ValueError(
                f"edge_index={edge_index} out of range [0,{self.edges_per_copy})."
            )
        return gauge_copy * self.edges_per_copy + edge_index

    def decode(self, site: int) -> tuple[int, int]:
        """
        Decode a flattened site index into `(gauge_copy, edge_index)`.

        Inverse of :meth:`site`. Given a flattened index `site` in the 1D configuration, returns the
        gauge-copy block and the within-block edge index:

            gauge_copy = site // edges_per_copy
            edge_index = site % edges_per_copy

        :param site: Flattened site index. Must satisfy
            `0 <= site < edges_per_copy * gauge_dimensions`.

        :return: A tuple `(gauge_copy, edge_index)` where:
            - `gauge_copy` is in `[0, gauge_dimensions)`,
            - `edge_index` is in `[0, edges_per_copy)`.

        :raises ValueError: If `site` is out of range.
        """

        if site < 0 or site >= self.edges_per_copy * self.gauge_dimensions:
            raise ValueError(
                f"site={site} out of range [0,{self.edges_per_copy * self.gauge_dimensions})."
            )
        gauge_copy = site // self.edges_per_copy
        edge_index = site % self.edges_per_copy
        return gauge_copy, edge_index
