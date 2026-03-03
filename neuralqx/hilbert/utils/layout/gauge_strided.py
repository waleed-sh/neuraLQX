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

This module defines :class:`~neuralqx.hilbert.utils.layout.StridedGaugeCopyLayout`, a small helper that
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
from typing import Iterator

from neuralqx.utils.deprecation import deprecated
from neuralqx.utils.deprecation import deprecated_class
from ._abstract_layout import AbstractBasisLayout


@dataclass(frozen=True, slots=True)
class GaugeCoord:
    """
    Structured coordinate for a gauge-copy strided layout.

    This coordinate is used by :class:`StridedGaugeCopyLayout` and represents a site as

    - `gauge_copy`: which gauge-copy block the site belongs to
    - `edge_index`: position inside that copy block

    The coordinate is intentionally small and immutable so that it is safe to pass around
    in indexing-heavy code.
    """

    gauge_copy: int

    edge_index: int


@dataclass(frozen=True, slots=True)
class StridedGaugeCopyLayout(AbstractBasisLayout[GaugeCoord]):
    r"""
    Strided gauge-copy layout with contiguous copy blocks.

    This is the concrete implementation of the current layout convention used by neuralqx
    for multiple gauge copies (i.e. U(1)^N gauge group structure).

    If each gauge copy contains :math:`E` edge degrees of freedom and there are :math:`G`
    copies, the flattened configuration has length

    .. math::

        N = E G

    and is stored as contiguous blocks

    .. math::

        (\sigma^{(0)}_0, \ldots, \sigma^{(0)}_{E-1},
         \sigma^{(1)}_0, \ldots, \sigma^{(1)}_{E-1},
         \ldots,
         \sigma^{(G-1)}_0, \ldots, \sigma^{(G-1)}_{E-1}).

    The structured coordinate is :class:`GaugeCoord(gauge_copy, edge_index)` and the
    flattening map is

    .. math::

        s = gE + e.

    This class exposes two API layers

    - Generic layout API
       - :meth:`site_of`
       - :meth:`coord_of`

    - Convenience API
       - :meth:`encode(gauge_copy, edge_index)`
       - :meth:`decode(site) -> (gauge_copy, edge_index)`
    """

    edges_per_copy: int
    """Number of edge sites in a single gauge copy (E)."""

    gauge_dimensions: int
    """Number of gauge copies (G)."""

    def __post_init__(self) -> None:
        """
        Validate constructor parameters.

        :raises TypeError: If any parameter is not an integer.
        :raises ValueError: If any parameter is not strictly positive.
        """
        self._validate_positive_int("edges_per_copy", self.edges_per_copy)
        self._validate_positive_int("gauge_dimensions", self.gauge_dimensions)

    @property
    def size(self) -> int:
        """
        Total number of flattened sites.

        :return: `edges_per_copy * gauge_dimensions`.
        """
        return self.edges_per_copy * self.gauge_dimensions

    @property
    def shape(self) -> tuple[int, int]:
        """
        Logical rectangular extents of the structured coordinate grid.

        The returned shape is ordered as

        `(gauge_dimensions, edges_per_copy)`.

        :return: Tuple `(G, E)`.
        """
        return self.gauge_dimensions, self.edges_per_copy

    def validate_coord(self, coord: GaugeCoord) -> None:
        """
        Validate a :class:`GaugeCoord`.

        :param coord: Coordinate to validate.
        :raises TypeError: If `coord` is not a :class:`GaugeCoord`, or if one of its
            fields is not an integer.
        :raises IndexError: If a field is out of range.
        """
        if not isinstance(coord, GaugeCoord):
            raise TypeError(f"coord must be a GaugeCoord, got {type(coord).__name__}.")
        self.validate_indices(coord.gauge_copy, coord.edge_index)

    def validate_indices(self, gauge_copy: int, edge_index: int) -> None:
        """
        Validate raw coordinate components.

        :param gauge_copy: Gauge-copy index.
        :param edge_index: Edge index within a copy.
        :raises TypeError: If either value is not an integer.
        :raises IndexError: If either value is out of range.
        """
        self._validate_int("gauge_copy", gauge_copy)
        self._validate_int("edge_index", edge_index)

        if gauge_copy < 0 or gauge_copy >= self.gauge_dimensions:
            raise IndexError(
                f"gauge_copy={gauge_copy} out of range [0,{self.gauge_dimensions})."
            )
        if edge_index < 0 or edge_index >= self.edges_per_copy:
            raise IndexError(
                f"edge_index={edge_index} out of range [0,{self.edges_per_copy})."
            )

    def site_of(self, coord: GaugeCoord) -> int:
        """
        Encode a :class:`GaugeCoord` into a flattened site index.

        :param coord: Structured coordinate `(gauge_copy, edge_index)`.
        :return: Flattened site index `site = gauge_copy * edges_per_copy + edge_index`.
        :raises TypeError: If `coord` is invalid or has wrong field types.
        :raises IndexError: If `coord` is out of range.
        """
        self.validate_coord(coord)
        return coord.gauge_copy * self.edges_per_copy + coord.edge_index

    def coord_of(self, site: int) -> GaugeCoord:
        """
        Decode a flattened site index into a :class:`GaugeCoord`.

        :param site: Flattened site index in `[0, size)`.
        :return: `GaugeCoord(gauge_copy, edge_index)`.
        :raises TypeError: If `site` is not an integer.
        :raises IndexError: If `site` is out of range.
        """
        self.validate_site(site)
        gauge_copy = site // self.edges_per_copy
        edge_index = site % self.edges_per_copy
        return GaugeCoord(gauge_copy=gauge_copy, edge_index=edge_index)

    def encode(self, gauge_copy: int, edge_index: int) -> int:
        """
        Convenience encoder using raw integer indices.

        This is equivalent to

        `site_of(GaugeCoord(gauge_copy, edge_index))`.

        New code should prefer this over the legacy :meth:`site` wrapper because the
        argument order matches the tuple returned by :meth:`decode`.

        :param gauge_copy: Gauge-copy index in `[0, gauge_dimensions)`.
        :param edge_index: Edge index in `[0, edges_per_copy)`.
        :return: Flattened site index.
        :raises TypeError: If any argument is not an integer.
        :raises IndexError: If any argument is out of range.
        """
        self.validate_indices(gauge_copy, edge_index)
        return gauge_copy * self.edges_per_copy + edge_index

    @deprecated(
        func_name="StridedGaugeCopyLayout.decode",
        reason="""
        Use `coord_of(site)` instead.

        `coord_of(site)` returns a `GaugeCoord` object with explicit fields:
            coord.gauge_copy
            coord.edge_index

        If you still need a tuple, convert explicitly:
            c = layout.coord_of(site)
            (c.gauge_copy, c.edge_index)
        """,
    )
    def decode(self, site: int) -> tuple[int, int]:
        """
        Convenience decoder returning a plain tuple.

        :param site: Flattened site index in `[0, size)`.
        :return: Tuple `(gauge_copy, edge_index)`.
        :raises TypeError: If `site` is not an integer.
        :raises IndexError: If `site` is out of range.
        """
        coord = self.coord_of(site)
        return coord.gauge_copy, coord.edge_index

    @deprecated(
        func_name="StridedGaugeCopyLayout.site",
        reason="""
        Use `encode(gauge_copy, edge_index)` instead.

        Migration:
            old: layout.site(edge_index=e, gauge_copy=g)
            new: layout.encode(gauge_copy=g, edge_index=e)

        The new API aligns the argument order with `decode(site) -> (gauge_copy, edge_index)`
        and with the generic layout interface `site_of(GaugeCoord(...))`.
        """,
    )
    def site(self, edge_index: int, gauge_copy: int = 0) -> int:
        """
        Backward-compatible wrapper for older call sites.

        This preserves the historical argument order used in earlier code:

        `site(edge_index, gauge_copy=0)`

        New code should prefer

        `encode(gauge_copy, edge_index)`

        because it aligns with :meth:`decode`, which returns `(gauge_copy, edge_index)`.

        :param edge_index: Edge index in `[0, edges_per_copy)`.
        :param gauge_copy: Gauge-copy index in `[0, gauge_dimensions)`.
        :return: Flattened site index.
        :raises TypeError: If any argument is not an integer.
        :raises IndexError: If any argument is out of range.
        """
        return self.encode(gauge_copy=gauge_copy, edge_index=edge_index)

    def copy_slice(self, gauge_copy: int) -> slice:
        """
        Return the contiguous slice that selects one gauge-copy block.

        Example
        -------
        If `edges_per_copy = E`, then `copy_slice(g)` returns `slice(g*E, (g+1)*E)`.

        :param gauge_copy: Gauge-copy index.
        :return: Python slice selecting the corresponding flattened block.
        :raises TypeError: If `gauge_copy` is not an integer.
        :raises IndexError: If `gauge_copy` is out of range.
        """
        self._validate_int("gauge_copy", gauge_copy)
        if gauge_copy < 0 or gauge_copy >= self.gauge_dimensions:
            raise IndexError(
                f"gauge_copy={gauge_copy} out of range [0,{self.gauge_dimensions})."
            )
        start = gauge_copy * self.edges_per_copy
        return slice(start, start + self.edges_per_copy)

    def iter_coords(self) -> Iterator[GaugeCoord]:
        """
        Iterate over all structured coordinates in flattened-site order.

        This ordering matches the physical memory layout used by the strided convention.

        :yield: `GaugeCoord` instances in the order corresponding to flat sites
            `0, 1, ..., size-1`.
        """
        for g in range(self.gauge_dimensions):
            for e in range(self.edges_per_copy):
                yield GaugeCoord(gauge_copy=g, edge_index=e)


@deprecated_class(
    class_name="GaugeLayout",
    reason="""
    Use `StridedGaugeCopyLayout` instead.

    For the preferred API, use:
        - `encode(gauge_copy, edge_index)`
        - `decode(site) -> (gauge_copy, edge_index)`
        - or the layout-agnostic API `site_of(...)` / `coord_of(...)`.
    """,
)
class GaugeLayout(StridedGaugeCopyLayout):
    """Deprecated alias for StridedGaugeCopyLayout."""

    pass


__all__ = [
    "GaugeCoord",
    "StridedGaugeCopyLayout",
    "GaugeLayout",
]
