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


"""Flat degree-of-freedom metadata for graph Hilbert layouts.

``SiteDof`` records how one flat scalar site maps back to a graph entity and
component. Layouts use it to expand edge and vertex local-space declarations
into the exact flat ordering expected by state arrays.
"""

from __future__ import annotations

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field
from neuralqx.utils.typing import DofKind


class SiteDof(Struct):
    """Metadata describing one flat scalar site in a graph Hilbert layout.

    Args:
        site: Flat site index in the state vector.
        kind: Entity kind associated with the site.
        entity_index: Edge or vertex index within the graph.
        component: Component index for vector degrees of freedom.

    Attributes:
        site: Flat site index in the state vector.
        kind: Entity kind associated with the site.
        entity_index: Edge or vertex index within the graph.
        component: Component index for vector degrees of freedom.
    """

    site: int = field(static=True)
    """Flat site index in the state vector."""

    kind: DofKind = field(static=True)
    """Entity kind associated with the site."""

    entity_index: int = field(static=True)
    """Edge or vertex index within the graph."""

    component: int = field(static=True, default=0)
    """Component index for vector degrees of freedom."""

    def __post_init__(self) -> None:
        """Validates the site metadata fields."""
        if self.kind not in ("edge", "vertex"):
            raise ValueError(f"Unsupported DOF kind {self.kind!r}.")
        if self.site < 0:
            raise ValueError("site must be non-negative.")
        if self.entity_index < 0:
            raise ValueError("entity_index must be non-negative.")
        if self.component < 0:
            raise ValueError("component must be non-negative.")

    def __hash__(self) -> int:
        """Returns a structural hash for the site metadata."""
        return hash(
            (type(self), self.site, self.kind, self.entity_index, self.component)
        )


__all__ = ["DofKind", "SiteDof"]
