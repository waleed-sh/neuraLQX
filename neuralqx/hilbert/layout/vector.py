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


"""Vector and heterogeneous local-space declarations.

Graph-based Hilbert spaces may attach scalar or vector degrees of freedom to
edges and vertices. This module normalizes those declarations into reusable
objects so layout construction can treat scalar and vector sites uniformly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import EntitySpaces
from neuralqx.utils.typing import Vertex

from .abstract import AbstractLocalSpace
from .factory import as_local_space


class VectorRange(Struct):
    """Vector-valued degree of freedom with a shared scalar component space.

    Args:
        local_space: Local space used by each scalar component.
        components: Number of scalar components in the vector value.

    Attributes:
        local_space: Local space used by each scalar component.
        components: Number of scalar components in the vector value.
    """

    local_space: AbstractLocalSpace = field(static=True)
    """Local space used by each scalar component."""

    components: int = field(static=True)
    """Number of scalar components in the vector value."""

    def __post_init__(self) -> None:
        """Coerces the component space and validates the component count."""
        object.__setattr__(self, "local_space", as_local_space(self.local_space))
        if isinstance(self.components, bool) or not isinstance(self.components, int):
            raise TypeError("components must be an integer.")
        if self.components <= 0:
            raise ValueError("components must be positive.")

    def __hash__(self) -> int:
        """Returns a structural hash for the vector range."""
        return hash((type(self), self.local_space, self.components))

    @property
    def local_size(self) -> int:
        """Local size of each scalar component."""
        return self.local_space.local_size


class HeterogeneousLocalSpace(Struct):
    """Edge and vertex local-space specification for heterogeneous layouts.

    Args:
        edge: Default space for all graph edges.
        vertex: Default space for all graph vertices.
        edge_spaces: Optional per-edge overrides keyed by edge index or edge.
        vertex_spaces: Optional per-vertex overrides keyed by vertex index or
            vertex.

    Attributes:
        edge: Default space for all graph edges.
        vertex: Default space for all graph vertices.
        edge_spaces: Optional per-edge overrides keyed by edge index or edge.
        vertex_spaces: Optional per-vertex overrides keyed by vertex index or vertex.
    """

    edge: AbstractLocalSpace | VectorRange | None = field(static=True, default=None)
    """Default space for all graph edges."""
    vertex: AbstractLocalSpace | VectorRange | None = field(static=True, default=None)
    """Default space for all graph vertices."""
    edge_spaces: EntitySpaces = field(static=True, default=None)
    """Optional per-edge overrides keyed by edge index or edge."""
    vertex_spaces: EntitySpaces = field(static=True, default=None)
    """Optional per-vertex overrides keyed by vertex index or vertex."""

    def __post_init__(self) -> None:
        """Coerces defaults and entity-specific overrides."""
        edge = coerce_dof_space(self.edge)
        vertex = coerce_dof_space(self.vertex)
        edge_spaces = _coerce_entity_spaces(self.edge_spaces, kind="edge")
        vertex_spaces = _coerce_entity_spaces(self.vertex_spaces, kind="vertex")
        object.__setattr__(self, "edge", edge)
        object.__setattr__(self, "vertex", vertex)
        object.__setattr__(self, "edge_spaces", edge_spaces)
        object.__setattr__(self, "vertex_spaces", vertex_spaces)
        if (
            edge is None
            and vertex is None
            and not _has_any_space(edge_spaces)
            and not _has_any_space(vertex_spaces)
        ):
            raise ValueError("At least one of edge or vertex local spaces must be set.")

    def __hash__(self) -> int:
        """Returns a structural hash for the heterogeneous declaration."""
        return hash(
            (
                type(self),
                self.edge,
                self.vertex,
                self.edge_spaces,
                self.vertex_spaces,
            )
        )


def coerce_dof_space(value: Any) -> AbstractLocalSpace | VectorRange | None:
    """Coerces a scalar or vector degree-of-freedom declaration.

    Args:
        value: ``None``, a ``VectorRange``, a local-space object, or a sequence
            accepted by :func:`as_local_space`.

    Returns:
        Normalized local-space declaration or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, VectorRange):
        return value
    return as_local_space(value)


def dof_width(value: AbstractLocalSpace | VectorRange | None) -> int:
    """Returns the flat scalar width occupied by one graph entity.

    Args:
        value: Scalar, vector, or empty degree-of-freedom declaration.

    Returns:
        ``0`` for absent spaces, ``1`` for scalar spaces, or the vector
        component count for ``VectorRange`` values.
    """
    if value is None:
        return 0
    if isinstance(value, VectorRange):
        return value.components
    return 1


def scalar_space(value: AbstractLocalSpace | VectorRange) -> AbstractLocalSpace:
    """Returns the scalar component local space for a declaration.

    Args:
        value: Scalar local space or vector range.

    Returns:
        The scalar local space used by each stored flat site.
    """
    if isinstance(value, VectorRange):
        return value.local_space
    return value


def _coerce_entity_spaces(
    spaces: EntitySpaces,
    *,
    kind: str,
) -> tuple[tuple[int | Edge | Vertex, AbstractLocalSpace | VectorRange | None], ...]:
    """Normalizes per-entity local-space overrides.

    Args:
        spaces: Mapping or iterable of entity keys paired with spaces.
        kind: Entity kind, either ``"edge"`` or ``"vertex"``.

    Returns:
        Tuple of normalized entity keys paired with coerced spaces.
    """
    if spaces is None:
        return ()
    items = spaces.items() if isinstance(spaces, Mapping) else spaces
    return tuple(
        (_coerce_entity_key(key, kind), coerce_dof_space(space)) for key, space in items
    )


def _coerce_entity_key(key: Any, kind: str) -> int | Edge | Vertex:
    """Coerces an edge or vertex override key.

    Args:
        key: Integer entity index or edge/vertex-like value.
        kind: Entity kind, either ``"edge"`` or ``"vertex"``.

    Returns:
        Integer index, canonical edge, or canonical vertex.

    Raises:
        ValueError: If ``kind`` is unsupported.
    """
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    if kind == "edge":
        return Edge.from_like(key)
    if kind == "vertex":
        return Vertex.from_like(key)
    raise ValueError(f"Unsupported entity kind {kind!r}.")


def _has_any_space(
    spaces: tuple[tuple[Any, AbstractLocalSpace | VectorRange | None], ...],
) -> bool:
    """Checks whether any override supplies a local space.

    Args:
        spaces: Normalized per-entity overrides.

    Returns:
        ``True`` when at least one override value is not ``None``.
    """
    return any(space is not None for _, space in spaces)


__all__ = [
    "HeterogeneousLocalSpace",
    "VectorRange",
    "coerce_dof_space",
    "dof_width",
    "scalar_space",
]
