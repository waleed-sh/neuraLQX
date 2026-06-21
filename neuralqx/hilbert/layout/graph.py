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


"""Hilbert-owned graph layout and semantic state views.

Graph Hilbert spaces store states as flat arrays for JAX efficiency, while
callers often want edge and vertex views that follow graph semantics. This
module defines the layout object that bridges those representations and records
the component-major flat ordering used throughout the Hilbert API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.local import VectorRange
from neuralqx.hilbert.local.vector import coerce_dof_space
from neuralqx.hilbert.local.vector import dof_width
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field
from neuralqx.utils.typing import DofSpace
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import Vertex

from .dof import SiteDof


class HilbertStateView(Struct):
    """Semantic edge and vertex view of flat Hilbert states.

    Args:
        edges: Edge block or tuple of edge values from a flat state.
        vertices: Vertex block or tuple of vertex values from a flat state.

    Attributes:
        edges: Edge block or tuple of edge values from a flat state.
        vertices: Vertex block or tuple of vertex values from a flat state.
    """

    edges: Any | None = field(pytree=True, default=None)
    """Edge block or tuple of edge values from a flat state."""

    vertices: Any | None = field(pytree=True, default=None)
    """Vertex block or tuple of vertex values from a flat state."""


class GraphHilbertLayout(Struct):
    """Block layout owned by a graph Hilbert space.

    The flat order is ``[edge DOFs | vertex DOFs]``. Inside each block, vector
    components are component-major: all component-0 values first, then all
    component-1 values, and so on. Uniform-width DOFs view as dense arrays.
    Mixed-width DOFs view as tuples indexed by edge/vertex index.

    Args:
        graph: Graph whose edges and vertices define the semantic entities.
        edge_space: Default degree-of-freedom declaration for every edge.
        vertex_space: Default degree-of-freedom declaration for every vertex.
        edge_spaces: Optional per-edge degree-of-freedom declarations.
        vertex_spaces: Optional per-vertex degree-of-freedom declarations.

    Attributes:
        graph: Graph whose edges and vertices define the semantic entities.
        edge_space: Default degree-of-freedom declaration for every edge.
        vertex_space: Default degree-of-freedom declaration for every vertex.
        edge_spaces: Per-edge degree-of-freedom declarations after normalization.
        vertex_spaces: Per-vertex degree-of-freedom declarations after normalization.
    """

    graph: AbstractGraph = field(pytree=False)
    """Graph whose edges and vertices define the semantic entities."""

    edge_space: DofSpace | None = field(static=True, default=None)
    """Default degree-of-freedom declaration for every edge."""

    vertex_space: DofSpace | None = field(static=True, default=None)
    """Default degree-of-freedom declaration for every vertex."""

    edge_spaces: tuple[DofSpace | None, ...] | None = field(static=True, default=None)
    """Per-edge degree-of-freedom declarations after normalization."""

    vertex_spaces: tuple[DofSpace | None, ...] | None = field(static=True, default=None)
    """Per-vertex degree-of-freedom declarations after normalization."""

    def __post_init__(self) -> None:
        """Normalizes edge and vertex degree-of-freedom declarations."""
        edge_spaces = _entity_spaces(
            self.edge_spaces,
            self.edge_space,
            self.graph.n_edges,
        )
        vertex_spaces = _entity_spaces(
            self.vertex_spaces,
            self.vertex_space,
            self.graph.n_vertices,
        )
        if all(space is None for space in edge_spaces + vertex_spaces):
            raise ValueError("At least one edge or vertex local space must be set.")
        object.__setattr__(self, "edge_spaces", edge_spaces)
        object.__setattr__(self, "vertex_spaces", vertex_spaces)
        object.__setattr__(self, "edge_space", _uniform_space(edge_spaces))
        object.__setattr__(self, "vertex_space", _uniform_space(vertex_spaces))

    def __hash__(self) -> int:
        """Returns a structural hash for the graph layout."""
        return hash((type(self), self.graph, self.edge_spaces, self.vertex_spaces))

    @property
    def edge_widths(self) -> tuple[int, ...]:
        """Flat scalar width occupied by each graph edge."""
        return tuple(dof_width(space) for space in self.edge_spaces)

    @property
    def vertex_widths(self) -> tuple[int, ...]:
        """Flat scalar width occupied by each graph vertex."""
        return tuple(dof_width(space) for space in self.vertex_spaces)

    @property
    def edge_width(self) -> int:
        """Common positive scalar width for all edge degrees of freedom.

        Raises:
            ValueError: If edge widths are heterogeneous.
        """
        return _uniform_width(self.edge_widths, "edge")

    @property
    def vertex_width(self) -> int:
        """Common positive scalar width for all vertex degrees of freedom.

        Raises:
            ValueError: If vertex widths are heterogeneous.
        """
        return _uniform_width(self.vertex_widths, "vertex")

    @property
    def edge_offsets(self) -> tuple[int, ...]:
        """First flat component site for each edge degree of freedom."""
        return _entity_first_component_offsets(self.edge_widths)

    @property
    def vertex_offsets(self) -> tuple[int, ...]:
        """First flat component site for each vertex degree of freedom."""
        return _entity_first_component_offsets(self.vertex_widths)

    @property
    def edge_size(self) -> int:
        """Total number of flat scalar sites allocated to graph edges."""
        return sum(self.edge_widths)

    @property
    def vertex_size(self) -> int:
        """Total number of flat scalar sites allocated to graph vertices."""
        return sum(self.vertex_widths)

    @property
    def size(self) -> int:
        """Total flat state size for this graph layout."""
        return self.edge_size + self.vertex_size

    @property
    def dofs(self) -> tuple[SiteDof, ...]:
        """Flat scalar site metadata in layout order.

        Returns:
            Tuple describing each flat scalar site as an edge or vertex
            component.
        """
        out: list[SiteDof] = []
        site = 0
        for component in range(max(self.edge_widths, default=0)):
            for edge_index, width in enumerate(self.edge_widths):
                if width > component:
                    out.append(SiteDof(site, "edge", edge_index, component))
                    site += 1
        for component in range(max(self.vertex_widths, default=0)):
            for vertex_index, width in enumerate(self.vertex_widths):
                if width > component:
                    out.append(SiteDof(site, "vertex", vertex_index, component))
                    site += 1
        return tuple(out)

    def view(self, states: Any) -> HilbertStateView:
        """Views flat state arrays as semantic edge and vertex blocks.

        Args:
            states: State array whose trailing dimension equals
                :attr:`size`.

        Returns:
            ``HilbertStateView`` containing edge and vertex data. Uniform-width
            entities are represented by dense arrays, while mixed-width
            entities are represented by tuples.

        Raises:
            ValueError: If the trailing state dimension does not match the
                layout size.
        """
        arr = jnp.asarray(states)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]} != layout size {self.size}."
            )
        edges = self._view_entities(arr, start=0, widths=self.edge_widths)
        vertices = self._view_entities(
            arr,
            start=self.edge_size,
            widths=self.vertex_widths,
        )
        return HilbertStateView(edges=edges, vertices=vertices)

    def flatten(
        self,
        view: HilbertStateView | None = None,
        *,
        edges: Any | None = None,
        vertices: Any | None = None,
    ) -> jax.Array:
        """Flattens a semantic view or explicit edge and vertex blocks.

        Args:
            view: Optional semantic state view. When supplied, its ``edges`` and
                ``vertices`` blocks are used.
            edges: Explicit edge block or tuple of edge values.
            vertices: Explicit vertex block or tuple of vertex values.

        Returns:
            Flat state array in layout order.

        Raises:
            ValueError: If required blocks are missing or have incompatible
                trailing shapes.
        """
        if view is not None:
            edges = view.edges
            vertices = view.vertices

        parts = []
        edge_part = self._flatten_entities(edges, self.edge_widths, "edges")
        vertex_part = self._flatten_entities(vertices, self.vertex_widths, "vertices")
        if edge_part is not None:
            parts.append(edge_part)
        if vertex_part is not None:
            parts.append(vertex_part)
        if len(parts) == 1:
            return parts[0]
        return jnp.concatenate(parts, axis=-1)

    def edge_values(self, states: Any, edge: int | Any) -> jax.Array:
        """Returns the scalar or vector value associated with one edge.

        Args:
            states: Flat state array.
            edge: Edge index or edge-like value to inspect.

        Returns:
            Scalar value for scalar edge degrees of freedom, or a vector over
            the component axis for vector edge degrees of freedom.
        """
        edge_index = self._edge_index(edge)
        return self._entity_values(
            states,
            start=0,
            widths=self.edge_widths,
            entity_index=edge_index,
            kind="edge",
        )

    def vertex_values(self, states: Any, vertex: int | Any) -> jax.Array:
        """Returns the scalar or vector value associated with one vertex.

        Args:
            states: Flat state array.
            vertex: Vertex index or vertex-like value to inspect.

        Returns:
            Scalar value for scalar vertex degrees of freedom, or a vector over
            the component axis for vector vertex degrees of freedom.
        """
        vertex_index = self._vertex_index(vertex)
        return self._entity_values(
            states,
            start=self.edge_size,
            widths=self.vertex_widths,
            entity_index=vertex_index,
            kind="vertex",
        )

    def edge_site(self, edge: int | Any, component: int = 0) -> int:
        """Returns the flat site for an edge scalar component.

        Args:
            edge: Edge index or edge-like value.
            component: Component index within the edge degree of freedom.

        Returns:
            Flat scalar site index.
        """
        edge_index = self._edge_index(edge)
        return _entity_component_site(self.edge_widths, edge_index, component, "edge")

    def vertex_site(self, vertex: int | Any, component: int = 0) -> int:
        """Returns the flat site for a vertex scalar component.

        Args:
            vertex: Vertex index or vertex-like value.
            component: Component index within the vertex degree of freedom.

        Returns:
            Flat scalar site index.
        """
        vertex_index = self._vertex_index(vertex)
        return self.edge_size + _entity_component_site(
            self.vertex_widths,
            vertex_index,
            component,
            "vertex",
        )

    def site_to_edge(self, site: int) -> tuple[int, int]:
        """Decodes an edge site to ``(edge_index, component)``.

        Args:
            site: Flat site index inside the edge block.

        Returns:
            Edge index and component index represented by ``site``.

        Raises:
            IndexError: If ``site`` is not inside the edge block.
        """
        if site < 0 or site >= self.edge_size:
            raise IndexError(f"site={site} is not an edge site.")
        return _entity_from_site(site, self.edge_widths, "edge")

    def site_to_vertex(self, site: int) -> tuple[int, int]:
        """Decodes a vertex site to ``(vertex_index, component)``.

        Args:
            site: Flat site index in the full state array.

        Returns:
            Vertex index and component index represented by ``site``.

        Raises:
            IndexError: If ``site`` is not inside the vertex block.
        """
        rel = site - self.edge_size
        if rel < 0 or rel >= self.vertex_size:
            raise IndexError(f"site={site} is not a vertex site.")
        return _entity_from_site(rel, self.vertex_widths, "vertex")

    def _view_entities(
        self,
        arr: jax.Array,
        *,
        start: int,
        widths: tuple[int, ...],
    ) -> Any | None:
        """Builds semantic values for one edge or vertex block.

        Args:
            arr: Flat state array.
            start: Starting flat site of the entity block.
            widths: Per-entity scalar widths for the block.

        Returns:
            ``None`` for an empty block, a dense array for uniform-width
            entities, or a tuple for mixed-width entities.
        """
        if not widths or all(width == 0 for width in widths):
            return None

        total_width = sum(widths)
        uniform = _common_positive_width(widths)
        if uniform is not None:
            block = arr[..., start : start + total_width]
            if uniform == 1:
                return block.reshape((*arr.shape[:-1], len(widths)))
            component_major = block.reshape((*arr.shape[:-1], uniform, len(widths)))
            return jnp.swapaxes(component_major, -1, -2)

        out = []
        for entity_index, width in enumerate(widths):
            if width == 0:
                out.append(None)
            elif width == 1:
                out.append(
                    arr[
                        ...,
                        start
                        + _entity_component_site(widths, entity_index, 0, "entity"),
                    ]
                )
            else:
                sites = jnp.asarray(
                    [
                        start
                        + _entity_component_site(
                            widths, entity_index, component, "entity"
                        )
                        for component in range(width)
                    ],
                    dtype=jnp.int32,
                )
                out.append(jnp.take(arr, sites, axis=-1))
        return tuple(out)

    def _flatten_entities(
        self,
        values: Any | None,
        widths: tuple[int, ...],
        name: str,
    ) -> jax.Array | None:
        """Flattens one semantic edge or vertex block.

        Args:
            values: Dense block or tuple of entity values.
            widths: Per-entity scalar widths for the block.
            name: Block name used in validation errors.

        Returns:
            Flat array for the entity block, or ``None`` when the layout has no
            degrees of freedom in that block.
        """
        if not widths or all(width == 0 for width in widths):
            if values is not None:
                raise ValueError(
                    f"{name} were provided, but this layout has no {name}."
                )
            return None
        if values is None:
            raise ValueError(f"{name} are required by this layout.")

        uniform = _common_positive_width(widths)
        if uniform is not None:
            return self._flatten_uniform_entities(values, len(widths), uniform, name)
        return self._flatten_ragged_entities(values, widths, name)

    def _flatten_uniform_entities(
        self,
        values: Any,
        entities: int,
        width: int,
        name: str,
    ) -> jax.Array:
        """Flattens a uniform-width semantic entity block.

        Args:
            values: Dense edge or vertex values.
            entities: Number of entities represented by the block.
            width: Common scalar width of each entity.
            name: Block name used in validation errors.

        Returns:
            Flat component-major array for the block.
        """
        arr = jnp.asarray(values)
        if width == 1:
            if arr.shape[-1:] != (entities,):
                raise ValueError(
                    f"{name} trailing shape {arr.shape[-1:]} != ({entities},)."
                )
            return arr.reshape((*arr.shape[:-1], entities))
        if arr.shape[-2:] != (entities, width):
            raise ValueError(
                f"{name} trailing shape {arr.shape[-2:]} != ({entities}, {width})."
            )
        component_major = jnp.swapaxes(arr, -1, -2)
        return component_major.reshape((*arr.shape[:-2], entities * width))

    def _flatten_ragged_entities(
        self,
        values: Any,
        widths: tuple[int, ...],
        name: str,
    ) -> jax.Array:
        """Flattens a mixed-width semantic entity block.

        Args:
            values: Sequence containing one value per entity.
            widths: Per-entity scalar widths for the block.
            name: Block name used in validation errors.

        Returns:
            Flat component-major array containing all non-empty entity values.
        """
        if not isinstance(values, Sequence) or len(values) != len(widths):
            raise ValueError(f"{name} must be a sequence with {len(widths)} entries.")
        parts = []
        coerced = []
        for idx, (value, width) in enumerate(zip(values, widths, strict=True)):
            if width == 0:
                if value is not None:
                    raise ValueError(f"{name}[{idx}] must be None.")
                coerced.append(None)
                continue
            if value is None:
                raise ValueError(f"{name}[{idx}] is required.")
            arr = jnp.asarray(value)
            if width == 1:
                coerced.append(jnp.expand_dims(arr, axis=-1))
                continue
            if arr.shape[-1:] != (width,):
                raise ValueError(f"{name}[{idx}] trailing shape must be ({width},).")
            coerced.append(arr.reshape((*arr.shape[:-1], width)))
        for component in range(max(widths, default=0)):
            for value, width in zip(coerced, widths, strict=True):
                if width > component:
                    parts.append(value[..., component : component + 1])
        return jnp.concatenate(parts, axis=-1)

    def _entity_values(
        self,
        states: Any,
        *,
        start: int,
        widths: tuple[int, ...],
        entity_index: int,
        kind: str,
    ) -> jax.Array:
        """Extracts scalar or vector values for one entity.

        Args:
            states: Flat state array.
            start: Starting flat site of the entity block.
            widths: Per-entity scalar widths for the block.
            entity_index: Entity index within the block.
            kind: Entity kind used in validation errors.

        Returns:
            Scalar or vector values for the selected entity.
        """
        width = widths[entity_index]
        if width == 0:
            raise ValueError(f"This layout has no DOFs for that {kind}.")
        arr = jnp.asarray(states)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]} != layout size {self.size}."
            )
        if width == 1:
            return arr[
                ..., start + _entity_component_site(widths, entity_index, 0, kind)
            ]
        sites = jnp.asarray(
            [
                start + _entity_component_site(widths, entity_index, component, kind)
                for component in range(width)
            ],
            dtype=jnp.int32,
        )
        return jnp.take(arr, sites, axis=-1)

    def _edge_index(self, edge: int | Any) -> int:
        """Normalizes an edge index or edge-like value.

        Args:
            edge: Integer edge index or edge-like value.

        Returns:
            Integer edge index in the graph.
        """
        if isinstance(edge, int):
            if edge < 0 or edge >= self.graph.n_edges:
                raise IndexError(f"edge index {edge} out of range.")
            return edge
        return self.graph.edge_to_index(Edge.from_like(edge))

    def _vertex_index(self, vertex: int | Any) -> int:
        """Normalizes a vertex index or vertex-like value.

        Args:
            vertex: Integer vertex index or vertex-like value.

        Returns:
            Integer vertex index in the graph.
        """
        if isinstance(vertex, int):
            if vertex < 0 or vertex >= self.graph.n_vertices:
                raise IndexError(f"vertex index {vertex} out of range.")
            return vertex
        canonical = Vertex.from_like(vertex)
        try:
            return self.graph.vertices.index(canonical)
        except ValueError as exc:
            raise KeyError(f"Unknown graph vertex {canonical!r}.") from exc


def _entity_spaces(
    explicit_spaces: tuple[DofSpace | None, ...] | None,
    default_space: DofSpace | None,
    n_entities: int,
) -> tuple[DofSpace | None, ...]:
    """Resolves per-entity spaces from explicit overrides or a default.

    Args:
        explicit_spaces: Optional sequence of spaces with one entry per entity.
        default_space: Space used for every entity when explicit spaces are not
            supplied.
        n_entities: Number of graph entities in the block.

    Returns:
        Tuple containing one coerced space or ``None`` per entity.
    """
    if explicit_spaces is not None:
        if len(explicit_spaces) != n_entities:
            raise ValueError(
                f"Expected {n_entities} entity local spaces, got {len(explicit_spaces)}."
            )
        return tuple(coerce_dof_space(space) for space in explicit_spaces)
    return (coerce_dof_space(default_space),) * n_entities


def _uniform_space(spaces: tuple[DofSpace | None, ...]) -> DofSpace | None:
    """Returns a shared non-empty space when all entities use the same one.

    Args:
        spaces: Per-entity space declarations.

    Returns:
        Shared degree-of-freedom declaration, or ``None`` when spaces are empty
        or heterogeneous.
    """
    nonempty = tuple(space for space in spaces if space is not None)
    if not nonempty:
        return None
    first = nonempty[0]
    if all(space == first for space in nonempty) and len(nonempty) == len(spaces):
        return first
    return None


def _uniform_width(widths: tuple[int, ...], kind: str) -> int:
    """Returns a common positive width for an entity block.

    Args:
        widths: Per-entity scalar widths.
        kind: Entity kind used in validation errors.

    Returns:
        Common width, or zero when all widths are zero.

    Raises:
        ValueError: If positive widths are heterogeneous.
    """
    common = _common_positive_width(widths)
    if common is not None:
        return common
    if all(width == 0 for width in widths):
        return 0
    raise ValueError(f"{kind} DOF widths are heterogeneous; use {kind}_widths.")


def _common_positive_width(widths: tuple[int, ...]) -> int | None:
    """Returns the common positive width when every entity has one.

    Args:
        widths: Per-entity scalar widths.

    Returns:
        Shared positive width, or ``None`` when the block is empty or mixed.
    """
    positive = tuple(width for width in widths if width > 0)
    if not positive:
        return None
    first = positive[0]
    if len(positive) == len(widths) and all(width == first for width in positive):
        return first
    return None


def _entity_first_component_offsets(widths: tuple[int, ...]) -> tuple[int, ...]:
    """Computes first-component offsets for every entity.

    Args:
        widths: Per-entity scalar widths.

    Returns:
        Offset of component zero for each non-empty entity, and ``-1`` for
        entities without degrees of freedom.
    """
    return tuple(
        _entity_component_site(widths, entity_index, 0, "entity") if width > 0 else -1
        for entity_index, width in enumerate(widths)
    )


def _entity_component_site(
    widths: tuple[int, ...],
    entity_index: int,
    component: int,
    kind: str,
) -> int:
    """Computes a component-major site index inside one entity block.

    Args:
        widths: Per-entity scalar widths.
        entity_index: Entity index within the block.
        component: Component index within the entity.
        kind: Entity kind used in validation errors.

    Returns:
        Site index relative to the start of the entity block.
    """
    width = widths[entity_index]
    if width == 0:
        raise ValueError(f"This layout has no DOFs for that {kind}.")
    if component < 0 or component >= width:
        raise IndexError(f"component={component} out of range [0,{width}).")
    preceding_components = sum(min(width, component) for width in widths)
    preceding_entities = sum(1 for width in widths[:entity_index] if width > component)
    return preceding_components + preceding_entities


def _entity_from_site(
    site: int,
    widths: tuple[int, ...],
    kind: str,
) -> tuple[int, int]:
    """Decodes a component-major site index inside one entity block.

    Args:
        site: Site index relative to the start of the entity block.
        widths: Per-entity scalar widths.
        kind: Entity kind used in validation errors.

    Returns:
        Entity index and component index represented by ``site``.

    Raises:
        IndexError: If ``site`` does not correspond to an active entity
            component.
    """
    cursor = 0
    for component in range(max(widths, default=0)):
        for entity_index, width in enumerate(widths):
            if width <= component:
                continue
            if cursor == site:
                return entity_index, component
            cursor += 1
    raise IndexError(f"site={site} is not a {kind} site.")


GraphDofLayout = GraphHilbertLayout


__all__ = ["GraphDofLayout", "GraphHilbertLayout", "HilbertStateView"]
