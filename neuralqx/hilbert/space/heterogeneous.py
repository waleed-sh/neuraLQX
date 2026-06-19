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


"""Heterogeneous discrete Hilbert spaces.

Heterogeneous spaces attach site-dependent local domains to graph entities.
They support defaults for all edges or vertices plus per-entity overrides,
including scalar and vector degrees of freedom.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.constraint import AbstractDiscreteConstraint
from neuralqx.hilbert.constraint import IdentityConstraint
from neuralqx.hilbert.layout import GraphHilbertLayout
from neuralqx.hilbert.layout import HilbertStateView
from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.local import HeterogeneousLocalSpace
from neuralqx.hilbert.local import VectorRange
from neuralqx.hilbert.local.vector import scalar_space
from neuralqx.hilbert.utils import ensure_2d_trailing
from neuralqx.hilbert.utils import restore_trailing
from neuralqx.utils.struct import field
from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import Vertex

from .discrete import DiscreteHilbertSpace


class HeterogeneousDiscreteHilbert(DiscreteHilbertSpace):
    """Flat discrete space with site-dependent local domains.

    Args:
        graph: Graph whose edges and vertices define the Hilbert sites.
        local_space: Heterogeneous edge and vertex local-space declaration.
        constraint: Optional global constraint. Defaults to
            ``IdentityConstraint``.

    Attributes:
        graph: Graph whose edges and vertices define the Hilbert sites.
        local_space: Heterogeneous edge and vertex local-space declaration.
        constraint: Global constraint applied after local-domain checks.
        layout: Graph layout that maps semantic entities to flat sites.
        _local_spaces: Scalar local space for every flat site.
        _unique_local_spaces: Unique scalar local spaces used by the flat sites.
        _site_space_ids: Index of the unique local space used by each flat site.
        _groups: Flat site groups sharing each unique local space.
        _dtype: Common dtype used for state values.
    """

    graph: AbstractGraph = field(pytree=False)
    """Graph whose edges and vertices define the Hilbert sites."""

    local_space: HeterogeneousLocalSpace = field(static=True)
    """Heterogeneous edge and vertex local-space declaration."""

    constraint: AbstractDiscreteConstraint | None = field(static=True, default=None)
    """Global constraint applied after local-domain checks."""

    layout: GraphHilbertLayout = field(pytree=False, init=False, default=None)
    """Graph layout that maps semantic entities to flat sites."""

    _local_spaces: tuple[AbstractLocalSpace, ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    """Scalar local space for every flat site."""

    _unique_local_spaces: tuple[AbstractLocalSpace, ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    """Unique scalar local spaces used by the flat sites."""

    _site_space_ids: tuple[int, ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    """Index of the unique local space used by each flat site."""

    _groups: tuple[tuple[int, ...], ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    """Flat site groups sharing each unique local space."""

    _dtype: np.dtype = field(
        static=True,
        init=False,
        default_factory=lambda: np.dtype(np.float32),
        repr=False,
        compare=False,
    )
    """Common dtype used for state values."""

    def __post_init__(self) -> None:
        """Builds the layout, site-space groups, dtype, and constraint."""
        local_space = (
            self.local_space
            if isinstance(self.local_space, HeterogeneousLocalSpace)
            else HeterogeneousLocalSpace(edge=self.local_space)
        )
        edge_spaces = _resolve_edge_spaces(self.graph, local_space)
        vertex_spaces = _resolve_vertex_spaces(self.graph, local_space)
        layout = GraphHilbertLayout(
            self.graph,
            edge_spaces=edge_spaces,
            vertex_spaces=vertex_spaces,
        )
        spaces = _expand_layout_spaces(layout)
        constraint = (
            IdentityConstraint() if self.constraint is None else self.constraint
        )
        unique, site_ids, groups = _group_local_spaces(spaces)
        dtype = np.result_type(*(space.dtype for space in spaces))
        object.__setattr__(self, "local_space", local_space)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_local_spaces", spaces)
        object.__setattr__(self, "_unique_local_spaces", unique)
        object.__setattr__(self, "_site_space_ids", site_ids)
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_dtype", np.dtype(dtype))
        object.__setattr__(self, "constraint", constraint)
        constraint.validate_hilbert(self)

    def __hash__(self) -> int:
        """Returns a structural hash for the heterogeneous Hilbert space."""
        return hash((type(self), self.graph, self.local_space, self.constraint))

    @property
    def size(self) -> int:
        """Total number of flat scalar sites."""
        return len(self.local_spaces)

    @property
    def local_sizes(self) -> tuple[int, ...]:
        """Local dimension at every flat scalar site."""
        return tuple(space.local_size for space in self.local_spaces)

    @property
    def local_spaces(self) -> tuple[AbstractLocalSpace, ...]:
        """Scalar local space for every flat scalar site."""
        return self._local_spaces

    @property
    def dtype(self) -> np.dtype:
        """Common dtype used for state values."""
        return self._dtype

    @property
    def unique_local_spaces(self) -> tuple[AbstractLocalSpace, ...]:
        """Unique scalar local spaces used by flat sites."""
        return self._unique_local_spaces

    @property
    def site_space_ids(self) -> tuple[int, ...]:
        """Index of the unique local space used by each flat site."""
        return self._site_space_ids

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Flat site groups sharing each unique local space."""
        return self._groups

    def with_constraint(
        self,
        constraint: AbstractDiscreteConstraint,
    ) -> HeterogeneousDiscreteHilbert:
        """Returns a copy of this space with a different constraint.

        Args:
            constraint: Constraint to attach to the returned space.

        Returns:
            New heterogeneous Hilbert space sharing the same graph and local
            domain declaration.
        """
        return self.replace(constraint=constraint)

    def states_to_local_indices(self, states: Any) -> jax.Array:
        """Maps state values to per-site local indices.

        Args:
            states: State array whose trailing dimension equals :attr:`size`.

        Returns:
            Integer array with the same shape as ``states``.
        """
        arr, single, batch = ensure_2d_trailing(states, self.size)
        out = jnp.empty(arr.shape, dtype=jnp.int32)
        for local_space, sites in zip(
            self.unique_local_spaces, self.groups, strict=True
        ):
            site_idx = jnp.asarray(sites, dtype=jnp.int32)
            out = out.at[:, site_idx].set(
                local_space.values_to_indices(jnp.take(arr, site_idx, axis=-1))
            )
        return restore_trailing(out, single, batch)

    def local_indices_to_states(
        self, indices: Any, *, dtype: Any | None = None
    ) -> jax.Array:
        """Maps per-site local indices to state values.

        Args:
            indices: Integer index array whose trailing dimension equals
                :attr:`size`.
            dtype: Optional dtype for returned values.

        Returns:
            State-value array with the same shape as ``indices``.
        """
        idx, single, batch = ensure_2d_trailing(indices, self.size)
        out_dtype = self.dtype if dtype is None else np.dtype(dtype)
        out = jnp.empty(idx.shape, dtype=out_dtype)
        for local_space, sites in zip(
            self.unique_local_spaces, self.groups, strict=True
        ):
            site_idx = jnp.asarray(sites, dtype=jnp.int32)
            values = local_space.indices_to_values(
                jnp.take(idx, site_idx, axis=-1),
                dtype=out_dtype,
            )
            out = out.at[:, site_idx].set(values)
        return restore_trailing(out, single, batch)

    def local_states_valid(self, states: Any) -> jax.Array:
        """Checks local-domain membership at every flat site.

        Args:
            states: State array whose trailing dimension equals :attr:`size`.

        Returns:
            Boolean mask over the batch dimensions. Each flat site is checked
            against the scalar local space assigned to that site by the
            heterogeneous graph layout.
        """
        arr, single, _batch = ensure_2d_trailing(states, self.size)
        ok = jnp.ones(arr.shape, dtype=jnp.bool_)
        for local_space, sites in zip(
            self.unique_local_spaces, self.groups, strict=True
        ):
            site_idx = jnp.asarray(sites, dtype=jnp.int32)
            ok = ok.at[:, site_idx].set(
                local_space.contains(jnp.take(arr, site_idx, axis=-1))
            )
        valid = jnp.all(ok, axis=-1)
        return valid[0] if single else valid

    def view(self, states: Any) -> HilbertStateView:
        """Views flat heterogeneous states through their graph layout.

        Args:
            states: Flat state array whose trailing dimension equals
                :attr:`size`.

        Returns:
            :class:`HilbertStateView` containing edge and vertex blocks. Each
            entity block keeps the scalar width declared for that entity.
        """
        return self.layout.view(states)

    def flatten(self, view: HilbertStateView | None = None, **blocks: Any) -> jax.Array:
        """Flattens graph-layout blocks into the heterogeneous state layout.

        Args:
            view: Optional prebuilt graph-layout view.
            **blocks: Edge or vertex blocks accepted by the underlying layout
                when ``view`` is not supplied.

        Returns:
            Flat state array with trailing dimension :attr:`size`.
        """
        return self.layout.flatten(view, **blocks)

    def edge_values(self, states: Any, edge: int | Any) -> jax.Array:
        """Returns the state block attached to one graph edge.

        Args:
            states: Flat state array in this heterogeneous layout.
            edge: Edge index or graph edge label accepted by the layout.

        Returns:
            Scalar or vector edge values using that edge's declared local
            space, with leading batch axes preserved.
        """
        return self.layout.edge_values(states, edge)

    def vertex_values(self, states: Any, vertex: int | Any) -> jax.Array:
        """Returns the state block attached to one graph vertex.

        Args:
            states: Flat state array in this heterogeneous layout.
            vertex: Vertex index or graph vertex label accepted by the layout.

        Returns:
            Scalar or vector vertex values using that vertex's declared local
            space, with leading batch axes preserved.
        """
        return self.layout.vertex_values(states, vertex)


def _group_local_spaces(
    spaces: tuple[AbstractLocalSpace, ...],
) -> tuple[
    tuple[AbstractLocalSpace, ...], tuple[int, ...], tuple[tuple[int, ...], ...]
]:
    """Groups flat sites by equal scalar local space.

    Args:
        spaces: Scalar local space for every flat site.

    Returns:
        Unique local spaces, the unique-space id for each flat site, and the
        flat site groups for each unique space.
    """
    unique: list[AbstractLocalSpace] = []
    site_ids: list[int] = []
    groups: list[list[int]] = []
    for site, local_space in enumerate(spaces):
        try:
            idx = unique.index(local_space)
        except ValueError:
            idx = len(unique)
            unique.append(local_space)
            groups.append([])
        site_ids.append(idx)
        groups[idx].append(site)
    return tuple(unique), tuple(site_ids), tuple(tuple(group) for group in groups)


def _expand_layout_spaces(layout: GraphHilbertLayout) -> tuple[AbstractLocalSpace, ...]:
    """Expands a graph layout into scalar local spaces in flat order.

    Args:
        layout: Graph Hilbert layout to expand.

    Returns:
        Scalar local space for every flat site.
    """
    out: list[AbstractLocalSpace] = []
    for dof in layout.dofs:
        spaces = layout.edge_spaces if dof.kind == "edge" else layout.vertex_spaces
        space = spaces[dof.entity_index]
        if space is None:
            raise RuntimeError("layout.dofs cannot reference an empty local space.")
        out.append(scalar_space(space))
    return tuple(out)


def _resolve_edge_spaces(
    graph: AbstractGraph,
    local_space: HeterogeneousLocalSpace,
) -> tuple[AbstractLocalSpace | VectorRange | None, ...]:
    """Resolves default and override spaces for graph edges.

    Args:
        graph: Graph whose edges are being assigned spaces.
        local_space: Heterogeneous local-space declaration.

    Returns:
        Tuple with one scalar, vector, or empty declaration per graph edge.
    """
    spaces = [local_space.edge] * graph.n_edges
    for edge, space in local_space.edge_spaces:
        spaces[_edge_index(graph, edge)] = space
    return tuple(spaces)


def _resolve_vertex_spaces(
    graph: AbstractGraph,
    local_space: HeterogeneousLocalSpace,
) -> tuple[AbstractLocalSpace | VectorRange | None, ...]:
    """Resolves default and override spaces for graph vertices.

    Args:
        graph: Graph whose vertices are being assigned spaces.
        local_space: Heterogeneous local-space declaration.

    Returns:
        Tuple with one scalar, vector, or empty declaration per graph vertex.
    """
    spaces = [local_space.vertex] * graph.n_vertices
    for vertex, space in local_space.vertex_spaces:
        spaces[_vertex_index(graph, vertex)] = space
    return tuple(spaces)


def _edge_index(graph: AbstractGraph, edge: int | Edge) -> int:
    """Normalizes an edge override key to an edge index.

    Args:
        graph: Graph containing the target edge.
        edge: Integer edge index or canonical edge.

    Returns:
        Integer edge index.
    """
    if isinstance(edge, int):
        if edge < 0 or edge >= graph.n_edges:
            raise IndexError(f"edge index {edge} out of range.")
        return edge
    return graph.edge_to_index(edge)


def _vertex_index(graph: AbstractGraph, vertex: int | Vertex) -> int:
    """Normalizes a vertex override key to a vertex index.

    Args:
        graph: Graph containing the target vertex.
        vertex: Integer vertex index or canonical vertex.

    Returns:
        Integer vertex index.
    """
    if isinstance(vertex, int):
        if vertex < 0 or vertex >= graph.n_vertices:
            raise IndexError(f"vertex index {vertex} out of range.")
        return vertex
    try:
        return graph.vertices.index(vertex)
    except ValueError as exc:
        raise KeyError(f"Unknown graph vertex {vertex!r}.") from exc


__all__ = ["HeterogeneousDiscreteHilbert"]
