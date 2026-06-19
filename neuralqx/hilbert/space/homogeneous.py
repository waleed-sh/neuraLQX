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


"""Homogeneous discrete Hilbert spaces.

Homogeneous spaces attach the same scalar or vector local domain to every
enabled graph entity. They are the most compact representation for models whose
edges, vertices, or both share a uniform local Hilbert domain.
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
from neuralqx.hilbert.local import VectorRange
from neuralqx.hilbert.local.vector import coerce_dof_space
from neuralqx.hilbert.local.vector import scalar_space
from neuralqx.utils.struct import field

from .discrete import DiscreteHilbertSpace


class HomogeneousDiscreteHilbert(DiscreteHilbertSpace):
    """Flat discrete space with the same local domain on every enabled entity.

    Args:
        graph: Graph whose edges and vertices define the Hilbert sites.
        local_space: Scalar or vector local domain shared by enabled entities.
        edges: Whether graph edges carry degrees of freedom.
        vertices: Whether graph vertices carry degrees of freedom.
        constraint: Optional global constraint. Defaults to
            ``IdentityConstraint``.

    Attributes:
        graph: Graph whose edges and vertices define the Hilbert sites.
        local_space: Scalar or vector local domain shared by enabled entities.
        edges: Whether graph edges carry degrees of freedom.
        vertices: Whether graph vertices carry degrees of freedom.
        constraint: Global constraint applied after local-domain checks.
        layout: Graph layout that maps semantic entities to flat sites.
        _local_spaces: Scalar local space for every flat site.
        _dtype: Common dtype used for state values.
    """

    graph: AbstractGraph = field(pytree=False)
    """Graph whose edges and vertices define the Hilbert sites."""

    local_space: AbstractLocalSpace | VectorRange | Any = field(static=True)
    """Scalar or vector local domain shared by enabled entities."""

    edges: bool = field(static=True, default=True)
    """Whether graph edges carry degrees of freedom."""

    vertices: bool = field(static=True, default=False)
    """Whether graph vertices carry degrees of freedom."""

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

    _dtype: np.dtype = field(
        static=True,
        init=False,
        default_factory=lambda: np.dtype(np.float32),
        repr=False,
        compare=False,
    )
    """Common dtype used for state values."""

    def __post_init__(self) -> None:
        """Builds the graph layout, scalar site spaces, and constraint."""
        if not isinstance(self.edges, bool) or not isinstance(self.vertices, bool):
            raise TypeError("edges and vertices must be booleans.")
        if not self.edges and not self.vertices:
            raise ValueError("At least one of edges or vertices must be enabled.")
        local_space = coerce_dof_space(self.local_space)
        edge_space = local_space if self.edges else None
        vertex_space = local_space if self.vertices else None
        layout = GraphHilbertLayout(
            self.graph, edge_space=edge_space, vertex_space=vertex_space
        )
        local_spaces = _expand_layout_spaces(layout)
        constraint = (
            IdentityConstraint() if self.constraint is None else self.constraint
        )
        object.__setattr__(self, "local_space", local_space)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_local_spaces", local_spaces)
        object.__setattr__(
            self, "_dtype", np.result_type(*(space.dtype for space in local_spaces))
        )
        object.__setattr__(self, "constraint", constraint)
        constraint.validate_hilbert(self)

    def __hash__(self) -> int:
        """Returns a structural hash for the homogeneous Hilbert space."""
        return hash(
            (
                type(self),
                self.graph,
                self.local_space,
                self.edges,
                self.vertices,
                self.constraint,
            )
        )

    @property
    def size(self) -> int:
        """Total number of flat scalar sites."""
        return self.layout.size

    @property
    def local_sizes(self) -> tuple[int, ...]:
        """Local dimension at every flat scalar site."""
        return tuple(space.local_size for space in self._local_spaces)

    @property
    def local_size(self) -> int:
        """Shared scalar local dimension."""
        return scalar_space(self.local_space).local_size

    @property
    def dtype(self) -> np.dtype:
        """Common dtype used for state values."""
        return self._dtype

    @property
    def local_spaces(self) -> tuple[AbstractLocalSpace, ...]:
        """Scalar local space for every flat scalar site."""
        return self._local_spaces

    def with_constraint(
        self, constraint: AbstractDiscreteConstraint
    ) -> HomogeneousDiscreteHilbert:
        """Returns a copy of this space with a different constraint.

        Args:
            constraint: Constraint to attach to the returned space.

        Returns:
            New homogeneous Hilbert space sharing the same graph and local
            domain.
        """
        return self.replace(constraint=constraint)

    def states_to_local_indices(self, states: Any) -> jax.Array:
        """Maps state values to per-site local indices.

        Args:
            states: State array whose trailing dimension equals :attr:`size`.

        Returns:
            Integer array with the same shape as ``states``.
        """
        arr = jnp.asarray(states)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]}, expected {self.size}."
            )
        out = jnp.empty(arr.shape, dtype=jnp.int32)
        for site, local_space in enumerate(self.local_spaces):
            out = out.at[..., site].set(local_space.values_to_indices(arr[..., site]))
        return out

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
        idx = jnp.asarray(indices)
        if idx.shape[-1] != self.size:
            raise ValueError(
                f"Index trailing dimension {idx.shape[-1]}, expected {self.size}."
            )
        out_dtype = self.dtype if dtype is None else np.dtype(dtype)
        out = jnp.empty(idx.shape, dtype=out_dtype)
        for site, local_space in enumerate(self.local_spaces):
            out = out.at[..., site].set(
                local_space.indices_to_values(idx[..., site], dtype=out_dtype)
            )
        return out

    def local_states_valid(self, states: Any) -> jax.Array:
        """Checks local-domain membership at every flat site.

        Args:
            states: State array whose trailing dimension equals :attr:`size`.

        Returns:
            Boolean mask over the batch dimensions. The result checks only the
            shared scalar local space assigned to each enabled graph entity and
            does not evaluate any non-local constraint.
        """
        arr = jnp.asarray(states, dtype=self.dtype)
        if arr.shape[-1] != self.size:
            raise ValueError(
                f"State trailing dimension {arr.shape[-1]}, expected {self.size}."
            )
        ok = jnp.ones(arr.shape, dtype=jnp.bool_)
        for site, local_space in enumerate(self.local_spaces):
            ok = ok.at[..., site].set(local_space.contains(arr[..., site]))
        return jnp.all(ok, axis=-1)

    def view(self, states: Any) -> HilbertStateView:
        """Views flat homogeneous states through their graph layout.

        Args:
            states: Flat state array whose trailing dimension equals
                :attr:`size`.

        Returns:
            :class:`HilbertStateView` containing edge and vertex blocks with
            the scalar widths declared by this homogeneous layout.
        """
        return self.layout.view(states)

    def flatten(self, view: HilbertStateView | None = None, **blocks: Any) -> jax.Array:
        """Flattens graph-layout blocks into the homogeneous state layout.

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
            states: Flat state array in this homogeneous layout.
            edge: Edge index or graph edge label accepted by the layout.

        Returns:
            Scalar or vector edge values with the leading batch axes preserved.
        """
        return self.layout.edge_values(states, edge)

    def vertex_values(self, states: Any, vertex: int | Any) -> jax.Array:
        """Returns the state block attached to one graph vertex.

        Args:
            states: Flat state array in this homogeneous layout.
            vertex: Vertex index or graph vertex label accepted by the layout.

        Returns:
            Scalar or vector vertex values with leading batch axes preserved.
        """
        return self.layout.vertex_values(states, vertex)


def _expand_layout_spaces(layout: GraphHilbertLayout) -> tuple[AbstractLocalSpace, ...]:
    """Expands layout degree-of-freedom declarations to scalar site spaces.

    Args:
        layout: Graph Hilbert layout whose flat sites should be expanded.

    Returns:
        Scalar local space for every flat site in layout order.
    """
    out: list[AbstractLocalSpace] = []
    for dof in layout.dofs:
        spaces = layout.edge_spaces if dof.kind == "edge" else layout.vertex_spaces
        space = spaces[dof.entity_index]
        if space is None:
            raise RuntimeError("layout.dofs cannot reference an empty local space.")
        out.append(scalar_space(space))
    return tuple(out)


__all__ = ["HomogeneousDiscreteHilbert"]
