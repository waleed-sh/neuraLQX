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


"""Planar 2+1-dimensional U(1)^3 WCL model facade."""

from __future__ import annotations

from typing import Any

from neuralqx.operator.prebuilt import wcl
from neuralqx.operator.prebuilt.utils import graph_vertex
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_planar_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert

from .wcl_1d import LqxWCL1D


class LqxWCL3D(LqxWCL1D):
    r"""The planar U(1)^3 weak-coupling model in 2+1 spacetime dimensions.

    The model exposes cyclic holonomies, Gauss violation, planar vertex volume,
    and the Thiemann constraint built from pairs of minimal loops. The default
    Euclidean component is $H_E=\sum_v H_v$, and no Lorentzian component is
    supplied. The total constraint is therefore this linear sum. The quadratic
    property represents $\sum_v H_v H_v^\dagger$, with each vertex squared
    separately. `thiemann_quadratic_constraint` returns the corresponding
    local products as a list.

    Volume and holonomy construction require an explicit vertex or edge.
    Area and mass implementations are unavailable. BF curvature
    remains accessible through `curvature_constraint`.

    Lapses are optional and are applied only when a factory is called with
    `apply_lapse=True`. A missing vertex entry uses lapse one. Vertices and
    integer indices are passed through to the operator factories, while lapse
    lookup uses the exact supplied dictionary key.

    Args:
        hilbert (AbstractU1Hilbert): Three-copy U(1) Hilbert space on a planar graph.
        graph (AbstractGraph | None): Optional graph object, identical to `hilbert.graph`.
        lapses: Vertex-to-lapse mapping copied into the model.
    """

    model_name = "(2+1)-Euclidean LQG"
    """Descriptive name of the physical model."""

    lapses: dict[Any, float]
    """Optional real lapse values keyed by the vertex argument used in factory calls."""

    def __init__(
        self, hilbert, graph=None, *, lapses: dict[Any, float] | None = None
    ) -> None:
        """Initialize the model and validate its graph and parameter requirements.

        Arguments have the meanings documented on the model class. An explicitly
        supplied graph must be the graph object attached to the Hilbert space.
        """
        super().__init__(hilbert, graph)
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        require_planar_graph(self.graph, model=type(self).__name__)
        self.lapses = {} if lapses is None else dict(lapses)

    def __repr__(self) -> str:
        """Return the model type and its defining Hilbert-space and graph parameters."""
        return f"LqxWCL3D(hilbert={self.hilbert!r}, graph={self.graph!r}, n_lapses={len(self.lapses)})"

    def get_holonomy(
        self,
        edge: Any = None,
        *,
        adjoint: bool = False,
        dimension: int = 3,
        copy: int = 0,
    ):
        """Construct a cyclic single-edge holonomy.

        Args:
            edge: Graph edge or base-edge index. An explicit edge is required.
            adjoint: Reverse the cyclic shift when true.
            dimension: Use one selected component with `1`, or the sum of all
                three component holonomies with `3` in a U(1)^3 model.
            copy: Component selected when `dimension=1`.

        Returns:
            result (ComputationalOperator): A computational operator with one row target per selected component.
        """
        if edge is None:
            self._not_implemented("holonomy", "Select an edge with get_holonomy(edge).")
        if int(dimension) == 1:
            return wcl.U1HolonomyOperator(
                self.hilbert, edge, adjoint_flag=adjoint, copy=copy
            )
        if int(dimension) != 3:
            raise ValueError("LqxWCL3D supports dimension=1 or dimension=3.")
        return wcl.NDU1HolonomyOperator(self.hilbert, edge, adjoint_flag=adjoint)

    def minimal_loop_holonomy(
        self,
        loop: Any,
        *,
        adjoint: bool = False,
        dimension: int = 3,
        copy: int = 0,
    ):
        """Construct a cyclic holonomy around a dressed graph loop.

        Args:
            loop: Dressed loop, preferably from `graph.dressed_minimal_loops()`.
            adjoint: Reverse the loop shifts when true.
            dimension: Select one component with `1`, or sum three components
                with `3` in a U(1)^3 model.
            copy: Component selected when `dimension=1`.

        Returns:
            result (ComputationalOperator): A computational loop operator. Each branch shifts the entire loop
                in one gauge component.
        """
        if int(dimension) == 1:
            return wcl.MinimalLoopHolonomyOperator(
                self.hilbert,
                loop,
                copy=copy,
                adjoint_flag=adjoint,
            )
        if int(dimension) != 3:
            raise ValueError("LqxWCL3D supports dimension=1 or dimension=3.")
        return wcl.NDMinimalLoopHolonomyOperator(
            self.hilbert,
            loop,
            adjoint_flag=adjoint,
        )

    def curvature_constraint(self, loops: list[Any] | None = None):
        r"""Sum $2I-H_\alpha-H_\alpha^\dagger$ over the selected loops and gauge copies.

        Args:
            loops: Loop list to include. `None` selects all dressed minimal loops.

        Returns:
            result (ComputationalOperator): A Hermitian positive-semidefinite operator. An empty loop list gives
                the zero operator on this Hilbert space.
        """
        selected = self.graph.dressed_minimal_loops() if loops is None else loops
        result = self.identity(2.0 * len(selected) * self.gauge_dimensions)
        for loop in selected:
            h = self.minimal_loop_holonomy(loop, dimension=3)
            result = result - h - h.adjoint
        return result

    def gauss_constraint(self, *, modular: bool = False):
        """Construct the total squared Gauss-law violation over all vertices and copies.

        Args:
            modular: Reduce each residual to the local cyclic range before
                squaring when true.

        Returns:
            result (wcl.GaussConstraintOperator): A non-negative diagonal `GaussConstraintOperator`.
        """
        return wcl.GaussConstraintOperator(self.hilbert, modular=modular)

    gauss = gauss_constraint
    """Alias for `gauss_constraint`."""

    def get_volume(
        self, vertex: Any = None, *, squared: bool = False, sqrt: bool = False
    ):
        """Construct the planar volume at an explicitly selected vertex.

        Args:
            vertex: Graph vertex or its index. An explicit vertex is required.
            squared: Square the selected vertex volume.
            sqrt: Take the square root after squaring, if requested.

        Returns:
            result (ComputationalOperator): A diagonal volume operator in the model's dimensionless normalization.
        """
        if vertex is None:
            self._not_implemented("volume", "Select a vertex with get_volume(vertex).")
        return wcl.PlanarVolumeOperator(
            self.hilbert, vertex, squared=squared, sqrt=sqrt
        )

    def squared_volume(self, vertex: Any):
        """Construct the square of the planar vertex-volume eigenvalue.

        Args:
            vertex: Graph vertex or its index.

        Returns:
            result (wcl.PlanarVolumeOperator): A diagonal `PlanarVolumeOperator` with `squared=True`.
        """
        return self.get_volume(vertex, squared=True)

    def sqrt_volume(self, vertex: Any):
        """Construct the square root of the planar vertex-volume eigenvalue.

        Args:
            vertex: Graph vertex or its index.

        Returns:
            result (wcl.PlanarVolumeOperator): A diagonal `PlanarVolumeOperator` with `sqrt=True`, as used in
                the planar Thiemann commutators.
        """
        return self.get_volume(vertex, sqrt=True)

    def get_euclidean_constraint(
        self, vertex: Any = None, *, adjoint: bool = False, apply_lapse: bool = False
    ):
        r"""Construct $H_v$ at one vertex or $H_E=\sum_v H_v$ over the graph.

        Args:
            vertex: Graph vertex or its index. None selects every vertex.
            adjoint: Return the mathematical adjoint of the selected linear constraint.
            apply_lapse: Include the configured vertex lapses before summing.

        Returns:
            result (ComputationalOperator): A linear planar Thiemann constraint.
        """
        vertices = self.graph.vertices if vertex is None else (vertex,)
        return self.sum_constraint_terms(
            {
                graph_vertex(self.graph, v): self.thiemann_regularized_constraint(
                    v, adjoint=adjoint, apply_lapse=apply_lapse
                )
                for v in vertices
            }
        )

    def thiemann_regularized_constraint(
        self,
        vertex: Any,
        *,
        adjoint: bool = False,
        apply_lapse: bool = False,
    ):
        """Construct the Thiemann constraint at one vertex.

        Args:
            vertex: Graph vertex or its index.
            adjoint: Return the conjugate-transpose constraint when true.
            apply_lapse: Multiply by `lapses.get(vertex, 1.0)` when true.

        Returns:
            result (ComputationalOperator): A computational vertex operator. A vertex without contributing
                triangulation terms gives a zero-amplitude row.
        """
        lapse = self.lapses.get(vertex, 1.0) if apply_lapse else 1.0
        return wcl.Thiemann2p1VertexConstraintOperator(
            self.hilbert,
            vertex,
            adjoint_flag=adjoint,
            lapse=float(lapse),
        )

    def thiemann_quadratic_constraint(
        self,
        *,
        alternative: bool = False,
        apply_lapse: bool = False,
        lazy: bool = True,
    ):
        r"""Return the local quadratic Thiemann constraints.

        Each list entry is a positive operator, evaluated without expanding
        the two-factor connectivity. `Squared(A)` means $A^\dagger A$.
        The default expectation is $\|H_v^\dagger\psi\|^2$ for normalized $\psi$.

        Args:
            alternative: Return $H_v^\dagger H_v$ when true. The default
                returns $H_v H_v^\dagger$ as `Squared(H_v.adjoint)`.
            apply_lapse: Include each vertex's configured lapse before squaring.
            lazy: Must be true. Explicit connectivity expansion is not provided
                by this factory.

        Returns:
            result (list[Squared]): One `Squared` constraint per graph vertex.
        """
        if not lazy:
            raise NotImplementedError(
                "Only lazy squared WCL Thiemann constraints are currently exposed."
            )
        return [
            self._quadratic_of(
                self.thiemann_regularized_constraint(vertex, apply_lapse=apply_lapse),
                alternative=alternative,
            )
            for vertex in self.graph.vertices
        ]


__all__ = ["LqxWCL3D"]
