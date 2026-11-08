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


"""Non-planar 3+1-dimensional U(1)^3 WCL model facade."""

from __future__ import annotations

from typing import Any

from neuralqx.operator.prebuilt import wcl
from neuralqx.operator.prebuilt.utils import admissible_thiemann_3p1_vertices
from neuralqx.operator.prebuilt.utils import graph_vertex
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_nonplanar_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert

from .wcl_1d import LqxWCL1D


class LqxWCL4D(LqxWCL1D):
    r"""The non-planar U(1)^3 weak-coupling model in 3+1 spacetime dimensions.

    This facade supplies surface areas, vertex volumes, cyclic holonomies,
    Gauss violation, and the local Thiemann constraint. Admissible vertices are
    those with a loop-and-segment contribution. Its default Euclidean component
    is $H_E=\sum_v H_v$ over these vertices. No Lorentzian term is supplied,
    so the total is this linear sum. The quadratic property represents
    $\sum_v H_v H_v^\dagger$, with each vertex squared separately. The
    `thiemann_quadratic_constraint` factory returns these local products as a list.

    Area and volume construction require an explicit surface or vertex.
    Holonomy construction requires an explicit edge.
    A mass operator is not supplied. BF curvature remains available through
    `curvature_constraint`.
    A lapse is used only when `apply_lapse=True`, with a default of one for
    keys absent from `lapses`. Lapse lookup uses the supplied vertex key.

    Args:
        hilbert (AbstractU1Hilbert): Three-copy U(1) Hilbert space on a non-planar graph.
        graph (AbstractGraph | None): Optional graph object, identical to `hilbert.graph`.
        lapses: Vertex-to-lapse mapping copied into the model.
    """

    model_name = "(3+1)-Euclidean LQG"
    """Descriptive name of the physical model."""

    lapses: dict[Any, float]
    """Optional real lapse values keyed by the vertex argument used in factory calls."""

    _admissible_vertices: tuple[Any, ...]
    """Vertices with at least one loop-and-segment Thiemann contribution."""

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
        require_nonplanar_graph(self.graph, model=type(self).__name__)
        self.lapses = {} if lapses is None else dict(lapses)
        self._admissible_vertices = admissible_thiemann_3p1_vertices(hilbert)

    def __repr__(self) -> str:
        """Return the model type and its defining Hilbert-space and graph parameters."""
        return (
            f"LqxWCL4D(hilbert={self.hilbert!r}, graph={self.graph!r}, "
            f"n_admissible={len(self._admissible_vertices)}, n_lapses={len(self.lapses)})"
        )

    @property
    def admissible_vertices(self):
        """Graph vertices that contribute loop-and-segment terms to the 3+1 constraint."""
        return self._admissible_vertices

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
            raise ValueError("LqxWCL4D supports dimension=1 or dimension=3.")
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
            raise ValueError("LqxWCL4D supports dimension=1 or dimension=3.")
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

    def get_area(
        self,
        surface: tuple[Any, ...] | list[Any] | None = None,
        *,
        squared: bool = False,
    ):
        """Construct the area of a surface specified by its puncturing edges.

        Args:
            surface: Explicit sequence of puncturing graph edges or base-edge indices.
            squared: Sum squared puncture areas when true. This does not square
                the total surface area.

        Returns:
            result (wcl.AreaOperator): A diagonal `AreaOperator` on the model's three gauge copies.
        """
        if surface is None:
            self._not_implemented("area", "Select a surface with get_area(surface).")
        return wcl.AreaOperator(self.hilbert, tuple(surface), squared=squared)

    def area_difference_squared(self, edges: tuple[Any, Any] | list[Any]):
        """Construct the squared area difference of two punctures.

        Args:
            edges: Exactly two graph edges or base-edge indices.
        """
        return wcl.AreaDifferenceSquaredOperator(self.hilbert, tuple(edges))

    def area_difference_squared_surfaces(self, surfaces: tuple[Any, Any] | list[Any]):
        """Construct the squared difference of two total surface areas.

        Args:
            surfaces: Exactly two edge sequences describing the punctures.
        """
        first, second = surfaces
        return wcl.AreaDifferenceSquaredSurfacesOperator(
            self.hilbert, (tuple(first), tuple(second))
        )

    def get_volume(self, vertex: Any = None, *, squared: bool = False):
        """Construct the 3+1 vertex volume or its square.

        Args:
            vertex: Graph vertex or its index. An explicit vertex is required.
            squared: Square the selected vertex volume when true.

        Returns:
            result (ComputationalOperator): A diagonal `VolumeOperator`.
        """
        if vertex is None:
            self._not_implemented("volume", "Select a vertex with get_volume(vertex).")
        return wcl.VolumeOperator(self.hilbert, vertex, squared=squared)

    def get_euclidean_constraint(
        self, vertex: Any = None, *, adjoint: bool = False, apply_lapse: bool = False
    ):
        r"""Construct $H_v$ locally or sum $H_E=\sum_v H_v$ over admissible vertices.

        Args:
            vertex: Graph vertex or its index. None selects `admissible_vertices`.
            adjoint: Return the mathematical adjoint of the selected linear constraint.
            apply_lapse: Include the configured vertex lapses before summing.

        Returns:
            result (ComputationalOperator): A linear 3+1 Thiemann constraint.
                A graph with no admissible vertices gives the zero operator.
        """
        vertices = self.admissible_vertices if vertex is None else (vertex,)
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
        return wcl.Thiemann3p1VertexConstraintOperator(
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
            result (list[Squared]): One `Squared` constraint per admissible vertex.
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
            for vertex in self._admissible_vertices
        ]


__all__ = ["LqxWCL4D"]
