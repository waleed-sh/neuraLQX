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


"""Spherical Bojowald-Swiderski model facade."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from neuralqx.graph import HalfLadderGraph
from neuralqx.lqx.base import LQXModel
from neuralqx.operator.prebuilt import bojowald_swiderski as bosw
from neuralqx.operator.prebuilt import wcl
from neuralqx.operator.prebuilt.utils import k_vertex


class LqxSphericalBojowaldSwiderski(LQXModel):
    r"""Spherically symmetric Bojowald-Swiderski operators on a half-ladder graph.

    Radial edges carry $k$ labels and transverse edges carry $\mu$ labels.
    The facade provides radial triads, shifted local volumes, and the
    non-Hermitian 19-branch vertex Hamiltonian. Its Euclidean component is
    $H_E=\sum_v H_v$ over `graph.vertices_k[2:-2]`. No separate Lorentzian
    correction is supplied, so the automatic total is this linear operator.
    The quadratic property gives $\sum_v H_v H_v^\dagger$, with each vertex
    squared separately. `hamiltonian_list` returns these local products as a list.

    The default volume factory uses `shift=1`, whereas constructing
    `SphericalVolumeOperator` directly defaults to zero. The Hamiltonian uses
    cyclic integer labels with unit spacing. A positive even `delta` keeps
    both full and half shifts on that lattice.

    Volume and holonomy construction require an explicit vertex or edge.
    Area and mass implementations are unavailable.

    Args:
        hilbert (AbstractU1Hilbert): Single-copy U(1) Hilbert space on a `HalfLadderGraph`.
        graph (AbstractGraph | None): Optional graph object, identical to `hilbert.graph`.
        immirzi: Nonzero real Barbero-Immirzi parameter.
        delta: Positive even integer controlling the Hamiltonian shifts.
    """

    immirzi: float
    r"""Nonzero real Barbero-Immirzi parameter $\gamma$."""

    delta: int
    """Positive even integer setting radial and transverse shift sizes."""

    def __init__(
        self,
        hilbert,
        graph: HalfLadderGraph | None = None,
        *,
        immirzi: float = 1.0,
        delta: int = 2,
    ) -> None:
        """Initialize the model and validate its graph and parameter requirements.

        Arguments have the meanings documented on the model class. An explicitly
        supplied graph must be the graph object attached to the Hilbert space.
        """
        super().__init__(hilbert, graph)
        if not isinstance(self.graph, HalfLadderGraph):
            raise ValueError(
                "LqxSphericalBojowaldSwiderski requires HalfLadderGraph, "
                f"got {type(self.graph).__name__}."
            )
        if (
            isinstance(delta, bool)
            or not isinstance(delta, Integral)
            or delta <= 0
            or delta % 2 != 0
        ):
            raise ValueError("delta must be a positive even integer.")
        if float(immirzi) == 0.0:
            raise ValueError("immirzi must be non-zero.")
        self.immirzi = float(immirzi)
        self.delta = int(delta)

    def __repr__(self) -> str:
        """Return the model type and its defining Hilbert-space and graph parameters."""
        return (
            f"LqxSphericalBojowaldSwiderski(hilbert={self.hilbert!r}, "
            f"graph={self.graph!r}, immirzi={self.immirzi}, delta={self.delta})"
        )

    @property
    def vertices_k(self):
        """Ordered radial vertices of the half-ladder graph."""
        return self.graph.vertices_k

    @property
    def inner_vertices(self):
        """Radial vertices selected for the total constraint, `vertices_k[2:-2]`."""
        return self.graph.vertices_k[2:-2]

    def spherical_volume(self, vertex: Any, *, shift: float = 1.0):
        """Construct the local spherical volume with a radial-label shift.

        Args:
            vertex: Radial vertex or its index in `vertices_k`.
            shift: Constant added to `k_minus + k_plus` under the square root.
                This factory defaults to one. The operator class defaults to zero.

        Returns:
            result (bosw.SphericalVolumeOperator): A diagonal `SphericalVolumeOperator`.
        """
        return bosw.SphericalVolumeOperator(self.hilbert, vertex, shift=shift)

    def get_volume(self, vertex: Any = None, *, shift: float = 1.0):
        """Construct the spherical volume at an explicitly selected radial vertex.

        Args:
            vertex: Radial vertex or its index. An explicit vertex is required.
            shift: Constant added to the sum of adjacent radial labels.

        Returns:
            result (bosw.SphericalVolumeOperator): The selected diagonal vertex-volume operator.
        """
        if vertex is None:
            self._not_implemented("volume", "Select a vertex with get_volume(vertex).")
        return self.spherical_volume(vertex, shift=shift)

    def spherical_Ex(self, vertex: Any):
        """Construct the radial triad using the model's Immirzi parameter.

        Args:
            vertex: Radial vertex or its index in `vertices_k`.

        Returns:
            result (bosw.SphericalExOperator): A diagonal `SphericalExOperator` with eigenvalue `immirzi * K / 2`.
        """
        return bosw.SphericalExOperator(self.hilbert, vertex, gamma=self.immirzi)

    def spherical_hamiltonian_on_vertex(
        self,
        vertex: Any,
        *,
        include_gamma_terms: bool = True,
        adjoint: bool = False,
        outer_km_constant: int = 0,
        outer_kp_constant: int = 0,
    ):
        """Construct the spherical Hamiltonian on one radial vertex.

        Args:
            vertex: Radial vertex or its index in `vertices_k`.
            include_gamma_terms: Include the spin connection and its difference.
            adjoint: Return the conjugate-transpose Hamiltonian when true.
            outer_km_constant: Substitute for missing outer left radial labels.
            outer_kp_constant: Substitute for missing outer right radial labels.

        Returns:
            result (bosw.BojowaldSwiderskiVertexConstraintOperator): A `BojowaldSwiderskiVertexConstraintOperator` using the model's
                `delta` and Immirzi parameter.
        """
        return bosw.BojowaldSwiderskiVertexConstraintOperator(
            self.hilbert,
            vertex,
            include_gamma_terms=include_gamma_terms,
            delta=self.delta,
            outer_km_constant=outer_km_constant,
            outer_kp_constant=outer_kp_constant,
            immirzi=self.immirzi,
            adjoint_kernel=adjoint,
        )

    def hamiltonian_list(self, *, alternative: bool = False):
        r"""Return local quadratic Hamiltonian constraints on the selected inner vertices.

        The default is $Q_v=H_v H_v^\dagger$, represented by
        `Squared(H_v.adjoint)`. With `alternative=True`, the result is
        $Q_v=H_v^\dagger H_v$, represented by `Squared(H_v)`.
        The list contains one term for each entry of `inner_vertices`.

        Args:
            alternative: Select $H_v^\dagger H_v$ instead of $H_v H_v^\dagger$.

        Returns:
            result (list[Squared]): Local quadratic constraints for the expectation engine.
        """
        return [
            self._quadratic_of(
                self.spherical_hamiltonian_on_vertex(vertex), alternative=alternative
            )
            for vertex in self.inner_vertices
        ]

    def get_euclidean_constraint(
        self,
        vertex: Any = None,
        *,
        include_gamma_terms: bool = True,
        adjoint: bool = False,
        outer_km_constant: int = 0,
        outer_kp_constant: int = 0,
    ):
        r"""Construct the spherical Euclidean Hamiltonian locally or as $\sum_v H_v$.

        Args:
            vertex: Radial vertex or its index. None selects `inner_vertices`.
            include_gamma_terms: Include the spin connection and its radial difference.
            adjoint: Return the mathematical adjoint of the selected linear constraint.
            outer_km_constant: Substitute for missing outer left radial labels.
            outer_kp_constant: Substitute for missing outer right radial labels.

        Returns:
            result (ComputationalOperator): A vertex Hamiltonian or their linear sum.
                An empty inner-vertex selection gives the zero operator.
        """
        vertices = self.inner_vertices if vertex is None else (vertex,)
        return self.sum_constraint_terms(
            {
                k_vertex(self.graph, v)[0]: self.spherical_hamiltonian_on_vertex(
                    v,
                    include_gamma_terms=include_gamma_terms,
                    adjoint=adjoint,
                    outer_km_constant=outer_km_constant,
                    outer_kp_constant=outer_kp_constant,
                )
                for v in vertices
            }
        )

    def get_holonomy(self, edge: Any = None, *, adjoint: bool = False):
        """Construct a cyclic shift of one radial or transverse edge label.

        Args:
            edge: Graph edge or base-edge index. An explicit edge is required.
            adjoint: Reverse the shift when true.
        """
        if edge is None:
            self._not_implemented("holonomy", "Select an edge with get_holonomy(edge).")
        return wcl.U1HolonomyOperator(self.hilbert, edge, adjoint_flag=adjoint)

    def number(self, edge: Any):
        """Construct the label operator for one radial or transverse edge.

        Args:
            edge: Graph edge or base-edge index.
        """
        return wcl.NumberOperator(self.hilbert, edge)


LqxSphericalBoSw = LqxSphericalBojowaldSwiderski

__all__ = ["LqxSphericalBoSw", "LqxSphericalBojowaldSwiderski"]
