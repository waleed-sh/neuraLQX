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


"""One-dimensional U(1) WCL/BF-model facade."""

from __future__ import annotations

from typing import Any

from neuralqx.lqx.base import LQXModel
from neuralqx.operator.prebuilt import wcl
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_u1_hilbert


class LqxWCL1D(LQXModel):
    r"""U(1) BF-model operators and loop-curvature constraint.

    This facade requires one U(1) charge per edge. Its Euclidean component and
    automatic total constraint are the BF loop-curvature operator

    $$
    \mathcal C=\sum_{\alpha}(2I-H_\alpha-H_\alpha^\dagger),
    $$

    summed over the graph's dressed minimal loops. Cyclic cutoff holonomies
    make every summand Hermitian and positive semidefinite. The squared Gauss
    violation is available separately through `gauss_constraint` and is not
    included in the default curvature constraint.

    The `1D` name denotes the single gauge component. This class supplies the
    common operator factories inherited by the higher-dimensional WCL models.
    Lorentzian, area, volume, mass, and Thiemann defaults are unavailable here.
    Select an edge with `get_holonomy` and assign the result to `holonomy`
    when a persistent holonomy selection is needed. The quadratic property
    squares the total curvature constraint according to the shared model API.

    Args:
        hilbert (AbstractU1Hilbert): U(1) Hilbert space with `gauge_dimensions=1`.
        graph (AbstractGraph | None): Optional graph object, identical to `hilbert.graph`.
    """

    model_name = "U(1) BF Model"
    """Descriptive name of the physical model."""

    def __init__(self, hilbert, graph=None) -> None:
        """Initialize the model and validate its graph and parameter requirements.

        Arguments have the meanings documented on the model class. An explicitly
        supplied graph must be the graph object attached to the Hilbert space.
        """
        super().__init__(hilbert, graph)
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        if type(self) is LqxWCL1D:
            require_gauge_dimensions(hilbert, 1, operator=type(self).__name__)

    def __repr__(self) -> str:
        """Return the model type and its defining Hilbert-space and graph parameters."""
        return f"LqxWCL1D(hilbert={self.hilbert!r}, graph={self.graph!r})"

    def identity(self, const: complex | float = 1.0) -> wcl.IdentityOperator:
        """Return a scalar multiple of the identity on this model's Hilbert space.

        Args:
            const: Real or complex coefficient.
        """
        return wcl.IdentityOperator(self.hilbert, const)

    def number(self, edge: Any, *, copy: int = 0) -> wcl.NumberOperator:
        """Return the charge operator on one graph edge.

        Args:
            edge: Graph edge or base-edge index.
            copy: Zero-based U(1) component.
        """
        return wcl.NumberOperator(self.hilbert, edge, copy=copy)

    def get_holonomy(
        self,
        edge: Any = None,
        *,
        adjoint: bool = False,
        dimension: int = 1,
        copy: int = 0,
    ):
        """Construct a cyclic holonomy on the model's single U(1) component.

        Args:
            edge: Graph edge or base-edge index. An explicit edge is required.
            adjoint: Reverse the cyclic shift when true.
            dimension: Must be one for this model.
            copy: Component index, which must be zero in a single-copy space.
        """
        if edge is None:
            self._not_implemented("holonomy", "Select an edge with get_holonomy(edge).")
        if int(dimension) != 1:
            raise ValueError("LqxWCL1D only supports dimension=1 holonomies.")
        return wcl.U1HolonomyOperator(
            self.hilbert, edge, adjoint_flag=adjoint, copy=copy
        )

    def minimal_loop_holonomy(
        self,
        loop: Any,
        *,
        adjoint: bool = False,
        dimension: int = 1,
        copy: int = 0,
    ):
        """Construct a cyclic loop holonomy on the single U(1) component.

        Args:
            loop: Dressed loop or edge sequence.
            adjoint: Reverse the loop shifts when true.
            dimension: Must be one for this model.
            copy: Component index, which must be zero in a single-copy space.
        """
        if int(dimension) != 1:
            raise ValueError("LqxWCL1D only supports dimension=1 loop holonomies.")
        return wcl.MinimalLoopHolonomyOperator(
            self.hilbert,
            loop,
            copy=copy,
            adjoint_flag=adjoint,
        )

    def one_minus_half_holonomy(self, loop: Any, *, copy: int = 0):
        r"""Return $I-(H_\alpha+H_\alpha^\dagger)/2$ for one loop component.

        Args:
            loop: Dressed graph loop or edge sequence.
            copy: Zero-based U(1) component.
        """
        return wcl.OneMinusHalfHolonomyOperator(self.hilbert, loop, copy=copy)

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

    def curvature_constraint(self, loops: list[Any] | None = None):
        r"""Sum $2I-H_\alpha-H_\alpha^\dagger$ over the selected loops and gauge copies.

        Args:
            loops: Loop list to include. `None` selects all dressed minimal loops.

        Returns:
            result (ComputationalOperator): A Hermitian positive-semidefinite operator. An empty loop list gives
                the zero operator on this Hilbert space.
        """
        selected = self.graph.dressed_minimal_loops() if loops is None else loops
        result = None
        for loop in selected:
            term = 2.0 * self.one_minus_half_holonomy(loop)
            result = term if result is None else result + term
        return self.identity(0.0) if result is None else result

    def get_euclidean_constraint(self, loops: list[Any] | None = None):
        """Construct the BF curvature constraint used as this model's Euclidean component.

        Args:
            loops: Dressed loops to include. None selects the graph's minimal loops.

        Returns:
            result (ComputationalOperator): The sum of loop-curvature terms.
                An empty selection gives the zero operator.
        """
        return self.curvature_constraint(loops)

    def thiemann_regularized_constraint(self, *args, **kwargs):
        """Raise `NotImplementedError` because the single-copy BF model has no Thiemann constraint."""
        self._not_implemented("thiemann_regularized_constraint")

    def thiemann_quadratic_constraint(self, *args, **kwargs):
        """Raise `NotImplementedError` because the single-copy BF model has no quadratic Thiemann constraint."""
        self._not_implemented("thiemann_quadratic_constraint")


__all__ = ["LqxWCL1D"]
