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


"""Shared operator properties and constraint composition for LQX models."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import NoReturn

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator import AbstractOperator
from neuralqx.operator import ComputationalOperator
from neuralqx.operator import ScaledOperator
from neuralqx.operator import Squared
from neuralqx.operator import SumOfSquares
from neuralqx.operator import SumOperator
from neuralqx.operator import VertexSum
from neuralqx.operator._algebra import is_zero_scalar_like
from neuralqx.operator.prebuilt.wcl import IdentityOperator


class _UnavailableOperator(NotImplementedError):
    """An intentionally absent model operation, distinct from a failed construction.

    The total constraint may omit an absent Euclidean or Lorentzian component.
    It must still propagate errors raised inside an implemented component,
    including ordinary NotImplementedError exceptions from an unavailable
    backend. This exception identifies only the model's declared absence.
    """

    name: str
    """Property whose default implementation is unavailable."""

    def __init__(self, model: str, name: str, detail: str = "") -> None:
        """Record the absent operation and name the concrete requesting model."""
        self.name = name
        message = f"{model} does not implement a default for {name}."
        super().__init__(f"{message} {detail}" if detail else message)


class LQXModel:
    r"""Common operator interface for discrete loop-quantum-gravity models.

    Every model exposes assignable `euclidean_constraint`, `lorentzian_constraint`,
    `area`, `volume`, `holonomy`, `mass`, `constraint`, and `quadratic_constraint`
    properties. A property constructs and caches its default on first access.
    The corresponding `get_...` method constructs an operator with explicit
    parameters without replacing that selection. Assign None to a property
    to discard its selection and restore default construction on next access.

    The automatic total is $H=H_E+H_L$. If only one component is implemented,
    it is the total. If neither exists, accessing the total raises
    NotImplementedError naming the concrete model. A failed implementation
    is never silently treated as an absent contribution.

    The automatic quadratic constraint is $Q=\sum_v H_vH_v^\dagger$, where
    $H_v=H_v^E+H_v^L$. Components at the same vertex are combined before
    squaring. Different vertices are squared separately, without cross terms
    between them. `get_quadratic_constraint(alternative=True)` instead
    constructs $\sum_v H_v^\dagger H_v$. Both use lazy norm estimators.

    Explicit assignments to `constraint` and `quadratic_constraint` override
    their automatic definitions. Replacing either linear component clears
    both derived selections. Replacing the total clears its quadratic
    selection. Geometric properties have independent selections.

    Prebuilt geometric factories require an explicit vertex, surface, edge,
    or loop where the operator needs one. Such properties remain unset until
    the user assigns the result of the corresponding parameterized getter.
    Combining geometric operators over a region is the caller's choice.

    To subclass, override the relevant `get_...` methods and return operators
    on this model's Hilbert space. Inherited methods raise NotImplementedError
    with the subclass name. Use `_not_implemented` when an overridden factory
    intentionally has no default, for example because an edge must be selected.
    Construct vertex sums with `sum_constraint_terms` so the quadratic factory
    can match the two components by vertex. An unlabelled constraint is treated
    as one local block, as in the single-vertex QR model.

    Args:
        hilbert: Discrete Hilbert space shared by the model's operators.
        graph: The graph attached to `hilbert`. None uses that graph directly.
    """

    hilbert: DiscreteHilbertSpace
    """Discrete Hilbert space shared by the selected operators."""

    graph: AbstractGraph
    """Graph attached to the model Hilbert space."""

    _euclidean_constraint: ComputationalOperator | None
    """Selected or cached Euclidean component, with None requesting its default."""

    _lorentzian_constraint: ComputationalOperator | None
    """Selected or cached Lorentzian component, with None requesting its default."""

    _area: AbstractOperator | None
    """Selected area operator, or None before default construction."""

    _volume: AbstractOperator | None
    """Selected volume operator, or None before default construction."""

    _holonomy: AbstractOperator | None
    """Selected holonomy operator, or None when no holonomy has been selected."""

    _mass: AbstractOperator | None
    """Selected mass operator, or None before default construction."""

    _constraint: ComputationalOperator | None
    """Selected total constraint or cached sum of the available linear components."""

    _quadratic_constraint: AbstractOperator | None
    """Selected quadratic operator or cached sum of local positive products."""

    def __init__(
        self, hilbert: DiscreteHilbertSpace, graph: AbstractGraph | None = None
    ) -> None:
        """Bind the Hilbert space and graph and initialize empty operator selections.

        Raises:
            TypeError: If the Hilbert space does not provide a discrete basis.
            ValueError: If an explicit graph is not the Hilbert space's graph.
        """
        if not isinstance(hilbert, DiscreteHilbertSpace):
            raise TypeError(f"{type(self).__name__} requires a DiscreteHilbertSpace.")
        self.hilbert = hilbert
        self.graph = hilbert.graph if graph is None else graph
        if self.graph is not hilbert.graph:
            raise ValueError(
                "The supplied graph must be the graph attached to the Hilbert space."
            )
        self._euclidean_constraint = None
        self._lorentzian_constraint = None
        self._area = None
        self._volume = None
        self._holonomy = None
        self._mass = None
        self._constraint = None
        self._quadratic_constraint = None

    @property
    def gauge_dimensions(self) -> int:
        """Number of gauge components supplied by the model's Hilbert-space convention."""
        if not hasattr(self.hilbert, "gauge_dimensions"):
            self._not_implemented("gauge_dimensions")
        return int(self.hilbert.gauge_dimensions)

    @property
    def euclidean_constraint(self) -> ComputationalOperator:
        """Selected Euclidean component, lazily constructed by get_euclidean_constraint."""
        return self._get_cached_operator("euclidean_constraint")

    @euclidean_constraint.setter
    def euclidean_constraint(self, operator: ComputationalOperator | None) -> None:
        """Select a Euclidean component and clear the total and quadratic selections."""
        self._set_operator("euclidean_constraint", operator)

    @property
    def lorentzian_constraint(self) -> ComputationalOperator:
        """Selected Lorentzian component, lazily constructed by get_lorentzian_constraint."""
        return self._get_cached_operator("lorentzian_constraint")

    @lorentzian_constraint.setter
    def lorentzian_constraint(self, operator: ComputationalOperator | None) -> None:
        """Select a Lorentzian component and clear the total and quadratic selections."""
        self._set_operator("lorentzian_constraint", operator)

    @property
    def area(self) -> AbstractOperator:
        """Selected area operator, using get_area without arguments for its default."""
        return self._get_cached_operator("area")

    @area.setter
    def area(self, operator: AbstractOperator | None) -> None:
        """Select an area operator, or assign None to restore default construction."""
        self._set_operator("area", operator)

    @property
    def volume(self) -> AbstractOperator:
        """Selected volume operator, using get_volume without arguments for its default."""
        return self._get_cached_operator("volume")

    @volume.setter
    def volume(self, operator: AbstractOperator | None) -> None:
        """Select a volume operator, or assign None to restore default construction."""
        self._set_operator("volume", operator)

    @property
    def holonomy(self) -> AbstractOperator:
        """Selected holonomy, using the model default when one is defined."""
        return self._get_cached_operator("holonomy")

    @holonomy.setter
    def holonomy(self, operator: AbstractOperator | None) -> None:
        """Select a holonomy operator, or assign None to discard the selection."""
        self._set_operator("holonomy", operator)

    @property
    def mass(self) -> AbstractOperator:
        """Selected mass operator, using get_mass without arguments for its default."""
        return self._get_cached_operator("mass")

    @mass.setter
    def mass(self, operator: AbstractOperator | None) -> None:
        """Select a mass operator, or assign None to restore default construction."""
        self._set_operator("mass", operator)

    @property
    def constraint(self) -> ComputationalOperator:
        r"""Selected total constraint, defaulting to the available sum $H_E+H_L$."""
        return self._get_cached_operator("constraint")

    @constraint.setter
    def constraint(self, operator: ComputationalOperator | None) -> None:
        """Override the automatic total and clear the quadratic selection.

        Assign None to restore composition from the selected linear components.
        Assigning either component later also discards this total override.
        """
        self._set_operator("constraint", operator)

    @property
    def quadratic_constraint(self) -> AbstractOperator:
        r"""Selected quadratic constraint, defaulting to $\sum_v H_vH_v^\dagger$."""
        return self._get_cached_operator("quadratic_constraint")

    @quadratic_constraint.setter
    def quadratic_constraint(self, operator: AbstractOperator | None) -> None:
        """Override the automatic quadratic operator, or reset it with None.

        To select the alternative ordering, assign the result of
        `get_quadratic_constraint(alternative=True)`. Replacing the total or
        either linear component discards this selection.
        """
        self._set_operator("quadratic_constraint", operator)

    def get_euclidean_constraint(
        self, *args: Any, **kwargs: Any
    ) -> ComputationalOperator:
        """Construct the Euclidean component with model-specific parameters.

        Override this method to provide a model's default and parameterized
        Euclidean operators. The inherited implementation raises
        NotImplementedError naming the concrete subclass.
        """
        self._not_implemented("euclidean_constraint")

    def get_lorentzian_constraint(
        self, *args: Any, **kwargs: Any
    ) -> ComputationalOperator:
        """Construct the Lorentzian component, or raise the model-specific default error.

        A model with no Lorentzian prescription can inherit this method. Its
        automatic total then uses the Euclidean component alone.
        """
        self._not_implemented("lorentzian_constraint")

    def get_area(self, *args: Any, **kwargs: Any) -> AbstractOperator:
        """Construct an area operator, or raise NotImplementedError for this model."""
        self._not_implemented("area")

    def get_volume(self, *args: Any, **kwargs: Any) -> AbstractOperator:
        """Construct a volume operator, or raise NotImplementedError for this model."""
        self._not_implemented("volume")

    def get_holonomy(self, *args: Any, **kwargs: Any) -> AbstractOperator:
        """Construct a holonomy, or raise NotImplementedError for this model."""
        self._not_implemented("holonomy")

    def get_mass(self, *args: Any, **kwargs: Any) -> AbstractOperator:
        """Construct a mass operator, or raise NotImplementedError for this model."""
        self._not_implemented("mass")

    def get_constraint(self) -> ComputationalOperator:
        r"""Compose the selected linear components as $H=H_E+H_L$.

        An intentionally unavailable component is omitted. If neither exists,
        NotImplementedError names the requesting model. Construction errors
        inside an implemented component propagate to the caller.

        This method computes the automatic total even when `constraint` has
        been explicitly overridden. It does not replace that selection.

        Returns:
            result (ComputationalOperator): The sum, or the sole available component.
        """
        components = []
        for name in ("euclidean_constraint", "lorentzian_constraint"):
            try:
                components.append(getattr(self, name))
            except _UnavailableOperator as error:
                if error.name != name:
                    raise
        if not components:
            self._not_implemented(
                "constraint", "Neither linear component is available."
            )
        return self._sum_operators(components)

    def get_quadratic_constraint(
        self, *, alternative: bool = False
    ) -> Squared | SumOfSquares:
        r"""Sum the positive products of the selected vertex-local constraints.

        At each vertex, first combine $H_v=H_v^E+H_v^L$. The default is
        $Q=\sum_v H_vH_v^\dagger$, evaluated as a sum of
        `Squared(H_v.adjoint)` markers. With `alternative=True`, return
        $\sum_v H_v^\dagger H_v$. Cross terms between Euclidean and Lorentzian
        contributions at the same vertex remain inside their common square.
        There are no cross terms between different vertices.

        The local decomposition comes from `get_constraint_terms`, including
        any explicit total override. Calling this method does not change the
        quadratic property selection.

        Args:
            alternative: Select $H_v^\dagger H_v$ at each vertex when true.

        Returns:
            result (Squared | SumOfSquares): A lazy sum for the VQS norm estimator.
                A single local contribution uses Squared directly. An empty
                vertex set gives Squared of the zero operator.

        Raises:
            TypeError: If alternative is not a bool.
            NotImplementedError: If no total constraint is available.
            ValueError: If labelled and unlabelled contributions are mixed.
        """
        if not isinstance(alternative, bool):
            raise TypeError("alternative must be a bool.")
        terms = tuple(self.get_constraint_terms().values())
        if not terms:
            return self._quadratic_of(IdentityOperator(self.hilbert, 0.0))
        for term in terms:
            self._validate_operator("constraint", term)
        if len(terms) == 1:
            return self._quadratic_of(terms[0], alternative=alternative)
        return SumOfSquares(
            terms if alternative else tuple(term.adjoint for term in terms)
        )

    def sum_constraint_terms(
        self, terms: Mapping[Any, ComputationalOperator]
    ) -> ComputationalOperator:
        r"""Construct $\sum_v H_v$ while retaining each contribution's vertex.

        Use this helper in component getters or when assigning custom linear
        constraints. The Euclidean and Lorentzian mappings may have different
        vertex sets and orders. Matching labels are combined by the quadratic
        factory, and a missing component contributes zero at that vertex.

        Each value is one complete local contribution. A sum inside a value
        stays together when squared. The helper does not replace any property.

        Args:
            terms: Hashable vertex labels mapped to local linear operators.
                Use the same labels for both components, preferably the graph's
                vertex objects. None is reserved for unlabelled constraints.

        Returns:
            result (ComputationalOperator): A VertexSum, or zero for an empty mapping.
        """
        for operator in terms.values():
            self._validate_operator("constraint", operator)
        if not terms:
            return IdentityOperator(self.hilbert, 0.0)
        return VertexSum(tuple(terms.values()), vertices=tuple(terms))

    def get_constraint_terms(self) -> dict[Any, ComputationalOperator]:
        r"""Return the selected $H_v$ contributions, grouped by vertex label.

        VertexSum preserves the boundary of each local contribution. Ordinary
        sums and scalar multiples around these wrappers preserve that grouping,
        so matching Euclidean and Lorentzian terms are combined by label, never
        by their position in a list.

        A constraint without vertex labels is one block with key None. This
        supports single-vertex models and explicit unlabelled total overrides.
        Mixing labelled and unlabelled contributions is ambiguous and raises
        ValueError. Use `sum_constraint_terms` to locate a custom contribution.
        Subclasses may override this method to supply another decomposition.
        """
        total = self.constraint
        grouped: dict[Any, list[ComputationalOperator]] = {}

        def append(
            vertex: Any, operator: ComputationalOperator, coefficient: Any
        ) -> None:
            """Keep scalar weights inside their local product, omitting exact zeros."""
            if is_zero_scalar_like(coefficient):
                return
            term = (
                operator
                if is_zero_scalar_like(coefficient - 1)
                else coefficient * operator
            )
            grouped.setdefault(vertex, []).append(term)

        def collect(operator: ComputationalOperator, coefficient: Any = 1.0) -> None:
            """Descend through linear algebra but stop at each labelled local term."""
            if isinstance(operator, VertexSum):
                for vertex, term, weight in zip(
                    operator.vertices,
                    operator.operators,
                    operator.coefficients,
                    strict=True,
                ):
                    append(vertex, term, coefficient * weight)
            elif isinstance(operator, ScaledOperator):
                collect(operator.operator, coefficient * operator.coefficient)
            elif isinstance(operator, SumOperator):
                for term, weight in zip(
                    operator.operators, operator.coefficients, strict=True
                ):
                    collect(term, coefficient * weight)
            elif isinstance(operator, IdentityOperator) and is_zero_scalar_like(
                operator.const
            ):
                return
            else:
                append(None, operator, coefficient)

        collect(total)
        if None in grouped:
            if len(grouped) > 1:
                raise ValueError(
                    "Cannot combine vertex-labelled and unlabelled constraints. "
                    "Use sum_constraint_terms({vertex: operator}) for custom contributions."
                )
            return {None: total}
        return {vertex: self._sum_operators(terms) for vertex, terms in grouped.items()}

    def _not_implemented(self, name: str, detail: str = "") -> NoReturn:
        """Declare an unavailable default operation using the concrete subclass name."""
        raise _UnavailableOperator(type(self).__name__, name, detail)

    def _get_cached_operator(self, name: str) -> Any:
        """Return a selected operator, constructing and validating its default once."""
        operator = getattr(self, f"_{name}")
        if operator is None:
            operator = getattr(self, f"get_{name}")()
            self._validate_operator(name, operator)
            setattr(self, f"_{name}", operator)
        return operator

    def _set_operator(self, name: str, operator: AbstractOperator | None) -> None:
        """Validate a selection before replacing it and invalidate dependent selections."""
        if operator is not None:
            self._validate_operator(name, operator)
        setattr(self, f"_{name}", operator)
        if name in ("euclidean_constraint", "lorentzian_constraint"):
            self._constraint = None
            self._quadratic_constraint = None
        elif name == "constraint":
            self._quadratic_constraint = None

    def _validate_operator(self, name: str, operator: AbstractOperator) -> None:
        """Require an operator on this Hilbert space and composable linear constraints."""
        expected = (
            ComputationalOperator
            if name in ("euclidean_constraint", "lorentzian_constraint", "constraint")
            else AbstractOperator
        )
        if not isinstance(operator, expected):
            raise TypeError(
                f"{type(self).__name__}.{name} requires an {expected.__name__}."
            )
        if operator.hilbert != self.hilbert:
            raise ValueError(
                f"{type(self).__name__}.{name} must act on the model's Hilbert space."
            )

    def _sum_operators(
        self, operators: Iterable[ComputationalOperator]
    ) -> ComputationalOperator:
        """Form a flat sum, using a zero operator when the selected region is empty."""
        terms = tuple(operators)
        if not terms:
            return IdentityOperator(self.hilbert, 0.0)
        if len(terms) == 1:
            return terms[0]
        return SumOperator(terms)

    @staticmethod
    def _quadratic_of(
        operator: ComputationalOperator, *, alternative: bool = False
    ) -> Squared:
        r"""Represent $AA^\dagger$ by default, or $A^\dagger A$ for the alternative ordering."""
        if not isinstance(alternative, bool):
            raise TypeError("alternative must be a bool.")
        return Squared(operator if alternative else operator.adjoint)


__all__ = ["LQXModel"]
