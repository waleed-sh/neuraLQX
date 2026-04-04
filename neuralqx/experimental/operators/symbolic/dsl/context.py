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


"""Build-time expression context for symbolic operator DSL callbacks."""

from __future__ import annotations

from typing import Any

from neuralqx.experimental.operators.symbolic.dsl.selectors import SiteSelector
from neuralqx.experimental.operators.symbolic.dsl.selectors import emitted
from neuralqx.experimental.operators.symbolic.dsl.selectors import site
from neuralqx.experimental.operators.symbolic.dsl.selectors import symbol

from neuralqx.experimental.operators.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.experimental.operators.symbolic.ir.expressions import (
    coerce_amplitude_expr,
)

from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr
from neuralqx.experimental.operators.symbolic.ir.predicates import coerce_predicate_expr


class ExpressionContext:
    """
    Utility context passed to DSL callables at build time.

    This context only builds IR expression nodes and never captures Python-runtime callbacks.

    Example:
        >>> def my_amplitude(ctx):
        ...     i = ctx.site("i")
        ...     return ctx.sqrt(i.value + 1)
    """

    __slots__ = ()

    def const(self, value: complex | float | int) -> AmplitudeExpr:
        """Returns a constant amplitude expression."""
        return AmplitudeExpr.constant(value)

    def symbol(self, name: str) -> AmplitudeExpr:
        """Returns a free symbolic amplitude expression."""
        return symbol(name)

    def sqrt(self, operand: Any) -> AmplitudeExpr:
        """Returns a square-root amplitude expression."""
        return AmplitudeExpr.sqrt(operand)

    def conj(self, operand: Any) -> AmplitudeExpr:
        """Returns a complex-conjugate amplitude expression."""
        return AmplitudeExpr.conj(operand)

    def neg(self, operand: Any) -> AmplitudeExpr:
        """Returns a negated amplitude expression."""
        return AmplitudeExpr.neg(operand)

    def site(self, label: str) -> SiteSelector:
        """Returns a site selector by label."""
        return site(label)

    def emitted(self, label: str) -> SiteSelector:
        """Returns an emitted-state selector by label."""
        return emitted(label)

    def all_of(self, *operands: Any) -> PredicateExpr:
        """Builds logical conjunction over provided predicate values."""
        return PredicateExpr.and_(*operands)

    def any_of(self, *operands: Any) -> PredicateExpr:
        """Builds logical disjunction over provided predicate values."""
        return PredicateExpr.or_(*operands)

    def not_(self, operand: Any) -> PredicateExpr:
        """Builds logical negation over one predicate value."""
        return PredicateExpr.not_(operand)

    def eq(self, left: Any, right: Any) -> PredicateExpr:
        """Builds equality predicate expression."""
        return PredicateExpr.eq(left, right)

    def ne(self, left: Any, right: Any) -> PredicateExpr:
        """Builds inequality predicate expression."""
        return PredicateExpr.ne(left, right)

    def lt(self, left: Any, right: Any) -> PredicateExpr:
        """Builds strict-less-than predicate expression."""
        return PredicateExpr.lt(left, right)

    def le(self, left: Any, right: Any) -> PredicateExpr:
        """Builds less-than-or-equal predicate expression."""
        return PredicateExpr.le(left, right)

    def gt(self, left: Any, right: Any) -> PredicateExpr:
        """Builds strict-greater-than predicate expression."""
        return PredicateExpr.gt(left, right)

    def ge(self, left: Any, right: Any) -> PredicateExpr:
        """Builds greater-than-or-equal predicate expression."""
        return PredicateExpr.ge(left, right)

    def coerce_amplitude(self, value: Any) -> AmplitudeExpr:
        """Coerces one value into an amplitude expression."""
        return coerce_amplitude_expr(value)

    def coerce_predicate(self, value: Any) -> PredicateExpr:
        """Coerces one value into a predicate expression."""
        return coerce_predicate_expr(value)

    def pow(self, base: Any, exponent: Any) -> AmplitudeExpr:
        """Returns a power expression ``base ** exponent``."""
        return AmplitudeExpr.pow(base, exponent)

    def abs_(self, operand: Any) -> AmplitudeExpr:
        """Returns an absolute-value expression ``|operand|``."""
        return AmplitudeExpr.abs_(operand)

    def wrap_mod(self, operand: Any) -> AmplitudeExpr:
        """Returns a Hilbert-aware modulo-wrapped amplitude expression."""
        return AmplitudeExpr.wrap_mod(operand)

    def sq_norm(self, *components: Any) -> AmplitudeExpr:
        """
        Returns the squared L2 norm of *components*: ``c0² + c1² + ...``.

        Each component is coerced to an :class:`AmplitudeExpr` before squaring.
        Equivalent to ``sum(c * c for c in components)``.

        Args:
            *components: Two or more amplitude expressions (or coercible values).

        Returns:
            Amplitude expression for the squared norm.
        """
        if not components:
            raise ValueError("sq_norm requires at least one component.")
        exprs = [coerce_amplitude_expr(c) for c in components]
        result = AmplitudeExpr.mul(exprs[0], exprs[0])
        for e in exprs[1:]:
            result = AmplitudeExpr.add(result, AmplitudeExpr.mul(e, e))
        return result

    def norm2(self, *components: Any) -> AmplitudeExpr:
        """
        Returns the L2 norm of *components*: ``sqrt(c0² + c1² + ...)``.

        Args:
            *components: Two or more amplitude expressions (or coercible values).

        Returns:
            Amplitude expression for the Euclidean norm.
        """
        return AmplitudeExpr.sqrt(self.sq_norm(*components))

    def edge_value(
        self,
        edge_idx: int,
        gauge_copy: int,
        n_edges_per_copy: int,
    ) -> AmplitudeExpr:
        """
        Returns the charge at ``x[gauge_copy * n_edges_per_copy + edge_idx]``.

        This is the primary access pattern for U(1)^G (or SU(N)^G) gauge
        theories where the flat configuration array is laid out as G consecutive
        blocks of ``n_edges_per_copy`` entries each.

        Args:
            edge_idx: Zero-based edge index within one gauge copy.
            gauge_copy: Gauge copy index (0-based).
            n_edges_per_copy: Number of edge sites per gauge copy.

        Returns:
            Static-index amplitude expression reading ``x[g * E + e]``.
        """
        flat = int(gauge_copy) * int(n_edges_per_copy) + int(edge_idx)
        return AmplitudeExpr.static_index(flat)

    def emitted_edge_value(
        self,
        edge_idx: int,
        gauge_copy: int,
        n_edges_per_copy: int,
    ) -> AmplitudeExpr:
        """Returns the emitted/connected charge at x'[g*E + e]."""
        flat = int(gauge_copy) * int(n_edges_per_copy) + int(edge_idx)
        return AmplitudeExpr.static_emitted_index(flat)

    def edge_components(
        self,
        edge_idx: int,
        n_edges_per_copy: int,
        gauge_dim: int = 3,
    ) -> list[AmplitudeExpr]:
        """
        Returns all *gauge_dim* charge components for one edge.

        For a U(1)^G Hilbert space with G gauge copies, returns the list
        ``[x[0*E + e], x[1*E + e], …, x[(G-1)*E + e]]``.

        Args:
            edge_idx: Zero-based edge index.
            n_edges_per_copy: Number of edges per gauge copy (E).
            gauge_dim: Number of gauge copies (G, default 3 for U(1)^3).

        Returns:
            List of G static-index amplitude expressions.
        """
        return [
            self.edge_value(edge_idx, g, n_edges_per_copy) for g in range(gauge_dim)
        ]

    def edge_sq_norm(
        self,
        edge_idx: int,
        n_edges_per_copy: int,
        gauge_dim: int = 3,
    ) -> AmplitudeExpr:
        """
        Returns the squared L2 norm of the charge vector at *edge_idx*.

        Equivalent to ``sq_norm(*edge_components(edge_idx, ...))``.
        """
        return self.sq_norm(
            *self.edge_components(edge_idx, n_edges_per_copy, gauge_dim)
        )

    def edge_norm(
        self,
        edge_idx: int,
        n_edges_per_copy: int,
        gauge_dim: int = 3,
    ) -> AmplitudeExpr:
        """
        Returns the L2 norm of the charge vector at *edge_idx*.

        Equivalent to ``norm2(*edge_components(edge_idx, ...))``.
        """
        return self.norm2(*self.edge_components(edge_idx, n_edges_per_copy, gauge_dim))


__all__ = ["ExpressionContext"]
