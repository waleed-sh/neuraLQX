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


"""Typed predicate-expression IR nodes for symbolic operators."""

from __future__ import annotations

import dataclasses
from typing import Any

from .expressions import AmplitudeExpr
from .expressions import coerce_amplitude_expr

_PREDICATE_OPS: frozenset[str] = frozenset(
    {
        "const",
        "not",
        "and",
        "or",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    }
)


@dataclasses.dataclass(frozen=True, repr=False)
class PredicateExpr:
    """
    Typed boolean expression node for operator branch filtering.

    Attributes:
        op: Predicate operation name.
        args: Ordered operation arguments.
    """

    op: str
    args: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.op not in _PREDICATE_OPS:
            raise ValueError(
                f"Unsupported predicate-expression op: {self.op!r}. "
                f"Allowed: {sorted(_PREDICATE_OPS)}."
            )

    @classmethod
    def constant(cls, value: bool) -> "PredicateExpr":
        """Builds a constant predicate expression."""
        return cls(op="const", args=(bool(value),))

    @classmethod
    def not_(cls, operand: Any) -> "PredicateExpr":
        """Builds a logical-negation predicate."""
        return cls(op="not", args=(coerce_predicate_expr(operand),))

    @classmethod
    def and_(cls, *operands: Any) -> "PredicateExpr":
        """Builds a logical conjunction predicate."""
        if not operands:
            return cls.constant(True)
        normalized = tuple(coerce_predicate_expr(item) for item in operands)
        return cls(op="and", args=normalized)

    @classmethod
    def or_(cls, *operands: Any) -> "PredicateExpr":
        """Builds a logical disjunction predicate."""
        if not operands:
            return cls.constant(False)
        normalized = tuple(coerce_predicate_expr(item) for item in operands)
        return cls(op="or", args=normalized)

    @classmethod
    def eq(cls, left: Any, right: Any) -> "PredicateExpr":
        """Builds an equality predicate."""
        return cls(
            op="eq",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def ne(cls, left: Any, right: Any) -> "PredicateExpr":
        """Builds an inequality predicate."""
        return cls(
            op="ne",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def lt(cls, left: Any, right: Any) -> "PredicateExpr":
        """Builds a strict-less-than predicate."""
        return cls(
            op="lt",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def le(cls, left: Any, right: Any) -> "PredicateExpr":
        """Builds a less-than-or-equal predicate."""
        return cls(
            op="le",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def gt(cls, left: Any, right: Any) -> "PredicateExpr":
        """Builds a strict-greater-than predicate."""
        return cls(
            op="gt",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def ge(cls, left: Any, right: Any) -> "PredicateExpr":
        """Builds a greater-than-or-equal predicate."""
        return cls(
            op="ge",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    def __and__(self, other: Any) -> "PredicateExpr":
        return self.and_(self, other)

    def __rand__(self, other: Any) -> "PredicateExpr":
        return self.and_(other, self)

    def __or__(self, other: Any) -> "PredicateExpr":
        return self.or_(self, other)

    def __ror__(self, other: Any) -> "PredicateExpr":
        return self.or_(other, self)

    def __invert__(self) -> "PredicateExpr":
        return self.not_(self)

    def __str__(self) -> str:
        return _render_predicate(self)

    def __repr__(self) -> str:
        return f"PredicateExpr(op={self.op!r}, args={self.args!r})"


def _render_predicate(expr: "PredicateExpr") -> str:
    """Renders a PredicateExpr as a readable infix boolean string."""
    from .expressions import _render_amplitude, AmplitudeExpr

    op = expr.op
    args = expr.args

    if op == "const":
        return "true" if args[0] else "false"

    if op == "not":
        return f"!{_render_predicate(args[0])}"

    if op == "and":
        parts = [
            _render_predicate(a) if isinstance(a, PredicateExpr) else repr(a)
            for a in args
        ]
        return f"({' && '.join(parts)})"

    if op == "or":
        parts = [
            _render_predicate(a) if isinstance(a, PredicateExpr) else repr(a)
            for a in args
        ]
        return f"({' || '.join(parts)})"

    # Comparison ops: args are two AmplitudeExprs
    _OPS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    if op in _OPS:
        lhs = (
            _render_amplitude(args[0])
            if isinstance(args[0], AmplitudeExpr)
            else repr(args[0])
        )
        rhs = (
            _render_amplitude(args[1])
            if isinstance(args[1], AmplitudeExpr)
            else repr(args[1])
        )
        return f"({lhs} {_OPS[op]} {rhs})"

    # Fallback
    arg_strs = ", ".join(
        _render_predicate(a) if isinstance(a, PredicateExpr) else repr(a) for a in args
    )
    return f"{op}({arg_strs})"


def _collect_free_symbols_pred(expr: "PredicateExpr", result: "set[str]") -> None:
    """Recursively collects free symbol names from a PredicateExpr."""
    from .expressions import AmplitudeExpr, _collect_free_symbols

    for arg in expr.args:
        if isinstance(arg, PredicateExpr):
            _collect_free_symbols_pred(arg, result)
        elif isinstance(arg, AmplitudeExpr):
            _collect_free_symbols(arg, result)


def coerce_predicate_expr(value: Any) -> PredicateExpr:
    """
    Coerces user values into typed predicate-expression nodes.

    Args:
        value: Input predicate value.

    Returns:
        Typed predicate expression.

    Raises:
        TypeError: If ``value`` cannot be converted.
    """
    if isinstance(value, PredicateExpr):
        return value
    if isinstance(value, bool):
        return PredicateExpr.constant(value)
    if isinstance(value, AmplitudeExpr):
        raise TypeError(
            "Cannot use an AmplitudeExpr directly as a predicate. "
            "Use an explicit comparison, e.g. expr > 0."
        )
    raise TypeError(
        f"Cannot coerce {type(value)!r} into a PredicateExpr. "
        "Use bool values or PredicateExpr objects."
    )


__all__ = [
    "PredicateExpr",
    "coerce_predicate_expr",
    "_collect_free_symbols_pred",
    "_render_predicate",
]
