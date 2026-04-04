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


"""
Typed amplitude-expression IR nodes for symbolic operators.

This module defines a compact, hashable expression tree used to represent
matrix-element amplitudes in declarative symbolic operators.
"""

from __future__ import annotations

import dataclasses
from typing import Any

_AMPLITUDE_OPS: frozenset[str] = frozenset(
    {
        "const",
        "symbol",
        "neg",
        "sqrt",
        "conj",
        "add",
        "sub",
        "mul",
        "div",
        "pow",  # base^exp, element-wise power
        "abs_",  # |operand|, absolute value
        "static_index",  # x[flat_index], reads source configuration
        "static_emitted_index",  # x'[flat_index], reads emitted/connected configuration
        "wrap_mod",  # wrap operand using Hilbert local_states modulo semantics
    }
)


def _freeze(v: Any) -> Any:
    """Recursively convert lists to tuples for hashability."""
    if isinstance(v, list):
        return tuple(_freeze(i) for i in v)
    return v


@dataclasses.dataclass(frozen=True, repr=False)
class AmplitudeExpr:
    """
    Typed expression node for operator matrix elements.

    Attributes:
        op: Expression operation name.
        args: Ordered operation arguments (frozen tuple).
    """

    op: str
    args: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.op not in _AMPLITUDE_OPS:
            raise ValueError(
                f"Unsupported amplitude-expression op: {self.op!r}. "
                f"Allowed: {sorted(_AMPLITUDE_OPS)}."
            )

    @classmethod
    def constant(cls, value: complex | float | int) -> "AmplitudeExpr":
        """Builds a constant-value expression node."""
        if isinstance(value, bool):
            raise TypeError(
                "Boolean values are not valid amplitude constants. "
                "Use numeric literals instead."
            )
        return cls(op="const", args=(_freeze(value),))

    @classmethod
    def symbol(cls, name: str) -> "AmplitudeExpr":
        """Builds a symbol-reference expression node."""
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Amplitude symbols must be non-empty strings.")
        return cls(op="symbol", args=(normalized,))

    @classmethod
    def neg(cls, operand: Any) -> "AmplitudeExpr":
        """Builds a unary negation expression node."""
        return cls(op="neg", args=(coerce_amplitude_expr(operand),))

    @classmethod
    def sqrt(cls, operand: Any) -> "AmplitudeExpr":
        """Builds a square-root expression node."""
        return cls(op="sqrt", args=(coerce_amplitude_expr(operand),))

    @classmethod
    def conj(cls, operand: Any) -> "AmplitudeExpr":
        """Builds a complex-conjugate expression node."""
        return cls(op="conj", args=(coerce_amplitude_expr(operand),))

    @classmethod
    def add(cls, left: Any, right: Any) -> "AmplitudeExpr":
        """Builds an addition expression node."""
        return cls(
            op="add",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def sub(cls, left: Any, right: Any) -> "AmplitudeExpr":
        """Builds a subtraction expression node."""
        return cls(
            op="sub",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def mul(cls, left: Any, right: Any) -> "AmplitudeExpr":
        """Builds a multiplication expression node."""
        return cls(
            op="mul",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def div(cls, left: Any, right: Any) -> "AmplitudeExpr":
        """Builds a division expression node."""
        return cls(
            op="div",
            args=(coerce_amplitude_expr(left), coerce_amplitude_expr(right)),
        )

    @classmethod
    def pow(cls, base: Any, exponent: Any) -> "AmplitudeExpr":
        """Builds a power expression node (base ** exponent)."""
        return cls(
            op="pow",
            args=(coerce_amplitude_expr(base), coerce_amplitude_expr(exponent)),
        )

    @classmethod
    def abs_(cls, operand: Any) -> "AmplitudeExpr":
        """Builds an absolute-value expression node (|operand|)."""
        return cls(op="abs_", args=(coerce_amplitude_expr(operand),))

    @classmethod
    def static_index(cls, flat_index: int) -> "AmplitudeExpr":
        """
        Builds a static-index read node (reads x[flat_index] at eval time).

        Unlike :meth:`symbol`, which is bound to a site-iterator label, a
        static-index node reads the full input configuration ``x`` at a
        compile-time-constant integer offset.  This is the primary mechanism
        for accessing *structured* Hilbert-space layouts (e.g. U(1)^G gauge
        theories where ``x[g * n_edges + e]`` holds the charge for gauge copy
        ``g`` at edge ``e``).

        The flat index is stored as a plain ``int`` in ``args[0]``.  The
        JAX lowerer resolves it by reading ``env["__x__"][flat_index]``.

        Args:
            flat_index: Non-negative integer flat index into the configuration
                array ``x``.
        """
        idx = int(flat_index)
        if idx < 0:
            raise ValueError(
                f"static_index requires a non-negative integer; got {flat_index!r}."
            )
        return cls(op="static_index", args=(idx,))

    @classmethod
    def static_emitted_index(cls, flat_index: int) -> "AmplitudeExpr":
        """Builds a static-index read node for the emitted/connected state x'[flat_index]."""
        idx = int(flat_index)
        if idx < 0:
            raise ValueError(
                f"static_emitted_index requires a non-negative integer; got {flat_index!r}."
            )
        return cls(op="static_emitted_index", args=(idx,))

    @classmethod
    def wrap_mod(cls, operand: Any) -> "AmplitudeExpr":
        """Builds a Hilbert-aware modulo-wrap node."""
        return cls(op="wrap_mod", args=(coerce_amplitude_expr(operand),))

    def __add__(self, other: Any) -> "AmplitudeExpr":
        return self.add(self, other)

    def __radd__(self, other: Any) -> "AmplitudeExpr":
        return self.add(other, self)

    def __sub__(self, other: Any) -> "AmplitudeExpr":
        return self.sub(self, other)

    def __rsub__(self, other: Any) -> "AmplitudeExpr":
        return self.sub(other, self)

    def __mul__(self, other: Any) -> "AmplitudeExpr":
        return self.mul(self, other)

    def __rmul__(self, other: Any) -> "AmplitudeExpr":
        return self.mul(other, self)

    def __truediv__(self, other: Any) -> "AmplitudeExpr":
        return self.div(self, other)

    def __rtruediv__(self, other: Any) -> "AmplitudeExpr":
        return self.div(other, self)

    def __neg__(self) -> "AmplitudeExpr":
        return self.neg(self)

    #
    #
    # Comparison operators yield PredicateExpr (imported lazily)

    def __lt__(self, other: Any) -> Any:
        from .predicates import PredicateExpr

        return PredicateExpr.lt(self, other)

    def __le__(self, other: Any) -> Any:
        from .predicates import PredicateExpr

        return PredicateExpr.le(self, other)

    def __gt__(self, other: Any) -> Any:
        from .predicates import PredicateExpr

        return PredicateExpr.gt(self, other)

    def __ge__(self, other: Any) -> Any:
        from .predicates import PredicateExpr

        return PredicateExpr.ge(self, other)

    def __str__(self) -> str:
        return _render_amplitude(self)

    def __repr__(self) -> str:
        return f"AmplitudeExpr(op={self.op!r}, args={self.args!r})"


def _render_amplitude(expr: "AmplitudeExpr") -> str:
    """Renders an AmplitudeExpr as a human-readable infix string."""
    op = expr.op
    args = expr.args

    if op == "const":
        v = args[0]
        if isinstance(v, complex):
            if v.imag == 0.0:
                v = v.real
            else:
                return repr(v)
        if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)

    if op == "symbol":
        name = str(args[0])
        parts = name.split(":")
        if len(parts) == 3:
            ns, label, field = parts
            if ns == "site":
                if field == "value":
                    return f"x[{label}]"
                if field == "index":
                    return label
                return f"x[{label}].{field}"
            if ns == "emit":
                if field == "value":
                    return f"x'[{label}]"
                if field == "index":
                    return label
                return f"x'[{label}].{field}"
        return f"%{name}"

    if op == "neg":
        inner = _render_amplitude(args[0])
        return f"-{inner}"

    if op == "sqrt":
        return f"sqrt({_render_amplitude(args[0])})"

    if op == "conj":
        return f"conj({_render_amplitude(args[0])})"

    if op == "abs_":
        return f"|{_render_amplitude(args[0])}|"

    if op == "wrap_mod":
        return f"wrap({_render_amplitude(args[0])})"

    if op == "static_index":
        return f"x[{args[0]}]"

    if op == "static_emitted_index":
        return f"x'[{args[0]}]"

    if op == "add":
        return f"({_render_amplitude(args[0])} + {_render_amplitude(args[1])})"

    if op == "sub":
        return f"({_render_amplitude(args[0])} - {_render_amplitude(args[1])})"

    if op == "mul":
        return f"({_render_amplitude(args[0])} * {_render_amplitude(args[1])})"

    if op == "div":
        return f"({_render_amplitude(args[0])} / {_render_amplitude(args[1])})"

    if op == "pow":
        return f"({_render_amplitude(args[0])}^{_render_amplitude(args[1])})"

    # Fallback for unknown ops
    arg_strs = ", ".join(
        _render_amplitude(a) if isinstance(a, AmplitudeExpr) else repr(a) for a in args
    )
    return f"{op}({arg_strs})"


def _collect_free_symbols(expr: "AmplitudeExpr", result: "set[str]") -> None:
    """Recursively collects free (non-iterator-bound) symbol names from an AmplitudeExpr."""
    if expr.op == "symbol":
        name = str(expr.args[0])
        parts = name.split(":")
        # Bound symbols follow "namespace:label:field" — site:i:value, emit:i:index, etc.
        if not (len(parts) == 3 and parts[0] in ("site", "emit")):
            result.add(name)
        return
    for arg in expr.args:
        if isinstance(arg, AmplitudeExpr):
            _collect_free_symbols(arg, result)
        elif isinstance(arg, tuple):
            for item in arg:
                if isinstance(item, AmplitudeExpr):
                    _collect_free_symbols(item, result)


def coerce_amplitude_expr(value: Any) -> AmplitudeExpr:
    """
    Coerces user values into typed amplitude-expression nodes.

    Args:
        value: Input expression value.

    Returns:
        Typed amplitude expression.

    Raises:
        TypeError: If ``value`` cannot be converted.
    """
    if isinstance(value, AmplitudeExpr):
        return value
    if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
        return AmplitudeExpr.constant(value)
    if isinstance(value, str):
        return AmplitudeExpr.symbol(value)
    raise TypeError(
        f"Cannot coerce {type(value)!r} into an AmplitudeExpr. "
        "Use numeric constants, symbol strings, or AmplitudeExpr values."
    )


__all__ = [
    "AmplitudeExpr",
    "coerce_amplitude_expr",
    "_collect_free_symbols",
    "_render_amplitude",
]
