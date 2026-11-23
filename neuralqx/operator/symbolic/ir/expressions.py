# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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
import numbers
from typing import Any

import numpy as np

from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

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

_UNSET_SYMBOL_DEFAULT = object()
_SYMBOL_DECLARATION_KEYS: frozenset[str] = frozenset({"default", "doc", "dtype"})


def _freeze(v: Any) -> Any:
    """Recursively convert lists to tuples for hashability."""
    if isinstance(v, list):
        return tuple(_freeze(i) for i in v)
    return v


def _normalize_symbol_name(name: str) -> str:
    """Normalizes one symbol name and validates non-empty content."""
    normalized = str(name).strip()
    if not normalized:
        raise NeuralqxValueError("Amplitude symbols must be non-empty strings.")
    return normalized


def _normalize_symbol_dtype(dtype: Any | None) -> str | None:
    """Normalizes one optional symbol dtype declaration.

    Args:
        dtype: Optional NumPy-compatible dtype specifier.

    Returns:
        Canonical dtype name, or ``None`` when no dtype is declared.
    """
    if dtype is None:
        return None
    try:
        return np.dtype(dtype).name
    except TypeError as exc:
        raise NeuralqxValueError(
            f"Unsupported symbol dtype declaration: {dtype!r}."
        ) from exc


def parse_symbol_declaration_args(args: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    """Parses a symbol-expression payload into name + declaration map.

    Supported payload forms:
      1. ``(name,)``
      2. ``(name, declaration_tuple)``

    ``declaration_tuple`` must be a stable tuple of ``(key, value)`` pairs.

    Args:
        args: ``AmplitudeExpr.args`` payload for a ``symbol`` node.

    Returns:
        Tuple ``(name, declaration_map)``.
    """
    if not args:
        raise NeuralqxValueError(
            "Symbol expression payload must contain at least one argument."
        )

    name = _normalize_symbol_name(str(args[0]))
    if len(args) == 1:
        return name, {}

    if len(args) != 2:
        raise NeuralqxValueError(
            "Symbol expression payload must have either one item (name) or two "
            "items (name + declaration tuple)."
        )

    raw_declaration = args[1]
    if not isinstance(raw_declaration, tuple):
        raise NeuralqxTypeError(
            "Symbol declaration payload must be a tuple of (key, value) pairs."
        )

    declaration: dict[str, Any] = {}
    for entry in raw_declaration:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise NeuralqxTypeError(
                "Each symbol declaration entry must be a two-item tuple."
            )
        key = str(entry[0]).strip()
        if not key:
            raise NeuralqxValueError(
                "Symbol declaration keys must be non-empty strings."
            )
        if key not in _SYMBOL_DECLARATION_KEYS:
            raise NeuralqxValueError(
                f"Unsupported symbol declaration key {key!r}. "
                f"Allowed: {sorted(_SYMBOL_DECLARATION_KEYS)!r}."
            )
        if key in declaration:
            raise NeuralqxValueError(f"Duplicate symbol declaration key {key!r}.")
        value = entry[1]
        if key == "doc":
            value = str(value).strip()
            if not value:
                continue
        elif key == "dtype":
            value = _normalize_symbol_dtype(value)
            if value is None:
                continue
        declaration[key] = value

    return name, declaration


@dataclasses.dataclass(frozen=True, repr=False)
class AmplitudeExpr:
    """
    Typed expression node for operator matrix elements.

    Attributes:
        op: Expression operation name.
        args: Ordered operation arguments (frozen tuple).

    Equality and inequality against numeric constants or symbol strings build
    symbolic predicates. Comparisons between two amplitude expressions retain
    structural equality so expression trees remain usable as dictionary keys
    and in compiler caches. Use ``PredicateExpr.eq`` or ``PredicateExpr.ne``
    for runtime equality comparisons between two amplitude expressions.
    """

    op: str
    args: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.op not in _AMPLITUDE_OPS:
            raise NeuralqxValueError(
                f"Unsupported amplitude-expression op: {self.op!r}. "
                f"Allowed: {sorted(_AMPLITUDE_OPS)}."
            )

    @classmethod
    def constant(cls, value: complex | float | int) -> AmplitudeExpr:
        """Builds a constant-value expression node."""
        if isinstance(value, bool):
            raise NeuralqxTypeError(
                "Boolean values are not valid amplitude constants. Use numeric literals instead."
            )
        return cls(op="const", args=(_freeze(value),))

    @classmethod
    def symbol(
        cls,
        name: str,
        *,
        default: Any = _UNSET_SYMBOL_DEFAULT,
        doc: str = "",
        dtype: Any | None = None,
    ) -> AmplitudeExpr:
        """Builds a symbol-reference expression node.

        Args:
            name: Symbol name.
            default: Optional default value used when the symbol is not supplied
                in the runtime evaluation environment.
            doc: Optional descriptive note for tooling and readability.
            dtype: Optional declared dtype for this symbol. If omitted and
                ``default`` is provided, dtype is inferred from ``default``.

        Raises:
            TypeError: If a provided ``default`` cannot be converted to the
                declared ``dtype``.
        """
        normalized_name = _normalize_symbol_name(name)
        normalized_dtype = _normalize_symbol_dtype(dtype)
        declaration: list[tuple[str, Any]] = []

        if default is not _UNSET_SYMBOL_DEFAULT:
            default_value = _freeze(default)
            inferred_dtype = np.asarray(default).dtype.name
            if normalized_dtype is None:
                normalized_dtype = inferred_dtype
            else:
                try:
                    np.asarray(default, dtype=np.dtype(normalized_dtype))
                except (TypeError, ValueError) as exc:
                    raise NeuralqxTypeError(
                        f"Default value for symbol {normalized_name!r} cannot be converted "
                        f"to declared dtype {normalized_dtype!r}."
                    ) from exc
            declaration.append(("default", default_value))

        normalized_doc = str(doc).strip()
        if normalized_doc:
            declaration.append(("doc", normalized_doc))

        if normalized_dtype is not None:
            declaration.append(("dtype", normalized_dtype))

        if declaration:
            return cls(op="symbol", args=(normalized_name, tuple(declaration)))
        return cls(op="symbol", args=(normalized_name,))

    def neg(self) -> AmplitudeExpr:
        """Builds a unary negation expression node."""
        return AmplitudeExpr(op="neg", args=(coerce_amplitude_expr(self),))

    def sqrt(self) -> AmplitudeExpr:
        """Builds a square-root expression node."""
        return AmplitudeExpr(op="sqrt", args=(coerce_amplitude_expr(self),))

    def conj(self) -> AmplitudeExpr:
        """Builds a complex-conjugate expression node."""
        return AmplitudeExpr(op="conj", args=(coerce_amplitude_expr(self),))

    def add(self, other: Any) -> AmplitudeExpr:
        """Builds an addition expression node."""
        return AmplitudeExpr(
            op="add",
            args=(coerce_amplitude_expr(self), coerce_amplitude_expr(other)),
        )

    def sub(self, other: Any) -> AmplitudeExpr:
        """Builds a subtraction expression node."""
        return AmplitudeExpr(
            op="sub",
            args=(coerce_amplitude_expr(self), coerce_amplitude_expr(other)),
        )

    def mul(self, other: Any) -> AmplitudeExpr:
        """Builds a multiplication expression node."""
        return AmplitudeExpr(
            op="mul",
            args=(coerce_amplitude_expr(self), coerce_amplitude_expr(other)),
        )

    def div(self, other: Any) -> AmplitudeExpr:
        """Builds a division expression node."""
        return AmplitudeExpr(
            op="div",
            args=(coerce_amplitude_expr(self), coerce_amplitude_expr(other)),
        )

    def pow(self, exponent: Any) -> AmplitudeExpr:
        """Builds a power expression node (base ** exponent)."""
        return AmplitudeExpr(
            op="pow",
            args=(coerce_amplitude_expr(self), coerce_amplitude_expr(exponent)),
        )

    def abs_(self) -> AmplitudeExpr:
        """Builds an absolute-value expression node (|operand|)."""
        return AmplitudeExpr(op="abs_", args=(coerce_amplitude_expr(self),))

    @classmethod
    def static_index(cls, flat_index: int) -> AmplitudeExpr:
        """
        Builds a static-index read node (reads x[flat_index] at eval time).

        Unlike :meth:`symbol`, which is bound to a site-iterator label, a
        static-index node reads the full input configuration ``x`` at a
        compile-time-constant integer offset.  This is the primary mechanism
        for accessing *structured* Hilbert space layouts (e.g. U(1)^G gauge
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
            raise NeuralqxValueError(
                f"static_index requires a non-negative integer; got {flat_index!r}."
            )
        return cls(op="static_index", args=(idx,))

    @classmethod
    def static_emitted_index(cls, flat_index: int) -> AmplitudeExpr:
        """Builds a static-index read node for the emitted/connected state x'[flat_index]."""
        idx = int(flat_index)
        if idx < 0:
            raise NeuralqxValueError(
                f"static_emitted_index requires a non-negative integer; got {flat_index!r}."
            )
        return cls(op="static_emitted_index", args=(idx,))

    def wrap_mod(self) -> AmplitudeExpr:
        """Builds a Hilbert-aware modulo-wrap node."""
        return AmplitudeExpr(op="wrap_mod", args=(coerce_amplitude_expr(self),))

    def __add__(self, other: Any) -> AmplitudeExpr:
        return self.add(other)

    def __radd__(self, other: Any) -> AmplitudeExpr:
        return AmplitudeExpr.add(other, self)

    def __sub__(self, other: Any) -> AmplitudeExpr:
        return self.sub(other)

    def __rsub__(self, other: Any) -> AmplitudeExpr:
        return AmplitudeExpr.sub(other, self)

    def __mul__(self, other: Any) -> AmplitudeExpr:
        return self.mul(other)

    def __rmul__(self, other: Any) -> AmplitudeExpr:
        return AmplitudeExpr.mul(other, self)

    def __truediv__(self, other: Any) -> AmplitudeExpr:
        return self.div(other)

    def __rtruediv__(self, other: Any) -> AmplitudeExpr:
        return AmplitudeExpr.div(other, self)

    def __neg__(self) -> AmplitudeExpr:
        return self.neg()

    #
    #
    # Scalar comparisons yield PredicateExpr (imported lazily).
    # Expression-to-expression equality retains the dataclass semantics.

    def __eq__(self, other: Any) -> Any:
        """Compares expression trees structurally or builds a scalar equality predicate."""
        if other.__class__ is self.__class__:
            return (self.op, self.args) == (other.op, other.args)
        if isinstance(other, (numbers.Number, str)) and not isinstance(
            other, (bool, np.bool_)
        ):
            from .predicates import PredicateExpr

            return PredicateExpr.eq(self, other)
        return NotImplemented

    def __ne__(self, other: Any) -> Any:
        """Compares expression trees structurally or builds a scalar inequality predicate."""
        if other.__class__ is self.__class__:
            return (self.op, self.args) != (other.op, other.args)
        if isinstance(other, (numbers.Number, str)) and not isinstance(
            other, (bool, np.bool_)
        ):
            from .predicates import PredicateExpr

            return PredicateExpr.ne(self, other)
        return NotImplemented

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


def _render_amplitude(expr: AmplitudeExpr) -> str:
    """Renders an AmplitudeExpr as a readable infix string."""
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
        name, _declaration = parse_symbol_declaration_args(args)
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


def _collect_free_symbols(expr: AmplitudeExpr, result: set[str]) -> None:
    """Recursively collects free (non-iterator-bound) symbol names from an AmplitudeExpr."""
    if expr.op == "symbol":
        name, declaration = parse_symbol_declaration_args(expr.args)
        parts = name.split(":")
        # Bound symbols follow "namespace:label:field" — site:i:value, emit:i:index, etc.
        if (
            not (len(parts) == 3 and parts[0] in ("site", "emit"))
            and "default" not in declaration
        ):
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
    if isinstance(value, (bool, np.bool_)):
        raise NeuralqxTypeError(
            "Boolean values are not valid amplitude constants. Use numeric literals instead."
        )
    if isinstance(value, numbers.Number):
        return AmplitudeExpr.constant(value)
    if isinstance(value, str):
        return AmplitudeExpr.symbol(value)
    raise NeuralqxTypeError(
        f"Cannot coerce {type(value)!r} into an AmplitudeExpr. "
        "Use numeric constants, symbol strings, or AmplitudeExpr values."
    )


__all__ = [
    "AmplitudeExpr",
    "_collect_free_symbols",
    "_render_amplitude",
    "coerce_amplitude_expr",
    "parse_symbol_declaration_args",
]
