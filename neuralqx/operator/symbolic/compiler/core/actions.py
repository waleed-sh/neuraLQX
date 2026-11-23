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


"""Shared ordered symbolic execution for connectivity analysis and kernel planning.

These nodes preserve arithmetic ordering and configuration casts. They describe
values, not algebraic equivalence classes; proofs and executable schedules are
separate consumers of the same sparse target representation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.operator.symbolic.ir.expressions import parse_symbol_declaration_args
from neuralqx.operator.symbolic.ir.predicates import PredicateExpr
from neuralqx.operator.symbolic.ir.update import UpdateOp
from neuralqx.utils.errors import NeuralqxValueError

_UNKNOWN = object()
_Node = tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class _Target:
    coordinates: tuple[tuple[int, _Node], ...]
    opaque: _Node | None = None

    def value(self, site: int) -> _Node:
        return dict(self.coordinates).get(site, ("read", site))


def _constant(value: Any) -> _Node:
    """Preserves scalar type information in a hashable expression key."""
    array = np.asarray(value)
    if array.ndim:
        return ("array", array.dtype.str, array.shape, array.tobytes())
    # Scalar equality alone conflates signed zeros. Preserve the bit pattern
    # while keeping dtype/value at the tail for exact-domain consumers.
    return (
        "const",
        type(value).__name__,
        array.tobytes(),
        array.dtype.str,
        array.item(),
    )


def _literal(node: _Node) -> Any:
    return node[-1] if node[0] == "const" else _UNKNOWN


def _freeze_payload(value: Any, bindings: Mapping[str, int]) -> _Node:
    """Builds complete structural keys, including arrays and nested operations."""
    if isinstance(value, (AmplitudeExpr, PredicateExpr)):
        return _expression(value, bindings)
    if isinstance(value, UpdateOp):
        return (
            "update",
            value.kind,
            tuple((k, _freeze_payload(v, bindings)) for k, v in value.params),
        )
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_freeze_payload(v, bindings) for v in value))
    if isinstance(value, dict):
        return (
            "mapping",
            tuple((k, _freeze_payload(v, bindings)) for k, v in sorted(value.items())),
        )
    return _constant(value)


def _expression(expr: Any, bindings: Mapping[str, int]) -> _Node:
    """Substitutes iterator indices while preserving arithmetic evaluation order."""
    if not isinstance(expr, (AmplitudeExpr, PredicateExpr)):
        return _constant(expr)
    if expr.op == "const":
        return _constant(expr.args[0])
    if isinstance(expr, AmplitudeExpr):
        if expr.op == "symbol":
            name, declaration = parse_symbol_declaration_args(expr.args)
            for label, site in bindings.items():
                if name == f"site:{label}:index":
                    # Reference iterator bindings are strong int32 values.
                    node = _constant(np.int32(site))
                    return (
                        ("convert", declaration["dtype"], node)
                        if "dtype" in declaration
                        else node
                    )
                if name == f"site:{label}:value":
                    node = ("read", site)
                    return (
                        ("convert", declaration["dtype"], node)
                        if "dtype" in declaration
                        else node
                    )
            if "default" in declaration and name != "__x__":
                default = declaration["default"]
                return _constant(
                    np.asarray(default, dtype=declaration["dtype"])
                    if "dtype" in declaration
                    else default
                )
            return ("symbol", name, _freeze_payload(declaration, bindings))
        if expr.op == "static_index":
            return ("read", int(expr.args[0]))
        if expr.op == "static_emitted_index":
            return ("emitted", int(expr.args[0]))
    prefix = "predicate:" if isinstance(expr, PredicateExpr) else ""
    args = tuple(_expression(arg, bindings) for arg in expr.args)
    node = (prefix + expr.op, *args)
    # Constant evaluation is used for indices and branch liveness only. Floating
    # arithmetic is deliberately not reassociated or folded into target keys.
    return node


def _cast(node: _Node) -> _Node:
    # A configuration read already has the configuration dtype.
    return node if node[0] in {"read", "cast"} else ("cast", node)


def _static_index(expr: Any, bindings: Mapping[str, int], size: int) -> int:
    node = _expression(expr, bindings)
    bounds = [0, 0, 2**63]
    value = _evaluate(node, {}, bounds)
    if value is _UNKNOWN:
        raise NeuralqxValueError("state-dependent update index")
    limit = min(bounds[2], 2**31 - 1)
    if not -limit <= bounds[0] <= bounds[1] <= limit:
        raise NeuralqxValueError("index arithmetic requires runtime dtype evaluation")
    index = int(value)
    if not -size <= index < size:
        raise NeuralqxValueError("out-of-range update index")
    return index % size


def _target(
    ops: tuple[UpdateOp, ...],
    bindings: Mapping[str, int],
    size: int,
    wrap_spec: tuple[int, int] | None = None,
) -> _Target:
    """Symbolically executes all update kinds over a sparse source-state map."""

    def run(program: tuple[UpdateOp, ...], state: dict[int, _Node]) -> dict[int, _Node]:
        state = dict(state)

        def read(site: int) -> _Node:
            return state.get(site, ("read", site))

        for op in program:
            if op.kind == "invalidate_branch":
                continue
            if op.kind == "scatter":
                for site, value in zip(
                    op.get("flat_indices"), op.get("values"), strict=False
                ):
                    if not -size <= int(site) < size:
                        raise NeuralqxValueError("out-of-range scatter index")
                    state[int(site) % size] = _cast(_expression(value, bindings))
            elif op.kind == "swap_sites":
                a = _static_index(op.get("site_a"), bindings, size)
                b = _static_index(op.get("site_b"), bindings, size)
                va, vb = read(a), read(b)
                state[a], state[b] = vb, va
            elif op.kind == "permute_sites":
                sites = tuple(_static_index(s, bindings, size) for s in op.get("sites"))
                values = tuple(read(s) for s in sites)
                for k, site in enumerate(sites):
                    state[site] = values[(k + 1) % len(sites)]
            elif op.kind == "cond_branch":
                predicate = _expression(op.get("predicate"), bindings)
                left = run(op.get("then_ops") or (), state)
                right = run(op.get("else_ops") or (), state)
                for site in left.keys() | right.keys():
                    left_value = left.get(site, ("read", site))
                    right_value = right.get(site, ("read", site))
                    state[site] = (
                        left_value
                        if left_value == right_value
                        else (
                            "select",
                            predicate,
                            left_value,
                            right_value,
                        )
                    )
            else:
                site = _static_index(op.get("site"), bindings, size)
                if op.kind == "write_site":
                    value = _expression(op.get("value"), bindings)
                elif op.kind in {"shift_site", "shift_mod_site"}:
                    value = ("add", read(site), _expression(op.get("delta"), bindings))
                    if op.kind == "shift_mod_site":
                        value = (
                            (
                                "wrap",
                                value,
                                _constant(wrap_spec[0]),
                                _constant(wrap_spec[1]),
                            )
                            if wrap_spec is not None
                            else ("wrap", value)
                        )
                elif op.kind == "affine_site":
                    value = (
                        "add",
                        ("mul", _expression(op.get("scale"), bindings), read(site)),
                        _expression(op.get("bias"), bindings),
                    )
                else:
                    raise NeuralqxValueError("unrecognized update")
                state[site] = _cast(value)
        return state

    try:
        state = run(ops, {})
        return _Target(
            tuple(sorted((s, v) for s, v in state.items() if v != ("read", s)))
        )
    except (TypeError, ValueError, OverflowError):
        # Opaque programs can still be value-numbered. The instantiated spelling
        # is included, so distinct dynamic programs are never merged by accident.
        return _Target((), _freeze_payload(ops, bindings))


def _dependencies(node: _Node) -> frozenset[int]:
    if node[0] == "read":
        return frozenset((node[1],))
    return frozenset().union(
        *(_dependencies(n) for n in node[1:] if isinstance(n, tuple))
    )


def _evaluate(node: _Node, values: Mapping[int, int], bounds: list[int]) -> Any:
    """Evaluates the exact-integer abstract domain, returning unknown otherwise."""
    kind = node[0]
    value: Any
    if kind == "read":
        value = values.get(node[1], _UNKNOWN)
    elif kind == "const":
        value = _literal(node)
        if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
            return _UNKNOWN
        if value != int(value):
            return _UNKNOWN
        if isinstance(value, float) and len(bounds) > 2:
            precision = min(24, np.finfo(np.dtype(node[-2])).nmant + 1)
            bounds[2] = min(bounds[2], 2**precision)
        elif len(bounds) > 2 and np.dtype(node[-2]).kind in "iu":
            integer_dtype: Any = np.dtype(node[-2])
            bounds[2] = min(bounds[2], np.iinfo(integer_dtype).max, 2**31 - 1)
        value = int(value)
    elif kind == "cast":
        return _evaluate(node[1], values, bounds)
    elif kind == "select":
        pred = _evaluate(node[1], values, bounds)
        if pred is _UNKNOWN:
            return _UNKNOWN
        return _evaluate(node[2] if pred else node[3], values, bounds)
    else:
        args = [
            _evaluate(arg, values, bounds) for arg in node[1:] if isinstance(arg, tuple)
        ]
        if kind in {"predicate:and", "predicate:or"}:
            decisive = 0 if kind == "predicate:and" else 1
            if any(arg is not _UNKNOWN and bool(arg) == bool(decisive) for arg in args):
                return bool(decisive)
            return (
                _UNKNOWN if any(arg is _UNKNOWN for arg in args) else not bool(decisive)
            )
        if any(arg is _UNKNOWN for arg in args):
            return _UNKNOWN
        if kind == "add":
            value = args[0] + args[1]
        elif kind == "sub":
            value = args[0] - args[1]
        elif kind == "mul":
            value = args[0] * args[1]
        elif kind == "neg":
            value = -args[0]
        elif kind == "abs_":
            value = abs(args[0])
        elif kind == "wrap" and len(args) == 3 and args[2] > 0:
            shifted = args[0] - args[1]
            bounds[0], bounds[1] = min(bounds[0], shifted), max(bounds[1], shifted)
            value = shifted % args[2] + args[1]
        elif kind == "predicate:not":
            return not args[0]
        elif kind.startswith("predicate:"):
            a, b = args
            return {
                "eq": a == b,
                "ne": a != b,
                "lt": a < b,
                "le": a <= b,
                "gt": a > b,
                "ge": a >= b,
            }[kind[10:]]
        else:
            return _UNKNOWN
    if value is not _UNKNOWN:
        bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
    return value
