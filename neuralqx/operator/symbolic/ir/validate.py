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


"""Structural and semantic validation for symbolic operator IR."""

from __future__ import annotations

from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.utils.errors import NeuralqxValueError

from .expressions import AmplitudeExpr
from .expressions import parse_symbol_declaration_args
from .predicates import PredicateExpr
from .program import SymbolicOperatorIR
from .term import SymbolicIRTerm
from .update import UpdateOp


def _iter_amplitude_symbols(expr: AmplitudeExpr) -> list[str]:
    """Collects all symbol names in an amplitude expression."""
    symbols: list[str] = []

    def visit(node: AmplitudeExpr) -> None:
        if node.op == "symbol":
            name, _declaration = parse_symbol_declaration_args(node.args)
            symbols.append(name)
            return
        for arg in node.args:
            if isinstance(arg, AmplitudeExpr):
                visit(arg)

    visit(expr)
    return symbols


def _iter_predicate_symbols(expr: PredicateExpr) -> list[str]:
    """Collects all symbol names in a predicate expression."""
    symbols: list[str] = []

    def visit_pred(node: PredicateExpr) -> None:
        if node.op == "const":
            return
        for arg in node.args:
            if isinstance(arg, PredicateExpr):
                visit_pred(arg)
            elif isinstance(arg, AmplitudeExpr):
                symbols.extend(_iter_amplitude_symbols(arg))

    visit_pred(expr)
    return symbols


def _validate_symbol_scope(term: SymbolicIRTerm, symbol: str) -> None:
    """Validates that a symbol references a label bound by the term iterator."""
    parts = symbol.split(":")
    if len(parts) != 3:
        # Free symbol (not a site reference), allowed
        return
    kind, label, _field = parts
    if kind not in {"site", "emit"}:
        raise NeuralqxValueError(
            f"Term {term.name!r} references symbol {symbol!r} with unknown "
            f"namespace {kind!r}. Only 'site' and 'emit' namespaces are supported."
        )
    labels = getattr(term.iterator, "labels", None)
    if labels is not None:
        bound_labels = {str(lbl) for lbl in labels}
    else:
        bound_labels = {term.iterator.label_a}
        if term.iterator.label_b is not None:
            bound_labels.add(term.iterator.label_b)
    if label not in bound_labels:
        raise NeuralqxValueError(
            f"Term {term.name!r} references site symbol {symbol!r} but the "
            f"label {label!r} is not bound by its iterator "
            f"(bound: {sorted(bound_labels)!r})."
        )
    if not bound_labels:
        raise NeuralqxValueError(
            f"Term {term.name!r} uses a global iterator but references "
            f"site symbol {symbol!r}. Global terms may not reference site DOFs."
        )


def _validate_update_op(term: SymbolicIRTerm, op: UpdateOp) -> None:
    """Validates one update operation."""
    if op.kind == "write_site":
        if op.get("site") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: write_site requires a 'site' parameter."
            )
        if op.get("value") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: write_site requires a 'value' parameter."
            )
    elif op.kind == "shift_site":
        if op.get("site") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: shift_site requires a 'site' parameter."
            )
        if op.get("delta") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: shift_site requires a 'delta' parameter."
            )
    elif op.kind == "shift_mod_site":
        if op.get("site") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: shift_mod_site requires a 'site' parameter."
            )
        if op.get("delta") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: shift_mod_site requires a 'delta' parameter."
            )
    elif op.kind == "swap_sites":
        if op.get("site_a") is None or op.get("site_b") is None:
            raise NeuralqxValueError(
                f"Term {term.name!r}: swap_sites requires 'site_a' and 'site_b'."
            )
    elif op.kind == "invalidate_branch":
        pass  # no required parameters


def _validate_term(term: SymbolicIRTerm) -> tuple[str, ...]:
    """Validates one IR term and returns the referenced symbols."""
    symbols: list[str] = []
    symbols.extend(_iter_predicate_symbols(term.predicate))

    for em in term.effective_emissions:
        symbols.extend(_iter_predicate_symbols(em.predicate))
        symbols.extend(_iter_amplitude_symbols(em.amplitude))
        for op in em.update_program.ops:
            _validate_update_op(term, op)
            for _key, value in op.params:
                if isinstance(value, AmplitudeExpr):
                    symbols.extend(_iter_amplitude_symbols(value))
                elif isinstance(value, tuple):
                    for item in value:
                        if isinstance(item, AmplitudeExpr):
                            symbols.extend(_iter_amplitude_symbols(item))

    for sym in symbols:
        _validate_symbol_scope(term, sym)

    return tuple(sorted(set(symbols)))


def validate_symbolic_ir(ir: SymbolicOperatorIR) -> dict:
    """
    Validates a SymbolicOperatorIR structurally and semantically.

    Args:
        ir: Operator IR to validate.

    Returns:
        Validation summary dictionary.

    Raises:
        ValueError: If the IR is structurally invalid.
    """
    debug_event(
        "validating symbolic ir",
        scope="ir",
        tag="IR",
        operator_name=ir.operator_name,
        mode=ir.mode,
        term_count=ir.term_count,
    )
    if ir.mode != "symbolic":
        summary = {"mode": ir.mode, "term_count": 0, "term_symbols": {}}
        debug_event(
            "skipped symbolic ir validation for non-symbolic mode",
            scope="ir",
            tag="IR",
            operator_name=ir.operator_name,
            mode=ir.mode,
        )
        return summary

    if not ir.terms:
        raise NeuralqxValueError(
            f"Symbolic operator IR {ir.operator_name!r} has no terms."
        )

    term_symbols: dict[str, tuple[str, ...]] = {}
    for term in ir.terms:
        term_symbols[term.name] = _validate_term(term)

    summary = {
        "mode": ir.mode,
        "term_count": ir.term_count,
        "term_symbols": term_symbols,
    }
    debug_event(
        "validated symbolic ir",
        scope="ir",
        tag="IR",
        operator_name=ir.operator_name,
        term_count=ir.term_count,
        symbol_count=sum(len(v) for v in term_symbols.values()),
    )
    return summary


__all__ = ["validate_symbolic_ir"]
