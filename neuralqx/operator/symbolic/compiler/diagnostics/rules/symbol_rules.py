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


"""Symbol and index-bound diagnostics rules."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    AbstractDiagnosticRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    DiagnosticRuleContext,
)
from neuralqx.operator.symbolic.compiler.diagnostics.traversals import (
    iter_term_amplitude_nodes,
)
from neuralqx.operator.symbolic.compiler.diagnostics.traversals import (
    iter_term_static_index_nodes,
)
from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.operator.symbolic.ir.expressions import parse_symbol_declaration_args


class UnresolvedFreeSymbolsRule(AbstractDiagnosticRule):
    """Reports unresolved free symbols present in the operator IR."""

    @property
    def name(self) -> str:
        return "unresolved_free_symbols"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        free_symbols = tuple(sorted(context.ir.free_symbols))
        if not free_symbols:
            return ()
        shown = ", ".join(f"%{name}" for name in free_symbols[:8])
        suffix = (
            "" if len(free_symbols) <= 8 else f", ... +{len(free_symbols) - 8} more"
        )
        return (
            DSLDiagnostic.create(
                code="NQXDSL-E001",
                severity="error",
                message=f"Unresolved free symbol(s): {shown}{suffix}.",
                operator_name=context.ir.operator_name,
                suggestion=(
                    "Replace with constants or bind symbols before compilation. "
                    "DSL compilation requires all free symbols to be resolved."
                ),
                context={"free_symbols": free_symbols},
            ),
        )


class StaticIndexBoundsRule(AbstractDiagnosticRule):
    """Reports static source/target index reads outside Hilbert bounds."""

    @property
    def name(self) -> str:
        return "static_index_bounds"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        hilbert_size = int(context.ir.hilbert_size)
        for term in context.ir.terms:
            for op_name, flat_index in iter_term_static_index_nodes(term):
                if 0 <= flat_index < hilbert_size:
                    continue
                diagnostics.append(
                    DSLDiagnostic.create(
                        code="NQXDSL-E002",
                        severity="error",
                        message=(
                            f"{op_name} uses flat index {flat_index}, but hilbert_size is "
                            f"{hilbert_size}."
                        ),
                        operator_name=context.ir.operator_name,
                        term_name=term.name,
                        suggestion="Use a static index in [0, hilbert_size).",
                        context={
                            "flat_index": flat_index,
                            "hilbert_size": hilbert_size,
                            "op_name": op_name,
                        },
                    )
                )
        return tuple(diagnostics)


def _is_direct_site_value_read(expr: AmplitudeExpr) -> bool:
    """Returns ``True`` when *expr* is a direct runtime site-value read."""
    if expr.op in {"static_index", "static_emitted_index"}:
        return True
    if expr.op != "symbol":
        return False
    name, _declaration = parse_symbol_declaration_args(expr.args)
    return (name.startswith("site:") and name.endswith(":value")) or (
        name.startswith("emit:") and name.endswith(":value")
    )


class PotentialZeroDivisionRule(AbstractDiagnosticRule):
    """Reports direct divisions by runtime state values that may be zero."""

    @property
    def name(self) -> str:
        return "potential_zero_division"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        for term in context.ir.terms:
            hits = 0
            for node in iter_term_amplitude_nodes(term):
                if node.op != "div":
                    continue
                denominator = node.args[1]
                if isinstance(
                    denominator, AmplitudeExpr
                ) and _is_direct_site_value_read(denominator):
                    hits += 1
            if hits == 0:
                continue
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-E003",
                    severity="error",
                    message=(
                        f"Detected {hits} division expression(s) whose denominator is a runtime "
                        "state value and may be zero."
                    ),
                    operator_name=context.ir.operator_name,
                    term_name=term.name,
                    suggestion=(
                        "Guard the division with a non-zero predicate (for example `site(i) != 0`) "
                        "or rewrite using an explicit conditional branch."
                    ),
                    context={"potential_zero_division_count": hits},
                )
            )
        return tuple(diagnostics)


__all__ = [
    "PotentialZeroDivisionRule",
    "StaticIndexBoundsRule",
    "UnresolvedFreeSymbolsRule",
]
