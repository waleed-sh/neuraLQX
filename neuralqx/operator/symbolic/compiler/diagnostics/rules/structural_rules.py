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


"""Structural diagnostics rules for symbolic operator terms."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    AbstractDiagnosticRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    DiagnosticRuleContext,
)
from neuralqx.operator.symbolic.ir.term import KBodyIteratorSpec


class ConstantFalsePredicateRule(AbstractDiagnosticRule):
    """Reports terms with branch predicates that are always false."""

    @property
    def name(self) -> str:
        return "constant_false_predicate"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        for term in context.ir.terms:
            if term.predicate.op == "const" and not bool(term.predicate.args[0]):
                diagnostics.append(
                    DSLDiagnostic.create(
                        code="NQXDSL-W101",
                        severity="warning",
                        message=(
                            "Term predicate is constant false; this term never emits connected states."
                        ),
                        operator_name=context.ir.operator_name,
                        term_name=term.name,
                        suggestion="Remove the term or adjust the predicate condition.",
                    )
                )
        return tuple(diagnostics)


class DuplicateEmissionRule(AbstractDiagnosticRule):
    """Reports duplicate emissions inside one term."""

    @property
    def name(self) -> str:
        return "duplicate_emissions"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        for term in context.ir.terms:
            emissions = term.effective_emissions
            if len(emissions) < 2:
                continue
            counts: dict[tuple, int] = {}
            for emission in emissions:
                key = (emission.update_program, emission.amplitude)
                counts[key] = counts.get(key, 0) + 1
            duplicates = sum(count - 1 for count in counts.values() if count > 1)
            if duplicates <= 0:
                continue
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-W103",
                    severity="warning",
                    message=(
                        f"Term contains {duplicates} duplicate emission branch(es) "
                        "with identical update/amplitude definitions."
                    ),
                    operator_name=context.ir.operator_name,
                    term_name=term.name,
                    suggestion="Remove duplicate emits or differentiate branch updates/amplitudes.",
                    context={"duplicate_count": duplicates},
                )
            )
        return tuple(diagnostics)


class MaxConnHintLowerBoundRule(AbstractDiagnosticRule):
    """Reports explicit max-connection hints below static emission upper bounds."""

    @property
    def name(self) -> str:
        return "max_conn_hint_lower_bound"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        for term in context.ir.terms:
            hint = term.max_conn_size_hint
            if hint is None or not isinstance(term.iterator, KBodyIteratorSpec):
                continue
            static_upper = len(term.iterator.index_sets) * len(term.effective_emissions)
            if hint >= static_upper:
                continue
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-W104",
                    severity="warning",
                    message=(
                        f"max_conn_size_hint={hint} is lower than static upper bound "
                        f"{static_upper} for this term."
                    ),
                    operator_name=context.ir.operator_name,
                    term_name=term.name,
                    suggestion="Increase max_conn_size hint to at least the static upper bound.",
                    context={
                        "max_conn_size_hint": hint,
                        "static_upper_bound": static_upper,
                    },
                )
            )
        return tuple(diagnostics)


class MissingBranchTagRule(AbstractDiagnosticRule):
    """Reports untagged branches on multi-emission terms."""

    @property
    def name(self) -> str:
        return "missing_branch_tags"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        for term in context.ir.terms:
            emissions = term.effective_emissions
            if len(emissions) <= 1:
                continue
            missing = sum(1 for emission in emissions if emission.branch_tag is None)
            if missing == 0:
                continue
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-I201",
                    severity="info",
                    message=(
                        f"Multi-emission term has {missing}/{len(emissions)} branch(es) without "
                        "diagnostic tags."
                    ),
                    operator_name=context.ir.operator_name,
                    term_name=term.name,
                    suggestion="Assign emit(..., tag='name') for clearer diagnostics output.",
                    context={"missing_tags": missing, "emission_count": len(emissions)},
                )
            )
        return tuple(diagnostics)


__all__ = [
    "ConstantFalsePredicateRule",
    "DuplicateEmissionRule",
    "MaxConnHintLowerBoundRule",
    "MissingBranchTagRule",
]
