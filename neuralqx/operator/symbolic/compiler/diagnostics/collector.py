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


"""Diagnostics collection and filtering orchestration."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.diagnostics.models import DiagnosticSeverity
from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    diagnostic_severity_at_least,
)
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    diagnostic_severity_rank,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules import DiagnosticRuleContext
from neuralqx.operator.symbolic.compiler.diagnostics.rules import (
    default_diagnostic_rules,
)


def run_default_diagnostics(
    *,
    operator,
    ir,
    options,
) -> tuple[DSLDiagnostic, ...]:
    """Runs all default diagnostics rules over one compilation input.

    Args:
        operator (AbstractSymbolicOperator): Source symbolic operator.
        ir (SymbolicOperatorIR): Symbolic operator IR.
        options (SymbolicCompilerOptions): Effective compiler options.

    Returns:
        Ordered tuple of diagnostics from all executed rules.
    """
    context = DiagnosticRuleContext(operator=operator, ir=ir, options=options)
    diagnostics: list[DSLDiagnostic] = []
    cap = max(1, int(options.max_diagnostics))
    for rule in default_diagnostic_rules():
        emitted = rule.run(context)
        if emitted:
            diagnostics.extend(emitted)
        if len(diagnostics) >= cap:
            break
    ordered = sorted(
        diagnostics[:cap],
        key=lambda item: (
            -diagnostic_severity_rank(item.severity),
            item.code,
            item.operator_name,
            "" if item.term_name is None else item.term_name,
        ),
    )
    return tuple(ordered)


def filter_diagnostics_by_minimum_severity(
    diagnostics: tuple[DSLDiagnostic, ...],
    *,
    minimum: DiagnosticSeverity,
) -> tuple[DSLDiagnostic, ...]:
    """Filters diagnostics at or above one severity threshold.

    Args:
        diagnostics: Input diagnostics tuple.
        minimum: Minimum severity threshold.

    Returns:
        Filtered diagnostics tuple.
    """
    return tuple(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic_severity_at_least(diagnostic.severity, minimum)
    )


__all__ = ["filter_diagnostics_by_minimum_severity", "run_default_diagnostics"]
