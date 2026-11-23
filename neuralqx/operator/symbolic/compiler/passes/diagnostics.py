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


"""DSL lint/diagnostics compiler pass."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.diagnostics import (
    count_diagnostics_by_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics import (
    filter_diagnostics_by_minimum_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics import format_diagnostics_block
from neuralqx.operator.symbolic.compiler.diagnostics import (
    normalize_diagnostic_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics import run_default_diagnostics
from neuralqx.operator.symbolic.compiler.passes.base import AbstractSymbolicPass
from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.operator.symbolic.errors import SymbolicDiagnosticsError


def _diagnostics_summary_payload(
    *,
    all_diagnostics,
    visible_diagnostics,
) -> dict[str, Any]:
    """Builds one summary payload for diagnostics pass metadata/analysis.

    Args:
        all_diagnostics: Full diagnostics tuple.
        visible_diagnostics: Diagnostics surviving severity filtering.

    Returns:
        Summary dictionary with counts and diagnostic payloads.
    """
    counts_all = count_diagnostics_by_severity(all_diagnostics)
    counts_visible = count_diagnostics_by_severity(visible_diagnostics)
    return {
        "total_count": len(all_diagnostics),
        "visible_count": len(visible_diagnostics),
        "total_info": counts_all["info"],
        "total_warning": counts_all["warning"],
        "total_error": counts_all["error"],
        "visible_info": counts_visible["info"],
        "visible_warning": counts_visible["warning"],
        "visible_error": counts_visible["error"],
        "diagnostics": tuple(diagnostic.as_dict() for diagnostic in all_diagnostics),
        "visible_diagnostics": tuple(
            diagnostic.as_dict() for diagnostic in visible_diagnostics
        ),
    }


def _should_raise_for_diagnostics(
    context: SymbolicCompilationContext, visible_diagnostics
) -> bool:
    """Determines whether diagnostics should fail compilation.

    Args:
        context: Compilation context.
        visible_diagnostics: Diagnostics surviving severity filtering.

    Returns:
        ``True`` when pass should raise, otherwise ``False``.
    """
    has_visible_errors = any(
        diagnostic.severity == "error" for diagnostic in visible_diagnostics
    )
    if has_visible_errors and context.options.strict_validation:
        return True
    has_visible_warnings = any(
        diagnostic.severity == "warning" for diagnostic in visible_diagnostics
    )
    return bool(has_visible_warnings and context.options.fail_on_warnings)


class SymbolicDiagnosticsPass(AbstractSymbolicPass):
    """Runs DSL lint/diagnostics checks before lowering."""

    @property
    def name(self) -> str:
        return "symbolic_diagnostics"

    def run(self, context: SymbolicCompilationContext) -> Mapping[str, Any] | None:
        if not context.options.diagnostics_enabled:
            summary = {
                "enabled": False,
                "total_count": 0,
                "visible_count": 0,
                "diagnostics": (),
                "visible_diagnostics": (),
            }
            context.set_analysis("dsl_diagnostics", summary["diagnostics"])
            context.set_analysis("dsl_diagnostics_summary", summary)
            return summary

        minimum_severity = normalize_diagnostic_severity(
            context.options.diagnostics_min_severity
        )
        diagnostics = run_default_diagnostics(
            operator=context.operator,
            ir=context.ir,
            options=context.options,
        )
        visible = filter_diagnostics_by_minimum_severity(
            diagnostics,
            minimum=minimum_severity,
        )
        summary = _diagnostics_summary_payload(
            all_diagnostics=diagnostics,
            visible_diagnostics=visible,
        )
        summary["enabled"] = True
        summary["minimum_severity"] = minimum_severity
        context.set_analysis("dsl_diagnostics", summary["diagnostics"])
        context.set_analysis("dsl_diagnostics_summary", summary)

        debug_event(
            "completed symbolic diagnostics pass",
            scope="passes",
            pass_name=self.name,
            tag="PASS",
            operator_name=context.ir.operator_name,
            total_count=summary["total_count"],
            visible_count=summary["visible_count"],
            visible_warning=summary["visible_warning"],
            visible_error=summary["visible_error"],
        )

        if visible:
            message = (
                f"Symbolic diagnostics for operator {context.ir.operator_name!r}:\n"
                f"{format_diagnostics_block(visible, max_items=20)}"
            )
            if _should_raise_for_diagnostics(context, visible):
                raise SymbolicDiagnosticsError(message)
            warnings.warn(message, category=UserWarning, stacklevel=3)

        return {
            "enabled": True,
            "minimum_severity": minimum_severity,
            "total_count": summary["total_count"],
            "visible_count": summary["visible_count"],
            "visible_warning": summary["visible_warning"],
            "visible_error": summary["visible_error"],
        }


__all__ = ["SymbolicDiagnosticsPass"]
