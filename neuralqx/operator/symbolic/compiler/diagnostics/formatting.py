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


"""Formatting helpers for DSL diagnostics."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    count_diagnostics_by_severity,
)

LINTING_DOCS_INDEX_URL = (
    "https://neuralqx.readthedocs.io/en/latest/concepts/operators/symbolic/"
)
LINTING_DOCS_MESSAGES_URL = (
    "https://neuralqx.readthedocs.io/en/latest/concepts/operators/symbolic/"
)


def linting_docs_url_for_code(code: str) -> str:
    """Builds a stable linting documentation URL for one diagnostic code.

    Args:
        code: Lint diagnostic code such as ``"NKDSL-E001"``.

    Returns:
        Absolute documentation URL pointing to the code subsection.
    """
    normalized = str(code).strip().lower()
    return f"{LINTING_DOCS_MESSAGES_URL}#lint-code-{normalized}"


def _format_example_preview(diagnostic: DSLDiagnostic) -> str | None:
    """Formats a concise example preview from diagnostic context.

    Args:
        diagnostic: Diagnostic finding.

    Returns:
        One descriptive example preview line or ``None``.
    """
    context = diagnostic.context_map()
    examples = context.get("examples")
    if not isinstance(examples, tuple) or not examples:
        return None
    first = examples[0]
    if not isinstance(first, dict):
        return None
    source = first.get("source_state")
    target = first.get("target_state")
    if source is None or target is None:
        return None
    suffix = ""
    illegal_positions = first.get("illegal_positions")
    if illegal_positions is not None:
        suffix = f", illegal_sites={illegal_positions}"
    return f"Example: source={source} -> target={target}{suffix}"


def format_diagnostic(diagnostic: DSLDiagnostic) -> str:
    """Formats one diagnostic finding for user-facing messages.

    Args:
        diagnostic: Diagnostic finding to format.

    Returns:
        Readable formatted text.
    """
    location = diagnostic.operator_name
    if diagnostic.term_name is not None:
        location = f"{location}.{diagnostic.term_name}"

    lines = [f"- [{diagnostic.code}] {diagnostic.severity.upper()} @ {location}"]
    lines.append(f"  {diagnostic.message}")

    preview = _format_example_preview(diagnostic)
    if preview is not None:
        lines.append(f"  {preview}")

    if diagnostic.suggestion is not None:
        lines.append(f"  Suggestion: {diagnostic.suggestion}")
    lines.append(f"  Docs: {linting_docs_url_for_code(diagnostic.code)}")

    return "\n".join(lines)


def format_diagnostics_block(
    diagnostics: tuple[DSLDiagnostic, ...],
    *,
    max_items: int = 20,
) -> str:
    """Formats multiple diagnostics into one multi-line readable block.

    Args:
        diagnostics: Ordered diagnostic tuple.
        max_items: Maximum number of diagnostics to render explicitly.

    Returns:
        Multi-line formatted diagnostics block.
    """
    if not diagnostics:
        return "No DSL diagnostics."
    shown = diagnostics[: max(0, int(max_items))]
    counts = count_diagnostics_by_severity(diagnostics)
    lines = [
        (
            f"Diagnostics summary: total={len(diagnostics)} "
            f"(errors={counts['error']}, warnings={counts['warning']}, info={counts['info']})"
        )
    ]
    lines.extend("\n" + format_diagnostic(diagnostic) for diagnostic in shown)
    if len(diagnostics) > len(shown):
        lines.append(f"... and {len(diagnostics) - len(shown)} more diagnostic(s).")

    lines.append(f"\nRead more: {LINTING_DOCS_MESSAGES_URL}")
    lines.append(f"Overview: {LINTING_DOCS_INDEX_URL}")
    return "\n".join(lines)


__all__ = ["format_diagnostic", "format_diagnostics_block", "linting_docs_url_for_code"]
