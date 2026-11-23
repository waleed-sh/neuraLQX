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


"""Typed diagnostic models and severity helpers for DSL linting."""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Literal

from neuralqx.utils.errors import NeuralqxValueError

DiagnosticSeverity = Literal["info", "warning", "error"]
_ALLOWED_DIAGNOSTIC_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "error"})
_DIAGNOSTIC_SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
}


def normalize_diagnostic_severity(value: str) -> DiagnosticSeverity:
    """Normalizes and validates one diagnostic-severity string.

    Args:
        value: Raw severity value.

    Returns:
        One normalized diagnostic severity value.

    Raises:
        ValueError: If the value is unsupported.
    """
    normalized = str(value).strip().lower()
    if normalized not in _ALLOWED_DIAGNOSTIC_SEVERITIES:
        raise NeuralqxValueError(
            f"Unsupported diagnostics severity {value!r}. "
            f"Allowed: {sorted(_ALLOWED_DIAGNOSTIC_SEVERITIES)!r}."
        )
    return normalized  # type: ignore[return-value]


def diagnostic_severity_rank(severity: DiagnosticSeverity) -> int:
    """Returns the integer rank of one diagnostic-severity value.

    Args:
        severity: Diagnostic severity value.

    Returns:
        Integer severity rank (higher means more severe).
    """
    return _DIAGNOSTIC_SEVERITY_RANK[severity]


def diagnostic_severity_at_least(
    severity: DiagnosticSeverity,
    minimum: DiagnosticSeverity,
) -> bool:
    """Checks whether *severity* is at least *minimum*.

    Args:
        severity: Candidate diagnostic severity.
        minimum: Threshold diagnostic severity.

    Returns:
        ``True`` when *severity* >= *minimum*, otherwise ``False``.
    """
    return diagnostic_severity_rank(severity) >= diagnostic_severity_rank(minimum)


def count_diagnostics_by_severity(
    diagnostics: tuple[DSLDiagnostic, ...],
) -> dict[str, int]:
    """Counts diagnostics grouped by severity.

    Args:
        diagnostics: Ordered diagnostics tuple.

    Returns:
        Mapping with ``info``/``warning``/``error`` counters.
    """
    counts = {"info": 0, "warning": 0, "error": 0}
    for diagnostic in diagnostics:
        counts[diagnostic.severity] += 1
    return counts


@dataclasses.dataclass(frozen=True, repr=False)
class DSLDiagnostic:
    """One DSL lint/diagnostic finding produced by compiler diagnostics.

    Attributes:
        code: Stable machine-readable diagnostic code.
        severity: Finding severity level.
        message: Readable summary message.
        operator_name: Source operator name.
        term_name: Optional term name associated with this finding.
        suggestion: Optional remediation hint.
        context: Optional frozen structured context payload.
    """

    code: str
    severity: DiagnosticSeverity
    message: str
    operator_name: str
    term_name: str | None = None
    suggestion: str | None = None
    context: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.code).strip():
            raise NeuralqxValueError("DSLDiagnostic.code must be a non-empty string.")
        if not str(self.message).strip():
            raise NeuralqxValueError(
                "DSLDiagnostic.message must be a non-empty string."
            )
        if not str(self.operator_name).strip():
            raise NeuralqxValueError(
                "DSLDiagnostic.operator_name must be a non-empty string."
            )
        normalized = normalize_diagnostic_severity(self.severity)
        object.__setattr__(self, "severity", normalized)
        if self.term_name is not None and not str(self.term_name).strip():
            raise NeuralqxValueError(
                "DSLDiagnostic.term_name must be non-empty when provided."
            )
        if self.suggestion is not None and not str(self.suggestion).strip():
            raise NeuralqxValueError(
                "DSLDiagnostic.suggestion must be non-empty when provided."
            )

    @classmethod
    def create(
        cls,
        *,
        code: str,
        severity: DiagnosticSeverity,
        message: str,
        operator_name: str,
        term_name: str | None = None,
        suggestion: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DSLDiagnostic:
        """Builds one diagnostic from user-friendly mapping values.

        Args:
            code: Stable machine-readable diagnostic code.
            severity: Finding severity level.
            message: Readable summary message.
            operator_name: Source operator name.
            term_name: Optional term name associated with this finding.
            suggestion: Optional remediation hint.
            context: Optional structured context payload.

        Returns:
            Frozen diagnostic object.
        """
        payload = tuple(sorted((context or {}).items()))
        return cls(
            code=str(code),
            severity=severity,
            message=str(message),
            operator_name=str(operator_name),
            term_name=None if term_name is None else str(term_name),
            suggestion=None if suggestion is None else str(suggestion),
            context=payload,
        )

    def context_map(self) -> dict[str, Any]:
        """Returns structured diagnostic context as a mutable dictionary.

        Returns:
            Mutable context mapping.
        """
        return dict(self.context)

    def as_dict(self) -> dict[str, Any]:
        """Serializes one diagnostic to a plain dictionary.

        Returns:
            Plain dictionary representation of this diagnostic.
        """
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "operator_name": self.operator_name,
            "term_name": self.term_name,
            "suggestion": self.suggestion,
            "context": self.context,
        }

    def format_line(self) -> str:
        """Formats one diagnostic as a concise single-line string.

        Returns:
            One descriptive single-line summary.
        """
        location = self.operator_name
        if self.term_name is not None:
            location = f"{location}.{self.term_name}"
        return f"[{self.code}] {self.severity.upper()} {location}: {self.message}"

    def __repr__(self) -> str:
        return (
            f"DSLDiagnostic("
            f"code={self.code!r}, "
            f"severity={self.severity!r}, "
            f"operator_name={self.operator_name!r}, "
            f"term_name={self.term_name!r})"
        )


__all__ = [
    "DSLDiagnostic",
    "DiagnosticSeverity",
    "count_diagnostics_by_severity",
    "diagnostic_severity_at_least",
    "diagnostic_severity_rank",
    "normalize_diagnostic_severity",
]
