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


"""Mutable symbolic compilation context shared across passes and lowerers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.options import (
    SymbolicCompilerOptions,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pass_report import (
    SymbolicPassReport,
)
from neuralqx.experimental.operators.symbolic.ir.program import SymbolicOperatorIR


class SymbolicCompilationContext:
    """
    Holds per-compilation mutable state across pipeline stages.

    The context is created by the compiler, mutated in-place by passes and
    lowerers, and finally read when packaging the compiled artifact.

    Args:
        operator: Source symbolic operator.
        ir: Symbolic operator IR extracted from the operator.
        options: Effective compiler options.
        metadata: Optional extra context metadata.
    """

    __slots__ = (
        "_analyses",
        "_ir",
        "_metadata",
        "_operator",
        "_options",
        "_pass_reports",
        "_selected_backend",
        "_selected_lowerer",
    )

    def __init__(
        self,
        *,
        operator: Any,
        ir: SymbolicOperatorIR,
        options: SymbolicCompilerOptions,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._operator = operator
        self._ir = ir
        self._options = options
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._analyses: dict[str, Any] = {}
        self._pass_reports: list[SymbolicPassReport] = []
        self._selected_backend: str | None = None
        self._selected_lowerer: str | None = None

    @property
    def operator(self) -> Any:
        """Returns the source symbolic operator."""
        return self._operator

    @property
    def ir(self) -> SymbolicOperatorIR:
        """Returns the symbolic operator IR."""
        return self._ir

    @property
    def options(self) -> SymbolicCompilerOptions:
        """Returns effective compiler options."""
        return self._options

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Returns context metadata mapping (read-only view)."""
        return self._metadata

    @property
    def selected_backend(self) -> str | None:
        """Returns the selected backend name, when assigned."""
        return self._selected_backend

    @property
    def selected_lowerer(self) -> str | None:
        """Returns the selected lowerer name, when assigned."""
        return self._selected_lowerer

    @property
    def pass_reports(self) -> tuple[SymbolicPassReport, ...]:
        """Returns pass reports in execution order."""
        return tuple(self._pass_reports)

    @property
    def analyses(self) -> Mapping[str, Any]:
        """Returns read-only analysis mapping."""
        return self._analyses

    def set_metadata(self, key: str, value: Any) -> None:
        """Stores one metadata value."""
        self._metadata[str(key)] = value

    def set_analysis(self, key: str, payload: Any) -> None:
        """Stores one analysis payload."""
        self._analyses[str(key)] = payload

    def analysis(self, key: str, *, default: Any = None) -> Any:
        """Returns one analysis payload or ``default``."""
        return self._analyses.get(str(key), default)

    def require_analysis(self, key: str) -> Any:
        """
        Returns a required analysis payload.

        Raises:
            ValueError: If the analysis key is absent.
        """
        normalized = str(key)
        if normalized not in self._analyses:
            raise ValueError(f"Required compiler analysis {normalized!r} is missing.")
        return self._analyses[normalized]

    def add_pass_report(self, report: SymbolicPassReport) -> None:
        """Appends one pass report entry."""
        self._pass_reports.append(report)

    def set_selected_backend(self, backend: str) -> None:
        """Stores selected backend name."""
        self._selected_backend = str(backend)

    def set_selected_lowerer(self, name: str) -> None:
        """Stores selected lowerer name."""
        self._selected_lowerer = str(name)

    def summary(self) -> dict[str, Any]:
        """Returns a summary dict for diagnostics and artifact packaging."""
        return {
            "operator_name": self._ir.operator_name,
            "mode": self._ir.mode,
            "hilbert_size": self._ir.hilbert_size,
            "selected_backend": self._selected_backend,
            "selected_lowerer": self._selected_lowerer,
            "analysis_keys": tuple(sorted(self._analyses)),
            "metadata_keys": tuple(sorted(self._metadata)),
            "pass_count": len(self._pass_reports),
        }


__all__ = ["SymbolicCompilationContext"]
