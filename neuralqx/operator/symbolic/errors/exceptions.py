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


"""Custom exception hierarchy for neuralqx.operator.symbolic."""

from __future__ import annotations

from textwrap import dedent

from neuralqx.utils.errors import neuralqxError


class SymbolicError(neuralqxError):
    """Base exception for symbolic operators, catchable as a neuraLQX error."""

    default_message: str = "A symbolic operator error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        hint: str | None = None,
        details: str | None = None,
    ) -> None:
        primary = message or self.default_message
        parts = [primary]
        if hint:
            parts.append(f"Hint: {hint}")
        if details:
            parts.append(f"Details: {details}")
        super().__init__("\n".join(parts))


class ConfigError(SymbolicError):
    """Base exception for symbolic configuration errors."""

    def __init__(self, message: str) -> None:
        super().__init__(dedent(message).strip())


class UnknownOptionError(ConfigError):
    """Raised when accessing or mutating an unknown symbolic option."""


class ConfigValidationError(ConfigError):
    """Raised when a symbolic configuration value fails validation."""


class SymbolicOperatorError(SymbolicError):
    """Base exception for symbolic operator construction and execution errors."""


class SymbolicOperatorExecutionError(SymbolicOperatorError):
    """Raised when a symbolic operator is executed before compilation."""

    default_message = "This symbolic operator cannot be executed before it is compiled."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message,
            hint="Call `.compile()` first, then run `get_conn_padded(...)` on the compiled operator.",
        )


class SymbolicCompilationError(SymbolicError):
    """Base exception for symbolic compilation failures."""


class SymbolicCompilerError(SymbolicCompilationError):
    """Raised when symbolic compilation fails at any pipeline stage."""

    default_message = "Symbolic compilation failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message,
            hint=(
                "Check the symbolic IR produced by `.to_ir()` and verify term predicates, "
                "update programs, and matrix elements."
            ),
        )


class SymbolicDiagnosticsError(SymbolicCompilationError):
    """Raised when diagnostics are configured to fail the compilation flow."""

    default_message = "Symbolic diagnostics reported blocking issues."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message,
            hint=(
                "Address reported diagnostics or relax diagnostics strictness via "
                "SymbolicCompilerOptions(diagnostics_min_severity=..., fail_on_warnings=...)."
            ),
        )


__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "SymbolicCompilationError",
    "SymbolicCompilerError",
    "SymbolicDiagnosticsError",
    "SymbolicError",
    "SymbolicOperatorError",
    "SymbolicOperatorExecutionError",
    "UnknownOptionError",
]
