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


"""Abstract base class for symbolic compiler passes."""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pass_report import (
    SymbolicPassReport,
)


class AbstractSymbolicPass(abc.ABC):
    """
    Abstract base for all symbolic compiler passes.

    Each concrete pass implements :meth:`run` which mutates the compilation
    context and optionally returns a metadata dictionary. The public
    :meth:`execute` wrapper times the pass, wraps any raised exception with
    context, records a :class:`~neuralqx.experimental.operators.symbolic.compiler.core.pass_report.SymbolicPassReport`
    on the context, and re-raises failures.

    Subclasses must define :attr:`name`.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique pass name used in reports and diagnostics."""

    @abc.abstractmethod
    def run(
        self,
        context: SymbolicCompilationContext,
    ) -> Mapping[str, Any] | None:
        """
        Executes this pass over the mutable compilation context.

        Args:
            context: Mutable compilation context.

        Returns:
            Optional metadata dictionary to include in the pass report.

        Raises:
            Exception: Any exception propagates after being wrapped with
                pass-name context by :meth:`execute`.
        """

    def execute(self, context: SymbolicCompilationContext) -> SymbolicPassReport:
        """
        Runs this pass with timing and report recording.

        Invokes :meth:`run`, records a :class:`SymbolicPassReport` on
        ``context``, and returns it.  Any exception raised by :meth:`run` is
        re-raised as-is after the timing guard.

        Args:
            context: Mutable compilation context.

        Returns:
            The recorded pass report.

        Raises:
            Exception: Propagated from :meth:`run`.
        """
        t0 = time.perf_counter()
        try:
            metadata = self.run(context)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Record a failed report so the artifact still carries pass history
            report = SymbolicPassReport.create(
                pass_name=self.name,
                duration_ms=elapsed_ms,
                metadata={"error": str(exc), "status": "failed"},
            )
            context.add_pass_report(report)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        report = SymbolicPassReport.create(
            pass_name=self.name,
            duration_ms=elapsed_ms,
            metadata=dict(metadata) if metadata else {},
        )
        context.add_pass_report(report)
        return report

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(name={self.name!r})"


__all__ = ["AbstractSymbolicPass"]
