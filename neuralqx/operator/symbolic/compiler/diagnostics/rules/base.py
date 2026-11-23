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


"""Base abstractions for diagnostics rules."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

from neuralqx.operator.symbolic.compiler.core.options import SymbolicCompilerOptions
from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.ir.program import SymbolicOperatorIR


@dataclasses.dataclass(frozen=True, repr=False)
class DiagnosticRuleContext:
    """Immutable context passed to diagnostics rules.

    Attributes:
        operator: Source symbolic operator.
        ir: Symbolic operator IR.
        options: Effective compiler options.
    """

    operator: Any
    ir: SymbolicOperatorIR
    options: SymbolicCompilerOptions


class AbstractDiagnosticRule(abc.ABC):
    """Abstract base class for one diagnostics rule."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique diagnostics rule identifier."""

    @abc.abstractmethod
    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        """Runs this rule and returns diagnostics.

        Args:
            context: Diagnostics rule execution context.

        Returns:
            Tuple of diagnostics emitted by this rule.
        """


__all__ = ["AbstractDiagnosticRule", "DiagnosticRuleContext"]
