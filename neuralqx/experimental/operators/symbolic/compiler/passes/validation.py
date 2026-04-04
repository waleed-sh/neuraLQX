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


"""Symbolic IR validation compiler pass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
    AbstractSymbolicPass,
)
from neuralqx.experimental.operators.symbolic.ir.validate import validate_symbolic_ir


class SymbolicValidationPass(AbstractSymbolicPass):
    """
    Validates symbolic operator IR before lowering.

    This pre-cache pass calls :func:`~neuralqx.experimental.operators.symbolic.ir.validate.validate_symbolic_ir`
    on the context IR. When ``strict_validation`` is enabled on the compiler
    options, any validation error is raised immediately. In non-strict mode the
    error is recorded in the pass metadata but execution continues, allowing the
    remaining pipeline to proceed.

    Analysis key written: ``"validation_summary"``
    """

    @property
    def name(self) -> str:
        return "symbolic_validation"

    def run(
        self,
        context: SymbolicCompilationContext,
    ) -> Mapping[str, Any] | None:
        strict = context.options.strict_validation
        try:
            summary = validate_symbolic_ir(context.ir)
        except (ValueError, TypeError) as exc:
            if strict:
                raise
            summary = {"error": str(exc), "valid": False}
            context.set_analysis("validation_summary", summary)
            return {"valid": False, "error": str(exc)}

        summary["valid"] = True
        context.set_analysis("validation_summary", summary)
        return {
            "valid": True,
            "term_count": summary.get("term_count", 0),
        }


__all__ = ["SymbolicValidationPass"]
