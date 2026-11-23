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


"""Default built-in emission clause implementations and registration."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import Any

from neuralqx.operator.symbolic.dsl.emissions.abstractions import AbstractEmissionClause
from neuralqx.operator.symbolic.dsl.emissions.registry import register_emission_clause
from neuralqx.operator.symbolic.dsl.emissions.types import EmissionClauseSpec

if TYPE_CHECKING:
    from neuralqx.operator.symbolic.dsl.context import ExpressionContext


class EmitIfEmissionClause(AbstractEmissionClause):
    """Built-in conditional emission clause implementing ``builder.emit_if(...)``."""

    clause_name = "emit_if"

    def build_emission(
        self,
        ctx: ExpressionContext,
        predicate: Any,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> EmissionClauseSpec:
        del ctx
        return EmissionClauseSpec(
            mode="emit_if",
            predicate=predicate,
            update=update,
            matrix_element=matrix_element,
            amplitude=amplitude,
            tag=tag,
        )


class EmitElseIfEmissionClause(AbstractEmissionClause):
    """Built-in conditional emission clause implementing ``builder.emit_elseif(...)``."""

    clause_name = "emit_elseif"

    def build_emission(
        self,
        ctx: ExpressionContext,
        predicate: Any,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> EmissionClauseSpec:
        del ctx
        return EmissionClauseSpec(
            mode="emit_elseif",
            predicate=predicate,
            update=update,
            matrix_element=matrix_element,
            amplitude=amplitude,
            tag=tag,
        )


class EmitElseEmissionClause(AbstractEmissionClause):
    """Built-in conditional emission clause implementing ``builder.emit_else(...)``."""

    clause_name = "emit_else"

    def build_emission(
        self,
        ctx: ExpressionContext,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> EmissionClauseSpec:
        del ctx
        return EmissionClauseSpec(
            mode="emit_else",
            update=update,
            matrix_element=matrix_element,
            amplitude=amplitude,
            tag=tag,
        )


_DEFAULT_EMISSION_CLAUSES_REGISTERED = False
"""Whether built-in emission clauses were already installed for this process."""

_DEFAULT_EMISSION_CLAUSES_LOCK = threading.RLock()
"""Lock protecting one-time registration of built-in emission clauses."""


def ensure_default_emission_clause_registrations() -> None:
    """
    Registers all built-in emission clauses exactly once.

    Returns:
        None
    """
    global _DEFAULT_EMISSION_CLAUSES_REGISTERED

    with _DEFAULT_EMISSION_CLAUSES_LOCK:
        if _DEFAULT_EMISSION_CLAUSES_REGISTERED:
            return

        register_emission_clause(EmitIfEmissionClause, replace=True)
        register_emission_clause(EmitElseIfEmissionClause, replace=True)
        register_emission_clause(EmitElseEmissionClause, replace=True)
        _DEFAULT_EMISSION_CLAUSES_REGISTERED = True


__all__ = [
    "EmitElseEmissionClause",
    "EmitElseIfEmissionClause",
    "EmitIfEmissionClause",
    "ensure_default_emission_clause_registrations",
]
