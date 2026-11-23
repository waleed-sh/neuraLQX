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


"""Default built-in predicate clause implementations and registration."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import Any

from neuralqx.operator.symbolic.dsl.predicates.abstractions import (
    AbstractPredicateClause,
)
from neuralqx.operator.symbolic.dsl.predicates.registry import register_predicate_clause

if TYPE_CHECKING:
    from neuralqx.operator.symbolic.dsl.context import ExpressionContext


class WherePredicateClause(AbstractPredicateClause):
    """Built-in predicate clause implementing ``builder.where(...)``."""

    clause_name = "where"
    """Fluent method name used to invoke this built-in predicate clause."""

    def build_predicate(self, ctx: ExpressionContext, predicate: Any) -> Any:
        """
        Returns the provided predicate unchanged.

        Args:
            ctx: Expression context (unused for pass-through behavior).
            predicate: Predicate value to attach to the current term.

        Returns:
            Any: Input predicate, to be coerced by the clause abstraction layer.
        """
        del ctx
        return predicate


_DEFAULT_PREDICATE_CLAUSES_REGISTERED = False
"""Whether built-in predicate clauses were already installed for this process."""

_DEFAULT_PREDICATE_CLAUSES_LOCK = threading.RLock()
"""Lock protecting one-time registration of built-in predicate clauses."""


def ensure_default_predicate_clause_registrations() -> None:
    """
    Registers all built-in predicate clauses exactly once.

    Returns:
        None
    """
    global _DEFAULT_PREDICATE_CLAUSES_REGISTERED

    with _DEFAULT_PREDICATE_CLAUSES_LOCK:
        if _DEFAULT_PREDICATE_CLAUSES_REGISTERED:
            return

        register_predicate_clause(WherePredicateClause, replace=True)
        _DEFAULT_PREDICATE_CLAUSES_REGISTERED = True


__all__ = [
    "WherePredicateClause",
    "ensure_default_predicate_clause_registrations",
]
