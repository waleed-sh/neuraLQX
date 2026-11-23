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


"""Dispatch helpers for invoking registered emission clauses."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from neuralqx.operator.symbolic.dsl.emissions.registry import resolve_emission_clause
from neuralqx.utils.errors import NeuralqxAttributeError

if TYPE_CHECKING:
    from neuralqx.operator.symbolic.dsl.operator import SymbolicDiscreteJaxOperator


def apply_emission_clause(
    builder: SymbolicDiscreteJaxOperator,
    clause_name: str,
    *args: Any,
    **kwargs: Any,
) -> SymbolicDiscreteJaxOperator:
    """
    Resolves and applies one registered emission clause.

    Args:
        builder: Builder receiving the clause invocation.
        clause_name: Registered fluent method name.
        *args: Clause-specific positional arguments.
        **kwargs: Clause-specific keyword arguments.

    Returns:
        SymbolicDiscreteJaxOperator: Builder for fluent chaining.

    Raises:
        AttributeError: If no emission clause is registered for *clause_name*.
    """
    clause_cls = resolve_emission_clause(clause_name)
    if clause_cls is None:
        raise NeuralqxAttributeError(f"Unknown emission clause {clause_name!r}.")
    return clause_cls(builder)(*args, **kwargs)


__all__ = ["apply_emission_clause"]
