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


"""Compatibility facade for iterator/predicate/emission DSL clause APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import overload

from neuralqx.operator.symbolic.dsl.emissions import AbstractEmissionClause
from neuralqx.operator.symbolic.dsl.emissions import apply_emission_clause
from neuralqx.operator.symbolic.dsl.emissions import available_emission_clause_names
from neuralqx.operator.symbolic.dsl.emissions import (
    ensure_default_emission_clause_registrations,
)
from neuralqx.operator.symbolic.dsl.emissions import register_emission_clause
from neuralqx.operator.symbolic.dsl.emissions import resolve_emission_clause
from neuralqx.operator.symbolic.dsl.iterators import AbstractIteratorClause
from neuralqx.operator.symbolic.dsl.iterators import apply_iterator_clause
from neuralqx.operator.symbolic.dsl.iterators import available_iterator_clause_names
from neuralqx.operator.symbolic.dsl.iterators import coerce_iterator_spec
from neuralqx.operator.symbolic.dsl.iterators import (
    ensure_default_iterator_clause_registrations,
)
from neuralqx.operator.symbolic.dsl.iterators import register_iterator_clause
from neuralqx.operator.symbolic.dsl.iterators import resolve_iterator_clause
from neuralqx.operator.symbolic.dsl.predicates import AbstractPredicateClause
from neuralqx.operator.symbolic.dsl.predicates import apply_predicate_clause
from neuralqx.operator.symbolic.dsl.predicates import available_predicate_clause_names
from neuralqx.operator.symbolic.dsl.predicates import (
    ensure_default_predicate_clause_registrations,
)
from neuralqx.operator.symbolic.dsl.predicates import register_predicate_clause
from neuralqx.operator.symbolic.dsl.predicates import resolve_predicate_clause
from neuralqx.utils.errors import NeuralqxTypeError


@overload
def register(
    clause_cls: type[Any],
    *,
    name: str | None = None,
    replace: bool = False,
) -> type[Any]: ...


@overload
def register(
    clause_cls: None = None,
    *,
    name: str | None = None,
    replace: bool = False,
) -> Callable[[type[Any]], type[Any]]: ...


def register(
    clause_cls: type[Any] | None = None,
    *,
    name: str | None = None,
    replace: bool = False,
) -> type[Any] | Callable[[type[Any]], type[Any]]:
    """
    Registers an iterator, predicate, or emission clause class.

    Args:
        clause_cls: Clause class to register. When ``None``, returns a decorator.
        name: Optional explicit fluent method name override.
        replace: Whether to replace an existing clause under the same name.
            Otherwise, duplicates warn and keep the existing implementation.

    Returns:
        type[Any] | Callable[[type[Any]], type[Any]]: Registered class or decorator.

    Raises:
        TypeError: If *clause_cls* is not a supported clause class.
    """

    def _decorator(cls: type[Any]) -> type[Any]:
        if issubclass(cls, AbstractIteratorClause):
            return register_iterator_clause(cls, name=name, replace=replace)
        if issubclass(cls, AbstractPredicateClause):
            return register_predicate_clause(cls, name=name, replace=replace)
        if issubclass(cls, AbstractEmissionClause):
            return register_emission_clause(cls, name=name, replace=replace)
        raise NeuralqxTypeError(
            "register(...) expects a subclass of AbstractIteratorClause "
            "or AbstractPredicateClause or AbstractEmissionClause."
        )

    if clause_cls is None:
        return _decorator
    return _decorator(clause_cls)


def ensure_default_clause_registrations() -> None:
    """
    Ensures all built-in iterator, predicate, and emission clauses are registered.

    Returns:
        None
    """
    ensure_default_iterator_clause_registrations()
    ensure_default_predicate_clause_registrations()
    ensure_default_emission_clause_registrations()


__all__ = [
    "AbstractEmissionClause",
    "AbstractIteratorClause",
    "AbstractPredicateClause",
    "apply_emission_clause",
    "apply_iterator_clause",
    "apply_predicate_clause",
    "available_emission_clause_names",
    "available_iterator_clause_names",
    "available_predicate_clause_names",
    "coerce_iterator_spec",
    "ensure_default_clause_registrations",
    "register",
    "register_emission_clause",
    "register_iterator_clause",
    "register_predicate_clause",
    "resolve_emission_clause",
    "resolve_iterator_clause",
    "resolve_predicate_clause",
]
