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


"""Registry operations for iterator clauses."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import overload

from neuralqx.operator.symbolic.dsl._clause_naming import normalize_clause_method_name
from neuralqx.operator.symbolic.warnings import RegistrationWarning
from neuralqx.utils.errors import NeuralqxTypeError

if TYPE_CHECKING:
    from neuralqx.operator.symbolic.dsl.iterators.abstractions import (
        AbstractIteratorClause,
    )


_REGISTRY_LOCK = threading.RLock()
"""Re-entrant lock guarding iterator-clause registry mutations."""

_ITERATOR_CLAUSES: dict[str, type[AbstractIteratorClause]] = {}
"""In-process map from fluent method name to iterator clause class."""


def resolve_iterator_clause(name: str) -> type[AbstractIteratorClause] | None:
    """
    Resolves one registered iterator clause by fluent method name.

    Args:
        name: Fluent method name.

    Returns:
        type[AbstractIteratorClause] | None: Registered clause class, or ``None``.
    """
    return _ITERATOR_CLAUSES.get(str(name))


def available_iterator_clause_names() -> tuple[str, ...]:
    """
    Lists iterator clause names currently registered.

    Returns:
        tuple[str, ...]: Sorted fluent method names.
    """
    return tuple(sorted(_ITERATOR_CLAUSES))


@overload
def register_iterator_clause(
    clause_cls: type[AbstractIteratorClause],
    *,
    name: str | None = None,
    replace: bool = False,
) -> type[AbstractIteratorClause]: ...


@overload
def register_iterator_clause(
    clause_cls: None = None,
    *,
    name: str | None = None,
    replace: bool = False,
) -> Callable[[type[AbstractIteratorClause]], type[AbstractIteratorClause]]: ...


def register_iterator_clause(
    clause_cls: type[AbstractIteratorClause] | None = None,
    *,
    name: str | None = None,
    replace: bool = False,
) -> (
    type[AbstractIteratorClause]
    | Callable[[type[AbstractIteratorClause]], type[AbstractIteratorClause]]
):
    """
    Registers one iterator clause class.

    Args:
        clause_cls: Clause class to register. When ``None``, returns a decorator.
        name: Optional explicit fluent method name override.
        replace: Whether to replace an existing clause under the same name.
            Otherwise, a duplicate emits :class:`neuralqx.operator.symbolic.RegistrationWarning`
            and keeps the existing implementation.

    Returns:
        type[AbstractIteratorClause] | Callable[..., type[AbstractIteratorClause]]:
            Provided class directly, or a class decorator. A duplicate keeps
            the existing registry entry without changing the provided class.

    Raises:
        TypeError: If the provided class does not inherit from
            :class:`AbstractIteratorClause`.
        ValueError: If the method name is invalid.
    """

    def _decorator(cls: type[AbstractIteratorClause]) -> type[AbstractIteratorClause]:
        from neuralqx.operator.symbolic.dsl.iterators.abstractions import (
            AbstractIteratorClause,
        )

        if not issubclass(cls, AbstractIteratorClause):
            raise NeuralqxTypeError(
                "Iterator clause must inherit from AbstractIteratorClause."
            )
        resolved_name = normalize_clause_method_name(name or cls.method_name())
        with _REGISTRY_LOCK:
            if resolved_name in _ITERATOR_CLAUSES and not replace:
                warnings.warn(
                    f"Iterator clause {resolved_name!r} is already registered. "
                    "Keeping the existing implementation. "
                    "Use replace=True to use the new implementation.",
                    RegistrationWarning,
                    stacklevel=3,
                )
                return cls
            cls.clause_name = resolved_name
            _ITERATOR_CLAUSES[resolved_name] = cls
        return cls

    if clause_cls is None:
        return _decorator
    return _decorator(clause_cls)


__all__ = [
    "available_iterator_clause_names",
    "register_iterator_clause",
    "resolve_iterator_clause",
]
