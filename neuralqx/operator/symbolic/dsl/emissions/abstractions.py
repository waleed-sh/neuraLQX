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


"""Emission clause abstraction classes."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Any

from neuralqx.operator.symbolic.dsl._clause_naming import normalize_clause_method_name
from neuralqx.operator.symbolic.dsl.emissions.types import coerce_emission_clause_spec

if TYPE_CHECKING:
    from neuralqx.operator.symbolic.dsl.context import ExpressionContext
    from neuralqx.operator.symbolic.dsl.operator import SymbolicDiscreteJaxOperator


class AbstractEmissionClause(abc.ABC):
    """
    Abstract base class for custom fluent emission clauses.

    Subclasses implement :meth:`build_emission` and can be registered into
    the emission-clause registry to become fluent builder methods.
    """

    clause_name: str | None = None
    """Optional method-name override for fluent method installation."""

    def __init__(self, builder: SymbolicDiscreteJaxOperator) -> None:
        """
        Initializes the emission clause bound to one builder instance.

        Args:
            builder: Builder instance this clause instance operates on.
        """
        self._builder = builder

    @property
    def builder(self) -> SymbolicDiscreteJaxOperator:
        """
        Returns the builder this clause instance is bound to.

        Returns:
            SymbolicDiscreteJaxOperator: Bound builder instance.
        """
        return self._builder

    @classmethod
    def method_name(cls) -> str:
        """
        Returns the fluent method name for the clause class.

        Returns:
            str: Validated fluent method name.
        """
        raw = cls.clause_name if cls.clause_name is not None else cls.__name__
        return normalize_clause_method_name(raw)

    @abc.abstractmethod
    def build_emission(self, ctx: ExpressionContext, *args: Any, **kwargs: Any) -> Any:
        """
        Builds one emission clause specification from user arguments.

        Args:
            ctx: Expression-context helper for constructing expressions.
            *args: Clause-specific positional arguments.
            **kwargs: Clause-specific keyword arguments.

        Returns:
            Any: Value coercible by :func:`coerce_emission_clause_spec`.
        """

    def __call__(self, *args: Any, **kwargs: Any) -> SymbolicDiscreteJaxOperator:
        """
        Executes the clause against the builder and appends emission behavior.

        Args:
            *args: Clause-specific positional arguments.
            **kwargs: Clause-specific keyword arguments.

        Returns:
            SymbolicDiscreteJaxOperator: Builder for fluent chaining.
        """
        from neuralqx.operator.symbolic.dsl.context import ExpressionContext

        spec = coerce_emission_clause_spec(
            self.build_emission(ExpressionContext(), *args, **kwargs)
        )
        return self.builder.append_emission_clause(spec, method_name=self.method_name())


__all__ = ["AbstractEmissionClause"]
