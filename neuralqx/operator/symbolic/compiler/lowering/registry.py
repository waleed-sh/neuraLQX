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


"""Registry for symbolic compiler lowerers."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.lowering.base import AbstractSymbolicLowerer
from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.utils.errors import NeuralqxRuntimeError
from neuralqx.utils.errors import NeuralqxTypeError


class SymbolicLowererRegistry:
    """
    Registry that maps compilation contexts to concrete lowerers.

    Lowerers are registered in priority order (first registered = highest
    priority).  When :meth:`resolve` is called, it returns the first lowerer
    whose :meth:`~AbstractSymbolicLowerer.supports` predicate returns ``True``
    for the given context.

    Args:
        lowerers: Initial ordered lowerer sequence (highest priority first).
    """

    def __init__(
        self,
        lowerers: list[AbstractSymbolicLowerer] | None = None,
    ) -> None:
        self._lowerers: list[AbstractSymbolicLowerer] = list(lowerers or [])

    def register(self, lowerer: AbstractSymbolicLowerer) -> None:
        """Appends *lowerer* to the registry (lowest priority)."""
        if not isinstance(lowerer, AbstractSymbolicLowerer):
            raise NeuralqxTypeError(
                f"Expected AbstractSymbolicLowerer, got {type(lowerer)!r}."
            )
        self._lowerers.append(lowerer)
        debug_event(
            "registered lowerer",
            scope="lowering",
            tag="LOWERING",
            lowerer_name=lowerer.name,
            backend=lowerer.backend,
            priority="last",
        )

    def register_first(self, lowerer: AbstractSymbolicLowerer) -> None:
        """Prepends *lowerer* to the registry (highest priority)."""
        if not isinstance(lowerer, AbstractSymbolicLowerer):
            raise NeuralqxTypeError(
                f"Expected AbstractSymbolicLowerer, got {type(lowerer)!r}."
            )
        self._lowerers.insert(0, lowerer)
        debug_event(
            "registered lowerer",
            scope="lowering",
            tag="LOWERING",
            lowerer_name=lowerer.name,
            backend=lowerer.backend,
            priority="first",
        )

    def resolve(
        self,
        context: SymbolicCompilationContext,
    ) -> AbstractSymbolicLowerer:
        """
        Returns the first registered lowerer that supports *context*.

        Args:
            context: Active compilation context.

        Returns:
            Matching lowerer.

        Raises:
            RuntimeError: If no registered lowerer supports the context.
        """
        for lowerer in self._lowerers:
            if lowerer.supports(context):
                debug_event(
                    "resolved lowerer from registry",
                    scope="lowering",
                    tag="LOWERING",
                    operator_name=context.ir.operator_name,
                    lowerer_name=lowerer.name,
                    backend=lowerer.backend,
                )
                return lowerer
        backend = context.selected_backend or context.options.backend_preference
        raise NeuralqxRuntimeError(
            f"No registered symbolic lowerer supports backend={backend!r} "
            f"for operator {context.ir.operator_name!r}. "
            f"Registered lowerers: {[lowerer.name for lowerer in self._lowerers]!r}."
        )

    @property
    def lowerer_names(self) -> tuple[str, ...]:
        """Returns registered lowerer names in priority order."""
        return tuple(lowerer.name for lowerer in self._lowerers)

    def __len__(self) -> int:
        return len(self._lowerers)

    def __repr__(self) -> str:
        return f"SymbolicLowererRegistry(lowerers={self.lowerer_names!r})"


__all__ = ["SymbolicLowererRegistry"]
