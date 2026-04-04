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


"""Abstract base class for symbolic operator lowerers."""

from __future__ import annotations

import abc

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
    SymbolicCompiledArtifact,
)


class AbstractSymbolicLowerer(abc.ABC):
    """
    Abstract base for backend-specific symbolic operator lowerers.

    A lowerer converts a compiled symbolic IR (held in a
    :class:`~neuralqx.experimental.operators.symbolic.compiler.core.context.SymbolicCompilationContext`)
    into an executable :class:`~neuralqx.operators.types.computational_operator.jax.ComputationalJaxOperator`
    subclass packaged inside a :class:`~neuralqx.experimental.operators.symbolic.compiler.core.artifact.SymbolicCompiledArtifact`.

    The compiler picks the lowerer via :meth:`supports`, then calls
    :meth:`lower`.

    Subclasses must implement:
        - :attr:`name` - unique lowerer identifier (used in artifact tagging).
        - :attr:`backend` - target backend name (e.g. ``"jax"``).
        - :meth:`supports` - returns ``True`` when this lowerer can handle
          the given context.
        - :meth:`lower` - performs the actual compilation.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this lowerer."""

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """Backend name this lowerer targets."""

    @abc.abstractmethod
    def supports(self, context: SymbolicCompilationContext) -> bool:
        """
        Returns whether this lowerer can handle the given context.

        Args:
            context: Active compilation context.

        Returns:
            ``True`` if this lowerer should be used, ``False`` otherwise.
        """

    @abc.abstractmethod
    def lower(
        self,
        context: SymbolicCompilationContext,
    ) -> "SymbolicCompiledArtifact":
        """
        Lowers the symbolic IR to an executable compiled artifact.

        The method should:

        1. Read analyses from *context* (e.g. ``"total_fanout"``,
           ``"fusion_groups"``).
        2. Build a concrete
           :class:`~neuralqx.operators.types.computational_operator.jax.ComputationalJaxOperator`
           subclass with a working ``_get_conn_padded`` implementation.
        3. Store the selected lowerer name on the context via
           :meth:`~neuralqx.experimental.operators.symbolic.compiler.core.context.SymbolicCompilationContext.set_selected_lowerer`.
        4. Package and return a
           :class:`~neuralqx.experimental.operators.symbolic.compiler.core.artifact.SymbolicCompiledArtifact`.

        Args:
            context: Mutable compilation context; may be further annotated
                during lowering.

        Returns:
            Compiled artifact wrapping the executable operator.
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self.name!r}, backend={self.backend!r})"
        )


__all__ = ["AbstractSymbolicLowerer"]
