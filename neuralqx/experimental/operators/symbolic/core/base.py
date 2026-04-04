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


"""Abstract base class for all symbolic operator implementations."""

from __future__ import annotations

import abc
from typing import Any

from netket.hilbert import DiscreteHilbert

from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator
from neuralqx.experimental.operators.symbolic.ir.program import SymbolicOperatorIR


class AbstractSymbolicOperator(ComputationalJaxOperator):
    """
    Abstract base class for all symbolic (DSL-defined) operators.

    Symbolic operators extend :class:`ComputationalJaxOperator` and declare
    their action through a typed IR rather than a hand-written JAX kernel.
    They **cannot** execute until the compiler has lowered them to a concrete
    JAX kernel via :meth:`neuralqx.experimental.operators.symbolic.compiler.SymbolicCompiler.compile`.

    Attempting to call :meth:`_get_conn_padded` before compilation raises
    :class:`~neuralqx.utils.errors.SymbolicOperatorExecutionError`.

    Args:
        hilbert: Discrete Hilbert space this operator is defined on.
        name: User-facing operator name.
        dtype_str: String label for the matrix-element dtype.
        is_hermitian: Whether this operator is declared Hermitian.
        metadata: Optional extra metadata dictionary.
    """

    __slots__ = ("_dtype_val", "_is_hermitian_val", "_metadata_dict", "_name_val")

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        *,
        name: str,
        dtype_str: str = "complex64",
        is_hermitian: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(hilbert)
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Operator name must be a non-empty string.")
        self._name_val: str = normalized
        self._dtype_val: str = str(dtype_str)
        self._is_hermitian_val: bool = bool(is_hermitian)
        self._metadata_dict: dict[str, Any] = dict(metadata) if metadata else {}

    @property
    def operator_name(self) -> str:
        """Returns user-facing operator name."""
        return self._name_val

    @property
    def dtype(self):
        """Returns matrix-element dtype."""
        import numpy as np

        return np.dtype(self._dtype_val)

    @property
    def is_hermitian(self) -> bool:
        """Returns whether this operator is declared Hermitian."""
        return self._is_hermitian_val

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns operator metadata dictionary."""
        return self._metadata_dict

    @abc.abstractmethod
    def to_ir(self) -> SymbolicOperatorIR:
        """Builds the symbolic action IR for this operator."""

    def _get_conn_padded(self, x):
        """Raises until this operator has been compiled."""
        from neuralqx.utils.errors import SymbolicOperatorExecutionError

        raise SymbolicOperatorExecutionError(
            f"Symbolic operator {self._name_val!r} cannot execute before "
            "compilation. Lower it through SymbolicCompiler.compile() first."
        )

    def __add__(self, other):
        if isinstance(other, AbstractSymbolicOperator):
            from neuralqx.experimental.operators.symbolic.core.sum import (
                SymbolicOperatorSum,
            )

            return SymbolicOperatorSum(
                hilbert=self.hilbert,
                terms=(self, other),
            )
        return super().__add__(other)

    def _apply_scalar(
        self, scalar: "int | float | complex"
    ) -> "AbstractSymbolicOperator":
        """
        Returns a new operator whose matrix elements are all multiplied by *scalar*.

        Subclasses override this to perform the actual term-level amplitude scaling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _apply_scalar."
        )

    def __mul__(self, scalar: Any) -> "AbstractSymbolicOperator":
        """
        Scales all matrix elements by *scalar*.

        Args:
            scalar: Real or complex numeric factor.

        Returns:
            New operator of the same concrete type with scaled amplitudes.
        """
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        return self._apply_scalar(scalar)

    def __rmul__(self, scalar: Any) -> "AbstractSymbolicOperator":
        return self.__mul__(scalar)

    def __neg__(self) -> "AbstractSymbolicOperator":
        return self.__mul__(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self._name_val!r}, "
            f"dtype={self._dtype_val!r}, "
            f"is_hermitian={self._is_hermitian_val}, "
            f"hilbert={self.hilbert!r})"
        )


__all__ = ["AbstractSymbolicOperator"]
