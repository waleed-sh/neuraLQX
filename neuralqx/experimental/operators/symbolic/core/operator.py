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


"""Unified symbolic operator type produced by the Operator DSL builder."""

from __future__ import annotations

from typing import Any

from netket.hilbert import DiscreteHilbert

from neuralqx.experimental.operators.symbolic.core.base import AbstractSymbolicOperator
from neuralqx.experimental.operators.symbolic.ir.program import SymbolicOperatorIR


def _resolve_dtype(a: str, b: str) -> str:
    """Promotes two dtype strings to their common type."""
    order = ("complex128", "complex64", "float64", "float32")
    for candidate in order:
        if candidate in (a, b):
            return candidate
    return a


def _scalar_str(scalar: Any) -> str:
    """Compact string for a numeric scalar used in operator name generation."""
    if isinstance(scalar, complex):
        return repr(scalar)
    if isinstance(scalar, float) and scalar == int(scalar) and abs(scalar) < 1e15:
        return str(int(scalar))
    return repr(scalar)


def _promote_dtype_for_scalar(dtype: str, scalar: Any) -> str:
    """Returns the dtype required to hold the product of dtype * scalar."""
    if isinstance(scalar, complex):
        if dtype in ("float32",):
            return "complex64"
        if dtype in ("float64",):
            return "complex128"
    return dtype


def _merge_metadata(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and out[k] != v:
            raise ValueError(
                f"Cannot merge symbolic operator metadata key {k!r} with different values."
            )
        out[k] = v
    return out


class SymbolicOperator(AbstractSymbolicOperator):
    """
    A symbolic operator built via the :class:`~neuralqx.experimental.operators.symbolic.dsl.op.DOperator` DSL.

    ``SymbolicOperator`` is the canonical result of ``DOperator(...).build()``.
    It holds an ordered list of typed IR terms and provides a ``.compile()``
    method to lower them to an executable
    :class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator`.

    Instances are **not** directly executable: calling ``get_conn_padded``
    before compilation raises
    :class:`~neuralqx.utils.errors.SymbolicOperatorExecutionError`.

    Attributes:
        name: User-facing operator name.
        hilbert: The NetKet Hilbert space.
        dtype: Matrix-element dtype.
        is_hermitian: Whether this operator is declared Hermitian.

    Example::

        op = (
            DOperator(hi, "hopping")
            .for_each_pair("i", "j")
            .where(site("i") > 0)
            .emit(shift("i", -1).shift("j", +1), amplitude=1.0)
            .build()
        )
        compiled = op.compile()
        xp, mels = compiled.get_conn_padded(x_batch)
    """

    __slots__ = ("_ir_terms",)

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        name: str,
        ir_terms: tuple,  # tuple[SymbolicIRTerm, ...]
        *,
        dtype_str: str = "complex64",
        is_hermitian: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            hilbert,
            name=name,
            dtype_str=dtype_str,
            is_hermitian=is_hermitian,
            metadata=metadata,
        )
        self._ir_terms: tuple = tuple(ir_terms)

    @property
    def name(self) -> str:
        """User-facing operator name."""
        return self._name_val

    @property
    def term_count(self) -> int:
        """Number of IR terms in this operator."""
        return len(self._ir_terms)

    @property
    def free_symbols(self) -> frozenset:
        """
        Returns the set of free (non-iterator-bound) symbol names across all terms.

        Free symbols are named parameters such as ``symbol("kappa")`` that must
        be resolved externally before or during compilation.
        """
        result: set = set()
        for t in self._ir_terms:
            result |= t.free_symbols
        return frozenset(result)

    def to_ir(self) -> SymbolicOperatorIR:
        """Builds the symbolic operator IR for compiler consumption."""
        return SymbolicOperatorIR.from_terms(
            operator_name=self._name_val,
            hilbert_size=int(self.hilbert.size),
            dtype_str=self._dtype_val,
            is_hermitian=self._is_hermitian_val,
            terms=self._ir_terms,
            metadata=self._metadata_dict if self._metadata_dict else None,
        )

    def estimate_max_conn_size(self) -> int:
        """Returns the aggregate static fanout bound across all terms."""
        from neuralqx.experimental.operators.symbolic.ir.term import KBodyIteratorSpec

        total = 0
        for t in self._ir_terms:
            if t.fanout_hint is not None:
                total += int(t.fanout_hint)
            else:
                if not isinstance(t.iterator, KBodyIteratorSpec):
                    raise TypeError(
                        f"Unsupported iterator type {type(t.iterator).__name__!r}; "
                        "expected KBodyIteratorSpec."
                    )
                E = len(t.effective_emissions)
                M = len(t.iterator.index_sets)
                total += M * E
        return max(1, total)

    def compile(
        self,
        *,
        backend: str = "jax",
        cache: bool = True,
        compiler: Any = None,
    ) -> Any:  # -> CompiledOperator
        """Lowers this symbolic operator to an executable :class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator`.

        Args:
            backend: Backend target (currently only ``"jax"`` is supported).
            cache: Whether to cache the compiled artifact in the process-level store.
            compiler: Optional :class:`~neuralqx.experimental.operators.symbolic.compiler.SymbolicCompiler`
                instance.  When ``None`` the module-level shared compiler is used.

        Returns:
            Executable :class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator`.
        """
        from neuralqx.experimental.operators.symbolic.compiler.compiler import (
            SymbolicCompiler,
        )
        from neuralqx.experimental.operators.symbolic.compiler.core.options import (
            SymbolicCompilerOptions,
        )

        c = compiler or SymbolicCompiler(
            options=SymbolicCompilerOptions(
                backend_preference=backend,
                cache_enabled=cache,
            )
        )
        return c.compile_operator(self)

    def _apply_scalar(self, scalar: "int | float | complex") -> "SymbolicOperator":
        from neuralqx.experimental.operators.symbolic.ir.expressions import (
            AmplitudeExpr,
        )
        from neuralqx.experimental.operators.symbolic.ir.term import _scale_ir_term

        scale_expr = AmplitudeExpr.constant(scalar)
        new_terms = tuple(_scale_ir_term(t, scale_expr) for t in self._ir_terms)
        is_hermitian = self._is_hermitian_val and not isinstance(scalar, complex)
        new_dtype = _promote_dtype_for_scalar(self._dtype_val, scalar)
        return SymbolicOperator(
            self.hilbert,
            f"({_scalar_str(scalar)} * {self.name})",
            new_terms,
            dtype_str=new_dtype,
            is_hermitian=is_hermitian,
            metadata=self._metadata_dict or None,
        )

    def __add__(self, other: Any) -> "SymbolicOperator":
        """
        Returns a new ``SymbolicOperator`` whose IR is the union of both.

        Args:
            other: Another :class:`SymbolicOperator` on the same Hilbert space.

        Returns:
            Combined :class:`SymbolicOperator` (terms are concatenated, not
            deduplicated).
        """
        if isinstance(other, SymbolicOperator) and other.hilbert == self.hilbert:
            combined = self._ir_terms + other._ir_terms
            merged_dtype = _resolve_dtype(self._dtype_val, other._dtype_val)
            merged_hermitian = self._is_hermitian_val and other._is_hermitian_val
            merged_metadata = _merge_metadata(self._metadata_dict, other._metadata_dict)
            return SymbolicOperator(
                self.hilbert,
                f"({self.name} + {other.name})",
                combined,
                dtype_str=merged_dtype,
                is_hermitian=merged_hermitian,
                metadata=merged_metadata or None,
            )
        # Fall through to ComputationalJaxOperator.__add__ (SumJax)
        return super().__add__(other)

    def __radd__(self, other: Any) -> "SymbolicOperator":
        if isinstance(other, SymbolicOperator):
            return other.__add__(self)
        return super().__radd__(other)

    def __repr__(self) -> str:
        return (
            f"SymbolicOperator("
            f"name={self.name!r}, "
            f"terms={self.term_count}, "
            f"dtype={self._dtype_val!r}, "
            f"hilbert={self.hilbert})"
        )


__all__ = ["SymbolicOperator"]
