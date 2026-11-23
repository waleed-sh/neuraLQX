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


"""Unified symbolic operator type produced by the Operator DSL builder."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator.symbolic.core.base import AbstractSymbolicOperator
from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.operator.symbolic.ir.program import SymbolicOperatorIR
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError


def _resolve_dtype(a: str, b: str) -> str:
    """Promotes two dtype strings to their common type."""
    return np.result_type(np.dtype(a), np.dtype(b)).name


def _scalar_str(scalar: Any) -> str:
    """Compact string for a numeric scalar used in operator name generation."""
    if isinstance(scalar, complex):
        return repr(scalar)
    if isinstance(scalar, float) and scalar == int(scalar) and abs(scalar) < 1e15:
        return str(int(scalar))
    return repr(scalar)


def _promote_dtype_for_scalar(dtype: str, scalar: Any) -> str:
    """Returns the dtype required to hold the product of dtype * scalar."""
    base = np.dtype(dtype)
    scalar_dtype = np.asarray(scalar).dtype
    if np.issubdtype(scalar_dtype, np.complexfloating):
        return np.result_type(base, scalar_dtype).name
    return base.name


def _merge_metadata(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and out[k] != v:
            raise NeuralqxValueError(
                f"Cannot merge symbolic operator metadata key {k!r} with different values."
            )
        out[k] = v
    return out


class SymbolicOperator(AbstractSymbolicOperator):
    """
    A symbolic operator built via the :class:`~neuralqx.operator.symbolic.dsl.op.SymbolicDiscreteJaxOperator` DSL.

    ``SymbolicOperator`` is the canonical result of ``SymbolicDiscreteJaxOperator(...).build()``.
    It holds an ordered list of typed IR terms and provides a ``.compile()``
    method to lower them to an executable
    :class:`~neuralqx.operator.symbolic.core.compiled.CompiledOperator`.

    Instances are **not** directly executable: calling ``get_conn_padded``
    before compilation raises
    :class:`~neuralqx.operator.symbolic.errors.SymbolicOperatorExecutionError`.

    Attributes:
        name: User-facing operator name.
        hilbert: The native neuraLQX Hilbert space.
        dtype: Matrix-element dtype.
        is_hermitian: Whether this operator is declared Hermitian.

    Example::

        op = (
            SymbolicDiscreteJaxOperator(hi, "hopping")
            .for_each_pair("i", "j")
            .where(site("i") > 0)
            .emit(shift("i", -1).shift("j", +1), matrix_element=1.0)
            .build()
        )
        compiled = op.compile()
        xp, mels = compiled.get_conn_padded(x_batch)
    """

    __slots__ = ("_ir_terms",)

    def __init__(
        self,
        hilbert: DiscreteHilbertSpace,
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

        Free symbols are unresolved named parameters such as ``symbol("kappa")``.
        Symbols declared with ``default=...`` are considered resolved and are not
        included in this set.
        """
        result: set = set()
        for t in self._ir_terms:
            result |= t.free_symbols
        return frozenset(result)

    def to_ir(self) -> SymbolicOperatorIR:
        """Builds the symbolic operator IR for compiler consumption."""
        debug_event(
            "materializing symbolic operator ir",
            scope="ir",
            tag="IR",
            operator_name=self._name_val,
            term_count=len(self._ir_terms),
            dtype=self._dtype_val,
        )
        ir = SymbolicOperatorIR.from_terms(
            operator_name=self._name_val,
            hilbert_size=int(self.hilbert.size),
            dtype_str=self._dtype_val,
            is_hermitian=self._is_hermitian_val,
            terms=self._ir_terms,
            metadata=self._metadata_dict if self._metadata_dict else None,
        )
        debug_event(
            "materialized symbolic operator ir",
            scope="ir",
            tag="IR",
            operator_name=ir.operator_name,
            term_count=ir.term_count,
            free_symbol_count=len(ir.free_symbols),
        )
        return ir

    def estimate_max_conn_size(self) -> int:
        """Returns the aggregate static max-connection bound across all terms."""
        from neuralqx.operator.symbolic.ir.term import KBodyIteratorSpec

        total = 0
        for t in self._ir_terms:
            if t.max_conn_size_hint is not None:
                total += int(t.max_conn_size_hint)
            else:
                if not isinstance(t.iterator, KBodyIteratorSpec):
                    raise NeuralqxTypeError(
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
        backend: str | None = None,
        operator_lowering: str | None = None,
        deduplicate_connected_components: bool | None = None,
        optimize_kernels: bool | None = None,
        cache: bool | None = None,
        compiler: Any = None,
    ) -> Any:  # -> CompiledOperator
        """Lowers this symbolic operator to the selected executable operator family.

        Args:
            backend: Backend target (currently only ``"jax"`` is supported).
            operator_lowering: Registered target name for this invocation.
                ``None`` uses the compiler default or ``neuralqx.operator.symbolic.config.LOWERING``.
                An explicit name does not change the default.
            deduplicate_connected_components: Whether to merge duplicate
                connected states and drop zero matrix elements. Static analysis
                derives the output capacity and plans residual runtime checks.
            optimize_kernels: Whether to use structured kernel planning. ``None``
                inherits the compiler default (enabled). Disable for reference lowering.
            cache: Whether to cache artifacts in the compiler's in-process store.
            compiler: Optional :class:`~neuralqx.operator.symbolic.compiler.SymbolicCompiler`
                instance. None constructs a compiler with the default pipeline.
                Options left as None inherit that compiler's settings.

        Returns:
            Executable operator instance for the selected lowering target.
        """
        from dataclasses import replace

        from neuralqx.operator.symbolic.compiler.compiler import SymbolicCompiler

        c = compiler if compiler is not None else SymbolicCompiler()
        overrides = {
            key: value
            for key, value in {
                "backend_preference": backend,
                "operator_lowering": operator_lowering,
                "deduplicate_connected_components": deduplicate_connected_components,
                "optimize_kernels": optimize_kernels,
                "cache_enabled": cache,
            }.items()
            if value is not None
        }
        debug_event(
            "compiling symbolic operator",
            scope="compile",
            tag="COMPILER",
            operator_name=self._name_val,
            backend=backend,
            operator_lowering=operator_lowering,
            deduplicate_connected_components=deduplicate_connected_components,
            cache=cache,
        )
        if not overrides:
            return c.compile_operator(self)
        return c.compile_operator(self, options=replace(c.options, **overrides))

    def _apply_scalar(self, scalar: int | float | complex) -> SymbolicOperator:
        from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
        from neuralqx.operator.symbolic.ir.term import _scale_ir_term

        scale_expr = AmplitudeExpr.constant(scalar)
        new_terms = tuple(_scale_ir_term(t, scale_expr) for t in self._ir_terms)
        is_hermitian = self._is_hermitian_val and not isinstance(scalar, complex)
        new_dtype = _promote_dtype_for_scalar(self._dtype_val, scalar)
        scaled = SymbolicOperator(
            self.hilbert,
            f"({_scalar_str(scalar)} * {self.name})",
            new_terms,
            dtype_str=new_dtype,
            is_hermitian=is_hermitian,
            metadata=self._metadata_dict or None,
        )
        debug_event(
            "scaled symbolic operator",
            scope="dsl",
            tag="DSL",
            source_operator=self.name,
            scalar=scalar,
            target_operator=scaled.name,
        )
        return scaled

    def __add__(self, other: Any):
        """Compose with another operator using native operator algebra."""
        return super().__add__(other)

    def __radd__(self, other: Any):
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
