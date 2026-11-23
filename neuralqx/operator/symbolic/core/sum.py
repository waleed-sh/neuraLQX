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


"""Additive composition of symbolic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator.symbolic.core.base import AbstractSymbolicOperator
from neuralqx.operator.symbolic.ir.program import SymbolicOperatorIR
from neuralqx.utils.errors import NeuralqxAttributeError
from neuralqx.utils.errors import NeuralqxValueError


def _term_dtype_name(term: AbstractSymbolicOperator) -> str:
    """
    Resolves one term's dtype name using public APIs with compatibility fallback.

    Args:
        term: Symbolic operator term.

    Returns:
        str: Normalized dtype name for the term.

    Raises:
        AttributeError: If no dtype-like attribute is available.
    """
    if hasattr(term, "dtype_str"):
        return np.dtype(term.dtype_str).name
    if hasattr(term, "dtype"):
        return np.dtype(term.dtype).name
    legacy = getattr(term, "_dtype_val", None)
    if legacy is not None:
        return np.dtype(legacy).name
    raise NeuralqxAttributeError(
        f"Cannot resolve dtype for term of type {type(term).__name__!r}."
    )


def _term_display_name(term: AbstractSymbolicOperator) -> str:
    """
    Resolves one term's display name with compatibility fallback.

    Args:
        term: Symbolic operator term.

    Returns:
        str: Best-effort descriptive term name.
    """
    if hasattr(term, "name"):
        return str(term.name)
    if hasattr(term, "operator_name"):
        return str(term.operator_name)
    legacy = getattr(term, "_name_val", None)
    if legacy is not None:
        return str(legacy)
    return type(term).__name__


def _flatten_terms(
    terms: Sequence[AbstractSymbolicOperator],
) -> tuple[AbstractSymbolicOperator, ...]:
    """Flattens nested symbolic operator sums while preserving order."""
    flat: list[AbstractSymbolicOperator] = []
    for t in terms:
        if isinstance(t, SymbolicOperatorSum):
            flat.extend(t.terms)
        else:
            flat.append(t)
    return tuple(flat)


def _resolve_dtype(
    terms: Sequence[AbstractSymbolicOperator], explicit: str | None
) -> str:
    """Resolves a common dtype string for an additive composition."""
    if explicit is not None:
        return np.dtype(explicit).name
    if not terms:
        return np.dtype("complex64").name
    resolved = np.dtype(_term_dtype_name(terms[0]))
    for term in terms[1:]:
        resolved = np.result_type(resolved, np.dtype(_term_dtype_name(term)))
    return np.dtype(resolved).name


class SymbolicOperatorSum(AbstractSymbolicOperator):
    """
    Additive composition of multiple symbolic operators sharing one Hilbert space.

    ``SymbolicOperatorSum`` is the canonical Hamiltonian-style container for
    DSL-defined operators. It preserves term ordering, flattens nested sums,
    and aggregates max-connection-size bounds across all contained terms.

    Args:
        hilbert: Shared Hilbert space.
        terms: Sequence of symbolic operator terms.
        name: Optional user-facing operator name.
        dtype_str: Optional explicit dtype override.
        is_hermitian: Optional Hermiticity override (defaults to ``True`` iff
            all contained terms are Hermitian).
        metadata: Optional metadata dictionary.
    """

    __slots__ = ("_terms",)

    def __init__(
        self,
        hilbert: DiscreteHilbertSpace,
        terms: Sequence[AbstractSymbolicOperator],
        *,
        name: str | None = None,
        dtype_str: str | None = None,
        is_hermitian: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        flattened = _flatten_terms(terms)
        if not flattened:
            raise NeuralqxValueError("SymbolicOperatorSum requires at least one term.")

        for t in flattened:
            if t.hilbert != hilbert:
                raise NeuralqxValueError(
                    f"All terms in SymbolicOperatorSum must share the same "
                    f"Hilbert space. Term {_term_display_name(t)!r} has a different hilbert."
                )

        resolved_name = (
            str(name).strip() if name and str(name).strip() else "symbolic_sum"
        )
        resolved_dtype = _resolve_dtype(flattened, dtype_str)
        resolved_hermitian = (
            bool(is_hermitian)
            if is_hermitian is not None
            else all(t.is_hermitian for t in flattened)
        )

        super().__init__(
            hilbert,
            name=resolved_name,
            dtype_str=resolved_dtype,
            is_hermitian=resolved_hermitian,
            metadata=metadata,
        )
        self._terms: tuple[AbstractSymbolicOperator, ...] = flattened

    @property
    def terms(self) -> tuple[AbstractSymbolicOperator, ...]:
        """Returns contained additive terms in declaration order."""
        return self._terms

    @property
    def free_symbols(self) -> frozenset:
        """Returns the union of free symbol names across all contained terms."""
        result: set = set()
        for t in self._terms:
            if hasattr(t, "free_symbols"):
                result |= t.free_symbols
        return frozenset(result)

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)

    def _apply_scalar(self, scalar: int | float | complex) -> SymbolicOperatorSum:
        from neuralqx.operator.symbolic.core.operator import _promote_dtype_for_scalar
        from neuralqx.operator.symbolic.core.operator import _scalar_str

        new_terms = tuple(t.apply_scalar(scalar) for t in self._terms)
        is_hermitian = self.is_hermitian and not isinstance(scalar, complex)
        new_dtype = _promote_dtype_for_scalar(self.dtype_str, scalar)
        return SymbolicOperatorSum(
            self.hilbert,
            new_terms,
            name=f"({_scalar_str(scalar)} * {self.name})",
            dtype_str=new_dtype,
            is_hermitian=is_hermitian,
        )

    def to_ir(self) -> SymbolicOperatorIR:
        """
        Builds one aggregate IR from all contained terms.

        All child IRs must be in ``symbolic`` mode.

        Returns:
            Aggregate symbolic operator IR.

        Raises:
            ValueError: If terms cannot be aggregated into one IR.
        """
        child_irs = tuple(t.to_ir() for t in self._terms)
        modes = {ir.mode for ir in child_irs}
        if modes != {"symbolic"}:
            raise NeuralqxValueError(
                f"Cannot aggregate term IRs with mixed modes: {modes!r}. "
                "All terms must be in 'symbolic' mode."
            )

        combined_terms = tuple(
            term for child_ir in child_irs for term in child_ir.terms
        )

        fingerprints = tuple(ir.static_fingerprint() for ir in child_irs)
        meta = dict(self.metadata)
        meta["child_ir_fingerprints"] = fingerprints

        return SymbolicOperatorIR.from_terms(
            operator_name=self.name,
            hilbert_size=int(self.hilbert.size),
            dtype_str=self.dtype_str,
            is_hermitian=self.is_hermitian,
            terms=combined_terms,
            metadata=meta,
        )

    def estimate_max_conn_size(self) -> int:
        """
        Returns the aggregate static max-connection bound across all terms.

        Returns:
            Sum of per-term max-connection upper bounds.
        """
        total = 0
        for t in self._terms:
            if hasattr(t, "estimate_max_conn_size"):
                total += int(t.estimate_max_conn_size())
            else:
                # Fallback: assume hilbert.size per unknown term
                total += int(self.hilbert.size)
        return max(1, total)

    def __repr__(self) -> str:
        return (
            f"SymbolicOperatorSum("
            f"name={self.name!r}, "
            f"term_count={len(self._terms)}, "
            f"dtype={self.dtype_str!r}, "
            f"is_hermitian={self.is_hermitian})"
        )


__all__ = ["SymbolicOperatorSum"]
