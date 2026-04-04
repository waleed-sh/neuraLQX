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


"""Additive composition of symbolic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from netket.hilbert import DiscreteHilbert

from neuralqx.experimental.operators.symbolic.core.base import AbstractSymbolicOperator
from neuralqx.experimental.operators.symbolic.ir.program import SymbolicOperatorIR


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
        return explicit
    dtypes = {t._dtype_val for t in terms}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    # Promote: complex128 > complex64 > float64 > float32
    for candidate in ("complex128", "float64", "complex64", "float32"):
        if candidate in dtypes:
            return candidate
    return "complex64"


class SymbolicOperatorSum(AbstractSymbolicOperator):
    """
    Additive composition of multiple symbolic operators sharing one Hilbert space.

    ``SymbolicOperatorSum`` is the canonical Hamiltonian-style container for
    DSL-defined operators. It preserves term ordering, flattens nested sums,
    and aggregates fanout bounds across all contained terms.

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
        hilbert: DiscreteHilbert,
        terms: Sequence[AbstractSymbolicOperator],
        *,
        name: str | None = None,
        dtype_str: str | None = None,
        is_hermitian: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        flattened = _flatten_terms(terms)
        if not flattened:
            raise ValueError("SymbolicOperatorSum requires at least one term.")

        for t in flattened:
            if t.hilbert != hilbert:
                raise ValueError(
                    f"All terms in SymbolicOperatorSum must share the same "
                    f"Hilbert space. Term {t._name_val!r} has a different hilbert."
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

    def _apply_scalar(self, scalar: "int | float | complex") -> "SymbolicOperatorSum":
        from neuralqx.experimental.operators.symbolic.core.operator import (
            _scalar_str,
            _promote_dtype_for_scalar,
        )

        new_terms = tuple(t._apply_scalar(scalar) for t in self._terms)
        is_hermitian = self._is_hermitian_val and not isinstance(scalar, complex)
        new_dtype = _promote_dtype_for_scalar(self._dtype_val, scalar)
        return SymbolicOperatorSum(
            self.hilbert,
            new_terms,
            name=f"({_scalar_str(scalar)} * {self._name_val})",
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
            raise ValueError(
                f"Cannot aggregate term IRs with mixed modes: {modes!r}. "
                "All terms must be in 'symbolic' mode."
            )

        combined_terms = tuple(
            term for child_ir in child_irs for term in child_ir.terms
        )

        fingerprints = tuple(ir.static_fingerprint() for ir in child_irs)
        meta = dict(self._metadata_dict)
        meta["child_ir_fingerprints"] = fingerprints

        return SymbolicOperatorIR.from_terms(
            operator_name=self._name_val,
            hilbert_size=int(self.hilbert.size),
            dtype_str=self._dtype_val,
            is_hermitian=self._is_hermitian_val,
            terms=combined_terms,
            metadata=meta,
        )

    def estimate_max_conn_size(self) -> int:
        """
        Returns the aggregate static fanout bound across all terms.

        Returns:
            Sum of per-term fanout upper bounds.
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
            f"name={self._name_val!r}, "
            f"term_count={len(self._terms)}, "
            f"dtype={self._dtype_val!r}, "
            f"is_hermitian={self._is_hermitian_val})"
        )


__all__ = ["SymbolicOperatorSum"]
