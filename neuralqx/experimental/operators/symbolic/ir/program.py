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


"""Top-level symbolic operator IR container."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .term import SymbolicIRTerm

_IR_MODES: frozenset[str] = frozenset({"symbolic", "jax_kernel"})


def _serialize_amplitude(expr: Any) -> Any:
    """Recursively serializes an AmplitudeExpr to a JSON-safe structure."""
    from .expressions import AmplitudeExpr

    if isinstance(expr, AmplitudeExpr):
        return {"op": expr.op, "args": [_serialize_amplitude(a) for a in expr.args]}
    if isinstance(expr, tuple):
        return [_serialize_amplitude(v) for v in expr]
    return expr


def _serialize_predicate(expr: Any) -> Any:
    """Recursively serializes a PredicateExpr to a JSON-safe structure."""
    from .predicates import PredicateExpr
    from .expressions import AmplitudeExpr

    if isinstance(expr, PredicateExpr):
        return {"op": expr.op, "args": [_serialize_predicate(a) for a in expr.args]}
    if isinstance(expr, AmplitudeExpr):
        return _serialize_amplitude(expr)
    if isinstance(expr, tuple):
        return [_serialize_predicate(v) for v in expr]
    return expr


def _serialize_update(program: Any) -> Any:
    """Serializes an UpdateProgram to a JSON-safe structure."""
    from .update import UpdateProgram, UpdateOp

    if not isinstance(program, UpdateProgram):
        return repr(program)
    ops = []
    for op in program.ops:
        params_dict = {}
        for k, v in op.params:
            params_dict[k] = _serialize_amplitude(v)
        ops.append({"kind": op.kind, "params": params_dict})
    return ops


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicOperatorIR:
    """
    Immutable symbolic operator IR container.

    Attributes:
        operator_name: Name of the operator this IR represents.
        mode: IR mode (``symbolic`` for DSL-built operators, ``jax_kernel``
            for direct JAX-kernel operators).
        hilbert_size: Size of the Hilbert space (number of sites).
        dtype_str: String representation of the matrix-element dtype.
        is_hermitian: Whether the source operator is declared Hermitian.
        terms: Declarative term tuple for ``symbolic`` mode.
        metadata: Optional stable metadata tuple.
    """

    operator_name: str
    mode: str
    hilbert_size: int
    dtype_str: str
    is_hermitian: bool
    terms: tuple = dataclasses.field(default_factory=tuple)
    metadata: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.operator_name.strip():
            raise ValueError("operator_name must be a non-empty string.")
        if self.mode not in _IR_MODES:
            raise ValueError(
                f"Unsupported IR mode: {self.mode!r}. Allowed: {sorted(_IR_MODES)}."
            )
        if self.hilbert_size <= 0:
            raise ValueError(
                f"hilbert_size must be a positive integer; got {self.hilbert_size!r}."
            )

    @classmethod
    def from_terms(
        cls,
        *,
        operator_name: str,
        hilbert_size: int,
        dtype_str: str,
        is_hermitian: bool,
        terms: tuple[SymbolicIRTerm, ...],
        metadata: dict[str, Any] | None = None,
    ) -> "SymbolicOperatorIR":
        """Builds declarative symbolic-mode operator IR."""
        if not terms:
            raise ValueError("Symbolic operator IR requires at least one term.")
        meta_tuple: tuple
        if metadata is None:
            meta_tuple = ()
        else:
            meta_tuple = tuple(sorted(metadata.items()))
        return cls(
            operator_name=str(operator_name),
            mode="symbolic",
            hilbert_size=int(hilbert_size),
            dtype_str=str(dtype_str),
            is_hermitian=bool(is_hermitian),
            terms=terms,
            metadata=meta_tuple,
        )

    @property
    def term_count(self) -> int:
        """Returns number of declarative terms."""
        return len(self.terms)

    def metadata_dict(self) -> dict[str, Any]:
        """Returns metadata in dictionary form."""
        return dict(self.metadata)

    def as_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation."""
        return {
            "operator_name": self.operator_name,
            "mode": self.mode,
            "hilbert_size": self.hilbert_size,
            "dtype_str": self.dtype_str,
            "is_hermitian": self.is_hermitian,
            "terms": [
                {
                    "name": t.name,
                    "iterator": {
                        "kind": t.iterator.kind,
                        "label_a": t.iterator.label_a,
                        "label_b": t.iterator.label_b,
                    },
                    "predicate": _serialize_predicate(t.predicate),
                    "update_program": _serialize_update(t.update_program),
                    "amplitude": _serialize_amplitude(t.amplitude),
                    "branch_tag": t.branch_tag,
                    "metadata": list(t.metadata),
                    "fanout_hint": t.fanout_hint,
                }
                for t in self.terms
            ],
            "metadata": list(self.metadata),
        }

    def static_fingerprint(self) -> str:
        """Returns a deterministic SHA-256 digest over the static IR payload."""
        raw = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def free_symbols(self) -> frozenset:
        """
        Returns the union of free symbol names across all terms.

        Free symbols are named parameters (e.g. ``symbol("kappa")``) that are
        not bound by any iterator label and must be resolved externally at
        evaluation time.
        """
        result: set = set()
        for term in self.terms:
            result |= term.free_symbols
        return frozenset(result)

    def __str__(self) -> str:
        """
        Returns a structured IR dump in the neuraLQX symbolic IR format.

        The format is inspired by LLVM IR / MLIR. Named operator blocks with
        typed terms, readable iterator descriptions, infix amplitude
        expressions, and pseudocode update programs.

        Example output::

            symbolic.operator @"hopping" [dtype=complex64, hermitian=false, hilbert_size=16] {
              ; 1 term(s)

              term #0 "0" [kbody, n_iter=256, fanout=256] {
                iterate: for (i, j) in [(0, 0), (0, 1), (0, 2), ... +253 more]
                where:   (x[i] > 0)
                emit #0:
                  update:    x'[i] = (x[i] + -1); x'[j] = (x[j] + 1)
                  amplitude: 1
              }

            }
        """
        hermitian_str = "true" if self.is_hermitian else "false"
        lines = [
            f'symbolic.operator @"{self.operator_name}" '
            f"[dtype={self.dtype_str}, hermitian={hermitian_str}, "
            f"hilbert_size={self.hilbert_size}] {{"
        ]

        fs = self.free_symbols
        if fs:
            fs_str = ", ".join(f"%{s}" for s in sorted(fs))
            lines.append(f"  ; {len(self.terms)} term(s), free symbols: [{fs_str}]")
        else:
            lines.append(f"  ; {len(self.terms)} term(s)")

        for idx, term in enumerate(self.terms):
            lines.append("")
            lines.extend(term._to_ir_lines(idx=idx, indent="  "))

        lines.append("")
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SymbolicOperatorIR("
            f"operator_name={self.operator_name!r}, "
            f"mode={self.mode!r}, "
            f"hilbert_size={self.hilbert_size}, "
            f"term_count={self.term_count}, "
            f"is_hermitian={self.is_hermitian})"
        )


__all__ = ["SymbolicOperatorIR"]
