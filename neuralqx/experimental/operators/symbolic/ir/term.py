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


"""Typed declarative symbolic operator-term IR structures."""

from __future__ import annotations

import dataclasses
from typing import Any

from .expressions import AmplitudeExpr
from .expressions import coerce_amplitude_expr

from .predicates import PredicateExpr
from .predicates import coerce_predicate_expr

from .update import UpdateProgram


@dataclasses.dataclass(frozen=True, repr=False)
class KBodyIteratorSpec:
    """
    Static K-body iterator over a pre-computed list of site-index tuples.

    This iterator evaluates a term kernel once per entry in *index_sets*.
    Each entry is a K-tuple of integer site indices that are bound to the
    corresponding element of *labels* inside the evaluation environment.

    For a single-site iterator over all N sites, use
    ``KBodyIteratorSpec(labels=("i",), index_sets=tuple((k,) for k in range(N)))``.
    For a static triplet iterator, provide the explicit list of
    ``(e1, e2, e3)`` triplets.  For a global (one-branch) term, use
    ``KBodyIteratorSpec(labels=(), index_sets=((),))``.

    Attributes:
        labels: Ordered tuple of K label strings bound per iteration.
        index_sets: M-tuple of K-tuples of int site indices.
    """

    labels: tuple  # tuple[str, ...]
    index_sets: tuple  # tuple[tuple[int, ...], ...]

    @property
    def kind(self) -> str:
        return "kbody"

    @property
    def label_a(self) -> str:
        return self.labels[0] if self.labels else ""

    @property
    def label_b(self) -> str | None:
        return self.labels[1] if len(self.labels) > 1 else None

    def _format_iterate_line(self, max_shown: int = 3) -> str:
        """Returns a compact readable description of the iteration domain."""
        if not self.labels:
            return "globally"
        n = len(self.index_sets)
        if len(self.labels) == 1:
            labels_str = f"({self.labels[0]},)"
        else:
            labels_str = f"({', '.join(self.labels)})"
        shown = self.index_sets[:max_shown]
        shown_str = ", ".join(str(t) for t in shown)
        if n > max_shown:
            return f"for {labels_str} in [{shown_str}, ... +{n - max_shown} more]"
        return f"for {labels_str} in [{shown_str}]"

    def __repr__(self) -> str:
        return (
            f"KBodyIteratorSpec(labels={self.labels!r}, "
            f"n_index_sets={len(self.index_sets)})"
        )


@dataclasses.dataclass(frozen=True, repr=False)
class EmissionSpec:
    """
    One output branch (connected state + matrix element) of a term.

    A single iterator evaluation can produce multiple branches, one per
    ``EmissionSpec`` in the parent term's ``emissions`` tuple.  This allows
    a plaquette term, for example, to emit both ``+`` and ``-`` connected
    states from the same site-tuple without splitting into two separate terms.

    Attributes:
        update_program: Site-update program mapping ``x → x'``.
        amplitude: Matrix-element expression evaluated in the source environment.
        branch_tag: Optional diagnostic tag for this emission slot.
    """

    update_program: UpdateProgram
    amplitude: AmplitudeExpr
    branch_tag: Any = None


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicIRTerm:
    """
    One primitive declarative symbolic operator term.

    Attributes:
        name: Term name.
        iterator: Iterator descriptor (``KBodyIteratorSpec``).
        predicate: Branch-selection predicate.
        update_program: Matrix-element update program.
        amplitude: Matrix-element expression.
        branch_tag: Optional branch tag for diagnostics.
        metadata: Optional stable term metadata tuple.
        fanout_hint: Optional static upper-bound hint on the number of connected
            states this term produces per input configuration.
        emissions: Optional multi-emission tuple that, when present, supersedes
            *update_program* and *amplitude*. Each entry is an :class:`EmissionSpec`
            representing one output branch per iterator evaluation.
    """

    name: str
    iterator: Any  # KBodyIteratorSpec
    predicate: PredicateExpr
    update_program: UpdateProgram
    amplitude: AmplitudeExpr
    branch_tag: Any = None
    metadata: tuple = dataclasses.field(default_factory=tuple)
    fanout_hint: int | None = None
    emissions: tuple | None = None  # tuple[EmissionSpec, ...] | None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Symbolic IR term name must be a non-empty string.")
        if self.fanout_hint is not None and self.fanout_hint <= 0:
            raise ValueError(
                f"fanout_hint must be a positive integer when provided; "
                f"received {self.fanout_hint!r}."
            )

    @property
    def effective_emissions(self) -> tuple:
        """
        Returns the active emission list for this term.

        When *emissions* is set, returns it directly. Otherwise wraps the
        legacy *update_program* / *amplitude* pair in a single-element tuple.
        """
        if self.emissions is not None:
            return self.emissions
        return (
            EmissionSpec(
                update_program=self.update_program,
                amplitude=self.amplitude,
                branch_tag=self.branch_tag,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        iterator: Any,
        predicate: Any,
        update_program: UpdateProgram,
        amplitude: Any,
        branch_tag: Any = None,
        metadata: dict[str, Any] | None = None,
        fanout_hint: int | None = None,
        emissions: tuple | None = None,
    ) -> "SymbolicIRTerm":
        """Builds a term from user-friendly values."""
        meta_tuple: tuple
        if metadata is None:
            meta_tuple = ()
        else:
            meta_tuple = tuple(sorted(metadata.items()))
        return cls(
            name=str(name),
            iterator=iterator,
            predicate=coerce_predicate_expr(predicate),
            update_program=update_program,
            amplitude=coerce_amplitude_expr(amplitude),
            branch_tag=branch_tag,
            metadata=meta_tuple,
            fanout_hint=fanout_hint,
            emissions=emissions,
        )

    def metadata_dict(self) -> dict[str, Any]:
        """Returns metadata in dictionary form."""
        return dict(self.metadata)

    @property
    def free_symbols(self) -> frozenset:
        """
        Returns the set of free (non-iterator-bound) symbol names in this term.

        Free symbols are those not bound by any site/emit iterator label, i.e. named
        parameters such as ``symbol("kappa")`` that must be resolved externally.
        """
        from .expressions import _collect_free_symbols
        from .predicates import _collect_free_symbols_pred
        from .update import _collect_free_symbols_from_ops

        result: set = set()
        _collect_free_symbols_pred(self.predicate, result)
        for em in self.effective_emissions:
            _collect_free_symbols(em.amplitude, result)
            _collect_free_symbols_from_ops(em.update_program.ops, result)
        return frozenset(result)

    def _to_ir_lines(self, idx: "int | None" = None, indent: str = "") -> "list[str]":
        """Formats this term as indented IR lines suitable for embedding in an operator dump."""
        name_part = f"#{idx} " if idx is not None else ""
        it = self.iterator
        it_kind = getattr(it, "kind", "unknown")
        n_iter = len(it.index_sets) if hasattr(it, "index_sets") else "?"
        fanout = self.fanout_hint if self.fanout_hint is not None else "?"
        iter_line = (
            it._format_iterate_line()
            if hasattr(it, "_format_iterate_line")
            else repr(it)
        )

        inner = indent + "  "
        lines = [
            f'{indent}term {name_part}"{self.name}" [{it_kind}, n_iter={n_iter}, fanout={fanout}] {{'
        ]
        lines.append(f"{inner}iterate: {iter_line}")
        lines.append(f"{inner}where:   {self.predicate}")

        for i, em in enumerate(self.effective_emissions):
            tag_str = f" [tag={em.branch_tag!r}]" if em.branch_tag is not None else ""
            lines.append(f"{inner}emit #{i}{tag_str}:")
            lines.append(f"{inner}  update:    {em.update_program}")
            lines.append(f"{inner}  amplitude: {em.amplitude}")

        lines.append(f"{indent}}}")
        return lines

    def __str__(self) -> str:
        return "\n".join(self._to_ir_lines(idx=None, indent=""))

    def __repr__(self) -> str:
        return (
            f"SymbolicIRTerm("
            f"name={self.name!r}, "
            f"iterator={self.iterator!r}, "
            f"fanout_hint={self.fanout_hint!r})"
        )


def _scale_ir_term(term: SymbolicIRTerm, scale_expr: AmplitudeExpr) -> SymbolicIRTerm:
    """
    Returns a new SymbolicIRTerm with all emission amplitudes multiplied by *scale_expr*.

    Handles both the multi-emission path (``term.emissions is not None``) and the
    single-emission path.
    """
    import dataclasses

    new_amp = AmplitudeExpr.mul(scale_expr, term.amplitude)

    new_emissions: "tuple | None" = None
    if term.emissions is not None:
        new_emissions = tuple(
            dataclasses.replace(
                em, amplitude=AmplitudeExpr.mul(scale_expr, em.amplitude)
            )
            for em in term.emissions
        )

    return dataclasses.replace(term, amplitude=new_amp, emissions=new_emissions)


__all__ = [
    "EmissionSpec",
    "KBodyIteratorSpec",
    "SymbolicIRTerm",
    "_scale_ir_term",
]
