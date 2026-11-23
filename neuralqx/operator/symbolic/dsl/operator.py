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


"""
Fluent operator builder, the primary entry point to the symbolic DSL.

The :class:`SymbolicDiscreteJaxOperator` class is the single entry point for constructing symbolic
quantum operators.

Usage
-----
::

    from neuralqx.operator.symbolic import SymbolicDiscreteJaxOperator
    from neuralqx.operator.symbolic.dsl import site, shift, swap, identity

    # diagonal operator
    N_e0 = (
        SymbolicDiscreteJaxOperator(hi, "N_e0", hermitian=True)
        .globally()
        .emit(identity(), matrix_element=my_sq_norm_expr)
        .build()
    )

    # single-site off-diagonal
    h_plus = (
        SymbolicDiscreteJaxOperator(hi, "h+")
        .for_each_site("e")
        .where(site("e") < cutoff)
        .emit(shift("e", +1))
        .build()
    )

    # hopping: compound update + matrix element from both DOFs
    hop = (
        SymbolicDiscreteJaxOperator(hi, "hopping")
        .for_each_pair("i", "j")
        .where(site("i") > 0)
        .emit(
            shift("i", -1).shift("j", +1),
            matrix_element=site("i").value * site("j").value,
        )
        .build()
    )

    # K-body: static triplet iterator
    vol = (
        SymbolicDiscreteJaxOperator(hi, "triplet_volume")
        .for_each(("e1", "e2", "e3"), over=triplet_index_sets)
        .emit(identity(), matrix_element=triple_product_expr)
        .build()
    )

    # multi-emission: two branches per iterator evaluation
    two_branch = (
        SymbolicDiscreteJaxOperator(hi, "two_branch")
        .for_each_site("i")
        .where(site("i").abs() < 2)
        .emit(shift("i", +1), matrix_element=+0.5)
        .emit(shift("i", -1), matrix_element=-0.5)
        .build()
    )

    # compile directly (skip explicit .build())
    compiled = (
        SymbolicDiscreteJaxOperator(hi, "my_op").for_each_site("i").emit(shift("i", +1)).compile()
    )

Iterator methods
----------------
Calling any ``for_each_*`` / ``globally`` method **seals** the current
in-progress term and begins a new one. Calling ``.where`` or ``.emit``
after is always associated with the most recent iterator call.

Multi-emission
--------------
Multiple ``.emit(...)`` calls on the same iterator scope produce multiple
output branches (``EmissionSpec`` entries) from a **single** iterator
evaluation. This avoids the overhead of iterating over sites twice and
keeps the semantic unit cohesive.

Connected-state dedup note
--------------------------
By default, if two terms (or two emissions within one term) produce the same
``x'``, those entries are merged and their matrix elements are summed. Final
zero-amplitude components are dropped before padding.

Set ``deduplicate_connected_components=False`` during ``.compile(...)`` to keep
raw per-branch connectivity output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.operator.symbolic.dsl.emissions.defaults import (
    ensure_default_emission_clause_registrations,
)
from neuralqx.operator.symbolic.dsl.emissions.dispatch import apply_emission_clause
from neuralqx.operator.symbolic.dsl.emissions.registry import (
    available_emission_clause_names,
)
from neuralqx.operator.symbolic.dsl.emissions.registry import resolve_emission_clause
from neuralqx.operator.symbolic.dsl.emissions.types import EmissionClauseSpec
from neuralqx.operator.symbolic.dsl.iterators.defaults import (
    ensure_default_iterator_clause_registrations,
)
from neuralqx.operator.symbolic.dsl.iterators.dispatch import apply_iterator_clause
from neuralqx.operator.symbolic.dsl.iterators.registry import (
    available_iterator_clause_names,
)
from neuralqx.operator.symbolic.dsl.iterators.registry import resolve_iterator_clause
from neuralqx.operator.symbolic.dsl.predicates.defaults import (
    ensure_default_predicate_clause_registrations,
)
from neuralqx.operator.symbolic.dsl.predicates.dispatch import apply_predicate_clause
from neuralqx.operator.symbolic.dsl.predicates.registry import (
    available_predicate_clause_names,
)
from neuralqx.operator.symbolic.dsl.predicates.registry import resolve_predicate_clause
from neuralqx.operator.symbolic.dsl.rewrite import _IDENTITY as _IDENTITY_UPDATE
from neuralqx.operator.symbolic.dsl.rewrite import Update
from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.operator.symbolic.ir.expressions import coerce_amplitude_expr
from neuralqx.operator.symbolic.ir.predicates import PredicateExpr
from neuralqx.operator.symbolic.ir.predicates import coerce_predicate_expr
from neuralqx.operator.symbolic.ir.term import EmissionSpec
from neuralqx.operator.symbolic.ir.term import KBodyIteratorSpec
from neuralqx.operator.symbolic.ir.term import SymbolicIRTerm
from neuralqx.operator.symbolic.ir.update import UpdateProgram
from neuralqx.utils.errors import NeuralqxAttributeError
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

if TYPE_CHECKING:
    from neuralqx.operator.symbolic import SymbolicOperator

ensure_default_iterator_clause_registrations()
ensure_default_predicate_clause_registrations()
ensure_default_emission_clause_registrations()


def _update_op_uses_shift_mod(op: Any) -> bool:
    if op.kind == "shift_mod_site":
        return True
    if op.kind == "cond_branch":
        then_ops = op.get("then_ops") or ()
        else_ops = op.get("else_ops") or ()
        return any(_update_op_uses_shift_mod(sub) for sub in then_ops) or any(
            _update_op_uses_shift_mod(sub) for sub in else_ops
        )
    return False


def _program_uses_shift_mod(program: UpdateProgram) -> bool:
    return any(_update_op_uses_shift_mod(op) for op in program.ops)


def _amplitude_uses_wrap_mod(expr: AmplitudeExpr) -> bool:
    if expr.op == "wrap_mod":
        return True
    return any(
        isinstance(arg, AmplitudeExpr) and _amplitude_uses_wrap_mod(arg)
        for arg in expr.args
    )


def _terms_use_shift_mod(terms: tuple[SymbolicIRTerm, ...]) -> bool:
    for term in terms:
        for em in term.effective_emissions:
            if _program_uses_shift_mod(em.update_program):
                return True
            if _amplitude_uses_wrap_mod(em.amplitude):
                return True
    return False


def _iter_amplitude_constants(expr: AmplitudeExpr):
    """Yields all constant payloads occurring in one amplitude expression tree."""
    if expr.op == "const":
        yield expr.args[0]
        return
    for arg in expr.args:
        if isinstance(arg, AmplitudeExpr):
            yield from _iter_amplitude_constants(arg)
        elif isinstance(arg, tuple):
            for item in arg:
                if isinstance(item, AmplitudeExpr):
                    yield from _iter_amplitude_constants(item)


def _resolve_matrix_element_dtype(
    base_dtype_str: str,
    terms: tuple[SymbolicIRTerm, ...],
) -> str:
    """
    Resolves the operator matrix-element dtype from user default + emissions.

    Complex constants in matrix-element expressions automatically promote
    the operator dtype using NumPy result-type promotion rules.
    """
    resolved = np.dtype(base_dtype_str)
    for term in terms:
        for em in term.effective_emissions:
            for const in _iter_amplitude_constants(em.amplitude):
                const_dtype = np.asarray(const).dtype
                if np.issubdtype(const_dtype, np.complexfloating):
                    resolved = np.result_type(resolved, const_dtype)
    return np.dtype(resolved).name


def _infer_shift_mod_spec_from_hilbert(hilbert: DiscreteHilbertSpace) -> dict[str, Any]:
    from neuralqx.operator.symbolic._hilbert import shift_mod_metadata

    return shift_mod_metadata(hilbert)


def _coerce_update(u: Any) -> UpdateProgram:
    """Normalises Update or UpdateProgram to UpdateProgram."""
    if isinstance(u, Update):
        return u.to_program()
    if isinstance(u, UpdateProgram):
        return u
    raise NeuralqxTypeError(
        f"Expected Update or UpdateProgram; got {type(u).__name__!r}."
    )


def _coerce_amplitude(a: Any) -> AmplitudeExpr:
    """Resolves callables, bare numbers, or AmplitudeExpr nodes."""
    if callable(a):
        from neuralqx.operator.symbolic.dsl.context import ExpressionContext

        return coerce_amplitude_expr(a(ExpressionContext()))
    return coerce_amplitude_expr(a)


#
#
# Internal term-in-progress


class _TermInProgress:
    """Mutable accumulator for one in-progress term definition."""

    __slots__ = (
        "_conditional_remaining",
        "_emissions",
        "_iterator",
        "_max_conn_size_hint",
        "_name",
        "_predicate",
    )

    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator
        self._predicate: PredicateExpr = PredicateExpr.constant(True)
        self._emissions: list[EmissionSpec] = []
        self._name: str | None = None
        self._max_conn_size_hint: int | None = None
        self._conditional_remaining: PredicateExpr | None = None

    @property
    def name(self) -> str | None:
        """Returns the optional user-assigned term name."""
        return self._name

    def set_name(self, name: str) -> None:
        """
        Sets the user-assigned term name.

        Args:
            name: Non-empty term name.
        """
        self._name = str(name)

    @property
    def max_conn_size_hint(self) -> int | None:
        """Returns the optional explicit max-connection-size hint."""
        return self._max_conn_size_hint

    def set_max_conn_size_hint(self, hint: int) -> None:
        """
        Sets an explicit max-connection-size hint.

        Args:
            hint: Positive max-connection-size bound.
        """
        self._max_conn_size_hint = int(hint)

    @property
    def predicate(self) -> PredicateExpr:
        """Returns the current composed predicate expression."""
        return self._predicate

    @property
    def predicate_op(self) -> str:
        """Returns the top-level predicate operator name."""
        return self._predicate.op

    def set_predicate(self, pred: Any) -> None:
        """
        Replaces the current term predicate.

        Args:
            pred: Predicate expression or coercible value.
        """
        self._predicate = coerce_predicate_expr(pred)

    @property
    def emission_count(self) -> int:
        """Returns the current emission count for this term."""
        return len(self._emissions)

    @property
    def has_open_conditional_chain(self) -> bool:
        """Returns whether an if/elseif chain is currently open."""
        return self._conditional_remaining is not None

    def close_conditional_chain(self) -> None:
        """Closes any currently open conditional emission chain."""
        self._conditional_remaining = None

    def _require_conditional_chain(self, method: str) -> PredicateExpr:
        if self._conditional_remaining is None:
            raise NeuralqxValueError(
                f".{method}() must follow .emit_if(...) or .emit_elseif(...) "
                "without intervening term modifiers."
            )
        return self._conditional_remaining

    def add_emission(
        self,
        update: Any,
        matrix_element: Any,
        tag: Any,
        *,
        predicate: Any = True,
        amplitude: Any | None = None,
    ) -> None:
        prog = _coerce_update(update)
        if amplitude is not None:
            matrix_element = amplitude
        amp = _coerce_amplitude(matrix_element)
        pred = coerce_predicate_expr(predicate)
        self._emissions.append(
            EmissionSpec(
                update_program=prog,
                amplitude=amp,
                branch_tag=tag,
                predicate=pred,
            )
        )
        debug_event(
            "registered term emission",
            scope="dsl",
            tag="DSL",
            update_kind_count=len(prog.ops),
            branch_tag=tag,
            emission_count=len(self._emissions),
            predicate_op=pred.op,
        )

    def add_conditional_if(
        self,
        predicate: Any,
        update: Any,
        matrix_element: Any,
        tag: Any,
        *,
        amplitude: Any | None = None,
    ) -> None:
        cond = coerce_predicate_expr(predicate)
        self.close_conditional_chain()
        self.add_emission(
            update=update,
            matrix_element=matrix_element,
            tag=tag,
            predicate=cond,
            amplitude=amplitude,
        )
        self._conditional_remaining = PredicateExpr.not_(cond)

    def add_conditional_elseif(
        self,
        predicate: Any,
        update: Any,
        matrix_element: Any,
        tag: Any,
        *,
        amplitude: Any | None = None,
    ) -> None:
        remaining = self._require_conditional_chain("emit_elseif")
        cond = coerce_predicate_expr(predicate)
        branch_pred = PredicateExpr.and_(remaining, cond)
        self.add_emission(
            update=update,
            matrix_element=matrix_element,
            tag=tag,
            predicate=branch_pred,
            amplitude=amplitude,
        )
        self._conditional_remaining = PredicateExpr.and_(
            remaining, PredicateExpr.not_(cond)
        )

    def add_conditional_else(
        self,
        update: Any,
        matrix_element: Any,
        tag: Any,
        *,
        amplitude: Any | None = None,
    ) -> None:
        remaining = self._require_conditional_chain("emit_else")
        self.add_emission(
            update=update,
            matrix_element=matrix_element,
            tag=tag,
            predicate=remaining,
            amplitude=amplitude,
        )
        self.close_conditional_chain()

    def to_ir_term(self, auto_name: str) -> SymbolicIRTerm:
        self.close_conditional_chain()
        if not self._emissions:
            name = self.name if self.name is not None else auto_name
            raise NeuralqxValueError(
                f"Term {name!r} has no emissions. Call .emit(...) before .build() / .compile()."
            )
        emissions_tuple = tuple(self._emissions)
        first = emissions_tuple[0]
        name = self.name if self.name is not None else auto_name

        # Auto-infer max_conn_size from iterator size x emission count when not user-set
        max_conn_size_hint = self.max_conn_size_hint
        if max_conn_size_hint is None and isinstance(self._iterator, KBodyIteratorSpec):
            max_conn_size_hint = len(self._iterator.index_sets) * len(emissions_tuple)

        return SymbolicIRTerm.create(
            name=name,
            iterator=self._iterator,
            predicate=self._predicate,
            update_program=first.update_program,
            amplitude=first.amplitude,
            branch_tag=first.branch_tag,
            emissions=emissions_tuple,
            max_conn_size_hint=max_conn_size_hint,
        )


#
#
#   Operator builder


class SymbolicDiscreteJaxOperator:
    """
    Fluent builder for declarative symbolic quantum operators.

    The builder accumulates one or more *terms*. Each term consists of an
    *iterator* (which sites to visit), an optional *predicate* (which visits
    to activate), and one or more *emissions* (how to rewrite the configuration
    and what matrix element to assign per active visit).

    Calling any iterator method (``for_each_site``, ``for_each_pair``, ...,
    ``globally``) **seals** the previous term (if any) and begins a new one.
    ``.where`` and ``.emit`` always target the current open term.

    Args:
        hilbert: Native :class:`~neuralqx.hilbert.DiscreteHilbertSpace`.
        name: Readable operator name (accessible as ``.name`` on the
            resulting :class:`~neuralqx.operator.symbolic.core.operator.SymbolicOperator`).
        dtype: Matrix-element dtype string (default ``"float64"``).
        hermitian: Whether to declare the operator Hermitian.
    """

    __slots__ = (
        "_completed_terms",
        "_current",
        "_dtype",
        "_hermitian",
        "_hilbert",
        "_name",
    )

    def __init__(
        self,
        hilbert: DiscreteHilbertSpace,
        name: str = "operator",
        *,
        dtype: str = "float64",
        hermitian: bool = False,
    ) -> None:
        name = str(name).strip()
        if not name:
            raise NeuralqxValueError("Operator name must be a non-empty string.")
        self._hilbert = hilbert
        self._name = name
        self._dtype = str(dtype)
        self._hermitian = bool(hermitian)
        self._completed_terms: list[SymbolicIRTerm] = []
        self._current: _TermInProgress | None = None
        debug_event(
            "created symbolic dsl builder",
            scope="dsl",
            tag="DSL",
            operator_name=self._name,
            dtype=self._dtype,
            hermitian=self._hermitian,
            hilbert_size=int(self._hilbert.size),
        )

    @property
    def hilbert(self) -> DiscreteHilbertSpace:
        """
        Returns the Hilbert space associated with this builder.

        Returns:
            DiscreteHilbertSpace: The builder Hilbert space.
        """
        return self._hilbert

    def open_term(self, iterator: Any) -> SymbolicDiscreteJaxOperator:
        """
        Opens a new in-progress term with the provided iterator.

        Args:
            iterator: Iterator descriptor for the new term.

        Returns:
            SymbolicDiscreteJaxOperator: This builder for fluent chaining.
        """
        return self._open_term(iterator)

    def append_predicate(
        self,
        predicate: PredicateExpr,
        *,
        method_name: str = "where",
    ) -> SymbolicDiscreteJaxOperator:
        """
        Appends one predicate to the current term.

        This is a public facade used by external predicate-clause modules to avoid
        accessing private builder methods directly.

        Args:
            predicate: Predicate expression to append.
            method_name: Name of the fluent method responsible for the append.

        Returns:
            SymbolicDiscreteJaxOperator: This builder for fluent chaining.
        """
        return self._append_predicate(predicate, method_name=method_name)

    def append_emission_clause(
        self,
        spec: EmissionClauseSpec,
        *,
        method_name: str,
    ) -> SymbolicDiscreteJaxOperator:
        """
        Appends emission behavior described by one normalized emission clause spec.

        Args:
            spec: Normalized emission clause action.
            method_name: Name of the fluent method responsible for the append.

        Returns:
            SymbolicDiscreteJaxOperator: This builder for fluent chaining.
        """
        return self._append_emission_clause(spec, method_name=method_name)

    #
    #
    #   Internal

    def _seal_current(self) -> None:
        """Finalises the current in-progress term and appends it."""
        if self._current is not None:
            term_name = str(len(self._completed_terms))
            ir_term = self._current.to_ir_term(term_name)
            self._completed_terms.append(ir_term)
            self._current = None
            debug_event(
                "sealed dsl term",
                scope="dsl",
                tag="DSL",
                term_name=ir_term.name,
                max_conn_size_hint=ir_term.max_conn_size_hint,
                emission_count=len(ir_term.effective_emissions),
            )

    def _open_term(self, iterator: Any) -> SymbolicDiscreteJaxOperator:
        """Seals any open term and starts a new one with *iterator*."""
        self._seal_current()
        self._current = _TermInProgress(iterator)
        debug_event(
            "opened dsl term",
            scope="dsl",
            tag="DSL",
            iterator_kind=getattr(iterator, "kind", type(iterator).__name__),
            labels=getattr(iterator, "labels", None),
        )
        return self

    def _require_open(self, method: str) -> _TermInProgress:
        if self._current is None:
            raise NeuralqxValueError(
                f".{method}() called before any iterator method "
                "(for_each_site / for_each_pair / for_each / globally). "
                "Declare an iterator first."
            )
        return self._current

    #
    #
    #   Iterator methods

    def globally(self) -> SymbolicDiscreteJaxOperator:
        """
        Sets a **global** iterator, one branch per configuration.

        Use this for diagonal operators (area, number, volume, ...) and for
        off-diagonal operators where the target sites are baked into the
        amplitude or update program via
        :func:`~neuralqx.operator.symbolic.ir.expressions.AmplitudeExpr.static_index`.

        Returns:
            This builder (for chaining).
        """
        return apply_iterator_clause(self, "globally")

    def for_each_site(self, label: str = "i") -> SymbolicDiscreteJaxOperator:
        """
        Iterates over **all** sites ``0 ... hilbert.size-1``.

        The site index is bound to *label* in the evaluation environment:
        ``site(label).value`` -> ``x[site_index]`` and
        ``site(label).index`` -> the integer site index.

        Args:
            label: Iterator label string (default ``"i"``).

        Returns:
            This builder (for chaining).
        """
        return apply_iterator_clause(self, "for_each_site", label)

    def for_each_pair(
        self,
        label_a: str = "i",
        label_b: str = "j",
    ) -> SymbolicDiscreteJaxOperator:
        """
        Iterates over all ordered pairs ``(i, j)`` with ``i, j ∈ [0, N)``.

        Includes diagonal pairs ``(i, i)``. To exclude them use
        ``.for_each_distinct_pair()`` or an explicit expression comparison,
        built with ``ExpressionContext.ne`` or ``PredicateExpr.ne``.

        Args:
            label_a: Primary site label.
            label_b: Secondary site label.

        Returns:
            This builder (for chaining).
        """
        return apply_iterator_clause(self, "for_each_pair", label_a, label_b)

    def for_each_distinct_pair(
        self,
        label_a: str = "i",
        label_b: str = "j",
    ) -> SymbolicDiscreteJaxOperator:
        """
        Iterates over all ordered pairs ``(i, j)`` with ``i, j ∈ [0, N)``.

        Excludes diagonal pairs ``(i, i)``. To include them, use
        ``.for_each_pair()``.

        Args:
            label_a: Primary site label.
            label_b: Secondary site label.

        Returns:
            This builder (for chaining).
        """
        return apply_iterator_clause(self, "for_each_distinct_pair", label_a, label_b)

    def for_each_triplet(
        self,
        label_a: str,
        label_b: str,
        label_c: str,
        *,
        over: Sequence[tuple[int, int, int]],
    ) -> SymbolicDiscreteJaxOperator:
        """
        Iterates over a **static list of ordered triplets**.

        Args:
            label_a: First site label.
            label_b: Second site label.
            label_c: Third site label.
            over: Sequence of ``(i, j, k)`` integer index triplets.

        Returns:
            This builder (for chaining).
        """
        return apply_iterator_clause(
            self,
            "for_each_triplet",
            label_a,
            label_b,
            label_c,
            over=over,
        )

    def for_each_plaquette(
        self,
        label_a: str,
        label_b: str,
        label_c: str,
        label_d: str,
        *,
        over: Sequence[tuple[int, int, int, int]],
    ) -> SymbolicDiscreteJaxOperator:
        """
        Iterates over a **static list of ordered plaquettes** (4-body).

        Args:
            label_a: Site label for the first corner.
            label_b: Site label for the second corner.
            label_c: Site label for the third corner.
            label_d: Site label for the fourth corner.
            over: Sequence of ``(i, j, k, l)`` integer index 4-tuples.

        Returns:
            This builder (for chaining).
        """
        return apply_iterator_clause(
            self,
            "for_each_plaquette",
            label_a,
            label_b,
            label_c,
            label_d,
            over=over,
        )

    def for_each(
        self,
        labels: Sequence[str],
        *,
        over: Sequence[Sequence[int]],
    ) -> SymbolicDiscreteJaxOperator:
        """
        Iterates over an **arbitrary static list of K-tuples**.

        This is the most general iterator method.  All other ``for_each_*``
        methods are convenience wrappers around this one.

        Args:
            labels: Sequence of K label strings.
            over: Sequence of K-tuples of integer site indices.
                Must be non-empty; all tuples must have length ``len(labels)``.

        Returns:
            This builder (for chaining).

        Raises:
            ValueError: If *over* is empty or any tuple has the wrong length.

        Example::

            # Graph-neighbourhood iterator from an adjacency list
            edges = [(src, dst) for src, nbrs in adj.items() for dst in nbrs]
            op = (
                SymbolicDiscreteJaxOperator(hi, "nbr_hop")
                .for_each(("src", "dst"), over=edges)
                .where(site("src") > 0)
                .emit(shift("src", -1).shift("dst", +1))
                .build()
            )
        """
        return apply_iterator_clause(self, "for_each", labels, over=over)

    #
    #
    #   Term annotation

    def named(self, name: str) -> SymbolicDiscreteJaxOperator:
        """
        Assigns a readable name to the current term.

        By default terms are named by their zero-based index (``"0"``, ``"1"``,
        ...).  Call ``.named(...)`` after an iterator method to override this with
        a descriptive label that appears in IR dumps and compiler diagnostics.

        Args:
            name: Non-empty string label for this term.

        Returns:
            This builder (for chaining).
        """
        term = self._require_open("named")
        term.close_conditional_chain()
        name = str(name).strip()
        if not name:
            raise NeuralqxValueError("Term name must be a non-empty string.")
        term.set_name(name)
        return self

    def max_conn_size(self, hint: int) -> SymbolicDiscreteJaxOperator:
        """
        Sets an explicit static max-connection-size hint for the current term.

        The hint is an upper bound on the total number of connected
        states this term produces per input configuration.  When not set, the
        DSL infers it automatically as ``n_iter x n_emissions``, a correct
        but conservative bound. Provide a tighter value when the predicate
        or physics guarantees fewer active branches (e.g. a holonomy operator
        with a hard cutoff always emits exactly 1 state).

        The hint is used by the compiler's buffer pre-allocation pass.

        Args:
            hint: Positive integer upper bound.

        Returns:
            This builder (for chaining).
        """
        term = self._require_open("max_conn_size")
        term.close_conditional_chain()
        hint = int(hint)
        if hint <= 0:
            raise NeuralqxValueError(
                f"max_conn_size hint must be a positive integer; got {hint!r}."
            )
        term.set_max_conn_size_hint(hint)
        return self

    def fanout(self, hint: int) -> SymbolicDiscreteJaxOperator:
        """Backward-compatible alias for :meth:`max_conn_size`."""
        return self.max_conn_size(hint)

    def _append_predicate(
        self,
        predicate: PredicateExpr,
        *,
        method_name: str = "where",
    ) -> SymbolicDiscreteJaxOperator:
        """Composes one predicate into the current term with logical AND."""
        term = self._require_open(method_name)
        term.close_conditional_chain()
        existing = term.predicate
        if existing.op == "const" and bool(existing.args[0]):
            term.set_predicate(predicate)
        else:
            term.set_predicate(PredicateExpr.and_(existing, predicate))
        debug_event(
            "updated term predicate",
            scope="dsl",
            tag="DSL",
            predicate_op=term.predicate_op,
            predicate_method=method_name,
        )
        return self

    def _append_emission_clause(
        self,
        spec: EmissionClauseSpec,
        *,
        method_name: str,
    ) -> SymbolicDiscreteJaxOperator:
        """Applies one emission-clause specification to the current term."""
        term = self._require_open(method_name)

        if spec.mode == "emit":
            term.close_conditional_chain()
            term.add_emission(
                update=_IDENTITY_UPDATE if spec.update is None else spec.update,
                matrix_element=spec.matrix_element,
                tag=spec.tag,
                amplitude=spec.amplitude,
            )
        elif spec.mode == "emit_if":
            term.add_conditional_if(
                predicate=spec.predicate,
                update=_IDENTITY_UPDATE if spec.update is None else spec.update,
                matrix_element=spec.matrix_element,
                tag=spec.tag,
                amplitude=spec.amplitude,
            )
        elif spec.mode == "emit_elseif":
            term.add_conditional_elseif(
                predicate=spec.predicate,
                update=_IDENTITY_UPDATE if spec.update is None else spec.update,
                matrix_element=spec.matrix_element,
                tag=spec.tag,
                amplitude=spec.amplitude,
            )
        elif spec.mode == "emit_else":
            term.add_conditional_else(
                update=_IDENTITY_UPDATE if spec.update is None else spec.update,
                matrix_element=spec.matrix_element,
                tag=spec.tag,
                amplitude=spec.amplitude,
            )
        else:
            raise NeuralqxValueError(
                f"Unsupported emission clause mode {spec.mode!r}. "
                "Expected one of: emit, emit_if, emit_elseif, emit_else."
            )

        debug_event(
            "applied emission clause",
            scope="dsl",
            tag="DSL",
            clause_method=method_name,
            clause_mode=spec.mode,
            emission_count=term.emission_count,
            branch_tag=spec.tag,
            has_open_conditional_chain=term.has_open_conditional_chain,
        )
        return self

    #
    #
    #   Predicate

    def where(self, predicate: Any) -> SymbolicDiscreteJaxOperator:
        """
        Sets the **branch predicate** for the current term.

        The predicate is evaluated in the iterator environment (``x``, site
        labels). Only branches where the predicate is ``True`` emit
        connected states, the rest contribute zero matrix elements.

        Multiple ``.where`` calls on the same term compose with logical AND::

            .where(site("i") > 0).where(site("j") < 2)
            # ↑ equivalent to .where((site("i") > 0) & (site("j") < 2))

        Args:
            predicate: :class:`~neuralqx.operator.symbolic.ir.predicates.PredicateExpr`
                or any value coercible to one (e.g. ``site("i").value > 0``).

        Returns:
            This builder (for chaining).
        """
        return apply_predicate_clause(self, "where", predicate)

    #
    #
    #   Emission

    def emit(
        self,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> SymbolicDiscreteJaxOperator:
        """
        Appends one **output branch** to the current term.

        Each call to ``.emit(...)`` on the same iterator scope adds one
        :class:`~neuralqx.operator.symbolic.ir.term.EmissionSpec` to the
        current term. Multiple emissions produce multiple connected states
        per iterator evaluation, e.g. raise *and* lower from the same site
        without splitting into two separate terms.

        Matrix-element semantics
        --------------------
        The matrix-element expression is evaluated in the *source* configuration
        environment ``(x, site_labels)``. There is no access to ``x'`` inside
        matrix-element expressions: ``<x|O|x'>`` is computed from ``x``, not ``x'``.

        Args:
            update: Site-rewrite program describing ``x -> x'``.  Accepts
                :class:`~neuralqx.operator.symbolic.dsl.rewrite.Update`,
                or :class:`~neuralqx.operator.symbolic.ir.update.UpdateProgram`.
                Pass ``None`` or :func:`~neuralqx.operator.symbolic.dsl.rewrite.identity`
                for diagonal (identity) updates.
            matrix_element: Matrix element: numeric constant, symbolic
                :class:`~neuralqx.operator.symbolic.ir.expressions.AmplitudeExpr`,
                or a callable ``(ExpressionContext) -> AmplitudeExpr``.
            amplitude: Deprecated alias of ``matrix_element``.
            tag: Optional diagnostic label for this emission branch.

        Returns:
            This builder (for chaining).
        """
        return self.append_emission_clause(
            EmissionClauseSpec(
                mode="emit",
                update=update,
                matrix_element=matrix_element,
                amplitude=amplitude,
                tag=tag,
            ),
            method_name="emit",
        )

    def emit_if(
        self,
        predicate: Any,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> SymbolicDiscreteJaxOperator:
        """
        Appends the ``if`` branch of a conditional emission chain.

        The branch emits only when *predicate* evaluates to true.
        Subsequent ``.emit_elseif(...)`` and ``.emit_else(...)`` calls can
        refine the same chain.
        """
        return apply_emission_clause(
            self,
            "emit_if",
            predicate,
            update,
            matrix_element=matrix_element,
            amplitude=amplitude,
            tag=tag,
        )

    def emit_elseif(
        self,
        predicate: Any,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> SymbolicDiscreteJaxOperator:
        """
        Appends an ``elseif`` branch to the current conditional emission chain.

        This method must directly follow ``emit_if(...)`` or another
        ``emit_elseif(...)`` on the same term.
        """
        return apply_emission_clause(
            self,
            "emit_elseif",
            predicate,
            update,
            matrix_element=matrix_element,
            amplitude=amplitude,
            tag=tag,
        )

    def emit_else(
        self,
        update: Any = None,
        *,
        matrix_element: Any = 1.0,
        amplitude: Any | None = None,
        tag: Any = None,
    ) -> SymbolicDiscreteJaxOperator:
        """
        Appends the ``else`` branch to the current conditional emission chain.

        This branch emits when all prior ``if`` / ``elseif`` predicates are false.
        """
        return apply_emission_clause(
            self,
            "emit_else",
            update,
            matrix_element=matrix_element,
            amplitude=amplitude,
            tag=tag,
        )

    #
    #
    #   Finalisation

    def build(self) -> SymbolicOperator:
        """
        Seals all open terms and returns a
        :class:`~neuralqx.operator.symbolic.core.operator.SymbolicOperator`.

        Returns:
            :class:`~neuralqx.operator.symbolic.core.operator.SymbolicOperator`
            ready for compilation.

        Raises:
            ValueError: If no terms have been defined, or the current open
                term has no emissions.
        """
        self._seal_current()
        if not self._completed_terms:
            raise NeuralqxValueError(
                "Cannot build an operator with zero terms. "
                "Add at least one iterator + emit() block."
            )
        debug_event(
            "building symbolic operator",
            scope="dsl",
            tag="DSL",
            operator_name=self._name,
            term_count=len(self._completed_terms),
        )

        metadata: dict[str, Any] = {}
        if _terms_use_shift_mod(tuple(self._completed_terms)):
            metadata.update(_infer_shift_mod_spec_from_hilbert(self._hilbert))
            debug_event(
                "inferred shift_mod metadata",
                scope="dsl",
                tag="DSL",
                metadata_keys=tuple(sorted(metadata)),
            )

        resolved_dtype = _resolve_matrix_element_dtype(
            self._dtype,
            tuple(self._completed_terms),
        )
        if resolved_dtype != self._dtype:
            debug_event(
                "promoted operator dtype from matrix-element constants",
                scope="dsl",
                tag="DSL",
                operator_name=self._name,
                requested_dtype=self._dtype,
                resolved_dtype=resolved_dtype,
            )

        from neuralqx.operator.symbolic.core.operator import SymbolicOperator

        op = SymbolicOperator(
            self._hilbert,
            self._name,
            tuple(self._completed_terms),
            dtype_str=resolved_dtype,
            is_hermitian=self._hermitian,
            metadata=metadata or None,
        )
        debug_event(
            "built symbolic operator",
            scope="dsl",
            tag="DSL",
            operator_name=op.name,
            term_count=op.term_count,
        )
        return op

    def compile(
        self,
        *,
        backend: str | None = None,
        operator_lowering: str | None = None,
        deduplicate_connected_components: bool | None = None,
        optimize_kernels: bool | None = None,
        cache: bool | None = None,
        compiler: Any = None,
    ) -> Any:
        """
        Convenience shortcut: ``.build().compile(...)``.

        Args:
            backend: Backend target. Currently only ``"jax"`` is supported.
            operator_lowering: Target name for this compilation, or ``None``
                to use ``neuralqx.operator.symbolic.config.LOWERING``. Does not change the default.
            deduplicate_connected_components: Whether to merge repeated targets.
            optimize_kernels: Whether to use structured kernel planning.
            cache: Whether to reuse in-process compiled artifacts. None inherits the compiler setting.
            compiler: Optional compiler instance with custom options or passes.

        Returns:
            Executable compiled operator instance.
        """
        debug_event(
            "compiling directly from dsl builder",
            scope="dsl",
            tag="DSL",
            operator_name=self._name,
            backend=backend,
            operator_lowering=operator_lowering,
            deduplicate_connected_components=deduplicate_connected_components,
            optimize_kernels=optimize_kernels,
            cache=cache,
        )
        return self.build().compile(
            backend=backend,
            operator_lowering=operator_lowering,
            deduplicate_connected_components=deduplicate_connected_components,
            optimize_kernels=optimize_kernels,
            cache=cache,
            compiler=compiler,
        )

    def __getattr__(self, name: str) -> Any:
        """
        Resolves dynamically-registered iterator/predicate/emission clause methods.

        This enables fluent user extensions such as ``builder.my_iterator(...)``
        without adding concrete methods to this class.
        """
        iterator_clause = resolve_iterator_clause(name)
        if iterator_clause is not None:

            def _bound_iterator(
                *args: Any, **kwargs: Any
            ) -> SymbolicDiscreteJaxOperator:
                return apply_iterator_clause(self, name, *args, **kwargs)

            _bound_iterator.__name__ = name
            _bound_iterator.__qualname__ = f"{type(self).__name__}.{name}"
            _bound_iterator.__doc__ = iterator_clause.__doc__
            return _bound_iterator

        predicate_clause = resolve_predicate_clause(name)
        if predicate_clause is not None:

            def _bound_predicate(
                *args: Any, **kwargs: Any
            ) -> SymbolicDiscreteJaxOperator:
                return apply_predicate_clause(self, name, *args, **kwargs)

            _bound_predicate.__name__ = name
            _bound_predicate.__qualname__ = f"{type(self).__name__}.{name}"
            _bound_predicate.__doc__ = predicate_clause.__doc__
            return _bound_predicate

        emission_clause = resolve_emission_clause(name)
        if emission_clause is not None:

            def _bound_emission(
                *args: Any, **kwargs: Any
            ) -> SymbolicDiscreteJaxOperator:
                return apply_emission_clause(self, name, *args, **kwargs)

            _bound_emission.__name__ = name
            _bound_emission.__qualname__ = f"{type(self).__name__}.{name}"
            _bound_emission.__doc__ = emission_clause.__doc__
            return _bound_emission

        raise NeuralqxAttributeError(
            f"{type(self).__name__!s} object has no attribute {name!r}"
        )

    def __dir__(self) -> list[str]:
        """
        Includes registered clause names in interactive completion output.
        """
        base = set(super().__dir__())
        base.update(available_iterator_clause_names())
        base.update(available_predicate_clause_names())
        base.update(available_emission_clause_names())
        return sorted(base)

    def __repr__(self) -> str:
        n_sealed = len(self._completed_terms)
        n_open = 1 if self._current is not None else 0
        return (
            f"{type(self).__name__}("
            f"name={self._name!r}, "
            f"dtype={self._dtype!r}, "
            f"terms_sealed={n_sealed}, "
            f"term_open={bool(n_open)})"
        )


SymbolicOperatorBuilder = SymbolicDiscreteJaxOperator


__all__ = [
    "SymbolicDiscreteJaxOperator",
    "SymbolicOperatorBuilder",
]
