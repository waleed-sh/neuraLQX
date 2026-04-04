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


"""
Fluent operator builder, the primary entry point to the symbolic DSL.

The :class:`DOperator` class is the single entry point for constructing symbolic
quantum operators.

Usage
-----
::

    from neuralqx.experimental.operators.symbolic import DOperator
    from neuralqx.experimental.operators.symbolic.dsl import site, shift, swap, identity

    # diagonal operator
    N_e0 = (
        DOperator(hi, "N_e0", hermitian=True)
        .globally()
        .emit(identity(), amplitude=my_sq_norm_expr)
        .build()
    )

    # single-site off-diagonal
    h_plus = (
        DOperator(hi, "h+")
        .for_each_site("e")
        .where(site("e") < cutoff)
        .emit(shift("e", +1))
        .build()
    )

    # hopping: compound update + amplitude from both DOFs
    hop = (
        DOperator(hi, "hopping")
        .for_each_pair("i", "j")
        .where(site("i") > 0)
        .emit(
            shift("i", -1).shift("j", +1),
            amplitude=site("i").value * site("j").value,
        )
        .build()
    )

    # K-body: static triplet iterator
    vol = (
        DOperator(hi, "triplet_volume")
        .for_each(("e1", "e2", "e3"), over=triplet_index_sets)
        .emit(identity(), amplitude=triple_product_expr)
        .build()
    )

    # multi-emission: two branches per iterator evaluation
    two_branch = (
        DOperator(hi, "two_branch")
        .for_each_site("i")
        .where(site("i").abs() < 2)
        .emit(shift("i", +1), amplitude=+0.5)
        .emit(shift("i", -1), amplitude=-0.5)
        .build()
    )

    # compile directly (skip explicit .build())
    compiled = DOperator(hi, "my_op").for_each_site("i").emit(shift("i", +1)).compile()

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

Branch-multiset note
--------------------
If two terms (or two emissions within one term) produce the same ``x'``,
both entries appear in the padded output with their own matrix elements.
The output is a branch **multiset**, not a canonical deduplicated row.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from netket.hilbert import DiscreteHilbert

from neuralqx.experimental.operators.symbolic.dsl.rewrite import Update
from neuralqx.experimental.operators.symbolic.dsl.rewrite import (
    _IDENTITY as _IDENTITY_UPDATE,
)

from neuralqx.experimental.operators.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.experimental.operators.symbolic.ir.expressions import (
    coerce_amplitude_expr,
)

from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr
from neuralqx.experimental.operators.symbolic.ir.predicates import coerce_predicate_expr

from neuralqx.experimental.operators.symbolic.ir.term import EmissionSpec
from neuralqx.experimental.operators.symbolic.ir.term import KBodyIteratorSpec
from neuralqx.experimental.operators.symbolic.ir.term import SymbolicIRTerm

from neuralqx.experimental.operators.symbolic.ir.update import UpdateProgram


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


def _infer_shift_mod_spec_from_hilbert(hilbert: DiscreteHilbert) -> dict[str, Any]:
    """
    Infer uniform wrapped-shift semantics from hilbert.local_states.

    Current contract:
      - finite local_states must exist
      - they must be 1D
      - they must be contiguous unit-spaced integers
        e.g. [-m_max, ..., m_max] or [0, 1, 2, 3]

    This exactly matches the current modulo-wrap semantics used by the
    computational operators.
    """
    local_states = getattr(hilbert, "local_states", None)
    if local_states is None:
        raise ValueError(
            "shift_mod requires a discrete Hilbert with finite local_states. "
            "This Hilbert exposes local_states=None."
        )

    states = np.asarray(local_states)
    if states.ndim != 1 or states.size == 0:
        raise ValueError(
            "shift_mod requires hilbert.local_states to be a non-empty 1D sequence."
        )

    # Require integer-valued local states
    states_i = states.astype(np.int64)
    if not np.array_equal(states, states_i):
        raise ValueError("shift_mod currently requires integer local_states.")

    # Require contiguous unit-spaced ascending values
    state_min = int(states_i[0])
    expected = np.arange(state_min, state_min + len(states_i), dtype=np.int64)
    if not np.array_equal(states_i, expected):
        raise ValueError(
            "shift_mod currently requires contiguous unit-spaced local_states, "
            "for example [-m_max, ..., m_max]. "
            f"Got {states_i.tolist()!r}."
        )

    return {
        "shift_mod_spec": {
            "version": "uniform_integer_wrap_v1",
            "state_min": state_min,
            "mod_span": int(len(states_i)),
            # included so the IR fingerprint/caches depend on the actual local basis
            "local_states": tuple(int(v) for v in states_i.tolist()),
        }
    }


def _coerce_update(u: Any) -> UpdateProgram:
    """Normalises Update or UpdateProgram to UpdateProgram."""
    if isinstance(u, Update):
        return u.to_program()
    if isinstance(u, UpdateProgram):
        return u
    raise TypeError(f"Expected Update or UpdateProgram; got {type(u).__name__!r}.")


def _coerce_amplitude(a: Any) -> AmplitudeExpr:
    """Resolves callables, bare numbers, or AmplitudeExpr nodes."""
    if callable(a):
        from neuralqx.experimental.operators.symbolic.dsl.context import (
            ExpressionContext,
        )

        return coerce_amplitude_expr(a(ExpressionContext()))
    return coerce_amplitude_expr(a)


#
#
# Internal term-in-progress


class _TermInProgress:
    """Mutable accumulator for one in-progress term definition."""

    __slots__ = ("_emissions", "_fanout_hint", "_iterator", "_name", "_predicate")

    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator
        self._predicate: PredicateExpr = PredicateExpr.constant(True)
        self._emissions: list[EmissionSpec] = []
        self._name: "str | None" = None
        self._fanout_hint: "int | None" = None

    def set_predicate(self, pred: Any) -> None:
        self._predicate = coerce_predicate_expr(pred)

    def add_emission(self, update: Any, amplitude: Any, tag: Any) -> None:
        prog = _coerce_update(update)
        amp = _coerce_amplitude(amplitude)
        self._emissions.append(
            EmissionSpec(
                update_program=prog,
                amplitude=amp,
                branch_tag=tag,
            )
        )

    def to_ir_term(self, auto_name: str) -> SymbolicIRTerm:
        if not self._emissions:
            name = self._name if self._name is not None else auto_name
            raise ValueError(
                f"Term {name!r} has no emissions. "
                "Call .emit(...) before .build() / .compile()."
            )
        emissions_tuple = tuple(self._emissions)
        first = emissions_tuple[0]
        name = self._name if self._name is not None else auto_name

        # Auto-infer fanout from iterator size x emission count when not user-set
        fanout_hint = self._fanout_hint
        if fanout_hint is None and isinstance(self._iterator, KBodyIteratorSpec):
            fanout_hint = len(self._iterator.index_sets) * len(emissions_tuple)

        return SymbolicIRTerm.create(
            name=name,
            iterator=self._iterator,
            predicate=self._predicate,
            update_program=first.update_program,
            amplitude=first.amplitude,
            branch_tag=first.branch_tag,
            emissions=emissions_tuple,
            fanout_hint=fanout_hint,
        )


#
#
#   Operator builder


class DOperator:
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
        hilbert: NetKet :class:`~netket.hilbert.DiscreteHilbert` space.
        name: Readable operator name (accessible as ``.name`` on the
            resulting :class:`~neuralqx.experimental.operators.symbolic.core.operator.SymbolicOperator`).
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
        hilbert: DiscreteHilbert,
        name: str = "operator",
        *,
        dtype: str = "float64",
        hermitian: bool = False,
    ) -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("Operator name must be a non-empty string.")
        self._hilbert = hilbert
        self._name = name
        self._dtype = str(dtype)
        self._hermitian = bool(hermitian)
        self._completed_terms: list[SymbolicIRTerm] = []
        self._current: _TermInProgress | None = None

    #
    #
    #   Internal

    def _seal_current(self) -> None:
        """Finalises the current in-progress term and appends it."""
        if self._current is not None:
            term_name = str(len(self._completed_terms))
            self._completed_terms.append(self._current.to_ir_term(term_name))
            self._current = None

    def _open_term(self, iterator: Any) -> "DOperator":
        """Seals any open term and starts a new one with *iterator*."""
        self._seal_current()
        self._current = _TermInProgress(iterator)
        return self

    def _require_open(self, method: str) -> _TermInProgress:
        if self._current is None:
            raise ValueError(
                f".{method}() called before any iterator method "
                "(for_each_site / for_each_pair / for_each / globally). "
                "Declare an iterator first."
            )
        return self._current

    #
    #
    #   Iterator methods

    def globally(self) -> "DOperator":
        """
        Sets a **global** iterator, one branch per configuration.

        Use this for diagonal operators (area, number, volume, ...) and for
        off-diagonal operators where the target sites are baked into the
        amplitude or update program via
        :func:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr.static_index`.

        Returns:
            This builder (for chaining).
        """
        return self._open_term(KBodyIteratorSpec(labels=(), index_sets=((),)))

    def for_each_site(self, label: str = "i") -> "DOperator":
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
        n = int(self._hilbert.size)
        return self._open_term(
            KBodyIteratorSpec(
                labels=(str(label),),
                index_sets=tuple((k,) for k in range(n)),
            )
        )

    def for_each_pair(
        self,
        label_a: str = "i",
        label_b: str = "j",
    ) -> "DOperator":
        """
        Iterates over all ordered pairs ``(i, j)`` with ``i, j ∈ [0, N)``.

        Includes diagonal pairs ``(i, i)``.  To exclude them add a predicate
        ``.where(site(label_a).index != site(label_b).index)``.

        Args:
            label_a: Primary site label.
            label_b: Secondary site label.

        Returns:
            This builder (for chaining).
        """
        n = int(self._hilbert.size)
        pairs = tuple((i, j) for i in range(n) for j in range(n))
        return self._open_term(
            KBodyIteratorSpec(labels=(str(label_a), str(label_b)), index_sets=pairs)
        )

    def for_each_triplet(
        self,
        label_a: str,
        label_b: str,
        label_c: str,
        *,
        over: Sequence[tuple[int, int, int]],
    ) -> "DOperator":
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
        return self.for_each(
            (str(label_a), str(label_b), str(label_c)),
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
    ) -> "DOperator":
        """
        Iterates over a **static list of ordered plaquettes** (4-body).

        Args:
            label_*: Site labels for the four corners.
            over: Sequence of ``(i, j, k, l)`` integer index 4-tuples.

        Returns:
            This builder (for chaining).
        """
        return self.for_each(
            (str(label_a), str(label_b), str(label_c), str(label_d)),
            over=over,
        )

    def for_each(
        self,
        labels: Sequence[str],
        *,
        over: Sequence[Sequence[int]],
    ) -> "DOperator":
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
                DOperator(hi, "nbr_hop")
                .for_each(("src", "dst"), over=edges)
                .where(site("src") > 0)
                .emit(shift("src", -1).shift("dst", +1))
                .build()
            )
        """
        labels_t = tuple(str(l) for l in labels)
        K = len(labels_t)
        index_sets = tuple(tuple(int(idx) for idx in row) for row in over)
        if not index_sets:
            raise ValueError("for_each: over= must not be empty.")
        for row in index_sets:
            if len(row) != K:
                raise ValueError(
                    f"for_each: each tuple in over= must have length {K} "
                    f"(one index per label); got length {len(row)}."
                )
        return self._open_term(
            KBodyIteratorSpec(labels=labels_t, index_sets=index_sets)
        )

    #
    #
    #   Term annotation

    def named(self, name: str) -> "DOperator":
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
        name = str(name).strip()
        if not name:
            raise ValueError("Term name must be a non-empty string.")
        term._name = name
        return self

    def fanout(self, hint: int) -> "DOperator":
        """
        Sets an explicit static fanout hint for the current term.

        The fanout hint is an upper bound on the total number of connected
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
        term = self._require_open("fanout")
        hint = int(hint)
        if hint <= 0:
            raise ValueError(f"fanout hint must be a positive integer; got {hint!r}.")
        term._fanout_hint = hint
        return self

    #
    #
    #   Predicate

    def where(self, predicate: Any) -> "DOperator":
        """
        Sets the **branch predicate** for the current term.

        The predicate is evaluated in the iterator environment (``x``, site
        labels). Only branches where the predicate is ``True`` emit
        connected states, the rest contribute zero matrix elements.

        Multiple ``.where`` calls on the same term compose with logical AND::

            .where(site("i") > 0).where(site("j") < 2)
            # ↑ equivalent to .where((site("i") > 0) & (site("j") < 2))

        Args:
            predicate: :class:`~neuralqx.experimental.operators.symbolic.ir.predicates.PredicateExpr`
                or any value coercible to one (e.g. ``site("i").value > 0``).

        Returns:
            This builder (for chaining).
        """
        term = self._require_open("where")
        existing = term._predicate
        new_pred = coerce_predicate_expr(predicate)
        if existing.op == "const" and bool(existing.args[0]):
            # Currently trivially true, replace
            term.set_predicate(new_pred)
        else:
            # Compose with AND
            term.set_predicate(PredicateExpr.and_(existing, new_pred))
        return self

    #
    #
    #   Emission

    def emit(
        self,
        update: Any = None,
        *,
        amplitude: Any = 1.0,
        tag: Any = None,
    ) -> "DOperator":
        """
        Appends one **output branch** to the current term.

        Each call to ``.emit(...)`` on the same iterator scope adds one
        :class:`~neuralqx.experimental.operators.symbolic.ir.term.EmissionSpec` to the
        current term. Multiple emissions produce multiple connected states
        per iterator evaluation, e.g. raise *and* lower from the same site
        without splitting into two separate terms.

        Amplitude semantics
        --------------------
        The *amplitude* expression is evaluated in the *source* configuration
        environment ``(x, site_labels)``. There is no access to ``x'`` inside
        amplitude expressions: ``<x|O|x'>`` is computed from ``x``, not ``x'``.

        Args:
            update: Site-rewrite program describing ``x -> x'``.  Accepts
                :class:`~neuralqx.experimental.operators.symbolic.dsl.rewrite.Update`,
                or :class:`~neuralqx.experimental.operators.symbolic.ir.update.UpdateProgram`.
                Pass ``None`` or :func:`~neuralqx.experimental.operators.symbolic.dsl.rewrite.identity`
                for diagonal (identity) updates.
            amplitude: Matrix element: numeric constant, symbolic
                :class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr`,
                or a callable ``(ExpressionContext) -> AmplitudeExpr``.
            tag: Optional diagnostic label for this emission branch.

        Returns:
            This builder (for chaining).
        """
        term = self._require_open("emit")
        if update is None:
            update = _IDENTITY_UPDATE
        term.add_emission(update, amplitude, tag)
        return self

    #
    #
    #   Finalisation

    def build(self) -> Any:  # -> SymbolicOperator
        """
        Seals all open terms and returns a
        :class:`~neuralqx.experimental.operators.symbolic.core.operator.SymbolicOperator`.

        Returns:
            :class:`~neuralqx.experimental.operators.symbolic.core.operator.SymbolicOperator`
            ready for compilation.

        Raises:
            ValueError: If no terms have been defined, or the current open
                term has no emissions.
        """
        self._seal_current()
        if not self._completed_terms:
            raise ValueError(
                "Cannot build an operator with zero terms. "
                "Add at least one iterator + emit() block."
            )

        metadata: dict[str, Any] = {}
        if _terms_use_shift_mod(tuple(self._completed_terms)):
            metadata.update(_infer_shift_mod_spec_from_hilbert(self._hilbert))

        from neuralqx.experimental.operators.symbolic.core.operator import (
            SymbolicOperator,
        )

        return SymbolicOperator(
            self._hilbert,
            self._name,
            tuple(self._completed_terms),
            dtype_str=self._dtype,
            is_hermitian=self._hermitian,
            metadata=metadata or None,
        )

    def compile(
        self, *, backend: str = "jax", cache: bool = True
    ) -> Any:  # -> CompiledOperator
        """
        Convenience shortcut: ``.build().compile(...)``.

        Returns:
            Executable :class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator`.
        """
        return self.build().compile(backend=backend, cache=cache)

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


__all__ = ["DOperator"]
