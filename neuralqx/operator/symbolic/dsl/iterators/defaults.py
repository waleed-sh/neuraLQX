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


"""Default built-in iterator clause implementations and registration."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.operator.symbolic.dsl.iterators.abstractions import AbstractIteratorClause
from neuralqx.operator.symbolic.dsl.iterators.registry import register_iterator_clause
from neuralqx.operator.symbolic.ir.term import KBodyIteratorSpec
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError


class GloballyIteratorClause(AbstractIteratorClause):
    """Built-in global iterator clause (`builder.globally()`)."""

    clause_name = "globally"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(
        self, hilbert: DiscreteHilbertSpace, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Returns the single global iterator row with no labels.

        Args:
            hilbert: Builder Hilbert space (unused for global iterator shape).
            *args: Must be empty.
            **kwargs: Must be empty.

        Returns:
            KBodyIteratorSpec: Global iterator spec with one empty row.

        Raises:
            TypeError: If positional or keyword arguments are provided.
        """
        del hilbert
        if args or kwargs:
            raise NeuralqxTypeError(
                "globally() does not accept positional or keyword arguments."
            )
        return KBodyIteratorSpec(labels=(), index_sets=((),))


class ForEachSiteIteratorClause(AbstractIteratorClause):
    """Built-in per-site iterator clause (`builder.for_each_site(...)`)."""

    clause_name = "for_each_site"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(self, hilbert: DiscreteHilbertSpace, label: str = "i") -> Any:
        """
        Builds an iterator over all sites in the Hilbert space.

        Args:
            hilbert: Builder Hilbert space.
            label: Site-label name exposed in DSL expressions.

        Returns:
            KBodyIteratorSpec: Iterator over rows ``(0,), (1,), ...``.
        """
        n = int(hilbert.size)
        return KBodyIteratorSpec(
            labels=(str(label),),
            index_sets=tuple((k,) for k in range(n)),
        )


class ForEachPairIteratorClause(AbstractIteratorClause):
    """Built-in ordered-pair iterator clause (`builder.for_each_pair(...)`)."""

    clause_name = "for_each_pair"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(
        self, hilbert: DiscreteHilbertSpace, label_a: str = "i", label_b: str = "j"
    ) -> Any:
        """
        Builds an iterator over all ordered pairs including diagonal pairs.

        Args:
            hilbert: Builder Hilbert space.
            label_a: First label bound in each row.
            label_b: Second label bound in each row.

        Returns:
            KBodyIteratorSpec: Iterator over rows ``(i, j)`` for all sites.
        """
        n = int(hilbert.size)
        pairs = tuple((i, j) for i in range(n) for j in range(n))
        return KBodyIteratorSpec(labels=(str(label_a), str(label_b)), index_sets=pairs)


class ForEachDistinctPairIteratorClause(AbstractIteratorClause):
    """Built-in ordered-pair iterator clause excluding diagonal pairs."""

    clause_name = "for_each_distinct_pair"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(
        self, hilbert: DiscreteHilbertSpace, label_a: str = "i", label_b: str = "j"
    ) -> Any:
        """
        Builds an iterator over all ordered pairs with ``i != j``.

        Args:
            hilbert: Builder Hilbert space.
            label_a: First label bound in each row.
            label_b: Second label bound in each row.

        Returns:
            KBodyIteratorSpec: Iterator over all distinct ordered site pairs.
        """
        n = int(hilbert.size)
        pairs = tuple((i, j) for i in range(n) for j in range(n) if i != j)
        return KBodyIteratorSpec(labels=(str(label_a), str(label_b)), index_sets=pairs)


class ForEachIteratorClause(AbstractIteratorClause):
    """Built-in generic static K-body iterator clause (`builder.for_each(...)`)."""

    clause_name = "for_each"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(
        self,
        hilbert: DiscreteHilbertSpace,
        labels: Sequence[str],
        *,
        over: Sequence[Sequence[int]],
    ) -> Any:
        """
        Builds an iterator from user-provided labels and static index rows.

        Args:
            hilbert: Builder Hilbert space (unused by this generic iterator).
            labels: Iterator label sequence.
            over: Static sequence of index rows.

        Returns:
            KBodyIteratorSpec: Iterator specification with normalized labels/rows.

        Raises:
            ValueError: If *over* is empty or row arity does not match labels.
        """
        del hilbert
        labels_t = tuple(str(label) for label in labels)
        k_arity = len(labels_t)
        index_sets = tuple(tuple(int(idx) for idx in row) for row in over)
        if not index_sets:
            raise NeuralqxValueError("for_each: over= must not be empty.")
        for row in index_sets:
            if len(row) != k_arity:
                raise NeuralqxValueError(
                    f"for_each: each tuple in over= must have length {k_arity} "
                    f"(one index per label); got length {len(row)}."
                )
        return KBodyIteratorSpec(labels=labels_t, index_sets=index_sets)


class ForEachTripletIteratorClause(AbstractIteratorClause):
    """Built-in convenience clause for static triplets."""

    clause_name = "for_each_triplet"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(
        self,
        hilbert: DiscreteHilbertSpace,
        label_a: str,
        label_b: str,
        label_c: str,
        *,
        over: Sequence[tuple[int, int, int]],
    ) -> Any:
        """
        Builds a static triplet iterator by delegating to :class:`ForEachIteratorClause`.

        Args:
            hilbert: Builder Hilbert space.
            label_a: First triplet label.
            label_b: Second triplet label.
            label_c: Third triplet label.
            over: Static triplet rows.

        Returns:
            KBodyIteratorSpec: Normalized triplet iterator specification.
        """
        return ForEachIteratorClause(self.builder).build_iterator(
            hilbert,
            (str(label_a), str(label_b), str(label_c)),
            over=over,
        )


class ForEachPlaquetteIteratorClause(AbstractIteratorClause):
    """Built-in convenience clause for static plaquettes (4-body rows)."""

    clause_name = "for_each_plaquette"
    """Fluent method name used to invoke this built-in iterator clause."""

    def build_iterator(
        self,
        hilbert: DiscreteHilbertSpace,
        label_a: str,
        label_b: str,
        label_c: str,
        label_d: str,
        *,
        over: Sequence[tuple[int, int, int, int]],
    ) -> Any:
        """
        Builds a static plaquette iterator by delegating to :class:`ForEachIteratorClause`.

        Args:
            hilbert: Builder Hilbert space.
            label_a: First plaquette label.
            label_b: Second plaquette label.
            label_c: Third plaquette label.
            label_d: Fourth plaquette label.
            over: Static plaquette rows.

        Returns:
            KBodyIteratorSpec: Normalized plaquette iterator specification.
        """
        return ForEachIteratorClause(self.builder).build_iterator(
            hilbert,
            (str(label_a), str(label_b), str(label_c), str(label_d)),
            over=over,
        )


_DEFAULT_ITERATOR_CLAUSES_REGISTERED = False
"""Whether built-in iterator clauses were already installed for this process."""

_DEFAULT_ITERATOR_CLAUSES_LOCK = threading.RLock()
"""Lock protecting one-time registration of built-in iterator clauses."""


def ensure_default_iterator_clause_registrations() -> None:
    """
    Registers all built-in iterator clauses exactly once.

    Returns:
        None
    """
    global _DEFAULT_ITERATOR_CLAUSES_REGISTERED

    with _DEFAULT_ITERATOR_CLAUSES_LOCK:
        if _DEFAULT_ITERATOR_CLAUSES_REGISTERED:
            return

        register_iterator_clause(GloballyIteratorClause, replace=True)
        register_iterator_clause(ForEachSiteIteratorClause, replace=True)
        register_iterator_clause(ForEachPairIteratorClause, replace=True)
        register_iterator_clause(ForEachDistinctPairIteratorClause, replace=True)
        register_iterator_clause(ForEachIteratorClause, replace=True)
        register_iterator_clause(ForEachTripletIteratorClause, replace=True)
        register_iterator_clause(ForEachPlaquetteIteratorClause, replace=True)

        _DEFAULT_ITERATOR_CLAUSES_REGISTERED = True


__all__ = [
    "ForEachDistinctPairIteratorClause",
    "ForEachIteratorClause",
    "ForEachPairIteratorClause",
    "ForEachPlaquetteIteratorClause",
    "ForEachSiteIteratorClause",
    "ForEachTripletIteratorClause",
    "GloballyIteratorClause",
    "ensure_default_iterator_clause_registrations",
]
