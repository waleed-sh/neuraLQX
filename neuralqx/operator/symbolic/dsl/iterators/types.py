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


"""Type helpers and coercion helpers for iterator clauses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import TypeAlias

from neuralqx.operator.symbolic.ir.term import KBodyIteratorSpec
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

IteratorLabels: TypeAlias = tuple[str, ...]
"""Canonical immutable tuple of iterator label names."""

IteratorIndexRows: TypeAlias = tuple[tuple[int, ...], ...]
"""Canonical immutable tuple of site-index rows."""

IteratorSpecTuple: TypeAlias = tuple[Sequence[str], Sequence[Sequence[int]]]
"""Two-tuple form accepted by iterator clauses: ``(labels, over)``."""


def normalize_index_rows(
    labels: IteratorLabels, index_sets: Sequence[Sequence[int]]
) -> IteratorIndexRows:
    """
    Normalizes a sequence of index rows into canonical immutable form.

    Args:
        labels: Iterator labels that define row arity.
        index_sets: Sequence of rows where each row contains site indices.

    Returns:
        IteratorIndexRows: Normalized tuple-of-tuples index rows.

    Raises:
        ValueError: If no rows are provided or row arity does not match labels.
    """
    k_arity = len(labels)
    rows = tuple(tuple(int(idx) for idx in row) for row in index_sets)
    if not rows:
        raise NeuralqxValueError(
            "Iterator clause must return at least one index tuple."
        )
    for row in rows:
        if len(row) != k_arity:
            raise NeuralqxValueError(
                f"Iterator clause produced tuple length {len(row)} for labels of length {k_arity}."
            )
    return rows


def coerce_iterator_spec(spec: Any) -> KBodyIteratorSpec:
    """
    Coerces user iterator-clause outputs into :class:`KBodyIteratorSpec`.

    Accepted shapes:
    1. ``KBodyIteratorSpec`` instance.
    2. ``(labels, over)`` two-tuple.
    3. Any object exposing ``labels`` and ``index_sets`` attributes.

    Args:
        spec: Iterator-clause output.

    Returns:
        KBodyIteratorSpec: Canonical iterator specification.

    Raises:
        TypeError: If *spec* cannot be interpreted as an iterator specification.
        ValueError: If the provided rows are empty or arity-mismatched.
    """
    if isinstance(spec, KBodyIteratorSpec):
        return spec

    if isinstance(spec, tuple) and len(spec) == 2:
        labels_raw, index_sets_raw = spec
        labels = tuple(str(label) for label in labels_raw)
        rows = normalize_index_rows(labels, index_sets_raw)
        return KBodyIteratorSpec(labels=labels, index_sets=rows)

    labels_attr = getattr(spec, "labels", None)
    index_sets_attr = getattr(spec, "index_sets", None)
    if labels_attr is not None and index_sets_attr is not None:
        labels = tuple(str(label) for label in labels_attr)
        rows = normalize_index_rows(labels, index_sets_attr)
        return KBodyIteratorSpec(labels=labels, index_sets=rows)

    raise NeuralqxTypeError(
        "Iterator clause must return KBodyIteratorSpec, (labels, over), "
        "or an object exposing labels/index_sets."
    )


__all__ = [
    "IteratorIndexRows",
    "IteratorLabels",
    "IteratorSpecTuple",
    "coerce_iterator_spec",
    "normalize_index_rows",
]
