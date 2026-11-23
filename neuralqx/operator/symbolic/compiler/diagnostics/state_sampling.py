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


"""Hilbert space sampling and validity helpers for diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax


def _safe_int(value: Any) -> int | None:
    """Converts one value to ``int`` when possible.

    Args:
        value: Candidate integer-like value.

    Returns:
        Integer value or ``None`` when conversion is not possible.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _indexable_state_count(hilbert: Any) -> int | None:
    """Reads cardinality only when Hilbert indexing is available.

    Args:
        hilbert: Discrete Hilbert space object.

    Returns:
        State count or ``None`` when indexing metadata is unavailable.
    """
    try:
        if not bool(getattr(hilbert, "is_indexable", True)):
            return None
        return _safe_int(getattr(hilbert, "n_states", None))
    except Exception:
        return None


def _state_rows(array_like: Any) -> np.ndarray | None:
    """Normalizes an array-like state container to shape ``[B, N]``.

    Args:
        array_like: Input state collection.

    Returns:
        2D NumPy array when normalization succeeds, otherwise ``None``.
    """
    try:
        arr = np.asarray(array_like)
    except Exception:  # pragma: no cover - defensive conversion guard
        return None
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    return None


def _random_source_states(hilbert: Any, cap: int) -> np.ndarray | None:
    """Request a reproducible sample using native bounded sampling.

    Args:
        hilbert: Discrete Hilbert space object.
        cap: Maximum number of source states to request.

    Returns:
        Sampled state rows or ``None`` when direct sampling is unavailable.
    """
    try:
        sampler = getattr(hilbert, "random_state", None)
        if not callable(sampler):
            return None
        key = jax.random.PRNGKey(0)
        rows = _state_rows(sampler(key, size=cap))
    except Exception:
        return None
    if rows is None or rows.shape[0] == 0:
        return None
    return rows[:cap]


def sample_source_states(
    hilbert: Any,
    *,
    max_samples: int,
) -> np.ndarray | None:
    """Samples source states used by dynamic connectivity diagnostics.

    Sampling priority:
    1. ``hilbert.random_state(...)`` with a fixed local seed
    2. A bounded ``hilbert.numbers_to_states(...)`` prefix when indexable
    3. ``hilbert.all_states()`` when its known cardinality fits the sample cap

    Native random-state generation uses bounded sampling attempts.
    Source sampling does not require the Hilbert space to be indexable.

    Args:
        hilbert: Discrete Hilbert space object.
        max_samples: Maximum number of states to return.

    Returns:
        Sampled state batch of shape ``[B, N]`` or ``None`` when unavailable.
    """
    cap = max(0, int(max_samples))
    if cap == 0:
        return None

    rows = _random_source_states(hilbert, cap)
    if rows is not None:
        return rows

    n_states = _indexable_state_count(hilbert)
    if n_states is None or n_states <= 0:
        return None

    try:
        numbers = np.arange(min(cap, n_states), dtype=np.int64)
        rows = _state_rows(hilbert.numbers_to_states(numbers))
        if rows is not None and rows.shape[0] > 0:
            return rows[:cap]
    except Exception:
        pass

    if n_states > cap:
        return None
    try:
        rows = _state_rows(hilbert.all_states())
    except Exception:
        return None
    if rows is None or rows.shape[0] == 0:
        return None
    return rows[:cap]


def build_hilbert_support_lookup(
    hilbert: Any,
    *,
    max_states: int,
) -> set[tuple[Any, ...]] | None:
    """Builds exact Hilbert support lookup for membership checks.

    Args:
        hilbert: Discrete Hilbert space object.
        max_states: Maximum Hilbert cardinality accepted for exact lookup.

    Returns:
        Set of valid state tuples when feasible, else ``None``.
    """
    cap = max(0, int(max_states))
    if cap == 0:
        return None
    n_states = _indexable_state_count(hilbert)
    if n_states is None or n_states <= 0 or n_states > cap:
        return None
    try:
        states = _state_rows(hilbert.all_states())
    except Exception:
        return None
    if states is None:
        return None
    return {tuple(np.asarray(row).tolist()) for row in states}


def evaluate_constraint_accepts_state(
    hilbert: Any,
    state: np.ndarray,
) -> bool | None:
    """Evaluates one Hilbert constraint predicate for a candidate state.

    Args:
        hilbert: Discrete Hilbert space object.
        state: Candidate connected state, shape ``[N]``.

    Returns:
        ``True``/``False`` when the constraint predicate is available.
        ``None`` when no usable constraint API is available.
    """
    constraint = getattr(hilbert, "constraint", None)
    if constraint is None or not callable(constraint):
        return None
    try:
        value = constraint(np.asarray(state)[None, :])
    except Exception:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return bool(arr[0])


def illegal_local_state_positions(
    state: np.ndarray,
    local_states: Any,
    *,
    hilbert: Any = None,
) -> tuple[int, ...]:
    """Finds indices with values outside the Hilbert local-state basis.

    Args:
        state: Candidate connected state, shape ``[N]``.
        local_states: Hilbert local-state sequence.

    Returns:
        Tuple of illegal site indices. Empty when all values are valid.
    """
    spaces = getattr(hilbert, "local_spaces", None)
    if spaces is not None:
        from neuralqx.operator.symbolic._hilbert import illegal_local_positions

        return illegal_local_positions(hilbert, state)
    if local_states is None:
        return ()
    basis = np.asarray(local_states)
    if basis.ndim != 1 or basis.size == 0:
        return ()

    flat = np.asarray(state).reshape(-1)
    illegal: list[int] = []
    for idx, value in enumerate(flat):
        matches = np.isclose(value, basis)
        if not bool(np.any(matches)):
            illegal.append(int(idx))
    return tuple(illegal)


def state_to_tuple(state: np.ndarray) -> tuple[Any, ...]:
    """Converts one state vector to a hashable tuple.

    Args:
        state: State vector.

    Returns:
        Hashable state tuple.
    """
    return tuple(np.asarray(state).tolist())


__all__ = [
    "build_hilbert_support_lookup",
    "evaluate_constraint_accepts_state",
    "illegal_local_state_positions",
    "sample_source_states",
    "state_to_tuple",
]
