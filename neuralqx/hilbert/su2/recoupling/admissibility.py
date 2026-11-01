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


"""SU(2) doubled-spin admissibility in a left-comb recoupling basis.

The functions in this module operate on doubled-spin labels, so integer parity
encodes half-integer spin structure. Python helpers support enumeration and
diagnostics, while JAX helpers support vectorized constraint and proposal
kernels.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import jax
import jax.numpy as jnp


def coupling_ok_py(a: int, b: int, c: int) -> bool:
    """Returns whether doubled spins can form an SU(2) trivalent node.

    Args:
        a: First doubled-spin label.
        b: Second doubled-spin label.
        c: Candidate coupled doubled-spin label.

    Returns:
        ``True`` when the labels satisfy the triangle inequalities and parity
        rule for SU(2) coupling.
    """
    a_i = int(a)
    b_i = int(b)
    c_i = int(c)
    return abs(a_i - b_i) <= c_i <= a_i + b_i and ((a_i + b_i + c_i) & 1) == 0


def coupling_ok(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Evaluates SU(2) trivalent admissibility with JAX arrays.

    Args:
        a: First doubled-spin label or array of labels.
        b: Second doubled-spin label or array of labels.
        c: Candidate coupled doubled-spin label or array of labels.

    Returns:
        Boolean JAX array applying the same triangle and parity rules as
        :func:`coupling_ok_py`.
    """
    a_i = jnp.asarray(a, dtype=jnp.int32)
    b_i = jnp.asarray(b, dtype=jnp.int32)
    c_i = jnp.asarray(c, dtype=jnp.int32)
    triangle = (jnp.abs(a_i - b_i) <= c_i) & (c_i <= a_i + b_i)
    parity = ((a_i + b_i + c_i) % 2) == 0
    return triangle & parity


def expected_channel_count(valence: int) -> int:
    """Returns the number of left-comb intermediate labels for a vertex.

    Args:
        valence: Number of edge legs incident on the vertex.

    Returns:
        Number of internal recoupling channels required by the left-comb basis.

    Raises:
        ValueError: If ``valence`` is negative.
    """
    valence_i = int(valence)
    if valence_i < 0:
        raise ValueError("valence must be non-negative.")
    return max(valence_i - 3, 0)


def left_comb_valid_py(edge_two_j: Iterable[int], channels: Iterable[int]) -> bool:
    """Validates one vertex in the left-comb basis using Python integers.

    Args:
        edge_two_j: Doubled-spin labels on the edges incident to the vertex.
        channels: Candidate left-comb intermediate labels.

    Returns:
        ``True`` when the edge labels and channel labels form an admissible
        gauge-invariant vertex.
    """
    edges = tuple(int(x) for x in edge_two_j)
    ch = tuple(int(x) for x in channels)
    valence = len(edges)
    if len(ch) != expected_channel_count(valence):
        return False
    if valence == 0:
        return True
    if valence == 1:
        return edges[0] == 0
    if valence == 2:
        return edges[0] == edges[1]
    if valence == 3:
        return coupling_ok_py(edges[0], edges[1], edges[2])

    if not coupling_ok_py(edges[0], edges[1], ch[0]):
        return False
    for idx in range(1, len(ch)):
        if not coupling_ok_py(ch[idx - 1], edges[idx + 1], ch[idx]):
            return False
    return coupling_ok_py(ch[-1], edges[-2], edges[-1])


def left_comb_valid(edge_two_j: jax.Array, channels: jax.Array) -> jax.Array:
    """Validates one vertex in a JAX-compatible left-comb basis.

    Args:
        edge_two_j: One-dimensional array of incident edge doubled-spin labels.
        channels: One-dimensional array of left-comb channel labels.

    Returns:
        Boolean JAX scalar indicating whether the vertex satisfies the SU(2)
        coupling rules in the configured left-comb order.
    """
    edges = jnp.asarray(edge_two_j, dtype=jnp.int32)
    ch = jnp.asarray(channels, dtype=jnp.int32)
    valence = int(edges.shape[0])
    expected = expected_channel_count(valence)
    if int(ch.shape[0]) != expected:
        return jnp.asarray(False)
    if valence == 0:
        return jnp.asarray(True)
    if valence == 1:
        return edges[0] == 0
    if valence == 2:
        return edges[0] == edges[1]
    if valence == 3:
        return coupling_ok(edges[0], edges[1], edges[2])

    ok = coupling_ok(edges[0], edges[1], ch[0])
    for idx in range(1, expected):
        ok = ok & coupling_ok(ch[idx - 1], edges[idx + 1], ch[idx])
    return ok & coupling_ok(ch[-1], edges[-2], edges[-1])


def allowed_couplings(a: int, b: int, max_label: int) -> tuple[int, ...]:
    """Returns allowed coupled doubled-spin labels for two inputs.

    Args:
        a: First doubled-spin label.
        b: Second doubled-spin label.
        max_label: Upper bound for returned coupled labels.

    Returns:
        Tuple of doubled-spin labels ``c`` that satisfy SU(2) coupling with
        ``a`` and ``b`` and do not exceed ``max_label``.
    """
    lo = abs(int(a) - int(b))
    hi = min(int(a) + int(b), int(max_label))
    if hi < lo:
        return ()
    parity = (int(a) + int(b)) & 1
    return tuple(c for c in range(lo, hi + 1) if (c & 1) == parity)


def first_left_comb_channels(
    edge_two_j: Iterable[int],
    channel_max_labels: Iterable[int],
) -> tuple[int, ...] | None:
    """Returns the first admissible left-comb channel assignment.

    Args:
        edge_two_j: Doubled-spin labels on incident edges.
        channel_max_labels: Maximum allowed label for each channel site.

    Returns:
        Lexicographically first admissible channel tuple, an empty tuple for a
        valid vertex with no channels, or ``None`` when no admissible assignment
        exists.
    """
    edges = tuple(int(x) for x in edge_two_j)
    maxima = tuple(int(x) for x in channel_max_labels)
    valence = len(edges)
    if len(maxima) != expected_channel_count(valence):
        return None
    if valence <= 3:
        return () if left_comb_valid_py(edges, ()) else None

    n_channels = len(maxima)

    @lru_cache(maxsize=None)
    def feasible(idx: int, previous: int) -> bool:
        """Checks whether the remaining channel suffix has a solution."""
        if idx == n_channels:
            return coupling_ok_py(previous, edges[-2], edges[-1])
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        return any(
            feasible(idx + 1, candidate)
            for candidate in allowed_couplings(source, partner, maxima[idx])
        )

    out: list[int] = []
    previous = 0
    for idx in range(n_channels):
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        for candidate in allowed_couplings(source, partner, maxima[idx]):
            if feasible(idx + 1, candidate):
                out.append(candidate)
                previous = candidate
                break
        else:
            return None
    return tuple(out)


def count_left_comb_channels(
    edge_two_j: Iterable[int],
    channel_max_labels: Iterable[int],
) -> int:
    """Counts admissible channel assignments for fixed edge labels.

    Args:
        edge_two_j: Doubled-spin labels on incident edges.
        channel_max_labels: Maximum allowed label for each channel site.

    Returns:
        Number of left-comb channel tuples that satisfy all vertex coupling
        rules for the supplied edge sector.
    """
    edges = tuple(int(x) for x in edge_two_j)
    maxima = tuple(int(x) for x in channel_max_labels)
    valence = len(edges)
    if len(maxima) != expected_channel_count(valence):
        return 0
    if valence <= 3:
        return int(left_comb_valid_py(edges, ()))

    n_channels = len(maxima)

    @lru_cache(maxsize=None)
    def count_from(idx: int, previous: int) -> int:
        """Counts admissible channel suffixes from one intermediate label."""
        if idx == n_channels:
            return int(coupling_ok_py(previous, edges[-2], edges[-1]))
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        return sum(
            count_from(idx + 1, candidate)
            for candidate in allowed_couplings(source, partner, maxima[idx])
        )

    return count_from(0, 0)


def all_left_comb_channels(
    edge_two_j: Iterable[int],
    channel_max_labels: Iterable[int],
    *,
    limit: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Enumerates admissible left-comb channel tuples for fixed edge labels.

    Args:
        edge_two_j: Doubled-spin labels on incident edges.
        channel_max_labels: Maximum allowed label for each channel site.
        limit: Optional maximum number of tuples to return.

    Returns:
        Tuple of admissible channel tuples in lexicographic construction order.
        An empty tuple is returned when the sector is inadmissible or the limit
        excludes enumeration.
    """
    if limit is not None and limit < 1:
        return ()
    edges = tuple(int(x) for x in edge_two_j)
    maxima = tuple(int(x) for x in channel_max_labels)
    valence = len(edges)
    if len(maxima) != expected_channel_count(valence):
        return ()
    if valence <= 3:
        return ((),) if left_comb_valid_py(edges, ()) else ()

    n_channels = len(maxima)

    @lru_cache(maxsize=None)
    def feasible(idx: int, previous: int) -> bool:
        """Checks whether the remaining channel suffix has a solution."""
        if idx == n_channels:
            return coupling_ok_py(previous, edges[-2], edges[-1])
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        return any(
            feasible(idx + 1, candidate)
            for candidate in allowed_couplings(source, partner, maxima[idx])
        )

    out: list[tuple[int, ...]] = []

    def visit(idx: int, previous: int, prefix: tuple[int, ...]) -> None:
        """Depth-first enumeration of admissible channel assignments."""
        if limit is not None and len(out) >= limit:
            return
        if idx == n_channels:
            if coupling_ok_py(previous, edges[-2], edges[-1]):
                out.append(prefix)
            return
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        for candidate in allowed_couplings(source, partner, maxima[idx]):
            if feasible(idx + 1, candidate):
                visit(idx + 1, candidate, (*prefix, candidate))

    visit(0, 0, ())
    return tuple(out)


__all__ = [
    "allowed_couplings",
    "all_left_comb_channels",
    "count_left_comb_channels",
    "coupling_ok",
    "coupling_ok_py",
    "expected_channel_count",
    "first_left_comb_channels",
    "left_comb_valid",
    "left_comb_valid_py",
]
