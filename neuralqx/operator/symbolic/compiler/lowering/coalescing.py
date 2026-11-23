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


"""Exact JAX assembly kernels selected by static connectivity analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.symbolic.compiler.core.connectivity import (
    ConnectivityDeduplicationPlan,
)
from neuralqx.operator.symbolic.compiler.core.connectivity import collision_strategy


def _proofs_apply(dtype: Any, plan: ConnectivityDeduplicationPlan) -> bool:
    """Checks that the configuration dtype exactly represents proof intermediates."""
    if (
        not -plan.proof_integer_limit
        <= plan.proof_min
        <= plan.proof_max
        <= plan.proof_integer_limit
    ):
        return False
    if jnp.issubdtype(dtype, jnp.integer):
        info = np.iinfo(dtype)
        return bool(info.min <= plan.proof_min and plan.proof_max <= info.max)
    if jnp.issubdtype(dtype, jnp.floating):
        exact = 2 ** (jnp.finfo(dtype).nmant + 1)  # type: ignore[no-untyped-call]
        return bool(-exact <= plan.proof_min and plan.proof_max <= exact)
    return False


def compact_connections(
    xp: Any, mels: Any, output_size: int | None = None
) -> tuple[Any, Any]:
    """Stably packs nonzero entries using a prefix sum and one destination map.

    Args:
        xp: Target configurations with shape ``(K, H)``.
        mels: Matrix elements with shape ``(K,)``.
        output_size: Proven output capacity. Defaults to ``K``.

    Returns:
        Nonzero connections followed by zero padding at the requested capacity.
    """
    size = mels.shape[0] if output_size is None else output_size
    if mels.shape[0] == 0:
        return jnp.zeros((size, xp.shape[-1]), xp.dtype), jnp.zeros((size,), mels.dtype)
    if mels.shape[0] == size == 1:
        return jnp.where((mels != 0)[:, None], xp, jnp.zeros_like(xp)), mels
    keep = mels != 0
    rank = jnp.cumsum(keep, dtype=jnp.int32) - 1
    destinations = jnp.where(keep, rank, size)
    # All in-bounds destinations are unique. Out-of-bounds entries are dropped;
    # no nonzero value can be overwritten by an inactive branch.
    sources = jnp.full((size,), mels.shape[0], dtype=jnp.int32)
    sources = sources.at[destinations].set(
        jnp.arange(mels.shape[0], dtype=jnp.int32), mode="drop"
    )
    packed_xp = jnp.take(xp, sources, axis=0, mode="fill", fill_value=0)
    packed_mels = jnp.take(mels, sources, axis=0, mode="fill", fill_value=0)
    return packed_xp, packed_mels


def _pairwise_mels(keys: Any, mels: Any) -> Any:
    size = mels.shape[0]
    ids = jnp.arange(size, dtype=jnp.int32)
    equal = jnp.all(keys[:, None, :] == keys[None, :, :], axis=-1)
    # Include the diagonal explicitly so NaN-containing targets remain separate
    # entries, as they do in the sorting path.
    equal = equal | (ids[:, None] == ids[None, :])
    leaders = jnp.min(jnp.where(equal, ids[None, :], size), axis=1)
    return jnp.zeros_like(mels).at[leaders].add(mels)


def _sorted_mels(keys: Any, mels: Any) -> Any:
    size = mels.shape[0]
    if keys.shape[-1] == 0:
        return jnp.zeros_like(mels).at[0].set(jnp.sum(mels))
    # Stable lexicographic sorting makes the first member of each equal-key run
    # its first original occurrence. Scattering sums back to that occurrence
    # removes the need for an inverse permutation or a second ordering sort.
    perm = jnp.lexsort(keys.T[::-1]).astype(jnp.int32)
    sorted_keys = keys[perm]
    start = jnp.concatenate(
        (
            jnp.ones((1,), dtype=bool),
            jnp.any(sorted_keys[1:] != sorted_keys[:-1], axis=1),
        )
    )
    ids = jnp.arange(size, dtype=jnp.int32)
    run_starts = jax.lax.associative_scan(jnp.maximum, jnp.where(start, ids, 0))
    leaders = perm[run_starts]
    return jnp.zeros_like(mels).at[leaders].add(mels[perm])


def _workspace_mels(keys: Any, mels: Any, size: int) -> Any:
    """Accumulates a small proven key domain in a direct-address workspace."""
    count = mels.shape[0]
    buckets = (
        keys[:, 0].astype(jnp.int32)
        if keys.shape[-1]
        else jnp.zeros((count,), jnp.int32)
    )
    first = jnp.full((size,), count, dtype=jnp.int32)
    first = first.at[buckets].min(jnp.arange(count, dtype=jnp.int32))
    sums = jnp.zeros((size,), dtype=mels.dtype).at[buckets].add(mels)
    return jnp.zeros_like(mels).at[first].add(sums, mode="drop")


def coalesce_mels(keys: Any, mels: Any, *, key_bits: int | None = None) -> Any:
    """Aggregates equal exact keys into their first original occurrence."""
    if mels.shape[0] <= 1:
        return mels
    strategy = collision_strategy(mels.shape[0], keys.shape[-1], key_bits=key_bits)
    if strategy == "workspace":
        assert key_bits is not None
        return _workspace_mels(keys, mels, 2**key_bits)
    if strategy == "pairwise":
        return _pairwise_mels(keys, mels)
    return _sorted_mels(keys, mels)


def pack_integer_keys(keys: Any, ranges: tuple[tuple[int, int], ...]) -> Any:
    """Packs proven coordinate ranges into collision-free 32-bit words.

    This is an injective bit-field encoding, not a probabilistic fingerprint.
    Every coordinate retains all bits required by its statically proven range.
    """
    bit_count = sum((hi - lo).bit_length() for lo, hi in ranges)
    words = [
        jnp.zeros((keys.shape[0],), dtype=jnp.uint32)
        for _ in range((bit_count + 31) // 32)
    ]
    offset = 0
    for site, (lo, hi) in enumerate(ranges):
        bits = (hi - lo).bit_length()
        if not bits:
            continue
        value = keys[:, site]
        if jnp.issubdtype(value.dtype, jnp.floating):
            value = value.astype(jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
        value = value.astype(jnp.uint32) - jnp.uint32(lo % 2**32)
        word, shift = divmod(offset, 32)
        words[word] = words[word] | (value << jnp.uint32(shift))
        if shift + bits > 32:
            words[word + 1] = words[word + 1] | (value >> jnp.uint32(32 - shift))
        offset += bits
    return (
        jnp.stack(words, axis=1)
        if words
        else jnp.zeros((keys.shape[0], 0), dtype=jnp.uint32)
    )


def coalesce_connected_components(xp: Any, mels: Any, valids: Any) -> tuple[Any, Any]:
    """Coalesces arbitrary targets without static connectivity information."""
    weighted = jnp.where(valids, mels, jnp.zeros_like(mels))
    return compact_connections(xp, coalesce_mels(xp, weighted))


def make_planned_coalescer(
    plan: ConnectivityDeduplicationPlan,
) -> Callable[[Any, Any, Any], tuple[Any, Any]]:
    """Builds a partially evaluated assembly kernel from one static plan.

    Args:
        plan: Static target classes and residual collision obligations.

    Returns:
        Callable accepting target rows, matrix elements, and validity flags.
    """
    finish = make_grouped_coalescer(plan)
    representatives = np.asarray([g[0] for g in plan.groups], dtype=np.int32)
    direct = len(plan.groups) == plan.raw_size and all(
        g == (i,) for i, g in enumerate(plan.groups)
    )
    by_width: dict[int, list[int]] = {}
    for i, group in enumerate(plan.groups):
        by_width.setdefault(len(group), []).append(i)
    buckets = tuple(
        (
            np.asarray(indices, dtype=np.int32),
            np.asarray([plan.groups[i] for i in indices], dtype=np.int32),
        )
        for indices in by_width.values()
    )
    bucket_order = [i for indices, _ in buckets for i in indices]
    restore = np.argsort(bucket_order).astype(np.int32)

    def coalesce(xp: Any, mels: Any, valids: Any) -> tuple[Any, Any]:
        if not plan.groups:
            return (
                jnp.zeros((plan.output_size, xp.shape[-1]), xp.dtype),
                jnp.zeros((plan.output_size,), mels.dtype),
            )
        weighted = jnp.where(valids, mels, jnp.zeros_like(mels))
        if direct:
            grouped_xp, grouped_mels = xp, weighted
        else:
            grouped_xp = xp[representatives]
            parts = []
            for _, members in buckets:
                values = weighted[members]
                parts.append(
                    values[:, 0] if members.shape[1] == 1 else jnp.sum(values, axis=1)
                )
            grouped_mels = jnp.concatenate(parts)[restore]
        return finish(grouped_xp, grouped_mels)

    return coalesce


def make_grouped_coalescer(
    plan: ConnectivityDeduplicationPlan,
) -> Callable[[Any, Any], tuple[Any, Any]]:
    """Builds exact assembly for targets whose amplitudes are already grouped.

    Args:
        plan: Target classes, collision obligations, and output capacity.

    Returns:
        Callable accepting one target and typed amplitude per target class.
    """

    def finish(grouped_xp: Any, grouped_mels: Any) -> tuple[Any, Any]:
        compatible = _proofs_apply(grouped_xp.dtype, plan)
        if compatible:
            for indices, columns, ranges in zip(
                plan.collision_components,
                plan.key_columns,
                plan.key_ranges,
                strict=False,
            ):
                idx = np.asarray(indices, dtype=np.int32)
                cols = np.asarray(columns, dtype=np.int32)
                whole = indices == tuple(range(len(plan.groups)))
                if whole:
                    keys = (
                        grouped_xp
                        if columns == tuple(range(grouped_xp.shape[-1]))
                        else grouped_xp[:, cols]
                    )
                    component_mels = grouped_mels
                else:
                    keys = grouped_xp[idx[:, None], cols[None, :]]
                    component_mels = grouped_mels[idx]
                if ranges is not None:
                    keys = pack_integer_keys(keys, ranges)
                key_bits = (
                    sum((hi - lo).bit_length() for lo, hi in ranges)
                    if ranges is not None
                    else None
                )
                component_mels = coalesce_mels(keys, component_mels, key_bits=key_bits)
                grouped_mels = (
                    component_mels
                    if whole
                    else grouped_mels.at[idx].set(
                        component_mels,
                        indices_are_sorted=True,
                        unique_indices=True,
                    )
                )
        else:
            # Overflow, low precision, and non-real configuration dtypes cannot
            # consume exact-integer disjointness proofs. Static expression
            # equivalence and its reduced output bound remain valid.
            grouped_mels = coalesce_mels(grouped_xp, grouped_mels)
        if plan.compact or not compatible:
            return compact_connections(grouped_xp, grouped_mels, plan.output_size)
        return grouped_xp, grouped_mels

    return finish


__all__ = [
    "coalesce_connected_components",
    "compact_connections",
    "make_grouped_coalescer",
    "make_planned_coalescer",
]
