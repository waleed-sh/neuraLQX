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


"""Exact, shape-preserving reduction of discrete operator connections.

These routines never use a floating point tolerance or a probabilistic hash.
They simplify *computed* rows; they cannot prove that an opaque operator kernel
has a zero branch before evaluating that kernel.
"""

from functools import partial

import jax.numpy as jnp

import jax


@jax.jit
def coalesce_connections(states, mels):
    """Sum identical targets in each row, then pack nonzero sums to the front.

    Accept ``(..., connections, sites)`` states and ``(..., connections)``
    amplitudes. Shapes are unchanged, including inside JIT. Zero slots remain
    padding. NaNs and arbitrarily small nonzero amplitudes are retained. The
    order of nonzero targets is lexicographic, not the input branch order.
    Support selection assumes fixed operator coefficients; use raw padded
    rows when differentiating coefficients through a cancellation point.
    """
    states, mels = jnp.asarray(states), jnp.asarray(mels)
    if states.shape[:-1] != mels.shape or mels.ndim < 1:
        raise ValueError("Expected states (..., connections, sites) and matching mels.")
    if mels.shape[-1] == 0:
        return states, mels

    def row(x, m):
        order = jnp.lexsort(tuple(x[:, i] for i in range(x.shape[-1] - 1, -1, -1)))
        x, m = x[order], m[order]
        starts = jnp.concatenate((jnp.ones(1, bool), jnp.any(x[1:] != x[:-1], axis=-1)))
        ids = jax.lax.associative_scan(
            jnp.maximum, jnp.where(starts, jnp.arange(len(m)), 0)
        )
        sums = jnp.zeros_like(m).at[ids].add(m)
        positions = jnp.nonzero(sums != 0, size=len(m), fill_value=0)[0]
        active = jnp.arange(len(m)) < jnp.count_nonzero(sums)
        return x[positions], jnp.where(active, sums[positions], 0)

    x, m = jax.vmap(row)(
        states.reshape((-1, *states.shape[-2:])), mels.reshape((-1, mels.shape[-1]))
    )
    return x.reshape(states.shape), m.reshape(mels.shape)


@partial(jax.jit, static_argnames=("local_sizes",))
def pack_local_indices(indices, *, local_sizes):
    """Losslessly pack bounded local indices into uint32 words (no basis rank).

    Each site's bits stay within a word. Spaces with a site requiring more
    than 32 bits use their original index columns instead. This works for
    constrained and non-indexable spaces without enumerating their basis.
    """
    widths = tuple((int(size) - 1).bit_length() for size in local_sizes)
    if max(widths, default=0) > 32:
        return indices
    words, offset = [], 0
    word = jnp.zeros(indices.shape[:-1], dtype=jnp.uint32)
    for site, width in enumerate(widths):
        if offset + width > 32:
            words.append(word)
            word, offset = jnp.zeros_like(word), 0
        if width:
            word = word | (indices[..., site].astype(jnp.uint32) << offset)
            offset += width
    return jnp.stack((*words, word), axis=-1)


__all__ = ["coalesce_connections"]
