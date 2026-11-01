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


"""Loop-sector SU(2) proposal kernels.

These kernels construct valid spin-network proposals on device. They adjust
edge doubled-spin labels, complete compatible vertex channels, and keep invalid
attempts from replacing the current state.
"""

from __future__ import annotations

from itertools import product

import jax.numpy as jnp

import jax
from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.su2.recoupling import coupling_ok

from .moves import EdgeSpinHeatbathMove
from .moves import LoopFundamentalMove
from .moves import MixedSU2Move
from .moves import VertexRecouplingMove


def graph_loop_edge_indices(graph: AbstractGraph) -> tuple[tuple[int, ...], ...]:
    """Resolves graph minimal loops to edge-index tuples.

    Args:
        graph: Graph whose minimal loops should be converted into flat edge
            indices.

    Returns:
        Tuple of loop descriptors. Each descriptor contains unique edge indices
        in the graph's edge order and can be used by SU(2) loop proposal
        kernels.
    """
    loops: list[tuple[int, ...]] = []
    for loop in graph.minimal_loops():
        edges = loop.edges if hasattr(loop, "edges") else tuple(loop)
        indices = tuple(dict.fromkeys(graph.edge_to_index(edge) for edge in edges))
        if indices:
            loops.append(indices)
    return tuple(loops)


def complete_states_from_edges(
    space, edge_values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Completes edge labels to the first admissible channel assignment.

    Args:
        space: SU(2) spin-network Hilbert space providing graph metadata and
            channel maxima.
        edge_values: Edge doubled-spin labels with trailing dimension
            ``space.n_edges``.

    Returns:
        Pair ``(states, valid)``. ``states`` contains completed flat
        spin-network states when completion succeeds and zeros otherwise.
        ``valid`` marks rows whose edge labels admit compatible left-comb
        channels at every vertex.
    """
    edges = jnp.asarray(edge_values, dtype=jnp.int32)
    single = edges.ndim == 1
    edge_rows = edges[None, :] if single else edges.reshape((-1, space.n_edges))
    batch = edge_rows.shape[0]

    state = jnp.zeros((batch, space.size), dtype=space.dtype)
    state = state.at[:, : space.n_edges].set(edge_rows.astype(space.dtype))
    valid = jnp.all((edge_rows >= 0) & (edge_rows <= int(space.two_j_max)), axis=-1)

    for vertex, edge_sites in enumerate(space.metadata.incident_edges_by_vertex):
        local_edges = edge_rows[:, jnp.asarray(edge_sites, dtype=jnp.int32)]
        channel_maxima = space.layout.vertex_channel_max_labels[vertex]
        channel_sites = space.layout.vertex_channel_sites[vertex]
        channels, ok = _first_left_comb_channels(local_edges, channel_maxima)
        valid = valid & ok
        if channel_sites:
            state = state.at[:, jnp.asarray(channel_sites, dtype=jnp.int32)].set(
                channels.astype(space.dtype)
            )

    zero = jnp.zeros_like(state)
    completed = jnp.where(valid[:, None], state, zero)
    return (completed[0] if single else completed), (valid[0] if single else valid)


def propose_loop(
    space,
    states: jax.Array,
    key: jax.Array,
    move: LoopFundamentalMove,
) -> jax.Array:
    """Applies valid fundamental loop proposals on device.

    Args:
        space: SU(2) spin-network Hilbert space being updated.
        states: Two-dimensional batch of valid flat states.
        key: JAX pseudo-random key.
        move: Loop proposal descriptor controlling fixed loop, direction, and
            retry count.

    Returns:
        Proposed state batch. Rows keep their original value when every
        attempted loop shift violates edge bounds or vertex admissibility.
    """
    arr = jnp.asarray(states, dtype=space.dtype)
    if not space.loop_edge_indices or space.two_j_max == 0:
        return arr

    batch = arr.shape[0]
    loop_sites, loop_mask = _loop_table(space)
    keys = jax.random.split(key, 2 * int(move.max_attempts))
    out = arr
    accepted = jnp.zeros((batch,), dtype=jnp.bool_)

    for attempt in range(int(move.max_attempts)):
        key_loop = keys[2 * attempt]
        key_dir = keys[2 * attempt + 1]
        loop_ids = _loop_ids(space, batch, key_loop, move.loop)
        directions = _directions(batch, key_dir, move.direction)
        selected_sites = loop_sites[loop_ids]
        selected_mask = loop_mask[loop_ids]
        safe_sites = jnp.where(selected_mask, selected_sites, 0)

        candidate_edges = arr[:, : space.n_edges]
        rows = jnp.arange(batch, dtype=jnp.int32)[:, None]
        increments = directions[:, None] * selected_mask.astype(jnp.int32)
        candidate_edges = candidate_edges.at[rows, safe_sites].add(increments)
        completed, valid = complete_states_from_edges(space, candidate_edges)
        take = (~accepted) & valid
        out = jnp.where(take[:, None], completed, out)
        accepted = accepted | take

    return out.astype(space.dtype)


def propose_vertex_recoupling(
    space,
    states: jax.Array,
    key: jax.Array,
    move: VertexRecouplingMove,
) -> jax.Array:
    """Resamples admissible recoupling channels at one vertex per row.

    Args:
        space: SU(2) spin-network Hilbert space being updated.
        states: Two-dimensional batch of valid flat states.
        key: JAX pseudo-random key.
        move: Vertex proposal descriptor selecting a fixed vertex or allowing a
            random vertex per row.

    Returns:
        Proposed state batch with edge labels unchanged. Rows whose selected
        vertex has no channel sites, or no admissible alternative in the
        candidate table, keep their original channel values.
    """
    arr = jnp.asarray(states, dtype=space.dtype)
    batch = arr.shape[0]
    key_vertex, key_choice = jax.random.split(key)
    if move.vertex is None:
        vertices = jax.random.randint(
            key_vertex, (batch,), 0, space.n_vertices, dtype=jnp.int32
        )
    else:
        vertices = jnp.full(
            (batch,),
            space.layout.vertex_position(move.vertex),
            dtype=jnp.int32,
        )
    choice_ids = jax.random.randint(
        key_choice,
        (batch,),
        0,
        jnp.iinfo(jnp.int32).max,
        dtype=jnp.int32,
    )

    out = arr
    for vertex, edge_sites in enumerate(space.metadata.incident_edges_by_vertex):
        channel_sites = space.layout.vertex_channel_sites[vertex]
        if not channel_sites:
            continue
        candidates = _channel_candidate_table(
            space.layout.vertex_channel_max_labels[vertex]
        )
        local_edges = arr[:, jnp.asarray(edge_sites, dtype=jnp.int32)]
        valid_candidates = _valid_channel_candidates(local_edges, candidates)
        counts = jnp.sum(valid_candidates, axis=-1).astype(jnp.int32)
        selected_id = jnp.mod(choice_ids, jnp.maximum(counts, 1))
        order = jnp.cumsum(valid_candidates.astype(jnp.int32), axis=-1) - 1
        selected_mask = valid_candidates & (order == selected_id[:, None])
        selected = selected_mask.astype(jnp.int32) @ candidates
        candidate_state = out.at[:, jnp.asarray(channel_sites, dtype=jnp.int32)].set(
            selected.astype(space.dtype)
        )
        take = (vertices == vertex) & (counts > 0)
        out = jnp.where(take[:, None], candidate_state, out)
    return out.astype(space.dtype)


def propose_edge_heatbath(
    space,
    states: jax.Array,
    key: jax.Array,
    move: EdgeSpinHeatbathMove,
) -> jax.Array:
    """Tries edge-label heatbath proposals and completes channels.

    Args:
        space: SU(2) spin-network Hilbert space being updated.
        states: Two-dimensional batch of valid flat states.
        key: JAX pseudo-random key.
        move: Edge heatbath descriptor controlling fixed edge, current-label
            handling, and retry count.

    Returns:
        Proposed state batch. A row changes only when a sampled edge-spin
        sector can be completed to admissible channels at all vertices.
    """
    arr = jnp.asarray(states, dtype=space.dtype)
    batch = arr.shape[0]
    key_edge, key_label = jax.random.split(key)
    if move.edge is None:
        edge_ids = jax.random.randint(
            key_edge,
            (batch, int(move.max_attempts)),
            0,
            space.n_edges,
            dtype=jnp.int32,
        )
    else:
        edge = (
            int(move.edge)
            if isinstance(move.edge, int)
            else space.graph.edge_to_index(move.edge)
        )
        edge_ids = jnp.full((batch, int(move.max_attempts)), edge, dtype=jnp.int32)
    labels = jax.random.randint(
        key_label,
        (batch, int(move.max_attempts)),
        0,
        int(space.two_j_max) + 1,
        dtype=jnp.int32,
    )

    out = arr
    accepted = jnp.zeros((batch,), dtype=jnp.bool_)
    rows = jnp.arange(batch, dtype=jnp.int32)
    original_edges = arr[:, : space.n_edges]
    for attempt in range(int(move.max_attempts)):
        edge = edge_ids[:, attempt]
        label = labels[:, attempt]
        current = original_edges[rows, edge]
        if not move.include_current:
            label = jnp.where(
                label == current,
                (label + 1) % (int(space.two_j_max) + 1),
                label,
            )
        candidate_edges = original_edges.at[rows, edge].set(label)
        completed, valid = complete_states_from_edges(space, candidate_edges)
        take = (~accepted) & valid
        out = jnp.where(take[:, None], completed, out)
        accepted = accepted | take
    return out.astype(space.dtype)


def propose_mixed_su2(
    space,
    states: jax.Array,
    key: jax.Array,
    move: MixedSU2Move,
) -> jax.Array:
    """Applies a per-row mixture of SU(2) proposal families.

    Args:
        space: SU(2) spin-network Hilbert space being updated.
        states: Two-dimensional batch of valid flat states.
        key: JAX pseudo-random key.
        move: Mixed proposal descriptor containing family weights and child
            move descriptors.

    Returns:
        Proposed state batch, with each row selected from the loop,
        vertex-channel, or edge heatbath proposal according to ``move.weights``.
    """
    arr = jnp.asarray(states, dtype=space.dtype)
    batch = arr.shape[0]
    weights = jnp.asarray(move.weights, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    key_family, key_loop, key_vertex, key_edge = jax.random.split(key, 4)
    families = jax.random.choice(key_family, 3, shape=(batch,), p=weights)

    loop_states = propose_loop(space, arr, key_loop, move.loop_move)
    vertex_states = propose_vertex_recoupling(space, arr, key_vertex, move.vertex_move)
    edge_states = propose_edge_heatbath(space, arr, key_edge, move.edge_move)
    out = jnp.where(families[:, None] == 0, loop_states, arr)
    out = jnp.where(families[:, None] == 1, vertex_states, out)
    out = jnp.where(families[:, None] == 2, edge_states, out)
    return out.astype(space.dtype)


def random_loop_warmup(
    space, key: jax.Array, batch: int, *, n_steps: int | None = None
) -> jax.Array:
    """Initializes valid SU(2) states by a loop-sector random walk.

    Args:
        space: SU(2) spin-network Hilbert space providing loop metadata and
            warmup configuration.
        key: JAX pseudo-random key.
        batch: Number of state rows to generate.
        n_steps: Optional override of the configured warmup length.

    Returns:
        Valid flat spin-network states. The walk starts at the vacuum sector
        and applies random fundamental loop attempts, so it never requires
        rectangular rejection sampling.
    """
    if batch == 0:
        return jnp.zeros((0, space.size), dtype=space.dtype)
    if not space.loop_edge_indices or space.two_j_max == 0:
        return jnp.zeros((batch, space.size), dtype=space.dtype)

    warmup = (
        n_steps
        if n_steps is not None
        else (
            int(space.init_warmup)
            if space.init_warmup is not None
            else max(32, 4 * max(space.n_edges, 1))
        )
    )
    key_loop, key_dir = jax.random.split(key)
    loop_ids = jax.random.randint(
        key_loop,
        (batch, warmup),
        minval=0,
        maxval=len(space.loop_edge_indices),
        dtype=jnp.int32,
    )
    directions = jnp.where(
        jax.random.bernoulli(key_dir, 0.5, (batch, warmup)),
        jnp.int32(1),
        jnp.int32(-1),
    )

    loop_sites, loop_mask = _loop_table(space)
    edge_values = jnp.zeros((batch, space.n_edges), dtype=jnp.int32)
    rows = jnp.arange(batch, dtype=jnp.int32)[:, None]
    for step in range(warmup):
        selected_sites = loop_sites[loop_ids[:, step]]
        selected_mask = loop_mask[loop_ids[:, step]]
        safe_sites = jnp.where(selected_mask, selected_sites, 0)
        increments = directions[:, step, None] * selected_mask.astype(jnp.int32)
        candidate = edge_values.at[rows, safe_sites].add(increments)
        _completed, valid = complete_states_from_edges(space, candidate)
        edge_values = jnp.where(valid[:, None], candidate, edge_values)

    completed, valid = complete_states_from_edges(space, edge_values)
    return jnp.where(valid[:, None], completed, jnp.zeros_like(completed)).astype(
        space.dtype
    )


def _loop_table(space) -> tuple[jax.Array, jax.Array]:
    """Builds padded loop-site and loop-mask tables."""
    width = max((len(loop) for loop in space.loop_edge_indices), default=1)
    sites = []
    masks = []
    for loop in space.loop_edge_indices:
        pad = width - len(loop)
        sites.append(tuple(loop) + (0,) * pad)
        masks.append((True,) * len(loop) + (False,) * pad)
    return (
        jnp.asarray(sites, dtype=jnp.int32),
        jnp.asarray(masks, dtype=jnp.bool_),
    )


def _loop_ids(space, batch: int, key: jax.Array, loop: int | None) -> jax.Array:
    """Selects loop indices for each chain row."""
    if loop is None:
        return jax.random.randint(
            key,
            (batch,),
            minval=0,
            maxval=len(space.loop_edge_indices),
            dtype=jnp.int32,
        )
    return jnp.full((batch,), int(loop) % len(space.loop_edge_indices), dtype=jnp.int32)


def _directions(batch: int, key: jax.Array, direction: int | None) -> jax.Array:
    """Selects loop update directions for each chain row."""
    if direction is None:
        return jnp.where(
            jax.random.bernoulli(key, 0.5, (batch,)),
            jnp.int32(1),
            jnp.int32(-1),
        )
    return jnp.full((batch,), int(direction), dtype=jnp.int32)


def _first_left_comb_channels(
    edge_two_j: jax.Array,
    channel_maxima: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Returns first valid channel assignments for edge-label rows."""
    edges = jnp.asarray(edge_two_j, dtype=jnp.int32)
    batch = edges.shape[0]
    valence = int(edges.shape[1])
    n_channels = len(channel_maxima)
    if n_channels != max(valence - 3, 0):
        return jnp.zeros((batch, 0), dtype=jnp.int32), jnp.zeros(
            (batch,), dtype=jnp.bool_
        )
    if valence == 0:
        return jnp.zeros((batch, 0), dtype=jnp.int32), jnp.ones(
            (batch,), dtype=jnp.bool_
        )
    if valence == 1:
        return jnp.zeros((batch, 0), dtype=jnp.int32), edges[:, 0] == 0
    if valence == 2:
        return jnp.zeros((batch, 0), dtype=jnp.int32), edges[:, 0] == edges[:, 1]
    if valence == 3:
        return (
            jnp.zeros((batch, 0), dtype=jnp.int32),
            coupling_ok(edges[:, 0], edges[:, 1], edges[:, 2]),
        )

    candidates = _channel_candidate_table(channel_maxima)
    valid = _valid_channel_candidates(edges, candidates)
    counts = jnp.sum(valid, axis=-1)
    first = jnp.argmax(valid, axis=-1)
    channels = candidates[first]
    return channels.astype(jnp.int32), counts > 0


def _channel_candidate_table(channel_maxima: tuple[int, ...]) -> jax.Array:
    """Builds the rectangular table of channel-label candidates."""
    maxima = tuple(int(value) for value in channel_maxima)
    if not maxima:
        return jnp.zeros((1, 0), dtype=jnp.int32)
    values = tuple(range(maximum + 1) for maximum in maxima)
    return jnp.asarray(tuple(product(*values)), dtype=jnp.int32)


def _valid_channel_candidates(edges: jax.Array, candidates: jax.Array) -> jax.Array:
    """Checks which channel candidates are admissible for edge rows."""
    valence = int(edges.shape[1])
    n_channels = int(candidates.shape[1])
    if n_channels == 0:
        if valence == 0:
            ok = jnp.ones((edges.shape[0],), dtype=jnp.bool_)
        elif valence == 1:
            ok = edges[:, 0] == 0
        elif valence == 2:
            ok = edges[:, 0] == edges[:, 1]
        else:
            ok = coupling_ok(edges[:, 0], edges[:, 1], edges[:, 2])
        return ok[:, None]

    ok = coupling_ok(edges[:, None, 0], edges[:, None, 1], candidates[None, :, 0])
    for idx in range(1, n_channels):
        ok = ok & coupling_ok(
            candidates[None, :, idx - 1],
            edges[:, None, idx + 1],
            candidates[None, :, idx],
        )
    return ok & coupling_ok(
        candidates[None, :, -1],
        edges[:, None, -2],
        edges[:, None, -1],
    )


__all__ = [
    "complete_states_from_edges",
    "graph_loop_edge_indices",
    "propose_edge_heatbath",
    "propose_loop",
    "propose_mixed_su2",
    "propose_vertex_recoupling",
    "random_loop_warmup",
]
