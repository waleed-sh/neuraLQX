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


"""Shared SU(2) proposals for Metropolis sampling and random initialization.

No global basis or intertwiner table is enumerated. Each proposal returns
its reverse/forward log probability ratio; acceptance belongs to the caller.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from functools import partial

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.hilbert.su2.recoupling import coupling_ok
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

DEFAULT_WEIGHTS = (0.45, 0.2, 0.25, 0.1)
DEFAULT_STAY_PROBABILITY = 0.05


@lru_cache(maxsize=64)
def parity_sector(space, parity):
    """Validate a fixed edge parity without enumerating the Hilbert space.

    Odd edges must have even incidence at every vertex, counting self-loop
    ends twice. Assigning doubled spins equal to these bits then gives an
    admissible starting spin assignment on any supported graph.
    """
    from neuralqx.hilbert.su2.spaces import SU2GaugeInvariantHilbert

    if not isinstance(space, SU2GaugeInvariantHilbert):
        raise NeuralqxTypeError("SU(2) proposals require SU2GaugeInvariantHilbert.")
    if parity is None:
        return None
    values = (
        (parity,) * space.n_edges
        if isinstance(parity, (int, np.integer))
        else tuple(parity)
    )
    if len(values) != space.n_edges or any(p not in (0, 1) for p in values):
        raise NeuralqxValueError(
            "edge_parity must be 0, 1, or one binary value per edge."
        )
    if any(p > space.two_j_max for p in values):
        raise NeuralqxValueError("edge_parity is incompatible with the spin cutoff.")
    if any(
        sum(values[e] for e in incident) % 2
        for incident in space.metadata.incident_edges_by_vertex
    ):
        raise NeuralqxValueError(
            "edge_parity violates even odd-spin incidence at a vertex."
        )
    return tuple(int(p) for p in values)


@lru_cache(maxsize=64)
def exchange_geometry(space, pairs):
    """Validate optional edge pairs once, outside the compiled proposal loop.

    None means implicit uniform selection of two distinct edges. No quadratic
    all-pairs table is allocated in that case.
    """
    if pairs is None:
        return None
    result = np.asarray(pairs, dtype=np.int64)
    if (
        result.ndim != 2
        or result.shape[1] != 2
        or len(result) == 0
        or np.any(result < 0)
        or np.any(result >= space.n_edges)
        or np.any(result[:, 0] == result[:, 1])
    ):
        raise NeuralqxValueError(
            "exchange_pairs must contain distinct, in-range edge indices."
        )
    result = result.astype(np.int32)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=64)
def _cycle_geometry(space):
    """Fundamental binary cycle basis of a spanning forest, including multiedges."""
    from neuralqx.hilbert.su2.spaces import SU2GaugeInvariantHilbert

    if not isinstance(space, SU2GaugeInvariantHilbert):
        raise NeuralqxTypeError("SU2MetropolisRule requires SU2GaugeInvariantHilbert.")
    parent = list(range(space.n_vertices))
    tree = [[] for _ in parent]
    chords = []

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for e, edge in enumerate(space.graph.edges):
        u, v = (space.metadata.vertex_index[x] for x in (edge.start, edge.end))
        ru, rv = root(u), root(v)
        if ru == rv:
            chords.append((u, v, e))
        else:
            parent[ru] = rv
            tree[u].append((v, e))
            tree[v].append((u, e))
    cycles = []
    for u, v, chord in chords:
        predecessors = {u: (u, -1)}
        queue = deque([u])
        while v not in predecessors:
            node = queue.popleft()
            for neighbor, edge in tree[node]:
                if neighbor not in predecessors:
                    predecessors[neighbor] = (node, edge)
                    queue.append(neighbor)
        cycle = [chord]
        while v != u:
            v, edge = predecessors[v]
            cycle.append(edge)
        cycles.append(tuple(sorted(cycle)))
    local_cycles = tuple(dict.fromkeys((*space.loop_edge_indices, *cycles)))
    basis = np.zeros((len(cycles), space.n_edges), dtype=np.int32)
    local = np.zeros((max(len(local_cycles), 1), space.n_edges), dtype=bool)
    for i, cycle in enumerate(cycles):
        basis[i, list(cycle)] = 1
    for i, cycle in enumerate(local_cycles):
        local[i, list(cycle)] = True
    basis.flags.writeable = local.flags.writeable = False
    return basis, local


def _vertex_channels(edges, *, channels=None, key=None):
    """Draw or score channels without enumerating coupled paths.

    A suffix of doubled spins has reachable total spins from
    max(sum mod 2, 2*max - sum) through sum, in steps of two. Intersecting
    this with the next triangle interval gives exactly the prefixes that
    can finish in a singlet. Sequential uniform choices need not yield a
    uniform distribution on whole intertwiners; log_p records the product.
    """
    batch, degree = edges.shape
    valid = jnp.all(edges >= 0, axis=-1) & (jnp.sum(edges, axis=-1) % 2 == 0)
    log_p = jnp.zeros(batch, dtype=jnp.float64)
    if degree < 4:
        if degree == 1:
            valid &= edges[:, 0] == 0
        elif degree == 2:
            valid &= edges[:, 0] == edges[:, 1]
        elif degree == 3:
            valid &= coupling_ok(edges[:, 0], edges[:, 1], edges[:, 2])
        return jnp.zeros((batch, 0), dtype=jnp.int32), jnp.where(valid, log_p, -jnp.inf)

    previous = edges[:, 0]
    chosen = []
    for slot in range(degree - 3):
        spin = edges[:, slot + 1]
        suffix = edges[:, slot + 2 :]
        total = jnp.sum(suffix, axis=-1)
        suffix_min = jnp.maximum(total % 2, 2 * jnp.max(suffix, axis=-1) - total)
        lower = jnp.maximum(jnp.abs(previous - spin), suffix_min)
        lower += (previous + spin - lower) % 2
        upper = jnp.minimum(previous + spin, total)
        count = jnp.maximum((upper - lower) // 2 + 1, 0)
        if channels is None:
            value = lower + 2 * jax.random.randint(
                jax.random.fold_in(key, slot),
                (batch,),
                0,
                jnp.maximum(count, 1),
                dtype=jnp.int32,
            )
        else:
            value = channels[:, slot]
        valid &= (
            (count > 0)
            & (value >= lower)
            & (value <= upper)
            & ((value - lower) % 2 == 0)
        )
        log_p -= jnp.log(jnp.maximum(count, 1).astype(jnp.float64))
        previous = value
        chosen.append(value)
    return jnp.stack(chosen, axis=-1).astype(jnp.int32), jnp.where(
        valid, log_p, -jnp.inf
    )


def _edge_log_prob(space, edges):
    counts = (space.two_j_max - edges % 2) // 2 + 1
    return -jnp.sum(jnp.log(jnp.maximum(counts, 1).astype(jnp.float64)), axis=-1)


@lru_cache(maxsize=64)
def _vertex_groups(space):
    """Vertices of equal valence share one batched coupling/PRNG program."""
    groups = {}
    for vertex, (edges, sites) in enumerate(
        zip(
            space.metadata.incident_edges_by_vertex,
            space.layout.vertex_channel_sites,
            strict=True,
        )
    ):
        groups.setdefault(len(edges), []).append((vertex, edges, sites))
    return tuple(
        (
            degree,
            tuple(v for v, _, _ in rows),
            tuple(e for _, e, _ in rows),
            tuple(s for _, _, s in rows),
        )
        for degree, rows in sorted(groups.items())
    )


def propose_metropolis(
    space,
    key,
    sigma,
    *,
    weights=DEFAULT_WEIGHTS,
    stay_probability=DEFAULT_STAY_PROBABILITY,
    coherent_loop_probability=0.0,
    edge_parity=None,
    exchange_pairs=None,
):
    """Return valid proposals and the selected component's Hastings correction.

    ``weights`` must be normalized and independent of the source state. Invalid
    attempts return the input state and zero correction, without retrying.
    The correction is log q(reverse)/q(forward) conditional on the selected
    move family and geometry. Accepting each component with its own correction
    gives a mixture of reversible MH kernels; it is not the ratio of proposal
    densities summed over all families. Returning valid states alone would not
    suffice: sequential channel draws are generally asymmetric.
    """
    weights = (*weights, 0.0) if len(weights) == 4 else weights
    sector = parity_sector(space, edge_parity)
    pairs = exchange_geometry(space, exchange_pairs) if weights[4] > 0 else None
    if weights[0] > 0 or (weights[3] > 0 and sector is None):
        basis, loops = _cycle_geometry(space)
    else:
        basis, loops = np.zeros((0, space.n_edges), np.int32), None
    batch = sigma.shape[0]
    if space.n_edges == 0 or space.two_j_max == 0:
        return sigma, jnp.zeros(batch, dtype=jnp.float64)
    keys = jax.random.split(key, 10)
    probabilities = jnp.asarray(
        (*(w * (1 - stay_probability) for w in weights), stay_probability)
    )
    family = jax.random.choice(keys[0], 6, shape=(batch,), p=probabilities)
    edges = sigma[:, : space.n_edges]
    rows = jnp.arange(batch)

    candidate_edges = edges
    touched_edges = jnp.zeros(edges.shape, dtype=bool)
    if weights[0] > 0:
        selected_loop = jnp.asarray(loops)[
            jax.random.randint(keys[1], (batch,), 0, len(loops), dtype=jnp.int32)
        ]
        signs = (
            2 * jax.random.bernoulli(keys[2], shape=edges.shape).astype(jnp.int32) - 1
        )
        if coherent_loop_probability > 0:
            coherent = jax.random.bernoulli(
                jax.random.fold_in(keys[2], 0), coherent_loop_probability, (batch, 1)
            )
            common = (
                2
                * jax.random.bernoulli(
                    jax.random.fold_in(keys[2], 1), shape=(batch, 1)
                ).astype(jnp.int32)
                - 1
            )
            signs = jnp.where(coherent, common, signs)
        delta = signs * (1 if sector is None else 2)
        candidate_edges = jnp.where(
            (family == 0)[:, None], edges + selected_loop * delta, edges
        )
        touched_edges |= (family == 0)[:, None] & selected_loop
    if weights[1] > 0:
        selected_edge = jax.random.randint(
            keys[3], (batch,), 0, space.n_edges, dtype=jnp.int32
        )
        parity = edges[rows, selected_edge] % 2
        count = (space.two_j_max - parity) // 2 + 1
        label = parity + 2 * jax.random.randint(
            keys[4], (batch,), 0, count, dtype=jnp.int32
        )
        edge_edges = edges.at[rows, selected_edge].set(label)
        candidate_edges = jnp.where((family == 1)[:, None], edge_edges, candidate_edges)
        touched_edges |= (family == 1)[:, None] & (
            jnp.arange(space.n_edges) == selected_edge[:, None]
        )
    selected_vertex = (
        jax.random.randint(keys[5], (batch,), 0, space.n_vertices, dtype=jnp.int32)
        if weights[2] > 0
        else jnp.full((batch,), -1, dtype=jnp.int32)
    )
    if weights[3] > 0:
        parity_key, spin_key = jax.random.split(keys[6])
        if sector is None:
            cycle_bits = jax.random.bernoulli(
                parity_key, shape=(batch, len(basis))
            ).astype(jnp.int32)
            global_parity = (cycle_bits @ jnp.asarray(basis)) % 2
        else:
            global_parity = jnp.broadcast_to(
                jnp.asarray(sector, dtype=jnp.int32), edges.shape
            )
        count = (space.two_j_max - global_parity) // 2 + 1
        global_edges = global_parity + 2 * jax.random.randint(
            spin_key, edges.shape, 0, count, dtype=jnp.int32
        )
        candidate_edges = jnp.where(
            (family == 3)[:, None], global_edges, candidate_edges
        )
    if weights[4] > 0 and space.n_edges > 1:
        if pairs is None:
            first = jax.random.randint(
                keys[8], (batch,), 0, space.n_edges, dtype=jnp.int32
            )
            second = jax.random.randint(
                keys[9], (batch,), 0, space.n_edges - 1, dtype=jnp.int32
            )
            second += second >= first
        else:
            selected = jnp.asarray(pairs)[
                jax.random.randint(keys[8], (batch,), 0, len(pairs), dtype=jnp.int32)
            ]
            first, second = selected[:, 0], selected[:, 1]
        exchanged = edges.at[rows, first].set(edges[rows, second])
        exchanged = exchanged.at[rows, second].set(edges[rows, first])
        candidate_edges = jnp.where((family == 4)[:, None], exchanged, candidate_edges)
        touched_edges |= (family == 4)[:, None] & (
            (jnp.arange(space.n_edges) == first[:, None])
            | (jnp.arange(space.n_edges) == second[:, None])
        )
    candidate = sigma.at[:, : space.n_edges].set(candidate_edges)
    valid = jnp.all(
        (candidate_edges >= 0) & (candidate_edges <= space.two_j_max), axis=-1
    )
    if sector is not None:
        valid &= jnp.all(candidate_edges % 2 == jnp.asarray(sector), axis=-1)
    correction = (
        jnp.where(
            family == 3,
            _edge_log_prob(space, edges) - _edge_log_prob(space, candidate_edges),
            0,
        )
        if weights[3] > 0
        else jnp.zeros(batch, dtype=jnp.float64)
    )
    for degree, vertices, incident, sites in _vertex_groups(space):
        incident_idx = jnp.asarray(incident, dtype=jnp.int32)
        sites_idx = jnp.asarray(sites, dtype=jnp.int32)
        n_vertices = len(vertices)
        n_channels = max(degree - 3, 0)
        touched = (
            jnp.any(touched_edges[:, incident_idx], axis=-1)
            | (family == 2)[:, None]
            & (selected_vertex[:, None] == jnp.asarray(vertices))
            | (family == 3)[:, None]
        )
        channels, new_p = _vertex_channels(
            candidate_edges[:, incident_idx].reshape((batch * n_vertices, degree)),
            key=jax.random.fold_in(keys[7], degree),
        )
        _, old_p = _vertex_channels(
            edges[:, incident_idx].reshape((batch * n_vertices, degree)),
            channels=sigma[:, sites_idx].reshape((batch * n_vertices, n_channels)),
        )
        channels = channels.reshape((batch, n_vertices, n_channels))
        candidate = candidate.at[:, sites_idx].set(
            jnp.where(touched[:, :, None], channels, sigma[:, sites_idx])
        )
        valid &= jnp.all(
            ~touched | jnp.isfinite(new_p.reshape((batch, n_vertices))), axis=-1
        )
        correction += jnp.sum(
            jnp.where(touched, (old_p - new_p).reshape((batch, n_vertices)), 0), axis=-1
        )
    return (
        jnp.where(valid[:, None], candidate, sigma),
        jnp.where(valid, correction, 0.0),
    )


@partial(jax.jit, static_argnames=("space", "batch", "n_steps", "edge_parity"))
def random_sector_warmup(space, key, batch, n_steps, edge_parity):
    """Construct and randomize valid states entirely inside a fixed parity sector.

    Minimal binary spins give a feasible singlet at every vertex. Channel
    generation and finite uniform-target warmup are compiled and never use a
    global basis. This is initialization, not a claim of equilibrium draws.
    """
    sector = parity_sector(space, edge_parity)
    states = jnp.zeros((batch, space.size), dtype=jnp.int32)
    states = states.at[:, : space.n_edges].set(jnp.asarray(sector, dtype=jnp.int32))
    for degree, vertices, incident, sites in _vertex_groups(space):
        channels, _ = _vertex_channels(
            states[:, jnp.asarray(incident, dtype=jnp.int32)].reshape(
                (batch * len(vertices), degree)
            ),
            key=jax.random.fold_in(key, degree),
        )
        states = states.at[:, jnp.asarray(sites, dtype=jnp.int32)].set(
            channels.reshape((batch, len(vertices), max(degree - 3, 0)))
        )
    if batch == 0 or n_steps <= 0 or space.n_edges == 0 or space.two_j_max == 0:
        return states

    def step(i, current):
        proposal_key, accept_key = jax.random.split(jax.random.fold_in(key, i + 10000))
        candidate, correction = propose_metropolis(
            space,
            proposal_key,
            current,
            edge_parity=sector,
            coherent_loop_probability=1.0,
        )
        log_u = jnp.log(
            jax.random.uniform(accept_key, (batch,), dtype=correction.dtype)
        )
        return jnp.where(
            (log_u < jnp.minimum(correction, 0))[:, None], candidate, current
        )

    states = step(0, states)
    return jax.lax.fori_loop(1, n_steps, step, states)


@partial(jax.jit, static_argnames=("space", "batch", "n_steps"))
def random_metropolis_warmup(space, key, batch, n_steps):
    """Generate valid states using independent finite-length uniform-target chains.

    The target is uniform over admissible spin networks, including distinct
    intertwiners. Finite warmup is an initializer, not an exact independent
    uniform draw. The vacuum is a legitimate outcome and is never filtered.
    """
    states = jnp.zeros((batch, space.size), dtype=jnp.int32)
    if batch == 0 or space.two_j_max == 0 or space.n_edges == 0 or n_steps <= 0:
        return states

    def step(i, current):
        proposal_key, accept_key = jax.random.split(jax.random.fold_in(key, i))
        candidate, correction = propose_metropolis(
            space, proposal_key, current, coherent_loop_probability=0.5
        )
        log_u = jnp.log(
            jax.random.uniform(accept_key, (batch,), dtype=correction.dtype)
        )
        accept = log_u < jnp.minimum(correction, 0.0)
        return jnp.where(accept[:, None], candidate, current)

    # The first draw makes the carry vary over the same manual mesh axes as
    # the RNG key. Starting a shard-mapped scan with constant zeros instead
    # changes the carry's axis type on its first iteration. This preserves
    # the exact draw sequence without assuming particular mesh-axis names.
    states = step(0, states)
    return jax.lax.fori_loop(1, n_steps, step, states)
