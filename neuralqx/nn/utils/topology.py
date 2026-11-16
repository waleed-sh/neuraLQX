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


"""Fixed graph construction and sparse neighbor aggregation."""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np

from .initialization import positive_int


def graph_edges(n_nodes, edges, *, directed=False):
    """Validate edges and return unique directed endpoints and inverse in-degrees.

    Undirected edges are expanded in both directions, with self edges included
    once. Duplicate edges are removed. This construction helper runs once on
    the host. Forward passes use only its resulting array buffers.
    """
    n_nodes = positive_int(n_nodes, "n_nodes")
    pairs = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Each edge must contain two integer node indices.")
        a, b = edge
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)):
            raise ValueError("Node indices must be integers.")
        if not (0 <= a < n_nodes and 0 <= b < n_nodes):
            raise ValueError("Node index lies outside [0, n_nodes).")
        pairs.add((int(a), int(b)))
        if not directed:
            pairs.add((int(b), int(a)))
    endpoints = np.asarray(sorted(pairs), dtype=np.int32).reshape((-1, 2))
    senders, receivers = endpoints[:, 0], endpoints[:, 1]
    degrees = np.bincount(receivers, minlength=n_nodes)
    inverse_degree = 1.0 / np.maximum(degrees, 1)
    return senders, receivers, inverse_degree


def mean_neighbors(inputs, senders, receivers, inverse_degree):
    """Average incoming node features with a sparse indexed reduction in JAX."""
    messages = jnp.zeros_like(inputs).at[..., receivers, :].add(inputs[..., senders, :])
    return messages * inverse_degree[:, None]


def connection_cones(
    n_nodes, senders, receivers, depth, *, itemsize=4, budget=8 * 1024**2
):
    """Build bounded dependency cones for a single changed node.

    Each layer records affected node indices, their positions in the previous
    cone, and incoming-neighbor weights from that cone. Edges are directed from
    sender to receiver. The self dependency is included at every layer, including
    for isolated nodes. Padding has index ``-1`` and zero weight.

    Return ``None`` when any cone exceeds a quarter of the graph, 32 nodes, or
    when the resulting tensor tables would exceed ``budget`` bytes. These are
    cost and memory limits only. Full message passing remains available for
    every graph. This helper executes once during model construction.
    """
    cap = min(32, n_nodes // 4)
    if cap < 1 or n_nodes * depth * (8 + itemsize) > budget:
        return None
    outgoing = [set() for _ in range(n_nodes)]
    incoming = [set() for _ in range(n_nodes)]
    for a, b in zip(senders, receivers, strict=True):
        outgoing[a].add(int(b))
        incoming[b].add(int(a))
    cones = [[(node,) for node in range(n_nodes)]]
    for _ in range(depth):
        following = []
        for previous in cones[-1]:
            nodes = set(previous)
            for node in previous:
                nodes.update(outgoing[node])
            if len(nodes) > cap:
                return None
            following.append(tuple(sorted(nodes)))
        cones.append(following)
    widths = [max(map(len, layer)) for layer in cones]
    nbytes = sum(n_nodes * b * (8 + a * itemsize) for a, b in pairwise(widths))
    if nbytes > budget:
        return None
    result = []
    for level in range(1, depth + 1):
        width, previous_width = widths[level], widths[level - 1]
        indices = np.full((n_nodes, width), -1, np.int32)
        self_slots = np.full_like(indices, -1)
        weights = np.zeros((n_nodes, width, previous_width), np.float64)
        for root, nodes in enumerate(cones[level]):
            previous = {node: slot for slot, node in enumerate(cones[level - 1][root])}
            for slot, node in enumerate(nodes):
                indices[root, slot] = node
                self_slots[root, slot] = previous.get(node, -1)
                for neighbor in incoming[node]:
                    if neighbor in previous:
                        weights[root, slot, previous[neighbor]] = 1 / len(
                            incoming[node]
                        )
        result.append((indices, self_slots, weights))
    return tuple(result)
