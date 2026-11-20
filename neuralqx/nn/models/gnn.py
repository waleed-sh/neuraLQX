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


"""Message-passing wavefunctions for general fixed graphs."""

import jax.numpy as jnp
from flax import nnx

from neuralqx.utils.errors import NeuralqxNotImplementedError
from neuralqx.utils.errors import NeuralqxValueError

from ..activations import activation_spec
from ..layers import GraphConv
from ..layers import Linear
from ..utils import ConnectionPolicy
from ..utils import as_rngs
from ..utils.initialization import compute_dtype
from ..utils.initialization import input_array
from ..utils.initialization import log_amplitude
from ..utils.initialization import positive_int
from ..utils.topology import connection_cones
from ..utils.topology import graph_edges
from ..utils.topology import mean_neighbors
from .base import LogWavefunction


class GNN(LogWavefunction):
    """Residual message-passing log wavefunction on a fixed graph.

    The configuration contains ``node_features`` consecutive coordinates for
    each node. A shared input map, mean-neighbor message layers and mean pooling
    produce one global log amplitude. Optional node embeddings allow the model
    to distinguish labeled nodes. Disable them to impose graph-relabeling
    invariance when topology and input features are relabeled together.

    The graph is stored as nontrainable endpoint buffers. Message passing uses
    sparse indexed reductions rather than a dense adjacency matrix. On sparse
    graphs, a single changed input can reuse source features outside its message
    dependency cone. Small graphs, broad cones and nonlocal activations retain
    ordinary evaluation. The optional cone tables occupy at most 8 MiB.
    Construct a new model when changing topology so these tables match its edges.
    """

    n_nodes: int
    """Number of graph nodes."""

    node_features: int
    """Number of consecutive input coordinates assigned to each node."""

    in_features: int
    """Total flattened configuration size."""

    senders: nnx.Variable
    """Source endpoint of each directed edge, stored outside the parameters."""

    receivers: nnx.Variable
    """Destination endpoint of each directed edge."""

    inverse_degree: nnx.Variable
    """Reciprocal incoming degree, with isolated nodes assigned one."""

    encoder: Linear
    """Shared projection of node input features."""

    node_embedding: nnx.Param | None
    """Optional learned embedding for each labeled node."""

    messages: nnx.List
    """Residual message-passing stages."""

    readout: Linear
    """Nonlinear projection of the pooled node features."""

    head: Linear
    """Log-amplitude and optional phase output."""

    activation: object
    """Resolved activation callable."""

    complex_output: bool
    """Whether to include an independent phase channel."""

    input_scale: float
    """Fixed scale applied to the raw configuration coordinates."""

    dtype: object
    """Computation dtype."""

    connection_nodes: nnx.List | None
    """Affected-node tables per layer, or ``None`` when local reuse is disabled."""

    connection_slots: nnx.List | None
    """Positions of affected self features in the previous layer's cone."""

    connection_weights: nnx.List | None
    """Incoming-neighbor weights from the previous cone to the current cone."""

    def __init__(
        self,
        n_nodes,
        edges,
        *,
        node_features=1,
        width=32,
        depth=2,
        directed=False,
        node_embeddings=True,
        activation="silu",
        complex_output=True,
        input_scale=1.0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        connection_policy=None,
        rngs,
    ):
        """Initialize a graph model from integer endpoint pairs.

        ``edges`` are undirected unless ``directed=True``. ``depth`` counts
        message-passing stages after the shared encoder. Node embeddings break
        otherwise enforced graph symmetries and are enabled by default.
        """
        self.n_nodes = positive_int(n_nodes, "n_nodes")
        self.node_features = positive_int(node_features, "node_features")
        self.in_features = self.n_nodes * self.node_features
        width = positive_int(width, "width")
        depth = positive_int(depth, "depth")
        self.input_scale = float(input_scale)
        if not 0 < self.input_scale < float("inf"):
            raise NeuralqxValueError("input_scale must be finite and positive.")
        dtype = compute_dtype(dtype, param_dtype)
        self.dtype = dtype
        self.complex_output = bool(complex_output)
        spec = activation_spec(activation, dtype=dtype)
        self.activation = spec.function
        senders, receivers, inv_degree = graph_edges(
            self.n_nodes, edges, directed=directed
        )
        self.senders = nnx.Variable(jnp.asarray(senders))
        self.receivers = nnx.Variable(jnp.asarray(receivers))
        self.inverse_degree = nnx.Variable(jnp.asarray(inv_degree, dtype=dtype))
        policy = connection_policy or ConnectionPolicy(max_changes=1, min_features=128)
        consider_reuse = (
            spec.elementwise
            and policy.mode != "off"
            and (policy.mode == "on" or self.in_features >= policy.min_features)
        )
        plan = (
            connection_cones(
                self.n_nodes,
                senders,
                receivers,
                depth,
                itemsize=jnp.dtype(dtype).itemsize,
            )
            if consider_reuse
            else None
        )
        if plan is None:
            self.connection_policy = ConnectionPolicy(mode="off")
            self.connection_nodes = self.connection_slots = self.connection_weights = (
                None
            )
        else:
            self.connection_policy = ConnectionPolicy(
                policy.mode, 1, policy.min_features
            )
            self.connection_nodes = nnx.List(
                [nnx.Variable(jnp.asarray(row[0])) for row in plan]
            )
            self.connection_slots = nnx.List(
                [nnx.Variable(jnp.asarray(row[1])) for row in plan]
            )
            self.connection_weights = nnx.List(
                [nnx.Variable(jnp.asarray(row[2], dtype=dtype)) for row in plan]
            )
        rngs = as_rngs(rngs)
        options = dict(
            rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        self.encoder = Linear(self.node_features, width, **options)
        self.node_embedding = (
            nnx.Param(
                nnx.initializers.normal(0.01)(
                    rngs.params(), (self.n_nodes, width), param_dtype
                )
            )
            if node_embeddings
            else None
        )
        self.messages = nnx.List(
            [
                GraphConv(width, width, activation=activation, **options)
                for _ in range(depth)
            ]
        )
        self.readout = Linear(width, width, **options)
        self.head = Linear(
            width,
            2 if complex_output else 1,
            kernel_init=nnx.initializers.normal(0.01),
            **options,
        )

    def __call__(self, states):
        """Evaluate flattened node features with arbitrary leading batch axes."""
        return self._readout(self.prepare_connections(states)[3])

    def _readout(self, pooled):
        pooled = self.activation(self.readout(pooled))
        return log_amplitude(self.head(pooled), self.complex_output)

    def prepare_connections(self, sources):
        """Cache node inputs, layer features and neighbor means for one evaluation."""
        x = input_array(sources, self.in_features, self.dtype) / self.input_scale
        x = x.reshape((*x.shape[:-1], self.n_nodes, self.node_features))
        h = self.encoder(x)
        if self.node_embedding is not None:
            h = h + jnp.asarray(self.node_embedding[...], self.dtype)
        h = self.activation(h)
        features = [h]
        means = []
        for layer in self.messages:
            neighbors = mean_neighbors(
                h, self.senders[...], self.receivers[...], self.inverse_degree[...]
            )
            h = (h + layer.from_aggregated(h, neighbors)) / 2**0.5
            features.append(h)
            means.append(neighbors)
        return x, tuple(features), tuple(means), h.mean(axis=-2)

    def apply_connections(self, cache, owners, sites, differences):
        """Recompute only affected nodes at each message-passing depth.

        Unchanged incoming messages are read from the source cache. Changed
        messages are added using the graph's exact normalized adjacency weights.
        The final feature differences update the global mean before the same
        nonlinear readout used by full evaluation.
        """
        if self.connection_nodes is None:
            raise NeuralqxNotImplementedError(
                "This graph does not have a bounded local-reuse plan."
            )
        inputs, features, means, pooled = cache
        root = sites[:, 0] // self.node_features
        coordinate = sites[:, 0] % self.node_features
        rows = jnp.arange(len(owners))
        node_input = (
            inputs[owners, root]
            .at[rows, coordinate]
            .add(jnp.asarray(differences[:, 0], self.dtype) / self.input_scale)
        )
        updated = self.encoder(node_input)
        if self.node_embedding is not None:
            updated += jnp.asarray(self.node_embedding[...], self.dtype)[root]
        delta = (self.activation(updated) - features[0][owners, root])[:, None, :]
        for level, layer in enumerate(self.messages):
            nodes = self.connection_nodes[level][...][root]
            slots = self.connection_slots[level][...][root]
            weights = self.connection_weights[level][...][root]
            safe_nodes = jnp.maximum(nodes, 0)
            self_delta = delta[rows[:, None], jnp.maximum(slots, 0)]
            own = features[level][owners[:, None], safe_nodes] + jnp.where(
                (slots >= 0)[..., None], self_delta, 0
            )
            neighbors = means[level][owners[:, None], safe_nodes] + jnp.einsum(
                "bij,bjf->bif", weights, delta, precision=layer.neighbor_map.precision
            )
            updated = (own + layer.from_aggregated(own, neighbors)) / 2**0.5
            delta = jnp.where(
                (nodes >= 0)[..., None],
                updated - features[level + 1][owners[:, None], safe_nodes],
                0,
            )
        return self._readout(pooled[owners] + delta.sum(axis=1) / self.n_nodes)
