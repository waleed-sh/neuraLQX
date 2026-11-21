---
title: Graph networks
---

# Graph networks

[`GNN`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.GNN) builds a log wavefunction through message passing on a fixed graph of input features. Each network node carries a group of consecutive configuration coordinates. Shared maps combine a node's own features with those of its neighbours, and a pooled readout produces the log amplitude.

The network graph and the physical graph have different roles. In loop quantum gravity, many labels live on physical edges, while the GNN expects labels on its input nodes. You must choose how those labels become network nodes. For example, physical edges can be nodes of a line graph, with two nodes connected when the corresponding edges share an endpoint.

## Define the node layout

The constructor requires `n_nodes` and integer endpoint pairs `edges`. With `node_features=q`, the final input width must be

$$
N=n_{\mathrm{nodes}}q.
$$

The expected order is all $q$ coordinates for node zero, then all $q$ coordinates for node one, and so on. `directed=False` is the default and adds both message directions. Duplicate endpoint pairs are removed. With `directed=True`, messages follow the supplied sender-to-receiver direction.

The U(1) Hilbert layout stores uniform gauge components in separate component blocks. It does not place all components of one edge consecutively. Therefore, setting `node_features=3` on an unmodified U(1)$^3$ configuration generally gives the wrong interpretation. Use `hilbert.layout.edge_site(edge, component=c)` to construct the desired gathering order in a custom wrapper. The [layout guide](../../hilbert/state-layouts.md) gives the full convention.

SU(2) edge spins and vertex channels require a similarly explicit design. A GNN constructor does not automatically convert a spin network into node features.

## Message passing and pooling

After a shared input map and activation, each message stage calculates the incoming-neighbour mean

$$
m_v^{(\ell)}
=\frac{1}{d_v^{\mathrm{in}}}
\sum_{u\to v}h_u^{(\ell)}.
$$

An isolated node receives a zero neighbour vector. A [`GraphConv`](../../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.GraphConv) supplies the update, combined with a residual contribution,

$$
h_v^{(\ell+1)}
=\frac{
h_v^{(\ell)}
+\rho\!\left(h_v^{(\ell)}W_s+m_v^{(\ell)}W_n+b\right)
}{\sqrt2}.
$$

The model averages the final node features, applies a nonlinear dense readout, and returns one log amplitude. `width=32` and `depth=2` are the defaults. Depth counts message stages after the input encoder.

## Put one U(1) charge on each feature node

This example uses one gauge component per physical edge, so the Hilbert vector already has one scalar for each feature node. It constructs the line-graph connections explicitly.

```python
from itertools import combinations

import jax
import neuralqx as nqx

a, b, c, d = (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)
graph = nqx.graph.Graph(
    [(a, b), (b, c), (c, d), (d, a)], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
endpoints = [{edge.start, edge.end} for edge in graph.edges]
feature_edges = [
    (i, j)
    for i, j in combinations(range(graph.n_edges), 2)
    if endpoints[i] & endpoints[j]
]
network = nqx.nn.GNN(
    graph.n_edges,
    feature_edges,
    node_features=1,
    width=16,
    depth=2,
    node_embeddings=True,
    rngs=6,
)
configurations = hilbert.random_state(jax.random.key(2), size=3)
print(network(configurations).shape)
print(network.in_features == hilbert.size)
```

```text
(3,)
True
```

This feature graph records incidence between physical edges. It does not encode tangent orientations, holonomies, or gauge constraints by itself. Those remain part of the physical setup and any additional feature design.

## Node embeddings and symmetry

Learned node embeddings are enabled by default. They let the model distinguish labelled nodes even when their neighbourhood structures agree. This avoids automatically restricting every calculation to the invariant sector of the feature graph.

With `node_embeddings=False` and pointwise activations, shared message maps and mean pooling make the result independent of a simultaneous relabelling of input nodes and topology. On a fixed topology, this imposes invariance under its node automorphisms.

A physical symmetry can also transform orientations, charge signs, or intertwiner data. The feature graph's automorphisms alone do not establish that transformation. Use an explicit [projector](../symmetries.md) when you need a specified action on the physical basis.

## Fixed topology and local reuse

Topology is stored in non-trainable sender and receiver buffers, and messages use indexed reductions rather than a dense adjacency matrix. Construct a new network when changing topology so all derived data agrees with it.

For suitable sparse graphs, a single changed input coordinate affects only a bounded message dependency region. The connected evaluator reuses unaffected source features and recomputes that region exactly. Small graphs, broad dependency regions, multiple changes, or nonlocal activations use the ordinary evaluation. This choice is controlled by the [connection policy](../models.md#exact-reuse-during-operator-evaluation), not by truncating physical correlations.
