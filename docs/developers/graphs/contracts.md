---
title: Graph contracts
---

# Graph contracts

The graph interface describes a directed keyed multigraph whose edges also have a deterministic numerical order. [`Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Graph) supplies the standard implementation. Subclassing it is the simplest way to add a family while retaining the existing loop, sign, and dual-graph conventions.

## Canonical vertices and edges

[`Vertex`](../../api/public/neuralqx/utils/typing/index.md#neuralqx.utils.typing.Vertex) stores three real coordinates. Coordinate triples supplied to a graph are converted to this representation. Equality uses those coordinates, so points intended to denote the same vertex must be constructed consistently. Two distinct logical vertices cannot be distinguished by assigning them identical coordinates. The edge-list constructor discovers vertices from endpoints, so it does not introduce isolated vertices that never occur in an edge.

[`Edge`](../../api/public/neuralqx/utils/typing/index.md#neuralqx.utils.typing.Edge) stores `start`, `end`, and a nonnegative integer `key`. Input may be entirely unkeyed pairs or entirely keyed edges. Do not mix these forms in one declaration. For unkeyed pairs, canonicalisation assigns keys in encounter order within each unordered endpoint pair. Oppositely oriented edges with the same endpoints therefore belong to the same parallel-edge group.

For explicitly keyed input, ensure that each physical edge is identified uniquely by its unordered endpoint pair and key. The constructor's canonicalisation is not a general duplicate-key audit. Repeating a key in the same group can make reverse lookup ambiguous and collapse entries in dictionaries. Use different keys for distinct parallel edges, including those stored in opposite directions.

## Ordering and lookup

`graph.mapping` maps stored canonical edges to indices in input order. `graph.inverse_mapping` reverses that map. [`edge_to_index`](../../api/public/neuralqx/graph/abstract_graph/index.md#neuralqx.graph.abstract_graph.AbstractGraph.edge_to_index) also accepts the reversed orientation of an edge with the same key. That convenience identifies the same degree of freedom. It does not encode a sign for the reversal.

The integer returned by this lookup is a graph edge index. It is a flat Hilbert configuration index only when the chosen layout has exactly one scalar degree of freedom on each edge and nothing else. For vector-valued, heterogeneous, or vertex-based spaces, use the [Hilbert layout](../hilbert/states.md) to locate the corresponding components.

Treat the graph as fixed after construction. Its edge and vertex objects are value objects, but some graph properties expose mutable Python collections. Editing those collections does not rebuild cached mappings, loops, and signs. Construct a new graph instead, then rebuild the objects that depend on it.

## A complete graph family

The following family places a directed cycle on a circle. Its edge order follows the vertex order around that circle. It inherits all graph operations and adds a method that prepares an oriented incidence matrix for numerical calculations.

```python
import math

import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

class PolygonGraph(nqx.graph.Graph):
    def __init__(self, n_vertices, *, radius=1.0):
        if type(n_vertices) is not int or n_vertices < 3:
            raise ValueError("n_vertices must be an integer of at least three")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        points = [
            (
                radius * math.cos(2 * math.pi * i / n_vertices),
                radius * math.sin(2 * math.pi * i / n_vertices),
                0.0,
            )
            for i in range(n_vertices)
        ]
        edges = [
            (points[i], points[(i + 1) % n_vertices], 0)
            for i in range(n_vertices)
        ]
        super().__init__(edges, is_planar=True)

    def incidence_matrix(self):
        vertex_index = {vertex: i for i, vertex in enumerate(self.vertices)}
        matrix = np.zeros((self.n_vertices, self.n_edges), dtype=np.int32)
        for edge, column in self.mapping.items():
            matrix[vertex_index[edge.start], column] -= 1
            matrix[vertex_index[edge.end], column] += 1
        return jnp.asarray(matrix)

graph = PolygonGraph(5)
incidence = graph.incidence_matrix()
for index, edge in enumerate(graph.edges):
    assert graph.mapping[edge] == index
    assert graph.inverse_mapping[index] == edge
    assert graph.edge_to_index((edge.end, edge.start, edge.key)) == index

assert [loop.length for loop in graph.minimal_loops()] == [5]
np.testing.assert_array_equal(incidence @ jnp.ones(5, dtype=jnp.int32), 0)

hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(-1, 1, 3)
)
states = hilbert.random_state(jax.random.key(2), size=4)

@jax.jit
def vertex_divergence(states, incidence):
    return states @ incidence.T

assert vertex_divergence(states, incidence).shape == (4, 5)
reconstructed = nqx.graph.Graph(graph.edges, is_planar=graph.is_planar)
assert reconstructed == graph
assert hash(reconstructed) == hash(graph)
print(graph.n_vertices, graph.n_edges, incidence.shape)
```

```text
5 5 (5, 5)
```

With the convention used here, $B_{ve}=+1$ at an edge's end and $-1$ at its start. Thus $Bq$ measures incoming minus outgoing edge labels. A uniform oriented cycle has zero divergence. The randomly drawn configurations in the example are not constrained to zero divergence. The calculation illustrates how to carry graph structure into a JAX kernel without making that kernel inspect Python edges.

Standard `Graph` equality and hashing use the planar flag and the ordered canonical edges and vertices. The subclass name does not create a different topology. If additional metadata in a custom family changes operator behaviour without changing those values, account for that metadata in the relevant consumer's static identity or in a carefully defined graph equality and hash. Do not mutate hash-relevant data after using an object as a key or as static compilation input.

## Implementing the abstract interface directly

[`AbstractGraph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.AbstractGraph) handles the common input boundary. After calling its constructor, a direct implementation must provide the following derived interface.

| Interface | Required interpretation |
| --- | --- |
| `mapping`, `inverse_mapping` | Mutually inverse edge-index maps with deterministic ordering. |
| `signs`, `dual_signs` | Orientation tables using the selected planar or nonplanar convention. |
| `dual_edges`, `dual_adjacency` | The line-graph dual, with primal edges serving as dual vertices. |
| `all_loops(length=None)`, `all_dressed_loops(length=None)` | Simple primal loops up to an inclusive length bound, with the standard traversal and dressing conventions. |
| `minimal_loops()`, `dressed_minimal_loops()` | The selected shortest primal loops and their dressing. |
| `minimal_dual_loops(edge_rep=False)`, `dressed_minimal_dual_loops()` | Shortest dual loops in the representations described by the base interface. |

Some consumers also use concrete `Graph` conveniences such as `connectivity` and plotting positions. Check the requirements of the operators and models that the implementation will support. Satisfying the abstract methods does not imply that every convenience of the concrete class has been implemented.

A specialised implementation can compute expensive structures lazily, but should preserve their meaning, ordering, and isolation from later caller mutation. Verify those properties against `Graph` on small examples before substituting a new algorithm. The next page explains the [derived-structure conventions](derived-structures.md) that those tests need to preserve.
