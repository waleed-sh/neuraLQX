---
title: Graphs
---

# Graphs

A graph specifies the vertices and edges on which you build a quantum system. In neuraLQX, it also supplies edge ordering, connectivity, loops, and the local orientation information used by graph-dependent operators. The [Hilbert space](../hilbert/index.md) adds the quantum labels and their admissibility conditions.

The main constructor is [`nqx.graph.Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Graph). You can give it your own edges or start from a supplied family such as [`Grid2D`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Grid2D) or [`K5Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.K5Graph). All of these graphs share the same interface.

## A first graph

Here is a triangle with three directed edges:

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (0, 1, 0)

graph = nqx.graph.Graph(
    edges=[(a, b), (b, c), (c, a)],
    is_planar=True,
)

print(graph.n_vertices, graph.n_edges)
print(graph.index_to_edge(0).start.as_tuple())
print(graph.index_to_edge(0).end.as_tuple())
```

```text
3 3
(0.0, 0.0, 0.0)
(1.0, 0.0, 0.0)
```

Each vertex is identified by three real coordinates, even in a planar graph. The constructor converts them to `Vertex` objects. An edge is directed from its first endpoint to its second and is stored as an `Edge` with a nonnegative integer key. The key distinguishes different edges joining the same pair of vertices. You can supply `(start, end, key)` triples explicitly, or let the constructor assign keys to endpoint pairs as above.

Edges also have zero-based indices, assigned in the order you supplied them. An edge key and an edge index have different roles. Neither is a quantum spin or charge. The [labelling guide](labels.md) explains these distinctions and how to move between edges and indices.

## Planar and non-planar conventions

`is_planar=True` selects two-dimensional plotting and local orientation signs for pairs of incident edges. `is_planar=False` selects three-dimensional plotting and signs for triples of incident edges. This flag chooses a convention. It does not test whether the graph is topologically planar or check that your coordinates form a suitable embedding for a particular operator.

Coordinates identify vertices and provide embedding information. Distances in a plot do not assign physical lengths or areas to the quantum geometry. Those quantities are described by [operators](../operators/index.md) and their action on a quantum state. The [embedding guide](embeddings.md) explains how coordinates, orientation signs, and plotting layouts relate.

## Explore the graph API

[Constructing graphs](construction.md) covers the supplied families, custom edge lists, and parallel edges. [Vertex and edge labels](labels.md) develops the representation and indexing conventions you will use when selecting edges or interpreting a configuration.

[Connectivity and loops](connectivity.md) shows how to inspect incident edges, find cycles, and work with loop orientations and the line-graph dual. [Embeddings and plotting](embeddings.md) covers reproducible embeddings and figures. For individual signatures and arguments, consult the [graph API reference](../../api/public/neuralqx/graph/index.md).
