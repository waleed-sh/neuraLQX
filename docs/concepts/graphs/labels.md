---
title: Vertex and edge labels
---

# Vertex and edge labels

Graph labels identify the vertices and edges you refer to in code. You will use them to select a vertex, inspect a loop, or specify which edges an operator acts on. Their relationship to array indices becomes especially important once you attach a [Hilbert space](../hilbert/index.md).

There are three different kinds of information to keep in view. A vertex has a coordinate label. An edge has two endpoints and a key, together with a separate position in the graph's edge ordering. Quantum labels, such as spins and charges, belong to a Hilbert space configuration and are assigned later.

## Vertices are coordinate triples

The canonical vertex type is `Vertex`, available from `neuralqx.utils.typing`. It stores three real coordinates as Python floating-point values. You can pass coordinate tuples directly to graph constructors and lookup methods, which convert them for you.

```python
import neuralqx as nqx
from neuralqx.utils.typing import Vertex

a = (0, 0, 0)
b = (2, 0, 0)
c = (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)

vertex = graph.vertices[0]

print(vertex.as_tuple())
print(vertex.x, vertex.y, vertex.z)
print(vertex == Vertex(0, 0, 0))
print(graph.valence(a))
```

```text
(0.0, 0.0, 0.0)
0.0 0.0 0.0
True
2
```

Coordinates determine identity through exact equality after conversion. For example, `(0, 0, 0)` and `(0.0, 0.0, 0.0)` identify the same vertex. There is no tolerance that merges nearby coordinates. If your coordinates come from a numerical calculation, reuse the same triples for shared endpoints so small roundoff differences do not accidentally create distinct vertices.

An integer such as `0` is not a vertex label, and a two-entry tuple is not a valid vertex. Even planar vertices need a third coordinate. To refer to the first vertex, use `graph.vertices[0]`.

`graph.vertices` follows the order in which endpoints first appear in the supplied edge list. The integer labels shown by [graph plotting](embeddings.md#plot-and-save-a-graph) are positions in this list. They are convenient display labels, while methods such as `valence` and `edges_at` receive a `Vertex` or coordinate triple.

## An edge contains endpoints and a key

The canonical `Edge` type stores `start`, `end`, and `key`. The first two fields are `Vertex` objects. The edge direction runs from `start` to `end`, and the nonnegative integer key distinguishes parallel links.

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
graph = nqx.graph.Graph([(a, b, 4), (a, b, 9)], is_planar=True)

edge = graph.edges[0]

print(edge.start.as_tuple())
print(edge.end.as_tuple())
print(edge.key)
print(graph.edge_to_index(edge))
```

```text
(0.0, 0.0, 0.0)
(1.0, 0.0, 0.0)
4
0
```

This edge has key `4` and graph index `0`. Keys identify links locally within an endpoint pair, so different pairs can reuse a key. Indices identify entries in the complete ordered edge list, from `0` to `graph.n_edges - 1`.

If you omit keys at construction, they are assigned in input order for each unordered endpoint pair. Both directions between a pair participate in that same count. See [parallel-edge construction](construction.md#include-parallel-edges) for examples and the rules for mixing input types.

An edge can be unpacked into `start, end, key`. Likewise, `edge.as_tuple()` returns a triple whose endpoints are still `Vertex` objects. To obtain coordinate triples for those endpoints, call `edge.start.as_tuple()` and `edge.end.as_tuple()` as above.

## Convert between edges and indices

Use `edge_to_index` when you have an edge and need its integer position. Use `index_to_edge` to recover the stored edge. The lookup accepts either an `Edge` or an endpoint tuple with an optional key.

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)

index = graph.edge_to_index((a, b, 0))
reverse_index = graph.edge_to_index((b, a, 0))
stored_edge = graph.index_to_edge(index)

print(index, reverse_index)
print(stored_edge == graph.edges[0])
print(stored_edge.start.as_tuple(), stored_edge.end.as_tuple())
```

```text
0 0
True
(0.0, 0.0, 0.0) (1.0, 0.0, 0.0)
```

The lookup recognises the reversed direction of a stored link, preserving its key. It does not reverse the edge in the graph. This is useful when converting a [loop traversal](connectivity.md#follow-a-loop-and-its-orientation) into edge indices, since a loop can traverse a stored edge in either direction. The resulting index alone does not record the traversal direction.

Leaving out the key in a lookup means key `0`. It does not mean any edge between those endpoints. For a link with key `4`, pass that key explicitly. An unknown edge or index raises an error rather than creating a new graph entry.

The dictionaries `graph.mapping` and `graph.inverse_mapping` expose the same stored correspondence. Their keys are canonical `Edge` objects and integer indices, respectively. Direct dictionary access does not convert tuples or try a reversed edge, so the methods above are usually the more convenient interface.

## Preserve the ordering when interpreting states

Edge indices are assigned by construction order, rather than by sorting coordinates or keys. Two edge lists with the same connectivity can therefore give the same link different indices:

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (0, 1, 0)
edges = [(a, b), (b, c), (c, a)]

first = nqx.graph.Graph(edges, is_planar=True)
second = nqx.graph.Graph(list(reversed(edges)), is_planar=True)

print(first.edge_to_index((a, b)))
print(second.edge_to_index((a, b)))
```

```text
0
2
```

This matters when an observable names edges by index, when you compare sampled configurations, or when you restore data from a previous run. Reconstruct the same ordered graph rather than assuming that equivalent connectivity implies equivalent array entries. For automatically keyed parallel edges, changing their input order can also change the keys assigned to individual entries.

The graph's edge ordering contributes to the Hilbert space layout, but it is not the entire layout. A model can have multiple degrees of freedom per edge or additional entries associated with vertices. In particular, SU(2) configurations include intertwiner data as well as edge-spin data. Use the [state-layout guide](../hilbert/state-layouts.md) to interpret the full configuration array.

## Graph labels and quantum labels

An edge key of `2` does not mean spin $j=2$, and edge index `0` does not mean a vanishing charge. Those numbers identify graph elements. The quantum label on an edge is a value in a configuration belonging to the selected Hilbert space, and it can vary as the sampler explores that space.

Similarly, moving a vertex in the embedding does not directly set its quantum volume. Coordinates supply identification and orientation information. [Geometric operators](../lqx/index.md) and the [variational state](../vqs/index.md) determine the quantities you measure. Continue with [connectivity and loops](connectivity.md) to use these labels when inspecting the graph's structure.
