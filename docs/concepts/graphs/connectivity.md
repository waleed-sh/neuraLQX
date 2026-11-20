---
title: Connectivity and loops
---

# Connectivity and loops

Many graph-based calculations need more than an edge list. You may want the edges incident to a vertex, a closed path on which to define a holonomy, or a neighbourhood connecting edge degrees of freedom. The graph API provides these structures using the [vertex and edge labels](labels.md) established at construction.

Edge directions matter when inspecting incidence and assigning orientation information. Cycle searches, however, allow a path to traverse an edge in either direction. Keeping these two uses of orientation separate makes the loop methods easier to interpret.

## Inspect the neighbourhood of a vertex

`edges_at(vertex)` returns all stored edges touching a vertex. `edges_to(vertex)` selects incoming edges, and `edges_from(vertex)` selects outgoing edges. The returned edges keep their original directions and their order in the graph's edge list.

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (1, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (a, c)], is_planar=True)

incoming = [graph.edge_to_index(edge) for edge in graph.edges_to(b)]
outgoing = [graph.edge_to_index(edge) for edge in graph.edges_from(b)]
incident = [graph.edge_to_index(edge) for edge in graph.edges_at(b)]

print(incoming)
print(outgoing)
print(incident)
print(graph.valence(b))
```

```text
[0]
[1]
[0, 1]
2
```

Here, edge `0` ends at `b` and edge `1` starts there. The diagonal edge from `a` to `c` does not touch `b`. Parallel edges count separately because they are distinct links.

`graph.connectivity` also exposes the incoming and outgoing lists for every canonical `Vertex`, under the keys `"incoming"` and `"outgoing"`. The methods above accept coordinate triples as well, so they are convenient when your script still uses the coordinates that constructed the graph.

`valence` counts incident stored edges. A self-loop is counted once by this method, even though it both starts and ends at the same vertex. This differs from the graph-theory degree convention that counts a self-loop twice.

## Find cycles of the length you need

`all_loops()` returns the graph's simple undirected cycles as `Loop` objects. A simple cycle visits no vertex twice apart from closing at its starting vertex. Rotations of the same traversal and its reverse are counted once. Distinct choices of parallel links can still produce distinct cycles.

You can restrict enumeration with `all_loops(length=...)`. The argument is an inclusive maximum length, measured in edges. It is not a request for cycles of exactly that length.

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (1, 1, 0)
d = (0, 1, 0)
graph = nqx.graph.Graph(
    [(a, b), (b, c), (c, d), (d, a), (a, c)],
    is_planar=True,
)

all_cycles = graph.all_loops()
short_cycles = graph.all_loops(length=3)
four_edge_cycles = [loop for loop in graph.all_loops(length=4) if loop.length == 4]

print([loop.length for loop in all_cycles])
print([loop.length for loop in short_cycles])
print([loop.length for loop in four_edge_cycles])
print([loop.length for loop in graph.minimal_loops()])
```

```text
[3, 3, 4]
[3, 3]
[4]
[3, 3]
```

The diagonal divides this square into two triangles, but the four-edge outer boundary is also a simple cycle. `all_loops()` includes it even though the diagonal is a chord. The filtered expression selects cycles of exactly four edges.

### What minimal means

`minimal_loops()` returns the cached shortest cycles across the graph. For a graph without self-loops, parallel-edge two-cycles take precedence whenever they exist. Otherwise, the method returns the cycles of globally shortest length. `graph.n_minimal_loops` gives their count.

This is not a selection of every face or elementary plaquette, and it is not a cycle basis. If one region has triangles and another has only longer cycles, the longer ones are absent from `minimal_loops()`. Choose the loops appropriate to the [operator](../operators/index.md) you are constructing, using bounded enumeration and filtering when needed.

The number of simple cycles can grow rapidly with graph size and connectivity. On a large graph, specify a useful length bound before asking for all cycles. `length=0` returns an empty list, while a negative bound is rejected.

## Follow a loop and its orientation

A `Loop` stores its traversal as an ordered tuple of `Edge` objects in `loop.edges`. Successive edges join head to tail, and the final edge returns to the start. `loop.vertices` includes this closing vertex, so a three-edge cycle has four entries in its vertex traversal.

The traversal direction can differ from the graph's stored edge direction. `all_dressed_loops()` records that relationship in one metadata entry per traversed edge:

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (1, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (a, c)], is_planar=True)

loop = graph.all_dressed_loops(length=3)[0]

print(loop.length)
print([vertex.as_tuple() for vertex in loop.vertices])
print([graph.edge_to_index(edge) for edge in loop.edges])
print([entry["type"] for entry in loop.dressing])
```

```text
3
[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]
[0, 1, 2]
['creation', 'creation', 'annihilation']
```

This traversal follows `a` to `b` to `c` and then back to `a`. Its final step reverses the graph's stored edge from `a` to `c`. The graph therefore contains this cycle even though its stored directions do not form a directed circuit.

The dressing uses `"creation"` when traversal follows the stored direction and `"annihilation"` when it goes against it. Each entry also retains the edge key. These names encode the loop-orientation convention. Their physical action is determined by the model or operator consuming the loop, so they should not be interpreted as universal bosonic ladder operators.

`loop.dressed_edges` pairs each traversed edge with its metadata. Ordinary loops from `all_loops()` have `dressing=None`. For the cached shortest cycles, `dressed_minimal_loops()` provides the corresponding dressed objects.

## Parallel edges and self-loops

Two distinct edges between the same pair of vertices form a two-edge cycle, even if their stored directions agree. An edge from a vertex back to itself forms a one-edge cycle in `all_loops()`.

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
graph = nqx.graph.Graph([(a, a), (a, b), (a, b)], is_planar=True)

print([loop.length for loop in graph.all_loops()])
print(graph.valence(a))
```

```text
[1, 2]
3
```

Use `all_loops()` when your selection must include self-loops. The minimal-loop routine does not enumerate one-edge cycles. As with any graph structure, support for constructing a link does not establish that a particular Hilbert space, sampler, or geometric operator supports every use of that link. Check the conventions of the component that consumes it.

## Work with the line-graph dual

The graph also constructs a *line graph*. Each original edge becomes a dual vertex, and two dual vertices are adjacent when their original edges share a vertex. In the API this structure is called the dual graph. It describes adjacency between links, rather than the faces or cells of a spatial decomposition.

For example, a three-edge star has no primal cycles. Its three edges all meet at the centre, so its line graph is a triangle:

```python
import neuralqx as nqx

graph = nqx.graph.SingleVertexGraph(3)
edge = graph.index_to_edge(0)
neighbours = graph.dual_adjacency[edge]

print(len(graph.all_loops()))
print(graph.n_dual_vertices, graph.n_dual_edges)
print(sorted(graph.map_edge_to_dual_vertex(item) for item in neighbours))
print(graph.dual_vertex_to_edge(0) == edge)
```

```text
0
3 3
[1, 2]
True
```

`dual_edges` stores pairs of original `Edge` objects, and `dual_adjacency` maps each original edge to its neighbours in the line graph. A dual-vertex index is the corresponding original edge index. `map_edge_to_dual_vertex` and `dual_vertex_to_edge` make that conversion explicit.

`minimal_dual_loops()` returns shortest cycles in this derived graph. Unlike the primal loop methods, its cycles are lists of keyed dual-edge triples. By default their endpoints are original `Edge` objects. With `edge_rep=True`, those endpoints are replaced by original edge indices.

You can display the line graph together with the original graph using `plot(include_dual=True)`. The [plotting guide](embeddings.md#show-the-line-graph-dual) demonstrates this overlay, and the [API reference](../../api/public/neuralqx/graph/index.md) documents the complete set of graph methods.
