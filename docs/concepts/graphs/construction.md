---
title: Constructing graphs
---

# Constructing graphs

You can build a graph from a supplied family or from an explicit sequence of directed edges. A family is convenient when its connectivity already matches your problem. An edge list gives you control over the vertices, edge directions, parallel edges, and order used throughout the calculation.

Both approaches produce objects with the interface introduced in the [graph overview](index.md). After construction, you can inspect their [labels](labels.md), [connectivity and loops](connectivity.md), or [embedding](embeddings.md) before passing the graph to a Hilbert space.

## Start from a graph family

For a rectangular lattice, [`Grid2D(m, n)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Grid2D) creates a grid with `m` by `n` vertices. The following graph has six vertices and seven edges:

```python
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 3)

print(graph.n_vertices, graph.n_edges)
print(graph.is_planar)
```

```text
6 7
True
```

The supplied families cover several common connectivities. The calls below are small examples of each constructor, with `nqx` denoting `import neuralqx as nqx`. Follow a constructor link to read its arguments and available options in the API reference.

| Constructor | Connectivity |
| --- | --- |
| [`nqx.graph.Grid2D(2, 3)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Grid2D) | Rectangular grid, open by default |
| [`nqx.graph.LadderGraph(4)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.LadderGraph) | Two open rails with four rungs |
| [`nqx.graph.CircularLadderGraph(4)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.CircularLadderGraph) | Two four-vertex cycles with rungs |
| [`nqx.graph.HalfLadderGraph(3)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.HalfLadderGraph) | Four-vertex chain with four outer links |
| [`nqx.graph.HexagonalLatticeGraph(2, 2)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.HexagonalLatticeGraph) | Finite patch of the hexagonal lattice |
| [`nqx.graph.TriangularLatticeGraph(2, 3)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.TriangularLatticeGraph) | Finite patch of the triangular lattice |
| [`nqx.graph.HypercubeGraph(3)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.HypercubeGraph) | Three-dimensional cube |
| [`nqx.graph.DodecahedralGraph()`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.DodecahedralGraph) | Dodecahedron with twenty vertices |
| [`nqx.graph.K5Graph()`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.K5Graph) | Five vertices joined pairwise |
| [`nqx.graph.SingleVertexGraph(4)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.SingleVertexGraph) | A centre with four outer vertices |

The name `SingleVertexGraph` refers to the central vertex being studied. The graph includes its outer endpoints, so this example has five vertices in total. You can choose the directions of its four edges with `orientation=[1, -1, 1, -1]`, where `1` points away from the centre and `-1` points towards it.

### Boundary conditions

The available boundary options depend on the family. For `Grid2D`, `periodic=True` adds wraparound connections in both lattice directions:

```python
import neuralqx as nqx

open_grid = nqx.graph.Grid2D(3, 3)
periodic_grid = nqx.graph.Grid2D(3, 3, periodic=True)

print(open_grid.n_vertices, open_grid.n_edges)
print(periodic_grid.n_vertices, periodic_grid.n_edges)
```

```text
9 12
9 18
```

A wraparound edge is added only when the corresponding dimension has more than two vertices. This avoids adding a second edge between the same two sites in a two-site direction. Periodicity changes connectivity. It does not automatically change the embedding convention or make the rectangular coordinates a geometric embedding of a torus. Check the [family reference](../../api/public/neuralqx/graph/index.md) for the boundary parameters of the graph you use.

## Supply your own edges

For a custom graph, pass an edge list to [`Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Graph). Write each vertex as a coordinate triple and each edge as a pair of endpoints. Here is a square with a diagonal:

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (1, 1, 0)
d = (0, 1, 0)

graph = nqx.graph.Graph(
    edges=[(a, b), (b, c), (c, d), (d, a), (a, c)],
    is_planar=True,
)

print(graph.n_vertices, graph.n_edges)
print(graph.edge_to_index((a, c)))
```

```text
4 5
4
```

The last edge runs from `a` to `c` and receives index `4`. The graph discovers its vertices from the endpoints in the edge list. There is no separate `vertices` argument for adding isolated vertices. Directions are retained exactly as supplied, and the list order determines the edge indices used by later components.

All coordinates have three entries. Use a zero third coordinate for an ordinary planar example. Distinct coordinate triples identify distinct vertices, while repeated triples refer to the same vertex. The Python names `a`, `b`, and so on are just convenient names in your script.

## Include parallel edges

A multigraph can contain several distinct edges joining the same pair of vertices. You distinguish them with edge keys. For an unkeyed edge list, the constructor assigns these keys automatically:

```python
import neuralqx as nqx

a = (0, 0, 0)
b = (1, 0, 0)
c = (0, 1, 0)

graph = nqx.graph.Graph(
    edges=[(a, b), (a, b), (b, a), (b, c)],
    is_planar=True,
)

print([edge.key for edge in graph.edges])
```

```text
[0, 1, 2, 0]
```

Counting starts at zero for each unordered pair of endpoints. The third edge above has the opposite direction, but it still joins the same two vertices and receives the next key. The edge from `b` to `c` starts a new sequence.

If the keys have a specific meaning in your setup, supply them explicitly. Every key must be a nonnegative Python integer. Use distinct keys for distinct links joining the same endpoint pair, including links with opposite directions. Keys do not need to be consecutive.

```python
import neuralqx as nqx
from neuralqx.utils.typing import Edge, Vertex

a = Vertex(0, 0, 0)
b = Vertex(1, 0, 0)

graph = nqx.graph.Graph(
    edges=[Edge(a, b, key=2), Edge(b, a, key=7)],
    is_planar=True,
)

print([edge.key for edge in graph.edges])
print(graph.edge_to_index((b, a, 7)))
```

```text
[2, 7]
1
```

An input edge list must be consistently keyed or unkeyed. Use only `(start, end)` pairs when asking the constructor to assign keys. For explicit keys, use `(start, end, key)` triples or `Edge` objects throughout. An `Edge` object already carries a key, including when you leave its default value at zero. Repeating `Edge(a, b)` therefore does not request two automatically numbered edges.

## Choose an embedding convention

The base `Graph` constructor uses `is_planar=False` by default. Set `is_planar=True` explicitly when you want planar conventions, as in the examples above.

Graph families express the choice through `non_planar` instead. For example, `Grid2D(2, 3, non_planar=True)` selects three-dimensional conventions and deterministically replaces the grid coordinates with three-dimensional vertex labels. `K5Graph()` already defaults to this convention. `HypercubeGraph(N)` defaults to it when `N > 2`.

These flags select orientation and plotting behaviour. They do not establish the topological planarity of the connectivity. For an embedding with coordinates you want to control precisely, construct `Graph` from those coordinates. The [embedding guide](embeddings.md) explains the signs computed from them and the optional random embedding.

## Keep the graph consistent with your calculation

Construct the graph before creating the [Hilbert space](../hilbert/index.md) and [operators](../operators/index.md) that depend on it. The graph prepares derived information such as index mappings, connectivity, and loops during construction. Editing its returned edge or vertex lists in place can leave that information inconsistent.

When you need a different graph, construct a new object and rebuild the dependent components. If you are reproducing an existing calculation, preserve the input edge order as well as the connectivity. The [labelling guide](labels.md) explains why that order matters when interpreting stored configurations.
