---
title: Derived structures
---

# Derived structures

Graph-dependent operators often consume loops, adjacency, or orientation tables rather than the raw edge list. A custom graph must preserve the meaning of those derived objects. This page makes the conventions explicit and gives small examples whose expected structure can be checked directly.

## Incidence and loop orientation

Stored edge orientation determines incoming and outgoing incidence. Loop traversal is a separate choice. A loop may traverse a stored edge in either direction, and the graph lookup still identifies the same physical edge through its key.

[`Loop`](../../api/public/neuralqx/utils/typing/index.md#neuralqx.utils.typing.Loop) stores a closed, ordered sequence of edges. Construction checks that consecutive endpoints meet and that the last edge returns to the first vertex. It does not establish that every edge belongs to a particular graph. A custom loop generator must also verify membership, including parallel-edge keys.

A dressed loop assigns `creation` when traversal follows the stored edge and `annihilation` when it opposes it. If $c_e$ is $+1$ for forward traversal and $-1$ for reverse traversal, closure implies

$$
B c=0,
\qquad
B_{ve}=\delta_{v,\operatorname{end}(e)}
-\delta_{v,\operatorname{start}(e)}.
$$

This is a useful structural test before constructing loop operators. It is independent of the local charge cutoff or the operator's numerical coefficients.

```python
import neuralqx as nqx
import numpy as np

a, b, c, d = (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)
graph = nqx.graph.Graph(
    [(a, b), (b, c), (c, d), (d, a), (a, c)], is_planar=True
)
vertex_index = {vertex: i for i, vertex in enumerate(graph.vertices)}
incidence = np.zeros((graph.n_vertices, graph.n_edges), dtype=int)
for edge, column in graph.mapping.items():
    incidence[vertex_index[edge.start], column] -= 1
    incidence[vertex_index[edge.end], column] += 1

assert [loop.length for loop in graph.all_loops()] == [3, 3, 4]
assert [loop.length for loop in graph.all_loops(length=3)] == [3, 3]
assert [loop.length for loop in graph.minimal_loops()] == [3, 3]

for loop in graph.all_dressed_loops():
    coefficients = np.zeros(graph.n_edges, dtype=int)
    for traversed, dressing in zip(loop.edges, loop.dressing):
        index = graph.edge_to_index(traversed)
        stored = graph.inverse_mapping[index]
        forward = traversed == stored
        assert dressing["type"] == ("creation" if forward else "annihilation")
        assert dressing["key"] == stored.key
        coefficients[index] += 1 if forward else -1
    np.testing.assert_array_equal(incidence @ coefficients, 0)

print([loop.length for loop in graph.all_loops()])
```

```text
[3, 3, 4]
```

The outer square remains a simple loop even though it has a diagonal chord. The two triangular loops are shorter, so the square does not occur in `minimal_loops()`.

## All loops and minimal loops

[`all_loops`](../../api/public/neuralqx/graph/graph/index.md#neuralqx.graph.graph.Graph.all_loops) enumerates simple loops. It includes self-loops of length one, loops made of two distinct parallel edges, and longer cycles with every distinct parallel-edge choice. Rotations and reverse traversals of the same keyed cycle are represented once. Repeated closed walks are not part of this collection.

The `length` argument is an inclusive upper bound, not a request for loops of exactly that length. Results are ordered deterministically by length and canonical edge sequence. [`all_dressed_loops`](../../api/public/neuralqx/graph/graph/index.md#neuralqx.graph.graph.Graph.all_dressed_loops) applies dressing to the same selection. Returned primal loop objects are detached from the cached minimal-loop data.

[`minimal_loops`](../../api/public/neuralqx/graph/graph/index.md#neuralqx.graph.graph.Graph.minimal_loops) follows a more specific convention. If parallel-edge two-cycles exist, it returns those exclusively. Otherwise it returns the global shortest cycles of the underlying simple graph. It does not provide a minimum cycle basis, all elementary faces, or a separate shortest-cycle collection for every connected component. Its specialised selection does not include self-loops. Use `all_loops` when length-one loops are required.

These distinctions affect operator sums. For example, adding a pair of parallel edges can change which loops appear in an operator built from `minimal_loops()`, even elsewhere in a disconnected graph. Choose an explicit loop collection when the intended physics requires a different selection.

## The line-graph dual

The dual used here is the line graph. Each primal edge becomes a dual vertex. Two dual vertices are adjacent when their corresponding primal edges share a primal endpoint. This construction is available for planar and nonplanar inputs and does not require identifying faces.

`dual_edges` contains pairs of canonical primal `Edge` objects. `dual_adjacency` uses those same objects as keys and neighbours. Dual plotting positions are primal edge midpoints. `minimal_dual_loops()` returns loops whose vertices are represented by primal edges, while `edge_rep=True` replaces those labels with their integer edge indices. These dual loops use the representations documented by the graph interface and should not be assumed to be primal `Loop` objects.

A three-edge star makes the distinction from a face dual particularly clear. Its primal graph is a tree, but its line graph is a triangle because every pair of primal edges meets at the centre.

```python
import neuralqx as nqx

origin = (0, 0, 0)
graph = nqx.graph.Graph(
    [(origin, (1, 0, 0)), (origin, (0, 1, 0)), (origin, (0, 0, 1))],
    is_planar=False,
)
assert graph.minimal_loops() == []
assert len(graph.dual_edges) == 3
assert len(graph.minimal_dual_loops()) == 1
assert len(graph.minimal_dual_loops(edge_rep=True)[0]) == 3

centre = graph.vertices[0]
e_x, e_y, e_z = graph.edges
assert graph.signs[centre][(e_x, e_y, e_z)] == 1
assert graph.signs[centre][(e_y, e_x, e_z)] == -1
print(len(graph.minimal_loops()), len(graph.minimal_dual_loops()))
```

```text
0 1
```

## Orientation signs and embeddings

For a nonplanar graph without parallel-edge ambiguity, the local sign is determined by outward tangent directions at a vertex,

$$
\epsilon_v(e_1,e_2,e_3)
=\operatorname{sgn}\det
\left[\boldsymbol t_{v,e_1},\boldsymbol t_{v,e_2},\boldsymbol t_{v,e_3}\right].
$$

The tangent points from $v$ to the opposite endpoint, regardless of the stored edge direction. Swapping two entries reverses the sign. Coplanar triples give zero, subject to the numerical tolerance used in the sign calculation. The star example fixes a nondegenerate positive orientation along the three coordinate axes.

Parallel edges share endpoint coordinates and therefore need a convention beyond their straight endpoint displacement. The current implementation uses small deterministic tangent perturbations to distinguish those edges when computing nonplanar signs. If an application needs a specific curved-edge embedding or prescribed tangents, establish the required sign convention explicitly instead of assuming that endpoint coordinates encode that extra geometry.

Planar sign tables use ordered edge pairs. Their convention combines clockwise ordering of outgoing geometric directions with incoming or outgoing factors for the stored edge orientations. They are not obtained by taking the three-dimensional determinant above with one argument removed. Preserve the supplied convention when extending algorithms that consume these tables.

Random embedding changes the coordinates before the derived structures are constructed. It can therefore change sign tables and the operators that use them. Retain the embedding seed and the realised geometry when reproducibility matters. The [embedding guide](../../concepts/graphs/embeddings.md) explains the public controls.

## Construction cost and validation

The standard constructor prepares minimal loops, dual structures, and signs. Full simple-cycle enumeration happens on demand, but the number of simple cycles can still grow very rapidly. Supply a length bound when the operator only needs short loops. A direct `AbstractGraph` implementation may use a specialised algorithm, provided it preserves the public interpretation and ordering.

Test disconnected graphs, reversed edges, parallel keys, and degenerate geometries when those are supported by a family. Validate loop membership before dressing, not only loop closure. For geometry-dependent operators, include a graph with known signs and a deliberately coplanar case. These checks make errors in topology and embedding visible before they appear as unexpected [operator matrix elements](../operators/connectivity.md).
