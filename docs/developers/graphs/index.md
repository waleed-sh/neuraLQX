---
title: Custom graphs
---

# Custom graphs

A custom graph supplies the topology and geometric conventions used by Hilbert spaces and graph-dependent operators. For most extensions, the work is to construct a deterministic sequence of oriented edges and let [`Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Graph) derive connectivity, loops, and orientation tables. A new graph family does not usually need new graph algorithms.

The [scientist guide](../../concepts/graphs/index.md) explains how to construct and inspect a graph. This section focuses on the additional responsibilities of a graph author, particularly stable indexing, the meaning of derived structures, and the boundary between Python topology and numerical JAX code.

## Topology, representation, and geometry

An edge has two endpoints, a stored orientation, and a key distinguishing parallel edges. A vertex is identified by its three coordinates. The order in which edges are supplied determines their indices, and first occurrence in those edges determines vertex order. These choices affect how a Hilbert space lays out its degrees of freedom.

Consequently, two isomorphic graphs need not define interchangeable configuration arrays. Reordering edges can reorder the state representation. Moving vertices can change orientation signs even when adjacency is unchanged. A graph family should document its ordering, coordinate convention, and any choices concerning parallel edges so that operators and saved states can be interpreted consistently.

The `is_planar` flag selects the planar geometric convention, including pairwise orientation signs and two-dimensional plotting coordinates. It is not a test of abstract graph planarity. A nonplanar convention uses three coordinates and signs associated with ordered triples of incident edges. Choose the convention required by the model, then construct coordinates that express the intended embedding.

## Choosing an implementation

A factory returning `Graph` is sufficient when only the edge list changes. A subclass of `Graph` is useful for a named family with validated constructor arguments, additional structural information, or methods that prepare arrays for an operator. In both cases, construct the complete topology before creating dependent Hilbert spaces or operators.

Implement [`AbstractGraph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.AbstractGraph) directly when a different representation or derived-structure algorithm is necessary. Its base constructor handles edge canonicalisation, vertex discovery, and optional embedding. The concrete implementation must supply the mappings, loop methods, dual adjacency, and sign tables expected by downstream code. The [graph contract](contracts.md) specifies those obligations and gives a complete family implemented through `Graph`.

Graph objects are Python topology objects, rather than JAX pytrees carrying trainable data. Use them during construction to prepare edge-index arrays, incidence matrices, or padded neighbourhood tables. Numerical kernels should then operate on those arrays. This avoids traversing Python dictionaries with traced configuration values inside a compiled calculation.

## Derived structures are part of the model

An operator may use only the shortest loops, all loops up to a length bound, or the graph's line-graph dual. These are distinct constructions. In particular, a line-graph dual is built from edge adjacency and is not a dual graph of planar faces. Choosing the wrong structure can change an operator while leaving the program syntactically valid.

The [derived-structures guide](derived-structures.md) explains these conventions and checks them on small graphs. It also shows how an oriented incidence matrix can verify that a dressed loop is closed, and how a three-dimensional example fixes the meaning of orientation signs.

When a graph family is complete, test its representation before embedding it in a full simulation. Mapping round trips, edge-key uniqueness, loop closure, and known sign values give direct evidence about the graph. A successful variational run is much less informative if one of those structural choices is wrong.
