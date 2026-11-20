---
title: Local spaces
---

# Local spaces

A local space specifies the allowed values of one quantum label. A graph Hilbert space places those local degrees of freedom on edges, vertices, or both. Separating these decisions lets you express a uniform charge model, vertex matter coupled to edge labels, or a model with different truncations in different parts of the graph.

This page develops that construction using the generic discrete spaces. For the supplied gauge theories, the [U(1)](u1.md) and [SU(2)](su2.md) constructors also provide the representation and constraints required by those models.

## Describe one scalar label

[`LocalRange`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.LocalRange) represents an arithmetic sequence. Its arguments are the first value, the step, and the **number of values**, rather than a stopping value:

$$
Q = \{a + k\Delta \mid k=0,\ldots,d-1\}.
$$

[`ExplicitLocalSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.ExplicitLocalSpace) supplies the values individually. Use it when the domain is irregular or when you need a particular local basis order.

```python
import neuralqx as nqx

charges = nqx.hilbert.LocalRange(start=-2, step=1, size=5)
selected = nqx.hilbert.ExplicitLocalSpace((-3, 0, 2))

print(charges.all_values().tolist())
print(selected.all_values().tolist())
print(charges.contains([-2, 0, 3]).tolist())
print(selected.values_to_indices([-3, 2]).tolist())
```

```text
[-2, -1, 0, 1, 2]
[-3, 0, 2]
[True, True, False]
[0, 2]
```

A value and its local index serve different purposes. In `selected`, the physical label `-3` has index `0`, and the physical label `2` has index `2`. Configurations store the label values. Index conversion is useful when an algorithm needs consecutive integers, for example to address a local operator matrix. The [layout page](state-layouts.md#values-local-indices-and-basis-numbers) develops this distinction.

Keep explicit values distinct so that each basis label has one local index. Integer labels are convenient for discrete charges, doubled spins, and occupation numbers. Arithmetic ranges can also contain floating point values, but a floating point label domain remains a finite discrete space. It does not define a continuous degree of freedom. Both constructors accept `dtype` when you need to control the stored value type.

## Homogeneous spaces

[`HomogeneousHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HomogeneousDiscreteHilbert), also exported as [`HomogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HomogeneousDiscreteHilbert), uses the same local declaration everywhere that degrees of freedom are enabled. By default, they live on edges, with no vertex labels.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(-1, 1, 3),
)
samples = hilbert.random_state(jax.random.key(1), size=8)

print(hilbert.size, hilbert.local_sizes)
print(hilbert.dimension)
print(hilbert.view(samples).edges.shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
4 (3, 3, 3, 3)
81
(8, 4)
True
```

Here homogeneous means that every enabled entity carries the same domain. It does not require a regular graph, equal vertex valences, a translation invariant network, or a spatially homogeneous physical state. Those are separate properties of the model and wavefunction.

Set `vertices=True` to place the same local degree of freedom on vertices as well as edges. Set `edges=False, vertices=True` for a model whose labels live only on vertices. At least one of the two must be enabled. When edge and vertex domains differ, use the heterogeneous construction below.

Without a constraint, a space with $N$ scalar sites and $d$ labels at each site has dimension $d^N$. Adding a [constraint](constraints.md) restricts the accepted combinations without changing the local declaration or the length of a configuration.

## Several components on each entity

[`VectorRange`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.VectorRange) places several independent copies of one scalar domain on an entity. For example, three charge components per edge can be declared as follows:

```python
import jax
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
edge_domain = nqx.hilbert.VectorRange(
    nqx.hilbert.LocalRange(-1, 1, 3),
    components=3,
)
hilbert = nqx.hilbert.HomogeneousHilbert(graph, edge_domain)
samples = hilbert.random_state(jax.random.key(2), size=5)

print(hilbert.size, hilbert.dimension)
print(samples.shape)
print(hilbert.view(samples).edges.shape)
print(hilbert.edge_values(samples, 0).shape)
```

```text
12 531441
(5, 12)
(5, 4, 3)
(5, 3)
```

There are four graph edges but twelve scalar sites. Each edge has $3^3=27$ possible component assignments before constraints. `VectorRange` supplies that product structure, while any relation among its components must be expressed separately.

The flat array groups all component zero values first, then all component one values, and so on. `view` presents the same data with an edge axis followed by a component axis. Use these helpers instead of assuming that consecutive flat entries belong to the same edge. See [State layouts](state-layouts.md#several-components-per-edge).

## Heterogeneous edge and vertex domains

[`HeterogeneousHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousDiscreteHilbert), also exported as [`HeterogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousDiscreteHilbert), accepts a [`HeterogeneousLocalSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousLocalSpace) declaration. Its `edge` and `vertex` arguments set the defaults for each kind of graph entity.

This example combines three charge labels on every edge with two occupation labels on every vertex:

```python
import jax
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
domains = nqx.hilbert.HeterogeneousLocalSpace(
    edge=nqx.hilbert.LocalRange(-1, 1, 3),
    vertex=nqx.hilbert.ExplicitLocalSpace((0, 1)),
)
hilbert = nqx.hilbert.HeterogeneousHilbert(graph, domains)
samples = hilbert.random_state(jax.random.key(3), size=6)
view = hilbert.view(samples)

print(hilbert.local_sizes)
print(hilbert.size, hilbert.dimension)
print(view.edges.shape, view.vertices.shape)
```

```text
(3, 3, 3, 3, 2, 2, 2, 2)
8 1296
(6, 4) (6, 4)
```

The unconstrained dimension is $3^4 2^4=1296$. More generally, if scalar site $i$ has $d_i$ possible labels, the product basis has dimension

$$
\dim\mathcal H_{\mathrm{product}} = \prod_{i=0}^{N-1} d_i.
$$

Use `local_sizes` to inspect all these dimensions. A single common local dimension is not sufficient for heterogeneous spaces. The numerical array still has one common `dtype`, obtained from the local domains, even when the meanings and allowed values of its entries differ.

This construction defines the basis for a possible edge and vertex model. It does not yet impose a Gauss relation between charge and occupation, or define their interaction. Those belong to [constraints](constraints.md) and [operators](../operators/index.md).

## Override individual edges or vertices

The `edge_spaces` and `vertex_spaces` mappings override the corresponding defaults. Keys can be graph indices or canonical edge and vertex objects. Use the [graph label conventions](../graphs/labels.md) when selecting them.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
domains = nqx.hilbert.HeterogeneousLocalSpace(
    edge=nqx.hilbert.LocalRange(-1, 1, 3),
    edge_spaces={
        0: nqx.hilbert.LocalRange(-2, 1, 5),
        3: None,
    },
    vertex_spaces={
        graph.vertices[0]: nqx.hilbert.ExplicitLocalSpace((0, 1)),
    },
)
hilbert = nqx.hilbert.HeterogeneousHilbert(graph, domains)
samples = hilbert.random_state(jax.random.key(4), size=5)
view = hilbert.view(samples)

print(hilbert.layout.edge_widths)
print(hilbert.layout.vertex_widths)
print(hilbert.local_sizes, hilbert.dimension)
print(view.edges[3] is None)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
(1, 1, 1, 0)
(1, 0, 0, 0)
(5, 3, 3, 2) 90
True
True
```

Edge zero has a larger charge cutoff. Edge three carries no label, and only the first vertex carries an occupation label. `None` removes the degree of freedom from the layout, not the edge or vertex from the graph. Downstream operators must support the resulting layout.

Overrides can also change the number of components by supplying a `VectorRange`. When entities have different widths, `view.edges` or `view.vertices` becomes a tuple indexed by graph position. This preserves each entity's actual shape, including `None` for an entity without labels. The [mixed layout example](state-layouts.md#entities-with-different-widths) shows how to inspect and reconstruct such configurations.

Once the local domains are in place, use a [constraint](constraints.md) to select the allowed combinations. For changes that require a new representation or specialised state generation, continue to the [custom Hilbert space guide](../../developers/hilbert/index.md).
