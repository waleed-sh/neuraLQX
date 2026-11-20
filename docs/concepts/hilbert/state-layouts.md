---
title: State layouts
---

# State layouts

Samplers and networks exchange arrays, while a scientific calculation refers to graph edges, charge components, and vertex labels. The state layout connects these two descriptions. It tells you where a quantity lives in a configuration and provides views that recover its graph meaning.

Understanding the layout is especially useful when inspecting samples, preparing an initial configuration, or checking an operator. You usually do not need to calculate flat offsets yourself. The Hilbert object exposes the relevant conversion methods.

## The final axis is one configuration

For a space with `hilbert.size == N`, a single configuration has shape `(N,)`. A batch has shape `(batch_size, N)`, and several leading batch axes are also supported. A call to `random_state(key, size=(2, 3))` therefore returns shape `(2, 3, N)`.

The argument named `size` in `random_state` specifies the leading batch shape. The property `hilbert.size` specifies the length of each configuration. Neither is the dimension of the basis. That separate count is discussed in the [overview](index.md#inspect-a-space-before-using-it).

[`GraphHilbertLayout`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.GraphHilbertLayout) is used by the generic homogeneous and heterogeneous spaces and by U(1). It places edge entries before vertex entries. The graph's [edge and vertex order](../graphs/labels.md) determines the order of entities within those blocks.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(0, 1, 3),
    vertices=True,
)
samples = hilbert.random_state(jax.random.key(5), size=(2, 3))
view = hilbert.view(samples)

print(samples.shape)
print(view.edges.shape, view.vertices.shape)
print(bool(jnp.array_equal(hilbert.flatten(view), samples)))
print(hilbert.layout.vertex_site(graph.vertices[0]))
```

```text
(2, 3, 8)
(2, 3, 4) (2, 3, 4)
True
4
```

The first vertex begins at flat site four because the four edge sites precede it. [`HilbertStateView`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HilbertStateView) holds the `edges` and `vertices` blocks. A block is `None` when that kind of entity carries no degrees of freedom.

## Several components per edge

With a uniform width of $g$ components on $E$ edges, the flat edge block follows

$$
\mathrm{site}(e,c)=cE+e.
$$

Thus, all component zero labels come first, followed by all component one labels. A view changes the axis order to `(..., E, g)`, which makes the components of an individual edge convenient to access. A uniform vertex block follows the same convention, after the full edge block.

The following example assembles an array from graph quantities. The chosen labels make the ordering visible, and every value belongs to the declared local domain.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.HeterogeneousHilbert(
    graph,
    nqx.hilbert.HeterogeneousLocalSpace(
        edge=nqx.hilbert.VectorRange(nqx.hilbert.LocalRange(0, 1, 12), 2),
        vertex=nqx.hilbert.LocalRange(0, 1, 12),
    ),
)
state = hilbert.flatten(
    edges=jnp.array([[0, 4], [1, 5], [2, 6], [3, 7]]),
    vertices=jnp.array([8, 9, 10, 11]),
)

print(state.tolist())
print(hilbert.edge_values(state, graph.edges[1]).tolist())
print(hilbert.layout.edge_site(1, component=1))
print(hilbert.layout.site_to_edge(5))
print(bool(hilbert.is_valid(state)))
```

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
[1, 5]
5
(1, 1)
True
```

Use `edge_values` and `vertex_values` to retrieve the labels of one graph entity. Use `layout.edge_site` or `layout.vertex_site` when an operation needs the flat scalar index, including a specific component. The inverse methods, `site_to_edge` and `site_to_vertex`, return the graph position and component.

Reshaping the flat edge block directly to `(E, g)` would give the wrong grouping. Prefer `view` and `flatten`, which also preserve any leading batch axes. These helpers organise data. They do not enforce a physical constraint or repair invalid label values, so check `is_valid` after assembling a configuration.

## Entities with different widths

A [heterogeneous declaration](local-spaces.md#override-individual-edges-or-vertices) can give one edge a scalar label and another several components. The flat block still visits components in order, skipping entities that do not have that component. Since no rectangular entity array can represent all these widths, the view contains a tuple.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.HeterogeneousHilbert(
    graph,
    nqx.hilbert.HeterogeneousLocalSpace(
        edge=nqx.hilbert.VectorRange(nqx.hilbert.LocalRange(0, 1, 2), 2),
        edge_spaces={0: nqx.hilbert.LocalRange(5, 1, 2)},
    ),
)
samples = hilbert.random_state(jax.random.key(6), size=3)
view = hilbert.view(samples)

print(hilbert.layout.edge_widths)
print(samples.shape)
print(type(view.edges).__name__)
print([values.shape for values in view.edges])
print(hilbert.layout.edge_site(1, component=1))
print(bool(jnp.array_equal(hilbert.flatten(view), samples)))
```

```text
(1, 2, 2, 2)
(3, 7)
tuple
[(3,), (3, 2), (3, 2), (3, 2)]
4
True
```

`view.edges[0]` contains one scalar per sample. `view.edges[1]` contains two components per sample. An entity with no local domain would have `None` in its tuple position. The tuple follows the full graph order even when some entities contribute no entries to the flat array.

This representation avoids padding the scientific configuration with artificial values. If your network requires a different arrangement, perform that transformation in the model while keeping the Hilbert configuration convention consistent with the sampler and operators.

## Values, local indices, and basis numbers

Three different integers can appear in a calculation. A **label value** identifies a local quantum number. A **local index** gives that value's position in its local domain. A **basis number** identifies a complete configuration in an indexed basis.

For a small unconstrained space, you can inspect all three:

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.ExplicitLocalSpace((-2, 0, 3)),
)
state = jnp.array([-2, 0, 3, -2])
local_indices = hilbert.states_to_local_indices(state)
number = hilbert.states_to_numbers(state, validate=True)

print(local_indices.tolist())
print(int(number))
print(hilbert.numbers_to_states(number).tolist())
print(hilbert.all_states(max_states=100).shape)
```

```text
[0, 1, 2, 0]
15
[-2, 0, 3, -2]
(81, 4)
```

The generic product ordering changes the rightmost local index fastest. Here the configuration number is $0\times3^3+1\times3^2+2\times3+0=15$. Different local domains use the corresponding mixed radices.

Check `is_valid` before local conversion if the input may contain invalid values. For global ranking, `validate=True` requests a membership check. Neither conversion should be used as a way of projecting arbitrary arrays into the allowed space.

Generic JAX basis ranking uses 32 bit integer indices and is available only for unconstrained products within its index range. Larger unconstrained products have `states_to_numbers_python` and `numbers_to_states_python`, which use Python integer arithmetic. This makes large labels representable, but it does not make full enumeration affordable. `all_states` allocates the complete basis and has a `max_states` guard.

Generic constrained spaces do not provide automatic basis ranking or enumeration. The [SU(2) exact basis option](su2.md#exact-bases-and-larger-calculations) is a specialised exception. See the [constraint guide](constraints.md#dimension-and-enumeration) for the implications of an unknown dimension.

## The SU(2) spin network layout

The gauge invariant SU(2) space has a specialised layout. Its first block contains edge labels $2j_e$, followed by vertex recoupling labels $2k$. Its view therefore exposes `edge_spins` and `vertex_channels`, rather than the generic `edges` and `vertices` fields.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
samples = hilbert.random_state(jax.random.key(7), size=4)
view = hilbert.view(samples)

print(samples.shape)
print(view.edge_spins.shape, view.vertex_channels.shape)
print(hilbert.layout.channel_site(graph.vertices[0], 0))
print(bool(jnp.array_equal(hilbert.flatten(view), samples)))
```

```text
(4, 15)
(4, 10) (4, 5)
10
True
```

The ten edges of K5 contribute ten spin entries. Each of its five vertices has valence four and contributes one recoupling entry. All fifteen entries are doubled spin labels, including the channels. Dividing edge labels by two gives the physical edge spins.

The [SU(2) guide](su2.md) explains the coupling rules, channel order, and the separate encoded link layout of the Peter–Weyl basis. That link basis stores one integer per edge encoding $(2j,2m,2n)$. Its integers have a different meaning from the edge entries of a gauge invariant spin network.
