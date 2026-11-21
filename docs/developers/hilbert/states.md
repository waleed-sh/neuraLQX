---
title: State representation
---

# State representation

The shared representation of a discrete configuration is an array with final dimension `hilbert.size`. Its entries are the physical labels of scalar degrees of freedom. Leading dimensions represent independent configurations, chain histories, or another batch arrangement. An operator, sampler, and network must all interpret that final axis in the same way.

The [scientist guide to state layouts](../../concepts/hilbert/state-layouts.md) introduces the supplied representations. When extending them, the main requirement is to keep the mapping between physical entities, flat sites, and local basis indices explicit.

## Three kinds of index

A graph edge index identifies an edge. A flat site index identifies one scalar entry in a configuration. A local basis index identifies one value in that site's allowed domain. They need not be equal.

For example, edge 2 can carry three components. Its component 1 has a flat site determined by the layout, and the physical charge stored there might be $-2$. If that site's domain is $\{-2,-1,0,1,2\}$, the charge $-2$ has local basis index zero.

Use `states_to_local_indices` and `local_indices_to_states` for the final conversion. These methods preserve the array shape. They are not general membership validators. In particular, an explicit local table can return an index for an unknown value through its lookup implementation, and dtype coercion can discard information before a membership check. Validate inputs in the intended label dtype before relying on converted indices.

## Graph layout ordering

[`GraphHilbertLayout`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.GraphHilbertLayout) stores the edge block before the vertex block. Within each block, components are ordered first. For $E$ edges with $r$ components each, the edge part is

$$
(x_{0,0},\ldots,x_{E-1,0},
  x_{0,1},\ldots,x_{E-1,1},\ldots).
$$

Thus the flat edge site for edge $e$ and component $c$ is $cE+e$ in this uniform case. Vertex entries follow the edge block with the same convention. For mixed widths, each component pass visits only entities that possess that component.

Use `layout.edge_site(...)` and `layout.vertex_site(...)` instead of reproducing that arithmetic in new operators. Their inverses, `site_to_edge` and `site_to_vertex`, recover the entity and component. This also avoids confusing graph labels with positional indices when the graph's ordering changes.

## Views preserve the physical organisation

The `view` method returns a [`HilbertStateView`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HilbertStateView) with `edges` and `vertices`. A block with a common scalar width is a dense array. Uniform vector blocks have trailing axes `(n_entities, components)`, even though their flat storage is ordered by component. A mixed-width block is a tuple, with `None` for absent entities.

The following space has two edge components on its first edge, one scalar on its second edge, and scalar labels on two of its three vertices. It verifies that the semantic view and flat array are exact inverses.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
graph = nqx.graph.Graph([(a, b), (b, c)], is_planar=True)
hilbert = nqx.hilbert.HeterogeneousDiscreteHilbert(
    graph,
    nqx.hilbert.HeterogeneousLocalSpace(
        edge=nqx.hilbert.VectorRange(nqx.hilbert.LocalRange(-1, 1, 3), 2),
        vertex=nqx.hilbert.LocalRange(0, 1, 2),
        edge_spaces={1: nqx.hilbert.ExplicitLocalSpace((0, 2, 5))},
        vertex_spaces={0: None},
    ),
)
indices = jnp.array([[0, 1, 2, 1, 0], [2, 2, 1, 0, 1]])
states = hilbert.local_indices_to_states(indices)
view = hilbert.view(states)

np.testing.assert_array_equal(hilbert.flatten(view), states)
np.testing.assert_array_equal(hilbert.states_to_local_indices(states), indices)
assert bool(jnp.all(hilbert.is_valid(states)))
assert view.vertices[0] is None
print(states.tolist())
print(view.edges[0].tolist())
print(hilbert.layout.edge_site(0, component=1))
```

```text
[[-1, 2, 1, 1, 0], [1, 5, 0, 0, 1]]
[[-1, 1], [1, 0]]
2
```

Use `edge_values` or `vertex_values` when a calculation needs one entity. A semantic view reorganises storage. It does not impose a constraint or create an independently sampled degree of freedom.

## Preserve batch axes in custom methods

For input `(..., N)`, a conversion returns `(..., N)`, while a validity predicate returns `(...)`. A custom method can flatten the batch to `(-1, N)` internally, but must restore the leading shape in its result. This matters when the same code receives both ordinary sample batches and connected configurations of shape `(batch, connections, N)`.

The current generic heterogeneous membership implementation handles a single configuration or a rank-two batch directly. When validating multiple leading axes through that implementation, flatten them explicitly and reshape the mask afterward. This also gives a useful reference pattern for a new implementation.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.HeterogeneousDiscreteHilbert(
    graph, nqx.hilbert.HeterogeneousLocalSpace(edge=(0, 2, 5))
)
states = hilbert.random_state(jax.random.key(2), size=(2, 3))

def validity(configurations):
    leading = configurations.shape[:-1]
    flat = configurations.reshape(-1, hilbert.size)
    return hilbert.is_valid(flat).reshape(leading)

mask = jax.jit(validity)(states)
assert bool(jnp.all(mask))
print(states.shape, mask.shape)
```

```text
(2, 3, 1) (2, 3)
```

## Configuration labels are not neural features

A network can turn labels into one-hot features, rescale them, or gather edge components into another arrangement. Those operations belong inside its evaluation. The configurations exchanged with operators and samplers must retain the Hilbert space representation.

For constrained spaces, retain whatever dependent labels the space defines. Do not independently shorten the network input and then pass that reduced array to an operator expecting the full layout. A genuine reduced representation needs compatible conversion, reconstruction, operator, and proposal implementations. The supplied [U(1)](../../concepts/hilbert/u1.md) and [SU(2)](../../concepts/hilbert/su2.md) spaces illustrate why that representation choice is part of the physical construction.
