---
title: Defining spaces
---

# Defining spaces

Begin with the meaning of one configuration. Decide which finite labels belong to each degree of freedom, whether the labels live on graph edges or vertices, and whether a global relation restricts their combinations. Most new spaces can then be assembled from existing local domains and layouts before any subclass is needed.

The [scientist guide to local spaces](../../concepts/hilbert/local-spaces.md) introduces the supplied constructors. This page explains how those constructors support extension work and what a new discrete implementation must provide.

## Scalar domains and vector components

[`LocalRange`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.LocalRange) describes values $a+si$ for $i=0,\ldots,d-1$, where `start=a`, `step=s`, and `size=d`. The step must be nonzero and the size positive. [`ExplicitLocalSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.ExplicitLocalSpace) stores a finite sequence in the chosen basis order. Use distinct values, since repeated labels cannot be inverted uniquely.

[`VectorRange`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.VectorRange) attaches several components with the same scalar domain to one entity. Its `components` determines the number of stored labels. Its `local_size` remains the dimension of each scalar component. An entity with $r$ independent components of local dimension $d$ contributes $d^r$ basis configurations before constraints.

All values in a configuration array share one dtype. For heterogeneous domains the implementation chooses a common dtype from the component spaces. Choose a representation that retains all labels exactly where exact equality matters. Converting physical labels to local indices is a separate operation, not a reason to replace the stored state with those indices everywhere.

## Homogeneous and heterogeneous constructions

[`HomogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HomogeneousDiscreteHilbert) uses a common domain for every enabled edge and vertex. Edges are enabled by default. Set `edges=False, vertices=True` for a vertex-only space, or enable both when both entity types carry the same domain.

For different defaults or per-entity choices, supply a [`HeterogeneousLocalSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousLocalSpace) to [`HeterogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousDiscreteHilbert). Its `edge` and `vertex` fields set defaults. `edge_spaces` and `vertex_spaces` override individual entities using graph indices or labels. A `None` declaration removes the degree of freedom from that entity.

```python
import jax
import neuralqx as nqx

a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
graph = nqx.graph.Graph([(a, b), (b, c)], is_planar=True)
binary = nqx.hilbert.LocalRange(0, 1, 2)

homogeneous = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.VectorRange(binary, components=2)
)
vertex_only = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, binary, edges=False, vertices=True
)
heterogeneous = nqx.hilbert.HeterogeneousDiscreteHilbert(
    graph,
    nqx.hilbert.HeterogeneousLocalSpace(
        edge=binary,
        vertex=nqx.hilbert.LocalRange(-1, 1, 3),
        edge_spaces={1: nqx.hilbert.ExplicitLocalSpace((0, 2, 5))},
        vertex_spaces={0: None},
    ),
)
samples = heterogeneous.random_state(jax.random.key(4), size=8)
assert bool(heterogeneous.is_valid(samples).all())
print(homogeneous.size, homogeneous.dimension)
print(vertex_only.size, vertex_only.dimension)
print(heterogeneous.local_sizes, heterogeneous.dimension)
```

```text
4 16
3 8
(2, 3, 3, 3) 54
```

The heterogeneous dimension is $2\times3\times3\times3=54$. Its graph has three vertices, but the first carries no degree of freedom. Its second edge uses three labels whose numerical spacing is irregular. Both distinctions are represented in the same flat configuration array and recorded in the [layout](states.md).

Attaching a [constraint](constraints.md) selects a subset of these configurations without changing their stored width. Use `with_constraint(...)` to obtain a new space. Keep the same constructed space object with its sampler and operators so that their representation choices remain aligned.

## A complete discrete space without a graph

Implement [`DiscreteHilbertSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.DiscreteHilbertSpace) directly when a graph layout is unnecessary. It requires local dimensions, a dtype, value-index conversions, and local membership checks. Concrete spaces must also expose `size` and `constraint`. Consumers such as the generic local sampler additionally use `local_spaces` to construct local updates.

The following binary register supplies those pieces and an explicit random initialiser. It inherits product-basis indexing, validity checks, and generic proposal dispatch. It deliberately represents an unconstrained register.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import field

class BinaryRegister(nqx.hilbert.DiscreteHilbertSpace):
    size: int = field(static=True)
    constraint: object = field(
        static=True, init=False, default_factory=nqx.hilbert.IdentityConstraint
    )

    def __post_init__(self):
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("size must be a positive Python integer")

    def __hash__(self):
        return hash((type(self), self.size))

    @property
    def dtype(self):
        return np.dtype("int32")

    @property
    def local_sizes(self):
        return (2,) * self.size

    @property
    def local_spaces(self):
        return (nqx.hilbert.LocalRange(0, 1, 2, dtype=self.dtype),) * self.size

    def _array(self, values):
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[-1] != self.size:
            raise ValueError("The trailing axis must equal size")
        return array

    def states_to_local_indices(self, states):
        return self._array(states).astype(jnp.int32)

    def local_indices_to_states(self, indices, *, dtype=None):
        return self._array(indices).astype(self.dtype if dtype is None else dtype)

    def local_states_valid(self, states):
        array = self._array(states)
        return jnp.all((array == 0) | (array == 1), axis=-1)

    def random_state(self, key, size=None, *, dtype=None, max_trials=1024):
        batch = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        indices = jax.random.randint(key, (*batch, self.size), 0, 2)
        return self.local_indices_to_states(indices, dtype=dtype)

def log_value(parameters, configurations):
    return parameters["theta"] * jnp.sum(configurations, axis=-1)

hilbert = BinaryRegister(3)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=2)
state = nqx.vqs.VariationalState(
    sampler=sampler, apply_fn=log_value,
    parameters={"theta": jnp.array(0.1)}, n_samples=8, seed=0,
)
samples = state.samples
assert bool(jnp.all(hilbert.is_valid(samples)))
np.testing.assert_array_equal(
    hilbert.states_to_numbers(hilbert.all_states()), jnp.arange(8)
)
print(hilbert.dimension)
print(samples.shape)
```

```text
8
(2, 4, 3)
```

The membership check rejects values outside the binary domain. The conversion methods assume valid inputs, since conversion and validation have different responsibilities. Do not infer validity solely from a successful integer cast.

The explicit `random_state` is necessary here. The generic product-space random implementation knows the supplied homogeneous and heterogeneous classes, not every possible direct subclass of the abstract discrete interface. The [random generation guide](random.md) explains this extension boundary and how a sampler obtains its initial configurations.

## Extending a local domain

A new [`AbstractLocalSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AbstractLocalSpace) must provide `local_size`, `all_values`, `indices_to_values`, `values_to_indices`, and `contains`. Its values and indices must use one consistent order. The generic graph-backed spaces also read its `dtype`, so expose that metadata as well. The inherited `random_indices` and `random_values` helpers use uniform local indices.

Keep local-domain descriptions immutable and hashable. They are used as static metadata and, in heterogeneous spaces, grouped by domain. A new representation should establish a real benefit over an explicit finite value table, such as avoiding a large materialised table or providing a useful exact conversion.

Before using it in a space, test both conversion directions on every local value and test membership on invalid values. Then check random generation, the intended sampler rule, and a simple diagonal operator. Some consumers construct host-side local tables during tracing, so a custom local domain also needs to work with the particular sampler metadata path you choose.
