---
title: Indexing and enumeration
---

# Indexing and enumeration

Indexing gives each basis configuration a stable integer label. It supports exact comparisons, small operator matrices, and a variational state's [array representation](../../concepts/vqs/arrays.md). A large simulation does not need to enumerate the basis merely to sample configurations or evaluate matrix free operators.

Keep three quantities distinct. `size` is the number of stored scalar labels. `unconstrained_dimension` is the product of their local dimensions. `dimension` is the exact number of allowed configurations when the constraint can provide it cheaply, and can otherwise be `None`. The `n_states` property requires a known dimension and raises when it is unavailable.

## Product-basis order

Let $i_r\in\{0,\ldots,d_r-1\}$ be the local basis index at flat site $r$. The generic product order ranks a configuration as

$$
\operatorname{rank}(x)
=\sum_{r=0}^{N-1}i_r(x)\prod_{s=r+1}^{N-1}d_s.
$$

The rightmost local index changes fastest. With dimensions $(2,3)$, the local-index rows are $(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)$. Physical labels can have different values or spacing. Their order comes from the local domains, not numerical sorting performed during ranking.

[`states_to_numbers`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.DiscreteHilbertSpace.states_to_numbers) and [`numbers_to_states`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.DiscreteHilbertSpace.numbers_to_states) implement the generic JAX conversion for small unconstrained product spaces.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
graph = nqx.graph.Graph([(a, b), (b, c)], is_planar=True)
hilbert = nqx.hilbert.HeterogeneousDiscreteHilbert(
    graph,
    nqx.hilbert.HeterogeneousLocalSpace(
        edge=(-1, 1), edge_spaces={1: (0, 2, 5)}
    ),
)
states = jnp.array([[1, 5], [-1, 2]], dtype=hilbert.dtype)
numbers = hilbert.states_to_numbers(states, validate=True)
restored = jax.jit(hilbert.numbers_to_states)(numbers)
np.testing.assert_array_equal(restored, states)
np.testing.assert_array_equal(
    hilbert.states_to_numbers(hilbert.all_states()), jnp.arange(6)
)
print(numbers.tolist())
print(hilbert.all_states().tolist())
```

```text
[5, 1]
[[-1, 0], [-1, 2], [-1, 5], [1, 0], [1, 2], [1, 5]]
```

The `validate=True` option performs a host-side validity check. Use it before compiled numerical work. Inside a compiled rank calculation, use already validated configurations and the ordinary conversion path. For JAX unranking, ensure the input integers lie in $[0,\dim\mathcal H)$, since the generic conversion is not a comprehensive bounds validator.

## Large integer labels do not require a full basis

The generic JAX ranking path uses 32-bit integers and requires an unconstrained product dimension no greater than $2^{31}-1$. `is_indexable` records that generic capability. It is not a statement that the whole basis would fit in memory.

[`states_to_numbers_python`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.DiscreteHilbertSpace.states_to_numbers_python) and [`numbers_to_states_python`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.DiscreteHilbertSpace.numbers_to_states_python) use arbitrary-size Python integer arithmetic for larger unconstrained product spaces. They are host-side operations, not differentiable or JIT-compatible numerical kernels. They can rank a few configurations without allocating the basis.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

edges = [((i, 0, 0), (i + 1, 0, 0)) for i in range(40)]
graph = nqx.graph.Graph(edges, is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
state = jnp.ones(hilbert.size, dtype=hilbert.dtype)
number = hilbert.states_to_numbers_python(state, validate=True)
np.testing.assert_array_equal(hilbert.numbers_to_states_python(number), state)
print(hilbert.is_indexable)
print(number)
```

```text
False
1099511627775
```

The label is $2^{40}-1$. Computing it is inexpensive compared with materialising $2^{40}$ configurations. `all_states(max_states=...)` explicitly enumerates the unconstrained basis and defaults to a guard of one million states. Increasing that guard authorises a larger allocation, not a more efficient representation. Dense operator matrices require substantially more memory again.

## Constrained counting and constrained indexing

A constraint with a known dimension does not automatically provide a rank or unrank algorithm. The generic implementation leaves constrained spaces non-indexable and rejects generic constrained enumeration, including through the Python integer helpers. Specialised physical spaces may provide their own capabilities, so consult the [U(1)](../../concepts/hilbert/u1.md) or [SU(2)](../../concepts/hilbert/su2.md) contract when using them.

For a tiny development reference, explicitly enumerate an unconstrained parent and filter its configurations. This produces a reference array in the parent's retained order. It does not install an indexing implementation on the constrained space.

```python
import jax.numpy as jnp
import neuralqx as nqx

edges = [((i, 0, 0), (i + 1, 0, 0)) for i in range(4)]
graph = nqx.graph.Graph(edges, is_planar=True)
parent = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
sector = parent.with_constraint(
    nqx.hilbert.LinearConstraint.sum(range(parent.size), target=2)
)
parent_basis = parent.all_states(max_states=16)
reference_basis = parent_basis[sector.is_valid(parent_basis)]
assert bool(jnp.all(jnp.sum(reference_basis, axis=-1) == 2))
print(reference_basis.shape)
print(sector.dimension, sector.is_indexable)
```

```text
(6, 4)
None False
```

Here the exact sector size is six, but `LinearConstraint` does not supply that count. The custom fixed-occupation constraint in [the constraint guide](constraints.md#define-a-reusable-constraint-class) does supply the binomial dimension. Neither declaration alone changes the generic indexing capability.

## Supplying a specialised constrained index

A complete constrained index needs a stable ordering, a map from every valid configuration to its rank, and its inverse. For fixed occupation, combinatorial counting can rank configurations without visiting all product states. For a gauge construction, the index may instead operate on independent coordinates and reconstruct the remaining labels.

Override the relevant space methods consistently, including `is_indexable`, rank, unrank, and enumeration where supported. Merely setting `is_indexable=True` does not replace the inherited functions' explicit rejection of constrained spaces. If a consumer calls a module-level indexing helper directly, overriding the method also does not change that separate helper. Test the complete consumer path you need, such as `to_array` or operator matrix conversion.

Check both inverse identities on a small sector,

$$
\operatorname{unrank}(\operatorname{rank}(x))=x,
\qquad
\operatorname{rank}(\operatorname{unrank}(r))=r,
$$

and verify that enumeration visits each valid configuration once. Document whether the implementation is compatible with JAX transformations, the integer range it supports, and what it does for invalid input. Stable indexing is also a scientific reproducibility choice, since a saved state vector only has meaning together with its basis order.
