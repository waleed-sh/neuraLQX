---
title: Writing and rearranging labels
---

# Writing and rearranging labels

Some actions are clearer as explicit assignments or rearrangements than as a sequence of numerical shifts. This page covers [`write`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.write), [`swap`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.swap), [`permute`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.permute), [`affine`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.affine), and [`scatter`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.scatter). Each changes one emission's target, while the coefficient remains a separate expression.

The examples share a three-edge U(1) space. Run the setup and subsequent blocks in order.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
source = jnp.array([1, 0, -1], dtype=hilbert.dtype)

def check_update(name, update, expected):
    operator = (
        s.SymbolicOperatorBuilder(hilbert, name)
        .globally()
        .emit(update, matrix_element=1)
        .compile()
    )
    targets, coefficients = operator.get_conn_padded(source)
    np.testing.assert_array_equal(targets, [expected])
    np.testing.assert_allclose(coefficients, [1])
    assert bool(jnp.all(hilbert.is_valid(targets)))
    return operator
```

## Assign a value

`write(i, value)` sets the selected coordinate to an explicit value or an [expression](../expressions.md). If $f$ is that expression, the rule is $q_i'=f(\sigma)$. The previous working value at $i$ is discarded.

```python
set_zero = check_update("set_zero", s.write(0, 0), [0, 0, -1])
copy_charge = check_update(
    "copy_charge", s.write(0, s.source_index(2)), [-1, 0, -1]
)
```

The copy reads the original third coordinate. Copying labels can map several input configurations to the same output, so a write is not generally invertible or Hermitian. Its meaning is still a valid matrix-row rule when the target belongs to the chosen basis. The [adjoint guide](../algebra.md#adjoints-and-inverse-connections) explains how reverse rows are constructed.

## Exchange and cyclically permute values

`swap(i, j)` exchanges the current values at two coordinates. `permute(i, j, k, ...)` performs a cyclic rotation in the supplied order,

$$
q_{i_0}'=q_{i_1},\quad
q_{i_1}'=q_{i_2},\quad\ldots,\quad
q_{i_{r-1}}'=q_{i_0}.
$$

The participating current values are captured before the writes are made. Thus the operation rotates or exchanges them without overwriting a value that is still needed.

```python
exchange = check_update("swap_ends", s.swap(0, 2), [-1, 0, 1])
rotation = check_update("rotate", s.permute(0, 1, 2), [0, -1, 1])
inverse_rotation = check_update(
    "rotate_back", s.permute(2, 1, 0), [-1, 1, 0]
)
```

Use distinct coordinate references in a permutation unless repeated-index writes are deliberately part of the definition. The ordinary interpretation as a permutation assumes distinct positions. `permute` requires at least two references and represents a cyclic rotation, rather than an arbitrary permutation table. More general permutations can be built from cycles or a [scatter](#write-several-fixed-coordinates).

A rearrangement preserves the multiset of labels, but that alone does not imply physical admissibility. Different sites can have different bases, and a permutation can violate a graph-dependent constraint even on a homogeneous space.

## Affine changes

`affine(i, scale=a, bias=b)` applies

$$
q_i^{\mathrm{new}}=a(\sigma)q_i^{\mathrm{current}}+b(\sigma).
$$

The scale and bias are evaluated from the source. The value being transformed is the current working target at that step. For a symmetric charge basis, changing the sign is a simple example.

```python
sign_flip = check_update(
    "sign_flip", s.affine(0, scale=-1, bias=0), [-1, 0, -1]
)
```

An affine formula does not guarantee closure of the local domain. For example, doubling a charge at its cutoff leaves the basis. It also retains the configuration dtype on assignment, so use values consistent with the representation rather than relying on truncation of fractional results.

## Write several fixed coordinates

`scatter(indices, values)` writes one expression for each static flat index. The two sequences must have equal lengths. It is useful when the action is naturally described by a fixed set of assignments.

```python
exchange_by_assignment = check_update(
    "exchange_by_assignment",
    s.scatter([0, 2], [s.source_index(2), s.source_index(0)]),
    [-1, 0, 1],
)
```

Every expression above reads the source, so the second assignment still sees the original first coordinate. The expressions are not reads from a target that has already been partly overwritten. Repeated indices are processed in order, with the last assignment determining the final value at that index. Prefer unique indices when describing a simultaneous physical rearrangement.

This fixed-index bulk write differs from passing a symbolic destination index to `write`. The latter may choose a destination from runtime labels, while `scatter` records its destination indices during construction. Fixed support usually gives the compiler more information for [connection planning](../../../../developers/symbolic/connectivity.md) and generic adjoints.

For the rules governing combinations of these operations, continue with [update chains and conditions](chaining.md).
