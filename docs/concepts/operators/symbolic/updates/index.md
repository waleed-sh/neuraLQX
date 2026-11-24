---
title: Configuration updates
---

# Configuration updates

An update determines the connected configuration produced by one emission. If the input is $\sigma$, the update computes a target $\sigma'=T(\sigma)$. Its coefficient is supplied separately by the emission's `matrix_element` expression.

Updates operate on configuration labels. They do not update neural network parameters, advance a sampler, or evolve a state in time. Their direction also follows the [matrix-row convention](../emissions.md#row-convention-and-ladder-operators).

The [`Update`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.Update) class stores an ordered program of elementary changes. The module-level functions below create such programs. The same methods, except the identity factory, can append changes to an existing program.

## Choose the operation

| Constructor | Change to the current target | Guide |
| --- | --- | --- |
| [`identity()`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.identity) | Leave all labels unchanged | Diagonal emissions below |
| [`shift(i, delta)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.shift) | Add a numerical displacement to one label | [Shifts](shifts.md) |
| [`hop(i, j, amount=...)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.hop) | Subtract at $i$ and add at $j$ | [Transport](shifts.md#transport-between-two-coordinates) |
| [`shift_mod(i, delta)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.shift_mod) | Move cyclically through the selected site's local basis | [Modular shifts](modular.md) |
| [`write(i, value)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.write) | Assign a label explicitly | [Writes](writes.md) |
| [`swap(i, j)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.swap) | Exchange two current labels | [Rearrangements](writes.md#exchange-and-cyclically-permute-values) |
| [`permute(i, j, ...)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.permute) | Rotate labels through the listed coordinates | [Rearrangements](writes.md#exchange-and-cyclically-permute-values) |
| [`affine(i, scale=..., bias=...)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.affine) | Multiply a current label and add an offset | [Affine updates](writes.md#affine-changes) |
| [`scatter(indices, values)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.scatter) | Write expressions at fixed flat indices | [Multiple writes](writes.md#write-several-fixed-coordinates) |
| [`Update.cond(...)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.Update.cond) | Choose between two update programs | [Conditional updates](chaining.md#choose-an-update-with-a-condition) |
| [`update.invalidate(...)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.Update.invalidate) | Mark the emission as inactive | [Invalidation](chaining.md#omit-a-connection) |

## Specify a destination coordinate

An update site reference can be an iterator label such as `"i"`, a selector such as `site("i")`, or an integer flat index such as `0`. An `AmplitudeExpr` can also compute an index, although state-dependent destination indices make analysis and generic adjoints more expensive. The expression must evaluate to a valid coordinate index, not a physical label.

For example, `shift("i", 1)` changes the coordinate bound to `i`. `shift(site("i").value, 1)` instead treats the stored value as a destination index. These are different operations. Prefer labels or fixed indices unless dynamic addressing is genuinely part of the model.

## A complete update in an operator

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
exchange = (
    s.SymbolicOperatorBuilder(hilbert, "exchange", hermitian=True)
    .globally()
    .emit(s.swap(0, 1), matrix_element=0.5)
    .compile()
)
source = jnp.array([1, -1], dtype=hilbert.dtype)
targets, coefficients = exchange.get_conn_padded(source)
np.testing.assert_array_equal(targets, [[-1, 1]])
np.testing.assert_allclose(coefficients, [0.5])
```

This exchange preserves the local domain because both coordinates use the same basis. On a heterogeneous space, exchanging values need not preserve either site's allowed labels. Updates do not automatically repair such a mismatch or project into a constrained space.

## Diagonal emissions

`identity()` returns an empty update program. It leaves the target equal to the input, so an emission with that update contributes to the diagonal. Omitting the update argument in `emit` has the same meaning. An identity update with coefficient zero contributes nothing, while one with coefficient $f(\sigma)$ defines the diagonal observable $f$.

An identity contribution is still a contribution during every iterator visit. If a global constant should appear once, use `globally()`. An all-sites iterator multiplies a constant diagonal coefficient by the number of sites.

## Order and validity

Appending an operation returns a new update program. It does not modify the earlier program. During execution, its operations act in order on a working target initially equal to the input. Source expressions continue to refer to the original input throughout that sequence.

Every active final target must belong to the operator's Hilbert space. Use [conditions](../conditions.md) to express valid boundaries and check [global admissibility](../spaces.md) separately from local membership. Configuration dtype is also significant. Assigning a fractional expression to integer labels does not create new fractional basis states, because writes retain the configuration dtype.
