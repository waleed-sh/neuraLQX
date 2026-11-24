---
title: Update chains and conditions
---

# Update chains and conditions

An update chain describes several changes to one connected configuration. Its operations run in order on a working target. This is different from writing several emissions, each of which starts from the original input and contributes a separate connection.

## Keep source reads and working values separate

Let $y^{(0)}=\sigma$ be the initial target. After step $k$, the next operation produces $y^{(k+1)}$. A shift uses the current target value, while a value expression built from a source selector continues to read $\sigma$,

$$
\begin{aligned}
\text{shift at }i:\quad
y_i^{(k+1)}&=y_i^{(k)}+f(\sigma),\\
\text{write at }i:\quad
y_i^{(k+1)}&=f(\sigma).
\end{aligned}
$$

This permits a later write to copy an original label even after the coordinate holding it has been changed in the working target.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)

base = s.write(0, 0)
chain = base.shift(0, 1).write(1, s.source_index(0))
assert base.to_program().op_count == 1
assert chain.to_program().op_count == 3

operator = (
    s.SymbolicOperatorBuilder(hilbert, "ordered_updates")
    .globally()
    .emit(chain, matrix_element=1)
    .compile()
)
targets, coefficients = operator.get_conn_padded(
    jnp.array([2, -1], dtype=hilbert.dtype)
)
np.testing.assert_array_equal(targets, [[1, 2]])
np.testing.assert_allclose(coefficients, [1])
```

The first coordinate becomes zero and then one. The final write copies its original value two into the second coordinate. Replacing the last source read with an emitted-state selector is not a way to read the intermediate one. Emitted reads are available for the matrix element only after the update program completes.

Appending methods return new [`Update`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.Update) objects, as the operation-count assertions demonstrate. Reusing `base` in another branch does not include the changes added to `chain`. The fluent operator builder itself is mutable, so do not infer the same immutability for a partially assembled operator.

## Choose an update with a condition

[`Update.cond`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.Update.cond) chooses one of two update programs using a source-based predicate. Both alternatives produce a configuration with the same shape. If `if_false` is omitted, that alternative is the identity update.

```python
reflect_positive = s.Update.cond(
    s.source_index(0) > 0,
    if_true=s.affine(0, scale=-1),
    if_false=s.identity(),
)
reflection = (
    s.SymbolicOperatorBuilder(hilbert, "reflect_positive")
    .globally()
    .emit(reflect_positive, matrix_element=1)
    .compile()
)
sources = jnp.array([[2, -1], [-2, 1]], dtype=hilbert.dtype)
targets, coefficients = reflection.get_conn_padded(sources)
np.testing.assert_array_equal(targets[:, 0], [[-2, -1], [-2, 1]])
np.testing.assert_allclose(coefficients, 1)
```

Both inputs produce an active connection. The second follows the identity alternative, which still carries coefficient one. An omitted alternative does not mean “emit nothing”. This distinction matters for projectors and boundary conditions.

The conditional can itself be followed by another update method. Subsequent changes act on the chosen working target. The predicate and update expressions continue to read the original source, not the partial result of an earlier operation.

## Omit a connection

To omit a contribution conditionally, prefer a term `where` or an `emit_if` without a matching alternative. For example, the following reflects only positive first coordinates and gives zero on the other rows.

```python
positive_only = (
    s.SymbolicOperatorBuilder(hilbert, "positive_only")
    .globally()
    .emit_if(
        s.source_index(0) > 0,
        s.affine(0, scale=-1),
        matrix_element=1,
    )
    .compile()
)
targets, coefficients = positive_only.get_conn_padded(sources)
np.testing.assert_allclose(coefficients[:, 0], [1, 0])
np.testing.assert_array_equal(targets[1, 0], sources[1])
```

[`invalidate`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.Update.invalidate) marks an entire emission as inactive when it appears in the outer update program. An optional reason records why the program was invalidated. It does not raise a runtime exception.

```python
inactive = (
    s.SymbolicOperatorBuilder(hilbert, "inactive")
    .globally()
    .emit(s.identity().invalidate(reason="disabled contribution"))
    .compile()
)
_, coefficients = inactive.get_conn_padded(sources)
np.testing.assert_allclose(coefficients, 0)
```

Keep invalidation at that outer level. The current evaluator does not propagate invalidation from inside a nested `Update.cond` alternative. Use emission conditions for conditional omission, as above, rather than placing `.invalidate()` in `if_false` and expecting that branch to disappear.

## Updates do not compose operators

A chain combines coordinate changes inside one matrix-row contribution. It has one final matrix element, evaluated after the complete update. It does not insert a sum over intermediate configurations or multiply the coefficients of two independent operators.

For the mathematical product $AB$, use [operator composition with `@`](../algebra.md). For a single elementary action that simultaneously changes several charges, use an update chain. Keeping these levels separate makes the definition correspond directly to the intended matrix elements.
