---
title: Conditions and comparisons
---

# Conditions and comparisons

Conditions specify the configurations and iterator visits on which a contribution exists. They can enforce truncation boundaries, select particular label sectors, or limit an all-pairs expression to a desired orientation. They must describe the operator's physical domain, rather than compensate for an unrelated invalid update after it has happened.

The language records conditions as [`PredicateExpr`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.PredicateExpr) objects. A comparison builds a predicate tree. It does not inspect a sampled configuration at definition time.

## Restrict an entire term

[`where`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.where) adds a condition shared by all emissions in the current term. Repeated calls combine with logical AND. Their order in the fluent definition does not create separate groups of emissions. A later `where` also belongs to the same term, so put shared conditions before the emissions for clarity.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
q = s.site("e").value

positive_sector = (
    s.SymbolicOperatorBuilder(hilbert, "positive_sector", hermitian=True)
    .for_each_site("e")
    .where(q > 0)
    .where(q <= 2)
    .emit(matrix_element=1)
    .compile()
)
np.testing.assert_allclose(
    np.diag(positive_sector.to_dense()), [0, 0, 0, 1, 1]
)
```

This is a projector onto positive charge for the one-coordinate example. On a larger space, an all-sites iterator would sum local projectors, giving the number of positive coordinates. The iterator is part of the mathematical definition.

## Compose logical expressions

Use `&` for conjunction, `|` for disjunction, and `~` for negation. Parenthesise comparisons before combining them, for example `(q > 0) & (q < 2)`. For a list of conditions, [`ExpressionContext`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.ExpressionContext) provides `all_of`, `any_of`, and `not_`.

```python
ctx = s.ExpressionContext()
nonzero_condition = ctx.any_of(q < 0, q > 0)
nonzero = (
    s.SymbolicOperatorBuilder(hilbert, "nonzero_sector", hermitian=True)
    .for_each_site("e")
    .where(nonzero_condition)
    .emit(matrix_element=1)
    .compile()
)
np.testing.assert_allclose(np.diag(nonzero.to_dense()), [1, 1, 0, 1, 1])
```

Do not use Python `and`, `or`, `not`, or a Python `if` to choose between symbolic conditions. Those operations ask Python to make a decision while building the operator. Likewise, write `(0 < q) & (q < 2)` instead of a chained comparison. Use an explicit comparison rather than passing a numerical expression directly to `where`.

## Compare values rather than expression trees

There is an important distinction in the current expression API. Comparing an `AmplitudeExpr` with a scalar, such as `q == 1`, constructs a runtime predicate. Comparing two `AmplitudeExpr` objects with `==` or `!=` compares their expression trees structurally. It does not compare their eventual numerical values.

For two state-dependent values, use `ctx.eq(left, right)` or `ctx.ne(left, right)`. The context also provides `lt`, `le`, `gt`, and `ge` when an explicit style is useful.

```python
pair_graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
pair_space = nqx.hilbert.U1Hilbert(pair_graph, cutoff=1)
equal_charge = (
    s.SymbolicOperatorBuilder(pair_space, "equal_charge", hermitian=True)
    .for_each(("i", "j"), over=[(0, 1)])
    .where(ctx.eq(s.site("i").value, s.site("j").value))
    .emit(matrix_element=1)
    .compile()
)
sources = jnp.array([[1, 1], [1, -1]], dtype=pair_space.dtype)
_, values = equal_charge.get_conn_padded(sources)
np.testing.assert_allclose(values.sum(axis=-1), [1, 0])
```

This example selects equal charges at runtime. Accidentally using `.value == .value` between the two different selector expressions would produce a constant false condition instead, which can be reported as [`NQXDSL-W101`](messages.md#lint-code-nqxdsl-w101).

## Shared guards and branch guards

If a term should emit both an upward and a downward move with different boundary conditions, use independent `emit_if` calls or separate terms. A shared guard requiring both boundaries to be satisfied would remove both moves at either endpoint. [Emissions and matrix elements](emissions.md#independent-emissions) demonstrates this with ladder operators.

For mutually exclusive alternatives, use `emit_if`, `emit_elseif`, and `emit_else`. For a single emission whose update follows one of two paths, use [`Update.cond`](updates/chaining.md#choose-an-update-with-a-condition). These choices describe different branching structures even when a simple example happens to produce the same result.

## Numerical safety and admissibility

A guard can express that a denominator is nonzero or an occupation is below a cutoff. It does not make the compiler a general theorem prover. In particular, the current [`NQXDSL-E003`](messages.md#lint-code-nqxdsl-e003) check reports direct division by a runtime site value without proving that a surrounding guard excludes zero. The message reference gives a finite-basis construction that avoids this ambiguity.

Use source reads in conditions. Target reads belong to a matrix element after an update has completed. Finally, local bounds are only part of validity. A move can satisfy every scalar bound while violating a vertex Gauss constraint. The [admissibility guide](spaces.md) explains how to check the whole resulting configuration.
