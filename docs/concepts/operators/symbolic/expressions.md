---
title: Site selectors and expressions
---

# Site selectors and expressions

A symbolic expression is a recipe for a number that depends on the current configuration. Constructing the expression does not read a sampled state. Its values are supplied later when the compiled operator evaluates a matrix row. This separation lets the same definition act on every configuration and batch.

The main numerical type is [`AmplitudeExpr`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.AmplitudeExpr). Despite its name, it also represents values used in conditions and updates. In an emission, it supplies an operator matrix element, not a wavefunction amplitude.

## Select a site value or its index

[`site("i")`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.site) returns a [`SiteSelector`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SiteSelector). An enclosing [iterator](iterators.md) binds `"i"` to a flat configuration index. `site("i").value` reads the label stored there, whereas `site("i").index` reads the index itself.

For a visit with `i=2` to a state `[1, -1, 0]`, these expressions evaluate to zero and two respectively. The name `"i"` has no geometric meaning by itself. Default execution binds the fields `value` and `index`. Attribute syntax does not create additional physical quantities such as an SU(2) spin, an edge length, or a vertex valence. Construct those quantities from the stored values and the [layout](spaces.md).

Use `.value` when doing arithmetic. Comparisons such as `site("i") > 0` are a supported shorthand for comparing its value. For equality between two expressions, use the explicit comparison helpers explained in [conditions and comparisons](conditions.md#compare-values-rather-than-expression-trees).

## Distinguish the input and completed target

[`emitted`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.emitted) reads the connected configuration after the current emission's entire update has completed. The source selector continues to read the original configuration.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)

operator = (
    s.SymbolicOperatorBuilder(hilbert, "source_and_target")
    .for_each_site("e")
    .where(s.site("e") < 2)
    .emit(
        s.shift("e", 1),
        matrix_element=s.site("e").value + s.emitted("e").value,
    )
    .compile()
)
targets, coefficients = operator.get_conn_padded(
    jnp.array([1], dtype=hilbert.dtype)
)
np.testing.assert_array_equal(targets, [[2]])
np.testing.assert_allclose(coefficients, [3])
```

This row has $q=1$, $q'=2$, and coefficient $q+q'=3$. A target selector belongs in the matrix element. It is not available as a way to inspect an intermediate update or to decide a source-based `where` condition. [Update chains](updates/chaining.md) explain how repeated changes to the same coordinate are evaluated.

For a term without iterator labels, [`source_index(k)`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.source_index) reads the fixed flat coordinate $k$. [`target_index(k)`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.target_index) reads that coordinate in the completed target. These indices are part of the definition and must satisfy $0\le k<\texttt{hilbert.size}$.

## Arithmetic and mathematical functions

Expressions support addition, subtraction, multiplication, division, and unary negation. Numbers can appear on either side. Mathematical helpers construct expression nodes rather than calling NumPy or JAX immediately.

| Operation | Symbolic expression | Meaning |
| --- | --- | --- |
| Power | `q.pow(p)` | $q^p$ |
| Square root | `q.sqrt()` | $\sqrt q$ |
| Complex conjugate | `q.conj()` | $q^*$ |
| Absolute value | `q.abs_()` | $|q|$ |
| Cyclic value remapping | `q.wrap_mod()` | Remap a value into a shared unit-spaced integer basis |

These are methods of [`AmplitudeExpr`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.AmplitudeExpr). Use `.pow(...)` for a scalar expression. Python `**` on an expression is not implemented. By contrast, `operator ** n` means a repeated matrix product, as described in [operator powers](algebra.md#powers-and-positive-quadratic-objectives).

```python
q = s.site("e").value
regularised_magnitude = (q.pow(2) + 1).sqrt()
complex_coefficient = (1 + 2j) * regularised_magnitude

diagonal = (
    s.SymbolicOperatorBuilder(hilbert, "complex_diagonal")
    .for_each_site("e")
    .emit(matrix_element=complex_coefficient.conj())
    .compile()
)
_, coefficients = diagonal.get_conn_padded(
    jnp.array([0], dtype=hilbert.dtype)
)
np.testing.assert_allclose(coefficients, [1 - 2j])
assert np.issubdtype(diagonal.dtype, np.complexfloating)
```

Complex constants in matrix-element expressions promote the operator dtype during construction. This does not change the dtype of configuration labels. Real square roots still require nonnegative arguments. A declared complex output dtype does not by itself turn an integer state read into a complex square-root argument. Keep expressions defined on the states where they are used and consult [numerical diagnostics](messages.md#lint-code-nqxdsl-e003) for divisions.

The site-free `.wrap_mod()` helper has narrower support than the [site-aware `shift_mod` update](updates/modular.md). Use it only with a homogeneous, unit-spaced integer basis. For an irregular or heterogeneous basis, let `shift_mod` use the selected site's local domain.

## Named constants

[`symbol`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.symbol) records a named value with an optional default, dtype, and description. Named values make an IR easier to inspect when a coefficient has physical meaning.

```python
coupling = s.symbol(
    "g", default=0.5, dtype="float64", doc="Quadratic charge coupling"
)
weighted_charge = (
    s.SymbolicOperatorBuilder(hilbert, "weighted_charge", hermitian=True)
    .for_each_site("e")
    .emit(matrix_element=coupling * s.site("e").value.pow(2))
    .compile()
)
_, coefficients = weighted_charge.get_conn_padded(
    jnp.array([2], dtype=hilbert.dtype)
)
np.testing.assert_allclose(coefficients, [2])
```

The standard compiled connectivity call accepts configurations, not an extra dictionary of symbol values. A free `symbol("g")` without a default is therefore unresolved and triggers [`NQXDSL-E001`](messages.md#lint-code-nqxdsl-e001). Supply a numeric coefficient or a default when constructing the definition. To change a coefficient embedded in the definition, construct and compile the new definition. For a scalar multiplying an already compiled operator, ordinary [native operator scaling](algebra.md) can be simpler.

## Construct a longer expression in a function

`matrix_element` also accepts a function that receives an [`ExpressionContext`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.ExpressionContext). The function runs during construction and returns an expression. It is not called once per Monte Carlo sample.

```python
def coefficient(ctx):
    q = ctx.site("e").value
    return ctx.sqrt(ctx.pow(q, 2) + ctx.const(1))

smooth_charge = (
    s.SymbolicOperatorBuilder(hilbert, "smooth_charge", hermitian=True)
    .for_each_site("e")
    .emit(matrix_element=coefficient)
    .compile()
)
_, coefficients = smooth_charge.get_conn_padded(
    jnp.array([0], dtype=hilbert.dtype)
)
np.testing.assert_allclose(coefficients, [1])
```

The context also provides `conj`, `abs_`, `neg`, `source_index`, `target_index`, and explicit comparison helpers. Its `sq_norm(*components)` computes $\sum_a c_a^2$, and `norm2` computes the square root of that sum. These are Euclidean norms for real components. For complex components, form $\sum_a c_a^*c_a$ or use absolute values explicitly.

The `edge_value`, `emitted_edge_value`, `edge_components`, `edge_sq_norm`, and `edge_norm` helpers use the fixed indexing formula $gE+e$ for gauge copy $g$ and edge $e$. They are appropriate when the chosen layout has those consecutive gauge blocks. They do not inspect an arbitrary layout to discover where a physical quantity is stored. The [Hilbert space examples](spaces.md) show how to select coordinates using the actual space instead.
