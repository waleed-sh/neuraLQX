---
title: Algebra and adjoints
---

# Algebra and adjoints

A compiled symbolic definition is a native [`ComputationalOperator`](../../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator). Its sums, products, adjoints, and expectation values use the same interfaces as a supplied physical operator. This lets you combine a symbolic term with an existing geometric operator or a custom JAX kernel.

There are two levels to distinguish. Expressions inside an emission calculate scalar matrix elements. Algebra between operator objects combines complete matrix actions, including any sum over intermediate configurations.

## Sums, scaling, and products

For operators $A,B$ on the same Hilbert space,

$$
(A+B)_{\sigma,\sigma'}=A_{\sigma,\sigma'}+B_{\sigma,\sigma'},
\qquad
(AB)_{\sigma,\sigma'}=\sum_\tau A_{\sigma,\tau}B_{\tau,\sigma'}.
$$

Use `A + B`, `c * A`, and `A @ B`. The right factor acts first on a ket. These combinations retain matrix free actions, although a product can have substantially more connections than either factor.

```python
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "charge", hermitian=True)
    .for_each_site("e")
    .emit(matrix_element=s.site("e").value)
    .build()
)
charge = definition.compile()
charge_squared = charge @ charge
combined = 2 * charge + charge_squared
np.testing.assert_allclose(combined.to_dense(), np.diag([-1, 0, 3]))
```

Scaling an uncompiled symbolic definition by a Python scalar preserves a symbolic definition. Adding two definitions with `+`, multiplying them with `@`, requesting their adjoint, or taking an operator power compiles them and uses native operator algebra. Do not expect `definition_a + definition_b` to retain one combined symbolic IR for later compiler analysis.

To keep several symbolic definitions in one IR, use [`SymbolicOperatorSum`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicOperatorSum), or place their terms in one builder.

```python
symbolic_sum = s.SymbolicOperatorSum(
    hilbert, (definition, 2 * definition), name="three_charges"
)
assert symbolic_sum.to_ir().term_count == 2
np.testing.assert_allclose(
    symbolic_sum.compile().to_dense(), np.diag([-3, 0, 3])
)
```

The aggregate retains each definition's contributions, allowing the symbolic compiler to analyse connections across those terms. It does not turn an operator product into a scalar expression.

## Powers and positive quadratic objectives

Three common squares have different meanings.

| Form | Meaning |
| --- | --- |
| `expression.pow(2)` | Square one scalar expression inside a coefficient or condition |
| `operator ** 2` | Form the matrix product $A^2=A@A$ |
| `operator.squared()` | Record the positive action norm $A^\dagger A$ for the expectation engine |

Operator powers currently accept positive Python integers. Zero, negative, and fractional powers are not implemented by this interface. Repeated products can increase connectivity, so a high power is not automatically cheap because the representation is matrix free.

```python
np.testing.assert_allclose((charge ** 2).to_dense(), np.diag([1, 0, 1]))
positive_objective = charge.squared()
assert isinstance(positive_objective, nqx.operator.Squared)
```

[`Squared`](../../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared) is an estimator marker rather than an operator exposing its own matrix rows. For a variational state,

$$
\langle A^\dagger A\rangle_\psi
=\frac{\lVert A\psi\rVert^2}{\lVert\psi\rVert^2}.
$$

Pass the marker to `state.expect` or a compatible driver to use the [squared estimator](../squared-estimators.md). If you need the product's connections explicitly, use `A.adjoint @ A`. For a non-Hermitian $A$, neither this product nor its estimator marker is generally equal to `A ** 2`. To represent $AA^\dagger$ through an action norm, use `A.adjoint.squared()`.

## Adjoints and inverse connections

The adjoint satisfies

$$
(A^\dagger)_{\sigma,\sigma'}
=\overline{A_{\sigma',\sigma}}.
$$

It reverses which rows and columns are connected and conjugates their coefficients. Negating a shift is sufficient only when it also gives the correct reversed coefficient and boundary conditions.

The generic symbolic adjoint searches candidate source configurations on the coordinates that the forward rules can modify. It checks the Hilbert constraint and sums matching forward contributions before conjugating. It does not need to enumerate the global basis, which makes local adjoints usable on otherwise nonindexable spaces.

The current candidate limit is 65,536. An update with broad support, or a state-dependent destination whose support cannot be narrowed, can exceed it. In that case, define the reverse rules explicitly and pair the compiled operators with [`with_adjoint`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.CompiledOperator.with_adjoint).

```python
occupations = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 3)
)
n = s.site("i").value
annihilation = (
    s.SymbolicOperatorBuilder(occupations, "annihilation")
    .for_each_site("i")
    .where(n < 2)
    .emit(s.shift("i", 1), matrix_element=(n + 1).sqrt())
    .compile()
)
creation = (
    s.SymbolicOperatorBuilder(occupations, "creation")
    .for_each_site("i")
    .where(n > 0)
    .emit(s.shift("i", -1), matrix_element=n.sqrt())
    .compile()
)
paired = annihilation.with_adjoint(creation)
np.testing.assert_allclose(
    paired.adjoint.to_dense(), np.asarray(annihilation.to_dense()).conj().T
)
np.testing.assert_allclose(
    (paired.adjoint @ paired).to_dense(), np.diag([0, 1, 2])
)
```

The supplied reverse operator must use the same Hilbert space. Pairing is a declaration by the author, so verify the reverse action independently on a small example. For complex coefficients, include their conjugation explicitly.

## Hermiticity is a declared property

`hermitian=True` tells the compiled operator that its adjoint is itself. It does not ask the compiler to prove this statement or symmetrise the rules. A diagonal operator is Hermitian when its diagonal coefficients are real, and a transition operator requires conjugate reversed connections.

For a small indexable space, compare `operator.to_dense()` with its conjugate transpose. For a larger space, check representative forward and reverse rows and the physical argument establishing the identity. The [direct operator composition guide](../../../developers/operators/composition.md) discusses the same contracts for custom kernels.
