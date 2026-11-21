---
title: Operator algebra
---

# Operator algebra

Operator algebra lets you express a physical construction using smaller pieces. You can add local contributions, introduce coupling constants, form products, and take adjoints. neuraLQX retains these expressions as compositions of operators and evaluates their connected configurations when needed.

This page concerns the linear algebra of [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) objects. [Squared objectives](squared-estimators.md) and [penalties](penalties.md) have additional meaning for the expectation engine and are combined differently.

## The basic operations

For computational operators on the same Hilbert space, the supported expressions are:

| Expression | Mathematical meaning | Result |
| --- | --- | --- |
| `A + B`, `A - B` | $A+B$, $A-B$ | [`SumOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.SumOperator) |
| `c * A`, `A * c`, `-A` | Scalar multiplication | [`ScaledOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ScaledOperator) |
| `A @ B` | Matrix product $AB$ | [`ProductOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ProductOperator) |
| `A ** n` for positive integer `n` | Repeated matrix product $A^n$ | The operator or a product |
| `A.adjoint` | Conjugate transpose $A^\dagger$ | The concrete adjoint implementation |
| `A.squared()` | Factorised positive objective $A^\dagger A$ | [`Squared`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared) |

Use `@` for operator multiplication. `A * B` is not an elementwise product or matrix product. Scalar division and negative or zero powers are not supplied by this algebra. Express division by a known nonzero scalar as multiplication by its reciprocal, and construct an identity explicitly when you need one.

The operators must have compatible Hilbert spaces, not merely equal configuration widths. The constructors check the shared space. A physical identification between different spaces belongs in the [embedding interface](restricted-spaces.md).

## Sums and scalar coefficients

A sum represents

$$
S=\sum_{i=0}^{n-1}c_i A_i.
$$

You can construct it with Python expressions or pass the operators and coefficients to `SumOperator`. A scalar coefficient passed to the constructor is broadcast to every term. A sequence provides one coefficient per term.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator import SumOperator
from neuralqx.operator.prebuilt.wcl import IdentityOperator, NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=0)
identity = IdentityOperator(hilbert)

expression = 2.0 * charge + 0.5 * identity
explicit = SumOperator((charge, identity), coefficients=(2.0, 0.5))
np.testing.assert_allclose(expression.to_dense(), explicit.to_dense())
print(np.diag(np.asarray(expression.to_dense())).tolist())
```

```text
[-1.5, 0.5, 2.5]
```

[`IdentityOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.IdentityOperator) makes the constant shift explicit. A bare nonzero scalar cannot be added to an operator. Python's `sum` works for a nonempty sequence because the initial zero is accepted. An empty Python sum remains the integer zero, so use `IdentityOperator(hilbert, const=0)` when an empty construction must still return an operator.

The sum's padded capacity is the sum of its terms' capacities. Duplicate targets remain separate in the padded output until coalesced or summed by an estimator. A mathematical cancellation therefore need not make the stored expression smaller.

## Product order

`A @ B` means $AB$. Applied to a ket, $B$ acts first. In the row convention used by connectivity, the product is assembled through

$$
(AB)_{\sigma,\sigma'}
=\sum_\tau A_{\sigma,\tau}B_{\tau,\sigma'}.
$$

The row traversal therefore starts with $A$, then follows $B$ from each intermediate configuration $\tau$. These are two views of the same matrix product.

A charge and a cyclic shift provide a small noncommuting example. Their order changes which charge value multiplies the transition.

```python
import neuralqx as nqx
import jax.numpy as jnp
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=0)
shift = U1HolonomyOperator(hilbert, edge=0)
configuration = jnp.array([0], dtype=jnp.int32)

_, charge_then_shift = (charge @ shift).get_conn_padded(configuration)
_, shift_then_charge = (shift @ charge).get_conn_padded(configuration)
print(charge_then_shift.tolist(), shift_then_charge.tolist())
np.testing.assert_allclose(
    (charge @ shift).to_dense(), charge.to_dense() @ shift.to_dense()
)
```

```text
[0.0] [1.0]
```

The padded capacity of a general product is the product of its factors' capacities. Repeated multiplication can therefore create many paths even when fewer distinct targets survive. Prefer a supplied specialised operator when it implements the desired action, and use the [squared estimator](squared-estimators.md) when you only need an action norm.

## Adjoints and Hermiticity

The adjoint is a property, so use `A.adjoint` without parentheses. It obeys

$$
(cA)^\dagger=c^*A^\dagger,
\qquad
(A+B)^\dagger=A^\dagger+B^\dagger,
\qquad
(AB)^\dagger=B^\dagger A^\dagger.
$$

The product implementation reverses the factor order and conjugates coefficients. It does not merely reverse individual shift directions while preserving the original ordering.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=0)
shift = U1HolonomyOperator(hilbert, edge=0)
operator = (1.0 + 2.0j) * (charge @ shift)

matrix = np.asarray(operator.to_dense())
np.testing.assert_allclose(operator.adjoint.to_dense(), matrix.conj().T)
print(charge.is_hermitian, shift.is_hermitian)
```

```text
True False
```

Hermiticity metadata is intentionally conservative. A product of two Hermitian factors is not generally Hermitian unless they commute. Even when you know a particular product is Hermitian, `ProductOperator` does not prove that fact automatically. For a small verification, compare a dense matrix with its conjugate transpose. For a reusable custom implementation, declare Hermiticity only when the action justifies it.

## Matrix squares and positive squares

For a non-Hermitian operator, three common expressions are different:

$$
A^2=AA,
\qquad Q=A^\dagger A,
\qquad \widetilde Q=AA^\dagger.
$$

`A ** 2` and `A @ A` represent the first. `A.squared()` represents the second as a factorised objective. `A.adjoint.squared()` represents the third. If you need the actual product connectivity or a matrix, use `A.adjoint @ A` or `A @ A.adjoint` explicitly.

These distinctions determine which residual is minimised. $\langle A^\dagger A\rangle$ measures $\lVert A\psi\rVert^2/\lVert\psi\rVert^2$, while $\langle AA^\dagger\rangle$ measures the action of $A^\dagger$. The [dedicated derivation](squared-estimators.md) explains their estimators and why factorisation matters.

## Keep separate constraints separate

[`SumOfSquares`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.SumOfSquares) represents

$$
Q=\sum_v A_v^\dagger A_v.
$$

It accepts a nonempty sequence of computational parents and squares each one separately. This differs from `(sum(parents)).squared()`, which also introduces cross terms between different parents. A sum inside one parent retains its own interference before that parent is squared.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator import SumOfSquares
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=0)
independent = SumOfSquares((charge, -charge))
cancelled = charge - charge

parents = [np.asarray(term.operator.to_dense()) for term in independent]
independent_matrix = sum(matrix.conj().T @ matrix for matrix in parents)
print(np.diag(independent_matrix).tolist())
print(np.diag(np.asarray((cancelled.adjoint @ cancelled).to_dense())).tolist())
```

```text
[2.0, 0.0, 2.0]
[0.0, 0.0, 0.0]
```

The two separate residuals cannot cancel each other, while their linear sum vanishes identically. For positive weights $w_v$, scale a parent by $\sqrt{w_v}$ before putting it in `SumOfSquares`, or wrap its squared objective in a [linear penalty](penalties.md#linear-penalties). Scaling a parent by $w_v$ would square that weight as well.

[`VertexSum`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.VertexSum) serves another purpose. It is a linear sum that retains explicit vertex labels, allowing an LQX model to combine Euclidean and Lorentzian contributions at the same vertex before squaring. Use canonical graph vertices and consistent labels across components. The [LQX structure guide](../lqx/structure.md#why-vertex-labels-matter) explains this grouping.

`Squared`, `SumOfSquares`, and penalty wrappers are accepted by the expectation and gradient APIs, but they do not participate in the computational `+` and `@` algebra. To add their objective contributions, pass a nonempty tuple or list to the state or driver. The [penalty examples](penalties.md#combine-objectives-and-optimise) show this in a complete calculation.
