---
title: Defining an operator
---

# Defining an operator

This page constructs one operator from its mathematical definition, checks its action, and measures it with a variational state. Run the Python blocks in order. The [overview](index.md) introduces the terminology, while the later pages explain each language feature independently.

We use three unconstrained U(1) edge charges $q_e\in\{-1,0,1\}$ on a triangle and define

$$
A=\sum_e Q_e^2-t\sum_e(R_e+R_e^\dagger),
\qquad t=\tfrac14.
$$

$Q_e$ is diagonal in the charge basis. The row action of $R_e$ advances the charge cyclically, $-1\mapsto0\mapsto1\mapsto-1$, leaving the other edges unchanged. Its adjoint advances in the opposite direction. This is a small model of diagonal charge costs and cyclic transitions. The wrap at the cutoff is an explicit part of this example, rather than a prescription for every truncated physical operator.

## Choose the space

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
t = 0.25
```

[`U1Hilbert`](../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.U1Hilbert) uses one coordinate per edge here. We deliberately leave gauge invariance unenforced because an independent shift of one edge need not preserve a vertex constraint. The [admissibility guide](spaces.md) explains how to design rules for constrained spaces.

## Add the diagonal and transition terms

```python
definition = (
    s.SymbolicOperatorBuilder(hilbert, "charge_and_cycles", hermitian=True)
    .for_each_site("e")
    .named("charge_cost")
    .emit(matrix_element=s.site("e").value.pow(2), tag="diagonal")
    .for_each_site("e")
    .named("cyclic_transitions")
    .emit(s.shift_mod("e", 1), matrix_element=-t, tag="forward")
    .emit(s.shift_mod("e", -1), matrix_element=-t, tag="reverse")
    .build()
)
operator = definition.compile()
```

Starting another iterator finishes the preceding term and starts a new term in the sum. The two emissions in `cyclic_transitions` share an iterator, but are separate contributions. Each starts from the original configuration. [`shift_mod`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.shift_mod) follows the declared local basis, so both transitions stay inside the three allowed charge values.

Names and tags identify terms and branches in the [IR](ir.md) and diagnostics. They do not change the matrix elements. We can declare this operator Hermitian because the diagonal contribution is real and every cyclic transition has its reverse with the conjugate coefficient. The compiler trusts that declaration.

## Inspect one row

```python
configuration = jnp.array([1, 0, -1], dtype=hilbert.dtype)
connected, matrix_elements = operator.get_conn_padded(configuration)
active = np.asarray(matrix_elements) != 0

for target, coefficient in zip(
    np.asarray(connected)[active], np.asarray(matrix_elements)[active]
):
    print(target.tolist(), float(coefficient))

assert bool(jnp.all(hilbert.is_valid(connected)))
np.testing.assert_allclose(matrix_elements.sum(), 0.5)
```

The diagonal entry is $1^2+0^2+(-1)^2=2$. There are six off-diagonal targets, each with coefficient $-1/4$, giving a row sum of $1/2$. This is the action on a constant wavefunction at that configuration, not an expectation value in a general state.

[`get_conn_padded`](../../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator.get_conn_padded) returns a fixed connection axis. Slots with zero matrix elements are padding and repeat the input configuration. Inspect nonzero entries when counting physical connections. Several contributions to the same diagonal target are combined, so there is one diagonal entry rather than one per edge.

## Check against an independent matrix

This example has only $3^3=27$ basis states, making a dense check practical. The reference below constructs the rows directly from the stated charge rule, without using the symbolic updates.

```python
basis = np.asarray(hilbert.all_states())
row_number = {tuple(row): i for i, row in enumerate(basis)}
reference = np.zeros((len(basis), len(basis)))

for row, source in enumerate(basis):
    reference[row, row] = np.sum(source**2)
    for edge in range(hilbert.size):
        for step in (-1, 1):
            target = source.copy()
            target[edge] = (int(source[edge]) + step + 1) % 3 - 1
            reference[row, row_number[tuple(target)]] -= t

dense = np.asarray(operator.to_dense())
np.testing.assert_allclose(dense, reference)
np.testing.assert_allclose(dense, dense.conj().T)
```

Matrix conversion is a verification tool for this small system. The compiled operator does not require global enumeration to evaluate sampled configurations. For larger spaces, compare local rows or known identities instead of allocating a dense matrix.

## Measure it with a variational state

```python
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(hilbert.size, hidden_features=(8,), rngs=3),
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=4,
)
estimate = state.expect(operator)
print(estimate)
assert bool(jnp.isfinite(estimate.mean))
```

[`VariationalState`](../../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) uses the generated connections in the ordinary local estimator,

$$
A_{\mathrm{loc}}(\sigma)
=\sum_{\sigma'} A_{\sigma,\sigma'}
\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.
$$

The same compiled operator can be passed to a [VMC driver](../../drivers/running.md). For this Hermitian model that would minimise its expectation. A non-Hermitian constraint usually needs a different objective, such as the [positive action norm](algebra.md#powers-and-positive-quadratic-objectives).

On small examples the compiler can emit [`NQXDSL-W401`](messages.md#lint-code-nqxdsl-w401), a structural estimate of connection assembly cost. It does not invalidate the operator or report a measured slowdown. Keep the mathematical checks above, and assess performance on the complete workload before changing deduplication options.
