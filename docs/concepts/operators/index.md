---
title: Operators
---

# Operators

An operator describes what you measure or how a quantum state is transformed. In an LQG calculation, a volume operator measures geometry, a holonomy changes the labels along an edge or loop, and a Hamiltonian constraint specifies the condition that a physical state should satisfy. These are different physical questions, but they share a mathematical language and a common interface in neuraLQX.

The [Hilbert space](../hilbert/index.md) tells you which basis configurations exist. An operator tells you how their amplitudes enter a measurement or transformation. The [variational state](../vqs/index.md) supplies those amplitudes. Keeping these ingredients separate lets you measure several observables on the same state or optimise different states against the same constraint.

**neuraLQX operators are matrix free by design.** All supplied physical operators can be used in variational calculations without constructing their matrix over the complete Hilbert space. They provide the connected configurations and matrix elements needed to evaluate their action.

## From a basis configuration to an operator action

Write a wavefunction in the discrete basis as

$$
|\psi\rangle=\sum_\sigma\psi(\sigma)|\sigma\rangle.
$$

A configuration $\sigma$ contains the charges, spins, or intertwiner labels described by your [state layout](../hilbert/state-layouts.md). It is one basis label, not the vector of all wavefunction amplitudes. For an operator $A$,

$$
(A\psi)(\sigma)
=\sum_{\sigma'}A_{\sigma,\sigma'}\psi(\sigma'),
\qquad
A_{\sigma,\sigma'}=\langle\sigma|A|\sigma'\rangle.
$$

The configurations $\sigma'$ with nonzero matrix elements are the configurations connected to $\sigma$. neuraLQX evaluates the required rows through these connections, using the wavefunction only at the configurations needed for the action.

A diagonal observable connects a configuration only to itself. For example, the charge operator on edge $e$ has eigenvalue $q_e$. An off-diagonal operator can connect different configurations, so its expectation also depends on their relative amplitudes and phases. The [application guide](application.md) derives the expectation value and local estimator from this distinction.

## Matrix free by design

If the Hilbert space has dimension $\mathcal D$, a full dense operator contains $\mathcal D^2$ entries. Even a sparse matrix stores connections across the complete basis. A matrix free operator instead retains a rule for calculating the action on the configurations requested by the simulation.

This distinction matters because the number of basis states can grow exponentially with the number of labels. For an unconstrained product of $N$ coordinates with $q$ possible values each, $\mathcal D=q^N$. A neural wavefunction avoids storing all $\mathcal D$ amplitudes, and matrix free operators avoid storing the global operator alongside it. [Monte Carlo expectations](application.md#from-the-expectation-value-to-the-local-estimator) combine sampled configurations, their operator connections, and network amplitude ratios without enumerating the full basis.

The same representation carries through [operator algebra](algebra.md). Expressions such as `A + B`, `c * A`, and `A @ B` retain their constituent operators and generate the required connections when evaluated. Products can still have many connected configurations, so their evaluation cost depends on the action being composed. The [squared estimator](squared-estimators.md) can avoid expanding a quadratic product's connectivity when only its expectation or gradient is needed.

Matrix free describes the representation of the full operator. Individual implementations can use local matrices or cached coefficients to calculate their entries. For example, the [SU(2) volume operator](../lqx/su2-volume/index.md#dense-spectral-evaluation) can diagonalise local intertwiner blocks without constructing the volume matrix over the complete spin network basis. Numerical approximations such as SRQ have their own [accuracy controls](../lqx/su2-volume/accuracy.md), separate from the matrix free representation.

Explicit [matrix conversion](application.md#matrices-batches-and-memory) remains available for small-space checks. You do not need that conversion before passing an operator to a variational state or driver.

## A first operator

[`NumberOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) measures one U(1) edge charge. Here we inspect its action on a single configuration.

```python
import neuralqx as nqx
import jax.numpy as jnp
from neuralqx.operator.prebuilt.wcl import NumberOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=graph.edges[0])
configuration = jnp.array([1, 0, -1], dtype=jnp.int32)

connected, matrix_elements = charge.get_conn_padded(configuration)
print(connected.tolist())
print(matrix_elements.tolist())
```

```text
[[1, 0, -1]]
[1.0]
```

The configuration is unchanged and the matrix element is one, its charge on the selected edge. This is an operator action on a basis configuration. Measuring the mean charge in a superposition additionally requires the wavefunction, which is why expectation values are evaluated through `state.expect(charge)`.

## Operators, observables, and objectives

A Hermitian operator represents an observable with real expectation values. Constraint operators need not be Hermitian, and minimising the expectation of a linear constraint need not enforce its action on the state. A common alternative is a positive quadratic objective,

$$
Q=A^\dagger A,
\qquad
\langle Q\rangle_\psi
=\frac{\lVert A\psi\rVert^2}{\lVert\psi\rVert^2}.
$$

This quantity vanishes precisely when $A|\psi\rangle=0$. The [`Squared`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared) wrapper records this factorisation so that the expectation engine can use the [squared estimator trick](squared-estimators.md). It avoids constructing the full product connectivity simply to measure an action norm.

A penalty can describe a different kind of objective. For instance, you might penalise an expectation value that is too small. Such a cost is a function $f(\langle A\rangle)$ of a state-dependent mean. It is not generally the expectation of a new matrix obtained by applying $f$ to $A$. The [penalties guide](penalties.md) explains the distinction and how to combine these costs with a constraint objective.

## The interfaces you will encounter

[`AbstractOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.AbstractOperator) provides the common Hilbert space, dtype, Hermiticity information, and adjoint interface. [`DiscreteOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator) adds matrix-row connectivity and small-space matrix conversion. [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) implements that connectivity through JAX kernels and supplies the usual operator algebra.

Most supplied physical operators are computational operators. You can add them, scale them, and multiply them with `@`. Objects such as `Squared` and [`PenaltyCost`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.PenaltyCost) instead describe how an objective should be evaluated. They do not expose matrix rows. The [algebra guide](algebra.md) shows which operations apply to each kind of object.

An operator's `.hilbert` identifies its basis. `.dtype` describes its matrix elements rather than its configuration labels. `.adjoint` returns the mathematical adjoint when the concrete operator implements it. `.is_hermitian` reports whether Hermiticity is known to the implementation. A false value can be conservative, especially for a product whose factors happen to commute.

## How this section is organised

Start with [constructing operators](construction.md) to select a Hilbert space, use supplied constructors, and build combinations. Continue with [applying operators](application.md) for connected configurations, local estimators, and expectations, then [operator algebra](algebra.md) for sums, products, adjoints, and local constraint grouping.

The [squared estimator page](squared-estimators.md) develops the factorised norm calculation in detail. [Penalties and objectives](penalties.md) shows how to turn measurements into optimisation terms. [Restricted spaces and residuals](restricted-spaces.md) explains what changes when a state lives inside a larger operator space.

Finally, the [prebuilt operator guide](prebuilt.md) provides a compact map of the supplied families and introduces charge projectors and correlations. Detailed geometric and Hamiltonian formulas remain in the [LQX model guides](../lqx/index.md), where their physical conventions can be read together.
