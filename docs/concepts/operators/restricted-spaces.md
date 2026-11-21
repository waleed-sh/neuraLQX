---
title: Restricted spaces and residuals
---

# Restricted spaces and residuals

Sometimes the wavefunction you can afford to represent lives in a smaller space than the operator you want to study. You might optimise at one spin cutoff while measuring a constraint defined at a larger cutoff, or restrict a calculation to a selected basis sector. In these situations, it matters whether you discard the action that leaves the smaller space or include it in the objective.

neuraLQX makes that choice explicit through a basis embedding. A linear restriction and a full squared residual are different constructions. This page explains both and shows how to retain the [squared estimator's factorisation](squared-estimators.md).

## Identify the two bases

Let $\mathcal H_s$ be the source space of the wavefunction and $\mathcal H_t$ the target space of the larger operator. A [`BasisEmbedding`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.BasisEmbedding) identifies each source basis state with a distinct target basis state,

$$
E|\sigma\rangle=|f(\sigma)\rangle,
\qquad E^\dagger E=I_s.
$$

The embedded wavefunction is zero outside that image. Its norm is unchanged, and $P=EE^\dagger$ projects the target space onto the source image.

By default, labels are left unchanged. This is appropriate for genuinely nested bases with the same layout and conventions. A shared configuration width alone does not establish that identification. The caller must ensure that every valid source configuration maps to a valid target configuration. In particular, changing a finite cyclic gauge constraint can change which labels are admissible, so nesting charge ranges does not automatically establish nesting of constrained spaces.

Optional `encode` and `decode` functions support explicit basis injections, including reordered labels or additional fixed degrees of freedom. They must preserve leading batch dimensions and use JAX operations. The interface represents a basis injection, not a general change of basis that maps one configuration to a superposition.

## Restrict a linear operator

For a target-space operator $A$, [`embed_operator(A, embedding)`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.embed_operator) returns an [`EmbeddedOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.EmbeddedOperator) with action

$$
B=E^\dagger A E.
$$

Its rows and connected configurations belong to the source space. Target connections outside the source image are removed. The original target operator still determines all matrix elements and cutoff conventions.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator import embed_operator
from neuralqx.operator.prebuilt.wcl import U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
small = nqx.hilbert.U1Hilbert(graph, cutoff=1)
large = nqx.hilbert.U1Hilbert(graph, cutoff=2)
embedding = nqx.hilbert.BasisEmbedding(small, large)

large_shift = U1HolonomyOperator(large, edge=0)
restricted = embed_operator(large_shift, embedding)
print(np.asarray(restricted.to_dense()).tolist())
```

```text
[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
```

The source labels are $-1,0,1$, while the target labels are $-2,-1,0,1,2$. The larger holonomy's row from charge $1$ connects to $2$, outside the source space, so that restricted row vanishes. A holonomy constructed directly at the smaller cutoff would instead wrap $1$ to $-1$. Restricting a larger operator therefore need not reproduce the native operator at the smaller cutoff.

## Restricting a square differs from squaring a restriction

Squaring the restricted action gives

$$
B^\dagger B=E^\dagger A^\dagger P A E.
$$

It measures only the component of $AE\psi$ that remains in the source image. The full target-space action norm is

$$
Q_{\mathrm{full}}=E^\dagger A^\dagger A E.
$$

The difference is the outgoing residual,

$$
Q_{\mathrm{out}}
=E^\dagger A^\dagger(I-P)AE,
\qquad
Q_{\mathrm{full}}=B^\dagger B+Q_{\mathrm{out}}.
$$

All three quadratic forms are nonnegative. A small retained residual alone does not show that the full constraint annihilates the embedded wavefunction. Part of its action may leave the represented space.

The two orders of construction express the choice directly. `embed_operator(A, embedding).squared()` measures the retained action. `embed_operator(A.squared(), embedding)` pulls back the full quadratic objective, including outgoing action. Passing the existing `Squared` marker preserves its factor and does not square it a second time.

The following small matrix check makes the difference explicit for a uniform source wavefunction.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator import embed_operator
from neuralqx.operator.prebuilt.wcl import U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
small = nqx.hilbert.U1Hilbert(graph, cutoff=1)
large = nqx.hilbert.U1Hilbert(graph, cutoff=2)
embedding = nqx.hilbert.BasisEmbedding(small, large)
action = U1HolonomyOperator(large, edge=0)

target_basis = embedding.embed(small.all_states())
target_indices = np.asarray(large.states_to_numbers(target_basis))
injection = np.eye(large.dimension)[:, target_indices]
psi = np.ones(small.dimension) / np.sqrt(small.dimension)
matrix = np.asarray(action.to_dense())
restricted_matrix = np.asarray(embed_operator(action, embedding).to_dense())

full = np.linalg.norm(matrix @ injection @ psi)**2
retained = np.linalg.norm(restricted_matrix @ psi)**2
outside = np.eye(large.dimension) - injection @ injection.T
outgoing = np.linalg.norm(outside @ matrix @ injection @ psi)**2
np.testing.assert_allclose(full, retained + outgoing)
print(round(full, 6), round(retained, 6), round(outgoing, 6))
```

```text
1.0 0.666667 0.333333
```

The full cyclic shift preserves the norm, while one third of this embedded state's norm is shifted outside the smaller space.

## Measure the full residual in a variational calculation

[`FullResidual`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.FullResidual) records $E^\dagger A^\dagger A E$ directly. Passing `Squared(A)` has the same meaning as passing $A$. The helper `embed_operator` chooses this route for a non-diagonal squared action. For a diagonal parent, which cannot leave the source image, it can retain a simpler restricted squared estimator.

[`outgoing_residual`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.outgoing_residual) extracts only the outgoing contribution from the original target-space squared objective. It expects the factorisation to be retained. An already expanded quadratic matrix no longer identifies which action's outputs should be classified as outgoing.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator import embed_operator, outgoing_residual
from neuralqx.operator.prebuilt.wcl import U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
small = nqx.hilbert.U1Hilbert(graph, cutoff=1)
large = nqx.hilbert.U1Hilbert(graph, cutoff=2)
embedding = nqx.hilbert.BasisEmbedding(small, large)
large_objective = U1HolonomyOperator(large, edge=0).squared()

full_objective = embed_operator(large_objective, embedding, n_probes=None)
outgoing = outgoing_residual(large_objective, embedding, n_probes=None)
network = nqx.nn.RBM(small.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(small, n_chains=4)
state = nqx.vqs.VariationalState(sampler, network, n_samples=128, seed=1)

full_stats = state.expect(full_objective)
outgoing_stats = state.expect(outgoing)
np.testing.assert_allclose(full_stats.mean, 1.0, atol=1e-6)
print(round(float(full_stats.mean), 6))
print(bool(outgoing_stats.mean >= 0))
```

```text
1.0
True
```

The network, sampler, and optimiser remain in the smaller space. The target operator supplies the larger action. The full objective already includes outgoing contributions, so adding `outgoing` to it would count those contributions twice.

`FullResidual(..., part='retained')` and `part='outgoing'` select the two output regions explicitly. Their exact expectations add to the full expectation. Finite estimates from different estimator forms can fluctuate differently, just as the [ordinary and squared local estimators](squared-estimators.md#same-expectation-different-local-values) do.

## Cost, sampling, and ordering

Residual estimators use auxiliary output proposals to reach configurations that Born sampling of the source state cannot visit. The default `n_probes=1` draws one proposal per sampled source configuration, with the appropriate proposal correction. Increasing it can reduce this additional sampling noise. `n_probes=None`, used above, sums all reachable outputs and removes that auxiliary noise at the cost of two-step connectivity. Source-state Monte Carlo sampling remains.

The action must provide its actual adjoint and use fixed operator coefficients for these estimators. Gradients differentiate the complete residual estimator rather than applying the ordinary Hermitian covariance shortcut. Both spaces may be constrained and non-indexable, since the residual calculation need not enumerate their full bases.

The ordering recorded in the target objective is preserved. If an LQX objective contains `Squared(H.adjoint)`, its full residual measures $H^\dagger E\psi$ and represents $E^\dagger HH^\dagger E$. Passing a `SumOfSquares` or an explicit objective sequence to `embed_operator` maps its terms separately and returns a tuple, preserving their local grouping.

Finally, the target operator must include the outputs you want to measure. No embedding can recover transitions that were already truncated by that operator. A larger cutoff, the intended boundary convention, and the physical basis identification should be chosen together.
