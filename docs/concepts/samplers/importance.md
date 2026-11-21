---
title: Importance sampling
---

# Importance sampling

Ordinary Born sampling follows $|\psi_\theta|^2$. This is a natural choice for quantum expectations, but it can give very little attention to configurations that contribute strongly to a particular estimator. Importance sampling changes the distribution used to draw configurations and accounts for that change when evaluating the physical expectation.

In neuraLQX, these policies are configured on the sampler and carried out through the [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState). The state retains the sampling density with the batch and uses it in expectations, gradients, and stochastic reconfiguration.

## Broaden the distribution with an exponent

[`ImportanceSampling`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.ImportanceSampling) uses a base distribution

$$
q_\alpha(\sigma)
\propto|\psi_\theta(\sigma)|^\alpha,
\qquad 0<\alpha\le2.
$$

For $\alpha<2$, amplitude differences are reduced and the distribution is broader. The physical state remains $\psi_\theta$, and the quantity being measured remains its Born expectation.

On the positive wavefunction support, define unnormalised weights

$$
w(\sigma)=|\psi_\theta(\sigma)|^{2-\alpha}.
$$

For an ordinary local estimator, the expectation becomes

$$
\langle A\rangle_\theta
=\frac{\mathbb E_{q_\alpha}[wA_{\mathrm{loc}}]}
{\mathbb E_{q_\alpha}[w]}
\approx
\frac{\sum_r w_rA_{\mathrm{loc}}(\sigma_r)}{\sum_r w_r}.
$$

This ratio avoids computing either distribution's normalisation. At finite sample size it can have bias, so convergence should be checked as the number of effectively independent samples grows.

## Use a fixed exponent

Set `adaptive=False` when you want the same exponent throughout the run. The following measures a charge probability with $\alpha=1.5$.

```python
import jax.numpy as jnp
import neuralqx as nqx
from neuralqx.operator.prebuilt import ChargeColoringOperator

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
policy = nqx.sampler.ImportanceSampling(exponent=1.5, adaptive=False)
sampler = nqx.sampler.MetropolisLocal(
    hilbert, n_chains=4, importance_sampling=policy
)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=37
)
projector = ChargeColoringOperator(hilbert, site=0, charge=0)

estimate = state.expect(projector)
weights = state.sample_weights
print(state.samples.shape, weights.shape)
print(bool(jnp.isclose(jnp.sum(weights), 1.0)))
print(bool(jnp.isfinite(estimate.mean)))
```

```text
(4, 64, 4) (4, 64)
True
True
```

The [`ChargeColoringOperator`](../../api/public/neuralqx/operator/prebuilt/index.md#neuralqx.operator.prebuilt.ChargeColoringOperator) still measures the physical probability of charge zero. The state supplies the density correction automatically. Averaging the raw zero-charge indicators without those weights would measure their frequency under $q_\alpha$ instead.

Keep the sampler's `machine_pow=2.0`, its default. The importance policy applies the desired exponent through the variational state's sampling callable. The constructor rejects combining an importance policy with a different `machine_pow`.

## Adaptive broadening

Passing `importance_sampling=True` enables the default adaptive policy. It starts at exponent `1.5`, with bounds `0.5` and `2.0`. Gradient evaluations use stochastic directional probes to adjust the exponent for later batches, aiming to reduce the variance of the gradient estimator. An already collected batch keeps its original density.

[`ImportanceSampling`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.ImportanceSampling) exposes the bounds, update frequency, step size, maximum change, and probe count. Adaptation is based on an independent-sample variance objective. It does not estimate Markov autocorrelation times and does not guarantee an improvement at every step. In the current implementation, adaptation is skipped for sums containing penalties, while their weighted expectations and gradients remain available.

`state.sampling_exponent` reports the exponent for the next base batch. `state.sample_measure.exponent` records the exponent belonging to the cached batch. These can differ after adaptation. Use the latter when interpreting an existing set of samples.

## Importance weights and effective sample size

If the normalised weights are $\omega_r=w_r/\sum_s w_s$, their effective sample size is

$$
M_{\mathrm{weights}}=\frac1{\sum_r\omega_r^2}.
$$

`state.sampling_ess` reports this quantity. It approaches the batch size when the weights are even and decreases when a few samples dominate. A very broad proposal can therefore spend effort on configurations with little Born weight.

This diagnostic only measures weight concentration. It does not account for correlations along a Markov chain, nor does it demonstrate that rare but important configurations were visited. Combine it with the [chain diagnostics](diagnostics.md). For ordinary Born sampling, the property returns the retained sample count, which likewise is not a claim of statistical independence.

## Reach configurations connected by an operator

Reducing $\alpha$ does not add support at an exact wavefunction zero. This matters for [squared residuals](../operators/squared-estimators.md#wavefunction-zeros-and-support), since $A\psi$ can be nonzero where $\psi$ is zero.

[`OperatorBlur`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.OperatorBlur) can extend the retained distribution through operator connections. For `Squared(A)`, it follows outgoing action from the columns of $A$, reaching configurations that can contribute to $\lVert A\psi\rVert^2$. The default move probability is `0.5`. Distinct nonzero destinations are used after duplicate connections have been combined, including cancellations. A configuration with no such destination stays unchanged.

This operation is applied after base-chain sampling. It preserves the underlying chain endpoints and tracks the resulting density rather than adding another Metropolis acceptance step. If $\rho$ is the unnormalised base density and $T(y\mid x)$ is the complete blurring transition, the retained density is

$$
\rho_{\mathrm{blur}}(y)=\sum_x T(y\mid x)\rho(x).
$$

The expectation engine uses this density with the appropriate numerator. At a wavefunction node, a squared residual cannot be handled by manually multiplying a zero Born weight by an undefined ordinary local ratio. Keep that calculation in the supported expectation interface.

```python
import jax.numpy as jnp
import neuralqx as nqx
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
action = NumberOperator(hilbert, edge=0) @ U1HolonomyOperator(hilbert, edge=0)
objective = action.squared()
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(
    hilbert,
    n_chains=4,
    operator_blur=nqx.sampler.OperatorBlur(objective, probability=0.5),
)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=128, n_discard_per_chain=32, seed=41
)
estimate = state.expect(objective)
print(state.samples.shape, state.sample_measure.log_density.shape)
print(bool(jnp.isfinite(estimate.mean)))
```

```text
(4, 32, 1) (4, 32)
True
```

The network in this small example has finite log amplitudes, so it demonstrates the interface without imposing nodes. For projected wavefunctions, the model must also provide well-defined derivatives on the configurations being evaluated. Connections chosen for one operator do not automatically cover the support required by an unrelated operator.

## Keep the density with the samples

Use `state.sample()` and `state.expect(...)` when these policies are enabled. Calling the sampler's low-level `sample` method directly only runs its supplied callable. It does not perform adaptive broadening, apply operator blurring, or attach the density needed by the variational-state estimators.

[`SamplingMeasure`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.SamplingMeasure) records the density for a cached batch. If importing externally generated samples into an importance-enabled state, use `state.set_samples(samples, log_density=...)` with the density that actually generated them. Changing the labels or density independently would change the meaning of the estimator.

These policies can be used with the supplied U(1) and SU(2) proposal rules. For new sampling algorithms, the [developer guide](../../developers/samplers/index.md) covers the interfaces needed to retain consistent probabilities and state updates.
