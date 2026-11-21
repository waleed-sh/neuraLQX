---
title: Expectations and gradients
---

# Expectations and gradients

A variational state evaluates observables by combining [operator connections](../operators/index.md#matrix-free-by-design) with the current wavefunction and sampled configurations. It can return the individual local values, their statistics, or the derivative of an objective with respect to the trainable parameters.

The ordinary formulas below assume Born sampling, `machine_pow=2`, and an operator that does not itself depend on the network parameters. [Importance sampling](../samplers/importance.md) adds the density correction needed when configurations come from another supported distribution.

## From amplitudes to a sampled expectation

For a wavefunction $\psi_\theta$ with norm $Z_\theta$, the expectation is

$$
E_A(\theta)
=\frac{1}{Z_\theta}
\sum_{\sigma,\sigma'}
\psi_\theta(\sigma)^*
A_{\sigma,\sigma'}\psi_\theta(\sigma').
$$

Factoring out the Born probability gives

$$
\begin{aligned}
E_A(\theta)
&=\mathbb E_{p_\theta}\!\left[A_{\mathrm{loc}}(\sigma)\right],
\\[6pt]
A_{\mathrm{loc}}(\sigma)
&=\sum_{\sigma'} A_{\sigma,\sigma'}
\exp\!\left[f_\theta(\sigma')-f_\theta(\sigma)\right].
\end{aligned}
$$

The local ratio is evaluated on configurations with nonzero wavefunction amplitude. The [operator application guide](../operators/application.md#from-the-expectation-value-to-the-local-estimator) derives this expression in detail.

For $M$ retained configurations, the sample mean is

$$
\widehat E_A
=\frac1M\sum_{r=1}^{M}A_{\mathrm{loc}}(\sigma_r).
$$

The operator connections inside each local value are still evaluated. Sampling replaces the sum over source configurations, while [chunking](batching.md) controls how those evaluations are scheduled.

## Inspect local values and statistics

[`local_estimators`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.AbstractVariationalState.local_estimators) returns a [`LocalEstimators`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.LocalEstimators) object. Its `.data` preserves the chain and time axes, and `.to_stats()` summarises those values. [`expect`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.AbstractVariationalState.expect) returns the statistics directly.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(hilbert.size, hidden_features=(8,), rngs=0),
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=1,
)
charge = NumberOperator(hilbert, edge=0)
observable = charge @ charge

local = state.local_estimators(observable)
stats = state.expect(observable)
np.testing.assert_allclose(stats.mean, jnp.mean(local.data), atol=1e-7)
print(local.data.shape)
print(int(stats.n_samples))
print(stats)
```

The local values have shape `(4, 32)` and there are 128 retained configurations. Both calls use the same cached samples at the same parameters.

[`Stats`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.Stats) exposes `mean`, `variance`, `error_of_mean`, `n_samples`, and `r_hat`, with `R_hat` as an alias. The variance describes the spread of local values. It is not the uncertainty of their mean. Keep chain axes intact so the reported error and split-chain diagnostic have their intended meaning, and use the [sampling diagnostics](../samplers/diagnostics.md#read-the-reported-statistics) when interpreting them.

For a complex local estimator, the variance uses squared magnitudes of deviations. A Hermitian observable has a real exact expectation, but an off-diagonal observable's finite-sample mean can have an imaginary component. A persistent imaginary part deserves a convergence and implementation check rather than being silently removed.

Passing a list or tuple to `expect` means the expectation of the sum. For separate measurements, evaluate each operator separately, for example `results = {name: state.expect(op) for name, op in observables.items()}`. The common cached batch is useful when comparing related observables.

## The gradient of a Hermitian expectation

For real parameter coordinates $\theta_i$, define the logarithmic derivative

$$
O_i(\sigma)
=\partial_{\theta_i}f_\theta(\sigma).
$$

When $A=A^\dagger$, the Born expectation has gradient

$$
\frac{\partial E_A}{\partial\theta_i}
=2\operatorname{Re}
\left[
\mathbb E_{p_\theta}\!\left[O_i^*A_{\mathrm{loc}}\right]
-\mathbb E_{p_\theta}\!\left[O_i^*\right]E_A
\right].
$$

The expression in brackets is the covariance force $F_i$. [`expect_and_forces`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.AbstractVariationalState.expect_and_forces) returns it, while [`expect_and_grad`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.AbstractVariationalState.expect_and_grad) returns the gradient in the convention consumed by the optimiser. Force leaves can be complex even when the corresponding parameters are real.

For a one-edge model $f_a(q)=a q$ with $q\in\{-1,0,1\}$, the state is uniform at $a=0$. The mean charge vanishes there, and

$$
\left.\frac{d\langle q\rangle}{da}\right|_{a=0}
=2\left(\langle q^2\rangle-\langle q\rangle^2\right)
=\frac43.
$$

This gives an exact small-basis check of the covariance estimator.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator

def log_amplitude(parameters, configurations):
    return parameters["a"] * configurations[..., 0]

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=1),
    apply_fn=log_amplitude,
    parameters={"a": jnp.array(0.0)},
    n_samples=3,
    seed=0,
)
state.samples = hilbert.all_states()[None, ...]
charge = NumberOperator(hilbert, edge=0)

stats, gradient = state.expect_and_grad(charge)
_, forces = state.expect_and_forces(charge)

def exact_mean(a):
    q = hilbert.all_states()[:, 0]
    probabilities = jax.nn.softmax(2 * a * q)
    return jnp.sum(probabilities * q)

reference = jax.grad(exact_mean)(state.parameters["a"])
np.testing.assert_allclose(gradient["a"], reference, atol=1e-7)
np.testing.assert_allclose(gradient["a"], 2 * forces["a"].real, atol=1e-7)
print(round(float(stats.mean.real), 6))
print(round(float(gradient["a"]), 6))
```

```text
0.0
1.333333
```

Each basis configuration has equal Born weight at the chosen parameter value, so the supplied batch evaluates the covariance exactly. That replacement is not valid at a generic nonuniform wavefunction. The statistics of this deliberately enumerated batch should not be interpreted as evidence of Monte Carlo convergence.

## Why differentiating a sampled mean is not sufficient

The distribution itself depends on the parameters. For a possibly parameter-dependent local quantity $L_\theta$, real-coordinate differentiation gives

$$
\begin{aligned}
\partial_i\mathbb E_{p_\theta}[L_\theta]
&=
\mathbb E_{p_\theta}[\partial_i L_\theta]
\\[6pt]
&\quad+
\mathbb E_{p_\theta}\!\left[
(L_\theta-\mathbb E_{p_\theta}[L_\theta])
\partial_i\log p_\theta
\right].
\end{aligned}
$$

Differentiating only the local values at a fixed sample batch misses the change in probability weights. For ordinary Hermitian expectations, the algebra reduces to the covariance formula above. For other estimators, the explicit derivative of $L_\theta$ must be retained.

This matters for [`Squared`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared) objectives. Their positive local values involve $|A_{\mathrm{loc}}|^2$, which depends on the network through amplitude ratios. `expect_and_grad` selects the required derivative path automatically. `expect_and_forces` does not accept this objective as an ordinary covariance-only calculation. See the [squared estimator guide](../operators/squared-estimators.md) for the support conditions and full derivation.

The same interface handles [expectation-level penalties](../operators/penalties.md) through their corresponding chain rule. Leave `use_covariance=None` for automatic selection. Forcing a covariance path does not make it valid for an incompatible objective.

## Apply one optimiser step

The returned gradient has the same pytree structure as `state.parameters`. A manual Optax update can therefore use the familiar `update` and `apply_updates` sequence.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), rngs=2),
    n_samples=128,
    sampler_seed=3,
)
action = (
    NumberOperator(hilbert, edge=0)
    + 0.3 * U1HolonomyOperator(hilbert, edge=0)
)
objective = action.squared()
optimiser = optax.sgd(learning_rate=0.01)
optimiser_state = optimiser.init(state.parameters)

stats, gradient = state.expect_and_grad(objective)
assert all(
    bool(jnp.all(jnp.isfinite(leaf)))
    for leaf in jax.tree_util.tree_leaves(gradient)
)
updates, optimiser_state = optimiser.update(
    gradient, optimiser_state, state.parameters
)
state.parameters = optax.apply_updates(state.parameters, updates)
print(jax.tree_util.tree_structure(gradient)
      == jax.tree_util.tree_structure(state.parameters))
```

```text
True
```

`stats` describes the wavefunction before this update. Assigning the new parameters invalidates the cached samples. The next expectation draws from the updated state. A single noisy step is not guaranteed to lower the next measured mean.

A [VMC driver](../drivers/index.md) manages this repeated process. If you supply `preconditioner=...` to `expect_and_grad`, its second result is the preconditioned direction rather than the raw gradient. The [SR guide](../optimisers/sr.md) explains that choice and its relation to `state.qgt()`.

## Complex parameters and alternative sampling

For a real objective and a complex parameter $z=u+iv$, the optimiser gradient is represented as

$$
g_z=\partial_u E+i\partial_v E.
$$

The VQS interfaces handle this convention. Complex output, complex parameters, and holomorphic differentiation remain separate choices, explained in [complex and holomorphic models](../networks/complex.md). A general non-Hermitian expectation can itself be complex. Choose a meaningful real objective, such as a Hermitian combination or an action norm, before treating it as a quantity to minimise.

For supported importance sampling, the state retains the proposal density associated with the cached batch and applies Born corrections in the estimators and gradients. Use `state.expect(...)` rather than taking an unweighted average of unrelated samples. `state.sample_weights` and `state.sampling_ess` expose the weights and their effective sample size, while the [importance sampling guide](../samplers/importance.md) explains their interpretation.

Explicit `samples=...` evaluates a supplied batch without advancing chains or replacing the cache. Ordinary external samples must represent the intended Born distribution. Importance samples require their matching density through `state.set_samples(..., log_density=...)`. Neither interface establishes validity or equilibration merely from an array's shape.
