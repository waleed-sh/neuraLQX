---
title: Samples and diagnostics
---

# Samples and diagnostics

A successful sampling call gives you configurations, but a useful Monte Carlo estimate also needs evidence that the chains have explored the relevant distribution. Validity, acceptance, movement, and agreement between chains answer different questions. Examining them together helps distinguish an inefficient sampler from a physical wavefunction that is strongly concentrated.

Start with a fixed set of network parameters when investigating sampling. During optimisation the target changes, so drift in an observable can reflect either training or incomplete equilibration. Once sampling is understood at fixed parameters, monitor the same quantities throughout the run.

## Preserve the chain and time axes

A batch has shape `(n_chains, chain_length, hilbert.size)`. The first axis distinguishes chains. The second records successive retained points along a chain. The last contains the labels of a complete configuration.

For $C$ chains of length $L$, an observable's local values have shape `(C, L)` and mean

$$
\overline A=\frac1{CL}\sum_{c=1}^{C}\sum_{t=1}^{L}A_{c,t}.
$$

Keeping the first two axes separate lets statistical routines compare chains and identify temporal variation. Flattening them into one long array discards that information.

The following checks a constructive U(1) run and measures the charge on one component. Since the charge operator is diagonal, its local value is simply that stored charge.

```python
import jax.numpy as jnp
import neuralqx as nqx
from neuralqx.utils.stats import statistics

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.U1GaugeInvariantHilbert(
    graph, cutoff=2, gauge_dimensions=3, auto_constraint=True
)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisU1Gauge(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=512, n_discard_per_chain=64, seed=43
)
samples, chain = state.sample()
site = hilbert.layout.edge_site(graph.edges[0], component=0)
charge_values = samples[..., site]
changed = jnp.any(samples[:, 1:, :] != samples[:, :-1, :], axis=-1)

print('Sample shape:', samples.shape)
print('All configurations valid:', bool(jnp.all(hilbert.is_valid(samples))))
print('Acceptance:', float(chain.acceptance))
print('Changed retained configurations:', float(jnp.mean(changed)))
print('Charge means by chain:', jnp.mean(charge_values, axis=1))
print('Charge statistics:', statistics(charge_values))
```

The batch has shape `(4, 128, 12)`. [`statistics`](../../api/public/neuralqx/utils/stats/statistics/index.md#neuralqx.utils.stats.statistics.statistics) treats axis zero as the chain axis by default. For an off-diagonal observable, obtain its local values from `state.local_estimators(operator)` or use `state.expect(operator)` directly, as shown in [applying operators](../operators/application.md#measure-an-operator-with-a-variational-state).

## Check validity first

`hilbert.is_valid(samples)` checks every complete configuration, including local bounds and constraints. For U(1), this tests the represented gauge relations. For SU(2), it checks spin bounds and vertex admissibility. A valid result establishes membership in the intended basis.

It does not establish the Born distribution. A batch containing only one valid configuration can pass every validity check while failing to explore a nontrivial state. This is especially relevant when a generic local rule is used on a constraint that requires collective motion.

Check the interpretation of labels as well. SU(2) stores doubled spins and channels, while the Peter–Weyl basis stores encoded edge indices. The [state layout guide](../hilbert/state-layouts.md) explains how to obtain the intended physical coordinates before interpreting a sample mean.

## Acceptance and actual movement

[`MetropolisSamplerState.acceptance`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSamplerState.acceptance) is the fraction of accepted proposals since the last reset. The counters include attempted moves during both discarded and retained portions of the chain. Before any proposals, acceptance is undefined and reported as `NaN`.

A high acceptance rate does not necessarily mean substantial motion. A replacement can redraw the same charge. An SU(2) vertex move can have no channel to change. An invalid specialised proposal can return the current configuration, which is then accepted as an unchanged state. The explicit SU(2) stay also contributes no motion.

The `changed` calculation above instead compares neighbouring retained configurations. It is useful for detecting a stationary chain, but it does not measure acceptance per attempt. Several proposals occur between retained points, and a chain can move away and return before the next point is stored.

There is no universal acceptance target for this mixture of proposal types. Compare the uncertainty of the observables you need for a given computation time. Very small accepted changes can give excellent acceptance and poor exploration, while a useful collective proposal can have a lower acceptance rate.

## Correlation reduces the information in a batch

For a stationary scalar observable, let $\rho(k)$ be its normalised autocorrelation at lag $k$ along a chain. One common convention for the integrated autocorrelation time is

$$
\tau_{\mathrm{int}}=\frac12+\sum_{k=1}^{\infty}\rho(k).
$$

For independent chains that are long compared with their correlation scale,

$$
\operatorname{Var}(\overline A)
\approx\frac{2\tau_{\mathrm{int}}}{CL}\operatorname{Var}(A),
\qquad
M_{\mathrm{effective}}\approx\frac{CL}{2\tau_{\mathrm{int}}}.
$$

This explains why doubling the number of stored points does not help much if the additional points repeat essentially the same information. Different observables can have different autocorrelation times on the same samples.

Increasing `sweep_size` separates retained points by more attempts, but also increases their cost. Increasing the number of chains improves parallel exploration, while increasing chain length gives each chain more time to traverse slow regions. Use these controls together with a suitable proposal. None of them fixes a disconnected move set.

The equations describe the statistical effect of correlation. neuraLQX does not report an automatically fitted `tau_int` field in [`Stats`](../../api/public/neuralqx/utils/stats/index.md#neuralqx.utils.stats.Stats). With multiple chains, its error estimate uses the spread of chain means. With a single chain, it falls back to the independent-sample formula $\sqrt{\operatorname{Var}(A)/M}$, which can underestimate uncertainty for correlated samples.

## Read the reported statistics

`Stats.mean` is the estimated expectation, `variance` describes the local estimator's fluctuations, and `error_of_mean` estimates uncertainty in the mean. Variance and error therefore answer different questions. A noisy local estimator can still have a precisely determined mean after sufficient effective sampling.

For complete batch statistics, `r_hat` compares split-chain means relative to the total local-estimator variance. It is a diagnostic of disagreement across parts of the chains, rather than a proof of convergence or a rank-normalised diagnostic. Values near one should be considered alongside the histories and starting conditions. The diagnostic is `NaN` when the data do not provide the required nonzero variance or sufficiently long multiple chains. A constant observable can therefore have an undefined diagnostic even when its value is known exactly.

For [squared estimators](../operators/squared-estimators.md#same-expectation-different-local-values), the variance refers to the chosen factorised estimator. For [importance sampling](importance.md), statistics use the weighted ratio estimator and its influence values. Neither variance should automatically be interpreted as a quantum variance of the underlying observable.

## Inspect spins and intertwiners separately

An SU(2) run can change channels frequently while its edge spins barely move, or explore spin sectors while visiting few intertwiner assignments. Both effects are hidden by a single total acceptance rate.

```python
import jax.numpy as jnp
import neuralqx as nqx
from neuralqx.utils.stats import statistics

hilbert = nqx.hilbert.SU2Hilbert(nqx.graph.K5Graph(), two_j_max=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisSU2(
    hilbert, n_chains=4, coherent_loop_probability=0.5
)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=64, seed=47
)
samples, _ = state.sample()
edge_labels = hilbert.edge_spins(samples)
channel_labels = hilbert.view(samples).vertex_channels
mean_spin = jnp.mean(edge_labels / 2.0, axis=-1)
mean_channel = jnp.mean(channel_labels / 2.0, axis=-1)

print('Edge and channel shapes:', edge_labels.shape, channel_labels.shape)
print('Mean edge spin:', statistics(mean_spin))
print('Mean intermediate spin:', statistics(mean_channel))
```

These are basis-label diagnostics on the K5 space, which has one stored channel per vertex. They complement physical measurements such as area, volume, and the constraint residual. A label average should not replace the physical observables relevant to your model.

If parity patterns barely change, examine loop and global moves. If spin magnitudes mix poorly at larger cutoffs, examine same-parity edge motion as well. If channel exploration is slow, examine vertex updates. The [SU(2) proposal guide](su2/moves.md) explains what each family can change and which attempts can become stays.

## Compare against a small exact distribution

On a small indexable space, compare sampled frequencies with a known target before increasing graph size or cutoff. A constant log amplitude gives a uniform Born distribution over complete basis states. The triangle below has three admissible SU(2) states at `two_j_max=2`, so each should approach probability one third after equilibration.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx


def log_amplitude(parameters, configurations):
    return jnp.zeros(configurations.shape[:-1])


a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=2, exact=True)
sampler = nqx.sampler.MetropolisSU2(
    hilbert, weights=(0.0, 0.0, 0.0, 1.0), n_chains=8, sweep_size=4
)
_, chain = sampler.sample(log_amplitude, None, key=53, chain_length=128)
samples, chain = sampler.sample(
    log_amplitude, None, state=chain, chain_length=512
)
indices = np.asarray(hilbert.states_to_numbers(samples)).reshape(-1)
frequencies = np.bincount(indices, minlength=hilbert.dimension) / len(indices)
reference = np.full(hilbert.dimension, 1.0 / hilbert.dimension)

print('Sampled probabilities:', np.round(frequencies, 3))
print('Exact probabilities:', np.round(reference, 3))
```

The global proposals themselves have unequal masses on these three states, as shown in the [global-move example](su2/moves.md#global-proposals). The Hastings correction is what restores the uniform target. Finite frequencies fluctuate, so compare several seeds and sample sizes rather than expecting exact equality from one batch.

## Decide whether a run is ready to interpret

Compare physical observables after increasing the discard, extending chain lengths, and changing initial seeds or starting states. Look for agreement within the estimated uncertainty and for stable behaviour as the sampling effort grows. An apparently small error from several chains trapped in the same region can still be misleading.

During optimisation, `state.samples` reuses the cached batch until it is replaced or invalidated. Call `state.sample()` to advance the chains deliberately. Repeatedly reading the cached array or measuring several observables on it does not create additional independent data.

If improving the supplied proposal settings is insufficient, use the [custom sampler guide](../../developers/samplers/index.md) to develop a move suited to the physical constraints. The scientific checks remain the same: the configurations must be valid, the proposal probabilities must be correct, and the finite run must explore the distribution well enough for the quantities you report.
