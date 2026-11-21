---
title: Metropolis sampling
---

# Metropolis sampling

Metropolis sampling builds a sequence of configurations whose stationary distribution is the probability law defined by your wavefunction. It only requires ratios of amplitudes, so it can work without enumerating the Hilbert space or normalising the neural network.

The proposal determines where a chain tries to go. The acceptance step determines whether it goes there. Keeping these two parts distinct is useful when choosing between [local, exchange, and gauge-preserving moves](transition-rules.md).

## The target probability

The model supplies a log amplitude $\ell_\theta(\sigma)=\log\psi_\theta(\sigma)$. [`MetropolisSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSampler) uses

$$
\log\widetilde p_\theta(\sigma)
=p\operatorname{Re}\ell_\theta(\sigma),
\qquad
\widetilde p_\theta(\sigma)=|\psi_\theta(\sigma)|^p.
$$

Here $p$ is `machine_pow`, whose default is `2.0`. This gives the Born distribution. The tilde indicates that the normalisation is omitted. A complex phase does not affect the sampling probability, although it can affect [off-diagonal operator expectations](../operators/application.md).

For ordinary quantum expectations, keep `machine_pow=2.0`. If you want a broader sampling distribution while preserving the Born objective, use the [importance-sampling interface](importance.md). Simply changing the power changes the distribution, and an ordinary unweighted sample average then measures that different distribution.

## Propose, compare, accept

Write $q(\sigma'\mid\sigma)$ for the probability of proposing $\sigma'$ from $\sigma$. The acceptance probability is

$$
a(\sigma\to\sigma')
=\min\left(
1,
\frac{\widetilde p_\theta(\sigma')}{\widetilde p_\theta(\sigma)}
\frac{q(\sigma\mid\sigma')}{q(\sigma'\mid\sigma)}
\right).
$$

The sampler evaluates the corresponding log ratio,

$$
\begin{aligned}
\Delta
={}&p\operatorname{Re}\bigl[\ell_\theta(\sigma')-\ell_\theta(\sigma)\bigr]\\
&+\log\frac{q(\sigma\mid\sigma')}{q(\sigma'\mid\sigma)}.
\end{aligned}
$$

It accepts when a uniform random number is smaller than $\min(1,e^\Delta)$. The transition rule supplies the final two terms, called the Hastings correction. For a symmetric proposal they cancel. The [bounded neighbour rule](transition-rules.md#moves-between-neighbouring-local-values) and the [SU(2) channel proposals](su2/moves.md#how-intertwiner-channels-are-drawn) show why that symmetry cannot always be assumed.

For two different configurations, the accepted transition satisfies detailed balance,

$$
\begin{aligned}
&p_\theta(\sigma)q(\sigma'\mid\sigma)a(\sigma\to\sigma')\\
&\qquad=p_\theta(\sigma')q(\sigma\mid\sigma')a(\sigma'\to\sigma).
\end{aligned}
$$

This makes the desired probability stationary. Convergence also requires that the moves can reach the relevant support and that the chain does not remain trapped in a periodic cycle. Valid configurations and a correct acceptance formula alone do not establish that every physical sector will be visited.

Some samplers first choose among several move families. The SU(2) sampler uses fixed family probabilities and applies the correction for the selected family. Each component is reversible, so their mixture is reversible too. You do not need to combine the proposal probabilities of all families yourself.

## Chains, sweeps, and retained samples

`n_chains` is the total number of parallel chains. For a retained chain length $L$ and $C$ chains, the batch contains $M=CL$ configurations and has shape `(C, L, hilbert.size)`.

`sweep_size` is the number of attempted proposals per chain between stored configurations. Its default is `hilbert.size`. A sweep does not guarantee that each coordinate is selected once, that every proposal is accepted, or that the next stored configuration is independent. In a gauge-preserving move, one attempt can change several labels at once.

In a [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState), `n_samples` requests the total retained count. If it is not divisible by the number of chains, it is adjusted to the nearest positive divisible count, with ties rounded upwards. Inspect `state.n_samples` and the actual array shape when the exact count matters.

`n_discard_per_chain` counts discarded stored points, so a discard of $B$ points costs $B\times\text{sweep size}$ attempted moves per chain. It is applied on every explicit `state.sample()` call. Its default is zero, so choose a discard deliberately for your problem.

You can alternatively specify `n_chains_per_device`. Do not supply both chain-count arguments. With neither supplied, the default is 16 chains per participating sample device. Device placement and larger runs are covered in [distributed simulations](../../distributed/sharding.md).

## Sampling an explicit wavefunction

Most simulations use the variational-state interface. The sampler can also work directly with a function that returns one log amplitude per configuration. This is useful for checking a known distribution or studying a fixed analytical ansatz.

For a single U(1) charge, choose

$$
\ell_\alpha(q)=-\frac{\alpha}{2}q^2,
\qquad p_\alpha(q)\propto e^{-\alpha q^2}.
$$

The example explicitly discards a first batch, then retains configurations and their unnormalised log probabilities.

```python
import jax.numpy as jnp
import neuralqx as nqx


def log_amplitude(alpha, configurations):
    return -0.5 * alpha * jnp.sum(configurations**2, axis=-1)


graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4, sweep_size=2)
alpha = 0.7

chain = sampler.reset(log_amplitude, alpha, key=12)
_, chain = sampler.sample(log_amplitude, alpha, state=chain, chain_length=32)
(samples, log_probabilities), chain = sampler.sample(
    log_amplitude,
    alpha,
    state=chain,
    chain_length=64,
    return_log_probabilities=True,
)
expected_logs = -alpha * jnp.sum(samples**2, axis=-1)
print(samples.shape, log_probabilities.shape)
print(bool(jnp.allclose(log_probabilities, expected_logs)))
```

```text
(4, 64, 1) (4, 64)
True
```

[`MetropolisLocal`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisLocal) supplies a symmetric replacement proposal. The function returns log amplitudes, so the sampler's default factor of two produces the probability exponent shown above. Returning a log probability instead would apply that factor twice.

The low-level `sample` method accepts either `chain_length` or `n_samples`. It returns the new sampler state alongside the batch. Pass that returned state into the next call to continue the chains. It does not add a thermalisation discard for you.

## Continue a chain when parameters change

The [`MetropolisSamplerState`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSamplerState) contains the current configurations, their log probabilities, the random state, and acceptance counters. Continuing with old probabilities after changing model parameters would compare two different targets. At the low level, call `sampler.reset(log_amplitude, new_parameters, state=chain)` before drawing from the changed wavefunction.

With the default `reset_chains=False`, reset keeps the configurations, recalculates their probabilities, and clears the acceptance counters. With `reset_chains=True`, it also draws new starting configurations. Reset does not rewind the random stream or thermalise the chains. A variational state manages probability refreshes when its parameters change through its normal update interfaces.

The random seed belongs to the sampling process. The network's initialisation seed is a separate choice. For example, `rngs=0` when constructing an RBM sets its initial parameters, while `seed=7` on the variational state sets the sampling stream. For a fixed configuration and execution environment, a repeated initial seed reproduces the run. Reusing an old sampler state also reuses its random state, so retain the newly returned state when you want fresh draws. The [JAX random-number guide](https://docs.jax.dev/en/latest/random-numbers.html#explicit-random-state) explains explicit keys and splitting.

## Nodes and initial support

A configuration with zero wavefunction amplitude has zero Born probability. Once a chain is on positive support, a proposal to a node is rejected. For supported projected models, neuraLQX attempts to move initial zero-density chains into finite support before retaining samples. This search is bounded and reports a failure if the proposal cannot find support. It does not replace the thermalisation discard.

If chains remain stuck, inspect the [proposal's reachability](transition-rules.md), the initial state, and the model's support. Increasing the sample count cannot recover a sector that the chosen moves never enter. [Samples and diagnostics](diagnostics.md) develops these checks in the context of a complete simulation.
