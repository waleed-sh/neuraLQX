---
title: Samplers
---

# Samplers

A variational wavefunction assigns an amplitude to every basis configuration in your Hilbert space. Even at a modest graph size, there can be far too many configurations to visit them all. Sampling makes expectation values accessible by spending more time in the configurations that carry more probability in the current state.

For an LQG calculation, one sample is a complete set of labels on the graph. In a [U(1) space](../hilbert/u1.md), these are edge charges in each gauge component. In a [gauge invariant SU(2) space](../hilbert/su2.md), they include edge spins and the intertwiner channels needed at the vertices. A sample is one basis configuration. The neural network supplies its amplitude, and an [operator](../operators/index.md) supplies the matrix elements used to measure an observable.

## Why we sample the wavefunction

Let $\psi_\theta(\sigma)$ denote the amplitude of configuration $\sigma$. For ordinary variational Monte Carlo, the sampling distribution is

$$
p_\theta(\sigma)
=\frac{|\psi_\theta(\sigma)|^2}{Z_\theta},
\qquad
Z_\theta=\sum_\sigma|\psi_\theta(\sigma)|^2.
$$

An observable can then be measured as an average over this distribution,

$$
\langle A\rangle_\theta
=\mathbb E_{p_\theta}[A_{\mathrm{loc}}(\sigma)]
\approx\frac1M\sum_{r=1}^{M}A_{\mathrm{loc}}(\sigma_r).
$$

The [operator application guide](../operators/application.md#from-the-expectation-value-to-the-local-estimator) derives the local estimator, including the wavefunction ratios for connected configurations. The sampler provides the outer configurations $\sigma_r$. The operator and variational state evaluate the local quantity at each of them.

You do not need to calculate $Z_\theta$. The [Metropolis algorithm](metropolis.md) compares probabilities through amplitude ratios, in which the normalisation cancels. This is what allows sampling to work with a neural wavefunction that can be evaluated at individual configurations but cannot be stored over its entire basis.

## How a sampling calculation proceeds

A Markov chain starts at a valid configuration and repeatedly proposes a new one. The proposal might change one unconstrained charge, reconstruct several charges to preserve a U(1) gauge condition, or update spins and intertwiner channels together. The sampler evaluates the wavefunction at the proposal and accepts or rejects it according to the Metropolis rule. A rejected move leaves the chain where it was.

Several chains run together. After a chosen number of attempted moves, the current configuration of each chain is stored. Repeating this process produces a batch with a chain axis and a time axis. Neighbouring stored configurations can remain correlated, which is why the number of stored samples alone does not determine the precision of an estimate.

The starting configurations are drawn using the Hilbert space's valid-state generator. That generator does not know the current neural wavefunction. The chains therefore need time to approach its probability distribution. Discarding an initial part of each chain is one way to reduce this initialisation bias. The [diagnostics guide](diagnostics.md) explains how to assess that choice instead of relying on a fixed discard count for every problem.

During optimisation, changing the network parameters changes the target distribution. neuraLQX normally keeps the current chain positions and refreshes their probabilities for the new parameters. The next batch follows the updated wavefunction. This lets successive optimisation steps reuse useful starting positions while still collecting new samples.

## A first sampled state

[`MetropolisSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSampler) chooses a default transition rule from the Hilbert space. Here it uses single-site replacements on an unconstrained U(1) space. A [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) connects the sampler to a small [`RBM`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.RBM).

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisSampler(hilbert, n_chains=8)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=7
)

samples, sampler_state = state.sample()
print(samples.shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
(8, 32, 4)
True
```

There are eight chains, each with 32 retained configurations. Every configuration contains four edge charges. The network has not been trained here, so these samples describe its initial wavefunction. Increasing the sample count improves the information available about that wavefunction. It does not itself optimise the state.

The sampler's `sweep_size` controls how many proposals separate stored configurations. The state's `n_samples` controls the total retained batch size, and `n_discard_per_chain` controls the discarded part of each newly collected batch. These settings solve different problems and are developed in the [Metropolis guide](metropolis.md#chains-sweeps-and-retained-samples).

## Choose moves that suit the space

For an unconstrained product space, a local replacement is often a useful starting point. For a constructive U(1) gauge space, the default instead updates independent charges and reconstructs the dependent ones. For a gauge invariant SU(2) space, it mixes spin and intertwiner moves with the appropriate proposal corrections. The [transition rule guide](transition-rules.md) compares these choices and explains when an exchange move is appropriate.

Continue with [U(1) sampling](u1.md) for charge replacement, modular neighbour moves, and gauge preservation. The [SU(2) guide](su2/index.md) explains the spin network sampler, all its proposal families, and its initialisation options. [Importance sampling](importance.md) changes where configurations are drawn while retaining the physical Born expectation through density weights.

Once a calculation runs, use [samples and diagnostics](diagnostics.md) to examine validity, chain motion, agreement between chains, and uncertainty. If the supplied rules do not express the updates you need, the [custom sampler guide](../../developers/samplers/index.md) is the place to implement and validate a new one.
