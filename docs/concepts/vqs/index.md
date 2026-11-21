---
title: Variational states (VQS)
---

# Variational states (VQS)

A variational quantum state brings a parameterised wavefunction into a simulation. The network defines how to evaluate an amplitude, but a calculation also needs a physical basis, a sampling distribution, persistent chains, and methods for measuring observables and differentiating objectives. [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) keeps these pieces together.

In a loop quantum gravity calculation, the [Hilbert space](../hilbert/index.md) determines the allowed charges, spins, and intertwiner channels. The [network](../networks/index.md) assigns amplitudes to those basis configurations. The [sampler](../samplers/index.md) explores them, while [operators](../operators/index.md) specify the quantities being measured or constrained. The VQS object holds the current wavefunction variables and coordinates their use.

## The state represented by the model

For a discrete basis, write

$$
|\psi_\theta\rangle
=\sum_\sigma \psi_\theta(\sigma)|\sigma\rangle,
\qquad
f_\theta(\sigma)=\log\psi_\theta(\sigma).
$$

The model returns $f_\theta$, which can be complex. Its real part controls the amplitude magnitude, and its imaginary part controls the phase. The normalised physical state is

$$
|\Psi_\theta\rangle
=\frac{|\psi_\theta\rangle}{\sqrt{Z_\theta}},
\qquad
Z_\theta=\sum_\sigma|\psi_\theta(\sigma)|^2.
$$

Most calculations never form this complete vector or calculate $Z_\theta$. With the usual sampler exponent `machine_pow=2`, configurations follow the Born distribution,

$$
p_\theta(\sigma)
=\frac{|\psi_\theta(\sigma)|^2}{Z_\theta}.
$$

Observable estimates combine these samples with [matrix free operator actions](../operators/index.md#matrix-free-by-design) and wavefunction ratios. The VQS object therefore provides a practical representation even when the complete basis is too large to enumerate. For a small indexable space, [`to_array()`](arrays.md) evaluates and normalises the full vector explicitly.

## A first calculation

This example measures the squared charge on one edge of a small U(1) graph. The model parameters are initialised once, and the state collects samples when the expectation is first requested.

```python
import neuralqx as nqx
from neuralqx.operator.prebuilt.wcl import NumberOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
network = nqx.nn.MLP(hilbert.size, hidden_features=(16,), rngs=7)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler,
    network,
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=11,
)

charge = NumberOperator(hilbert, edge=0)
estimate = state.expect(charge @ charge)
print(state.samples.shape)
print(estimate)
```

The sample array has shape `(4, 32, 3)`: four chains, thirty-two retained points per chain, and three edge labels per configuration. The printed statistics describe an estimate at the initial parameters. Constructing a VQS and measuring an observable does not optimise the network.

The basis in this example permits independent edge charges. For a gauge invariant calculation, construct the corresponding [U(1)](../hilbert/u1.md) or [SU(2)](../hilbert/su2.md) space and use its compatible sampler. The VQS interface remains the same.

## What the object keeps

`state.parameters` is the trainable parameter pytree. `state.model_state` contains fixed model collections, and `state.variables` combines both in the model framework's representation. A graph network's fixed topology, for example, belongs to its model state rather than to the trainable weights.

The sampler has its own state. Its chain endpoints, random keys, and acceptance counters are distinct from the cached batch in `state.samples`. Endpoints are where future sampling continues. The cached batch is the set of configurations used by the current measurements.

This distinction explains a useful behaviour: repeated ordinary expectations at unchanged parameters reuse the same sample batch. They do not silently collect more statistics. Calling `state.sample()` advances the chains and replaces that batch. Assigning new model variables invalidates it so the next measurement samples the updated wavefunction.

## Measurement and optimisation have different roles

`state.expect(operator)` returns expectation statistics. `state.local_estimators(operator)` exposes the individual sampled values. `state.expect_and_grad(objective)` returns statistics and a gradient with the same tree structure as the trainable parameters.

The state does not own an Optax optimiser's momentum or adaptive moments. A [VMC driver](../drivers/index.md) manages the optimisation loop and assigns updated parameters to the VQS. You can also make an update yourself through the [expectation and gradient interface](expectations.md). [Stochastic reconfiguration](../optimisers/sr.md) additionally uses the geometry of the represented wavefunction, available through `state.qgt()`.

A state is useful without a driver. You can load trained variables, inspect amplitudes, draw new samples, and evaluate several observables without changing the wavefunction. Conversely, changing an operator does not require constructing a new VQS when its Hilbert space remains compatible.

## Continue through the guides

[Creating a state](state.md) explains construction, random seeds, variables, and the sampling lifecycle. [Amplitudes and state vectors](arrays.md) covers `log_value`, normalisation, and exact small-basis comparisons. [Expectations and gradients](expectations.md) develops the estimators and shows a parameter update.

[Batching and chunking](batching.md) separates statistical sample counts from computational batch sizes. [Saving and loading](saving.md) explains checkpoints and chain continuation. [Transferring weights](transfer.md) covers reusing compatible parameters while retaining the destination's physical setup and sampling configuration.
