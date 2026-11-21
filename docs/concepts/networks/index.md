---
title: Neural networks
---

# Neural networks

A neural wavefunction assigns an amplitude to each configuration in your Hilbert space. The configuration might contain U(1) edge charges, SU(2) spins and intertwiner channels, or another set of discrete labels. The network supplies the coefficients of the quantum state in that basis. Its parameters are the variational freedom that the optimisation changes.

The network does not choose the physical basis or define the Hamiltonian. Those roles belong to the [Hilbert space](../hilbert/index.md) and [operators](../operators/index.md). A [variational state](../vqs/index.md) brings the network and sampler together, evaluates observables, and differentiates them. A [VMC driver](../drivers/index.md) then uses an [optimiser](../optimisers/index.md) to update the parameters.

## Represent a wavefunction through its logarithm

For basis configurations $\sigma$, write

$$
|\psi_\theta\rangle=\sum_\sigma \psi_\theta(\sigma)|\sigma\rangle,
\qquad
f_\theta(\sigma)=\log\psi_\theta(\sigma).
$$

Models return $f_\theta$, one scalar log amplitude per configuration. A real output describes a positive amplitude through $\psi_\theta=e^{f_\theta}$. A complex output can represent a phase,

$$
f_\theta(\sigma)=a_\theta(\sigma)+i\phi_\theta(\sigma),
\qquad
\psi_\theta(\sigma)=e^{a_\theta(\sigma)}e^{i\phi_\theta(\sigma)}.
$$

The model therefore returns neither a probability nor an expectation value. With the usual Born sampling convention,

$$
p_\theta(\sigma)
=\frac{e^{2\operatorname{Re}f_\theta(\sigma)}}{Z_\theta},
\qquad
Z_\theta=\sum_\sigma e^{2\operatorname{Re}f_\theta(\sigma)}.
$$

Most networks need not calculate $Z_\theta$. Metropolis sampling uses ratios, and a [local estimator](../operators/application.md#from-the-expectation-value-to-the-local-estimator) uses wavefunction ratios such as $\exp[f_\theta(\sigma')-f_\theta(\sigma)]$. Working with logarithms makes these operations convenient without repeatedly forming potentially very large or very small amplitudes.

The phase does not appear in the Born probability, but it matters for off-diagonal observables and their gradients. A model with real parameters can still have a complex output. The [complex and holomorphic guide](complex.md) explains why this is a separate choice from using complex parameters.

## Inputs follow the Hilbert layout

The final input axis is the flattened configuration axis, of length `hilbert.size`. A batch of shape `(batch, hilbert.size)` produces shape `(batch,)`. The supplied models also preserve additional leading axes, so `(chains, samples, hilbert.size)` produces `(chains, samples)`. A single configuration produces a scalar.

These coordinates are basis labels, not an arbitrary feature ordering. In a U(1) space, several coordinates can belong to the same edge. In an SU(2) spin network space, the vector includes both doubled edge spins and the stored intertwiner channels. Consult the [state layout](../hilbert/state-layouts.md) before interpreting a reshape or imposing a network symmetry.

A dense network can consume that vector directly. A convolutional or graph network needs an explicit correspondence between its sites or nodes and the configuration coordinates. Neither architecture discovers that correspondence from the length of the vector.

## Connect a network to a variational state

The [built-in networks](models.md) use Flax NNX by default. Constructing one creates its parameters, and calling it evaluates the current log amplitude. This example uses a small U(1) space and an [`MLP`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.MLP).

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.MLP(hilbert.size, hidden_features=(16, 16), rngs=7)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=64, sampler_seed=13
)

configurations = hilbert.random_state(jax.random.key(5), size=4)
direct = network(configurations)
through_state = state.log_value(configurations)
np.testing.assert_allclose(through_state, direct)
print(direct.shape)
print(network(configurations.reshape(2, 2, hilbert.size)).shape)
print(bool(jnp.iscomplexobj(direct)))
```

```text
(4,)
(2, 2)
True
```

[`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) extracts the model variables and manages the parameter values used by the simulation. Once training starts, inspect the current wavefunction through the variational state. The original NNX object is not the interface through which the driver updates the state's parameters.

The initialisation seed for an already constructed NNX model is its `rngs` argument. Supplying a different `seed` to the variational state does not reinitialise that object's weights. The sampler has its own random stream.

## What makes a model a valid wavefunction

At fixed parameters, the amplitude of one configuration must be independent of which other configurations happen to share its batch. It must also be reproducible across repeated evaluations. This matters because sampling, connected-state evaluation, and gradient calculations can split or rearrange batches.

Use deterministic forward passes. Training-mode dropout and batch statistics that depend on the current sample batch do not define the required fixed wavefunction. Feature normalisation within each configuration can be appropriate, provided that it preserves the model's intended parameter and symmetry properties. The [layers guide](layers.md) explains the supplied normalisation choices.

The forward calculation must work with JAX transformations. Custom NumPy calculations on traced arrays, data-dependent Python branching, and parameter creation during evaluation will interfere with that contract. The [custom model guide](custom-models.md) gives complete NNX, Linen, and explicit-function examples.

## Choose the representation deliberately

An architecture defines which patterns the variational family can represent. Increasing a width changes its capacity. Imposing a translation symmetry or a projector changes the family itself. A more restricted family can help when the restriction reflects the physical state you want, but it can also exclude the solution.

Start with the [built-in model overview](models.md) to compare the supplied families and their assumptions. Use [projectors](symmetries.md) when you want an explicit linear restriction of amplitudes, and check [complex and holomorphic models](complex.md) before selecting a differentiation mode. Advanced implementations can continue with the [developer guide to custom networks](../../developers/networks/index.md).
