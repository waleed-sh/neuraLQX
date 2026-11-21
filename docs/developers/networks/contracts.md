---
title: Model contracts
---

# Model contracts

A wavefunction model must define the same scalar log amplitude wherever it is evaluated. This includes raw sample batches, operator connections, projected configurations, and chunks chosen to control memory. The [scientist custom-model guide](../../concepts/networks/custom-models.md) introduces the supported model forms. Here the focus is on the contracts a developer should enforce and test.

## Input representation and output shape

For input shape `(..., N)`, where `N=hilbert.size`, return shape `(...)`. The final axis stores configuration labels, not necessarily local basis indices. All leading axes are batch axes. Remove only the final scalar output axis when a dense layer returns `(..., 1)`. An unrestricted `squeeze()` can also remove a batch axis of length one.

Use the [Hilbert layout](../hilbert/states.md) when features depend on graph entities or vector components. A graph with $E$ edges does not imply a state width $E$. Likewise, labels need not be consecutive integers starting at zero. For categorical features, convert labels with the Hilbert space's value-index mapping instead of treating the label itself as a table index.

Return $\log\psi_\theta$, rather than $\psi_\theta$, a probability, or a normalised batch of probabilities. If a real two-output head represents log modulus and phase, combine it into a single complex scalar $a_\theta(x)+i\phi_\theta(x)$. A configuration-independent additive constant changes wavefunction normalisation or global phase but cancels from the usual amplitude ratios. Normalising across the supplied batch is a different operation and violates the contract.

## Deterministic evaluation and batch independence

For any configuration $x$ and any two batches containing it, the corresponding output must be the same. Operations can mix sites, channels, or graph nodes within one configuration. They must not mix independent configurations to determine the wavefunction value.

This rules out active dropout and normalisation using current batch statistics during evaluation. Fixed inference statistics can be part of model state if they define a deterministic function. Generate random parameters at initialisation, and keep stochastic sampling in the sampler. The variational state currently does not propagate mutable model-collection updates through its forward pass.

The following NNX model has separate real log-modulus and phase parameters, plus a fixed feature scale. Its checks cover batch reshaping, singleton evaluation, chunking, compiled evaluation, and an analytical parameter derivative.

```python
from flax import nnx
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

class FeatureWavefunction(nqx.nn.LogWavefunction):
    def __init__(self, scale, *, rngs):
        self.scale = nnx.Variable(jnp.asarray(scale, dtype=jnp.float64))
        width = self.scale[...].shape[0]
        self.weights = nnx.Param(
            0.05 * jax.random.normal(rngs.params(), (width,), dtype=jnp.float64)
        )
        self.phase = nnx.Param(
            0.05 * jax.random.normal(rngs.params(), (width,), dtype=jnp.float64)
        )
        self.coupling = nnx.Param(jnp.asarray(0.01, dtype=jnp.float64))

    def __call__(self, states):
        features = states.astype(jnp.float64) * self.scale[...]
        total = jnp.sum(features, axis=-1)
        modulus = features @ self.weights[...] + self.coupling[...] * total**2
        phase = features @ self.phase[...]
        return modulus + 1j * phase

model = FeatureWavefunction((1.0, 0.5, 2.0), rngs=nnx.Rngs(4))
states = jnp.array([
    [[-1, 0, 1], [0, 1, -1], [1, 1, 0]],
    [[0, 0, 0], [-1, -1, 1], [1, 0, -1]],
], dtype=jnp.int32)
flat = states.reshape(-1, 3)
values = model(states)
compiled = nnx.jit(lambda network, x: network(x))(model, states)
assert values.shape == (2, 3)
np.testing.assert_allclose(compiled, values, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(model(flat).reshape(2, 3), values)
np.testing.assert_allclose(model(flat[:1]), model(flat)[:1])
np.testing.assert_allclose(
    jnp.concatenate([model(flat[:2]), model(flat[2:])]), model(flat)
)

gradient = nnx.grad(lambda network: jnp.real(network(flat)).sum())(model)
np.testing.assert_allclose(
    gradient["weights"][...], jnp.sum(flat * model.scale[...], axis=0),
    atol=1e-12,
)
assert "scale" not in gradient
assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(gradient))
print(values.shape)
```

```text
(2, 3)
```

The fixed scale is an NNX variable, while trainable quantities are `nnx.Param` leaves. Default NNX parameter differentiation selects the latter. The [parameter guide](parameters.md) explains how neuraLQX carries these two groups into a variational state.

## Differentiation and complex parameters

Ordinary models composed of JAX operations participate in automatic differentiation. For real parameters, a useful first check is a directional derivative of a real scalar function of the log amplitudes. Compare the automatic derivative along a small parameter perturbation with a central finite difference, using several step sizes to distinguish truncation error from floating-point error.

For complex parameters $\theta=u+iv$, holomorphic dependence requires $\partial\ell/\partial\overline\theta=0$. Equivalently, the real and imaginary parts satisfy the Cauchy-Riemann conditions in the region being differentiated. Complex dtype support in a layer or activation does not prove that condition. Conjugation, absolute value, and taking real parts generally introduce dependence on $\overline\theta$.

The example above uses real parameters to represent a complex output. It should be interpreted through that real parameterisation. Choose [complex differentiation and SR settings](../../concepts/networks/complex.md) from the actual model, and check [estimator gradients](../estimators/gradients.md) when adding a custom derivative rule. Supplying a custom JVP or VJP also requires verifying the transformations needed by the QGT or optimiser, rather than checking one forward pass alone.

## Layers and activation metadata

A custom layer should preserve its documented leading axes and make its parameter dtype explicit. The configuration dtype may be integer, even when the layer has floating or complex weights. Cast configuration features deliberately instead of allowing an integer input dtype to determine trainable parameter storage.

The [`activation`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.activation) decorator registers an activation with optional `elementwise` and `supports_complex` metadata. Those declarations describe behaviour that the author must establish. Marking a function as elementwise asserts that each output coordinate depends only on the corresponding input coordinate. It is not appropriate for a normalisation or reduction across features. Similarly, support for complex inputs does not declare holomorphicity.

[`to_linen`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.to_linen) can expose an NNX module class through Linen without duplicating its numerical implementation. Pass the class and constructor arguments, and provide RNGs through Linen initialisation. The bridge preserves the connected-state method for `LogWavefunction` subclasses. The [layer guide](../../concepts/networks/layers.md) covers the supplied construction shortcuts.

## Projection and the model's domain

A [projector](../../concepts/networks/symmetries.md) acts on amplitudes. For a row expansion,

$$
(P\psi_\theta)(x)=\sum_y P_{xy}\exp\!\left(\ell_\theta(y)\right),
$$

so the projected log amplitude is the logarithm of this coherent sum. Averaging the log amplitudes instead would define a different function. A custom [`RowProjector`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.RowProjector) must provide the required connected configurations and coefficients. A [`DiagonalProjector`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.DiagonalProjector) defines a support mask. The scientist [projector guide](../../concepts/networks/symmetries.md) contains runnable constructions and registration examples.

Test a purported projection for linearity and idempotence on a small amplitude vector. Also verify the base network on every configuration requested by the projector, not only the configurations retained by the sampler. Projection can introduce exact nodes, where a finite log amplitude and an ordinary log derivative do not exist. The native wrappers expose this projection information to consumers. A custom framework that implements projection must likewise report `has_projector` so support-sensitive algorithms can recognise it.

For model tests, distinguish expected negative-infinity values at exact nodes from accidental `NaN` or infinite arithmetic elsewhere. Verify amplitudes and physically relevant gradients on the intended support. The [squared-estimator discussion](../../concepts/operators/squared-estimators.md) explains why target-support coverage deserves particular attention when an operator acts outside that support.
