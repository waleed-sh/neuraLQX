---
title: Convolutional networks
---

# Convolutional networks

[`CNN`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.CNN) represents a log wavefunction on a fixed periodic lattice. Shared convolution kernels extract spatial features, and mean pooling turns them into a single log amplitude. With the supplied pointwise activations, this construction imposes translation invariance.

That symmetry is a physical modelling choice. Use this architecture when translations of the chosen input lattice should leave the wavefunction unchanged. A regular-looking physical graph alone does not establish that its flattened edge and channel labels have the required lattice ordering.

## Lattice layout and periodic boundaries

`lattice_shape` specifies the spatial dimensions. An integer describes a chain, while `(Lx, Ly)` describes a two-dimensional array. The input still has one final flattened axis, of length

$$
N=\prod_{\mu=1}^{d}L_\mu.
$$

It is reshaped in row-major order with **one scalar input channel per lattice site**. A tuple such as `(Lx, Ly, 3)` creates a third spatial axis. It does not declare three gauge components at each site.

For U(1)$^3$ configurations, or SU(2) configurations containing both edge and vertex data, choose an explicit representation before applying this model. The [Hilbert layout guide](../../hilbert/state-layouts.md) explains the stored coordinate order. A [custom network](../custom-models.md) can use the lower-level convolution layer with several feature channels.

## Convolutions and the readout

A convolutional stage has the form

$$
h^{(\ell+1)}_r
=\rho\!\left(
b_\ell+\sum_{\delta\in\mathcal K}
h^{(\ell)}_{r+\delta}W_{\ell,\delta}
\right),
$$

with indices interpreted periodically. After the last convolution, the model forms

$$
\bar h=\frac1N\sum_r h^{(L)}_r
$$

and applies a nonlinear dense readout followed by the scalar log-amplitude head. Translation permutes spatial features and leaves their mean unchanged. This gives the invariant output without summing separate evaluations over every translation.

`features=(16, 16)` is the default sequence of convolution widths. At least one convolution is required. `kernel_size=3` is an odd width shared by all spatial axes and stages. The built-in model uses circular padding and unit stride.

## Construct and check the symmetry

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

network = nqx.nn.CNN(
    (4, 4), features=(8, 8), kernel_size=3, rngs=5
)
configurations = (jnp.arange(32).reshape(2, 16) % 3) - 1
translated = jnp.roll(
    configurations.reshape(2, 4, 4), shift=1, axis=1
).reshape(2, 16)

values = network(configurations)
np.testing.assert_allclose(
    network(translated), values, rtol=1e-5, atol=1e-6
)
print(values.shape)
```

```text
(2,)
```

The example checks a translation along one lattice axis. It does not assert arbitrary permutation invariance, reflection symmetry, or a momentum sector with nontrivial translation phases. Those are different restrictions, and [projectors](../symmetries.md) provide one way to impose an explicitly chosen finite symmetry sector.

If the desired state can break translation symmetry, this pooled architecture excludes that possibility. A dense model or a custom position-dependent readout can represent a different family.

## Receptive fields and exact reuse

On a one-dimensional lattice with $L$ stages and kernel width $k$, the receptive-field radius is

$$
R=L\frac{k-1}{2}.
$$

Changing one input coordinate affects only a finite region of the final convolutional features. On sufficiently long chains, the connected evaluator recomputes that region, updates the spatial mean, and applies the same readout.

This route requires an appropriate elementwise activation and a chain long enough to contain the local update without covering the whole system. Multiple changed coordinates, higher-dimensional lattices, and unsupported local structures use ordinary convolution. The [connection policy](../models.md#exact-reuse-during-operator-evaluation) preserves the full wavefunction in either case.

The model accepts the shared dtype, complex-output, and precision controls. To change padding, stride, channel layout, or the readout itself, use [`Conv`](../../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Conv) in a [custom model](../custom-models.md).
