---
title: Multilayer perceptrons
---

# Multilayer perceptrons

[`MLP`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.MLP) is a dense network over the complete configuration vector. Each hidden unit can depend on every coordinate through the first affine map. It is a useful starting point when the labels do not have a simple translation structure or when you want to avoid imposing a spatial symmetry on the wavefunction.

The input width is `hilbert.size` in a simulation. It includes every component of a U(1) charge vector or every stored SU(2) spin and channel label. The network does not need to know what those coordinates mean, although that meaning matters when deciding whether the representation is suitable.

## The represented function

Let $s$ denote `input_scale`, and let $\rho$ be the selected activation. With row-vector conventions,

$$
\begin{aligned}
h^{(0)}&=x/s,\\
h^{(\ell+1)}&=\rho\!\left(h^{(\ell)}W_\ell+b_\ell\right),\\
o&=h^{(L)}W_{\mathrm{out}}+b_{\mathrm{out}}.
\end{aligned}
$$

With the default real parameters and `complex_output=True`, the returned log amplitude is $o_0+i o_1$. The final head has no additional activation. With `complex_output=False`, it returns $o_0$.

`hidden_features` specifies the width of each hidden layer. The default `(64, 64)` gives two hidden layers, and the constructor requires at least one. An activation can be a registered name or a JAX callable, as explained in [layers and activations](../layers.md).

## Construct and evaluate an MLP

```python
import jax.numpy as jnp
import neuralqx as nqx

network = nqx.nn.MLP(
    6,
    hidden_features=(32, 16),
    activation="tanh",
    input_scale=2.0,
    complex_output=True,
    dtype=jnp.float64,
    param_dtype=jnp.float64,
    rngs=4,
)
configurations = jnp.array([
    [0, 1, -1, 0, 2, -2],
    [1, 0, 1, -1, 0, 2],
], dtype=jnp.int32)

print(network(configurations).shape)
print(network(configurations[0]).shape)
print(bool(jnp.iscomplexobj(network(configurations))))
```

```text
(2,)
()
True
```

The labels are divided by two inside this network. The original Hilbert configurations and the operators acting on them retain their original labels. Scaling is a feature transformation, not a change of physical units or a new cutoff.

For bounded activations such as `tanh`, large input labels can place units in a saturated region. `input_scale` provides a fixed positive divisor to control the range entering the network. It is shared by all input coordinates. If edge spins and intertwiner channels need different transformations, define those explicitly in a [custom model](../custom-models.md).

## Capacity and imposed structure

The MLP does not automatically impose graph automorphisms, charge conjugation, or translation symmetry. Its weights can distinguish positions in the flattened vector. The Hilbert space still determines which configurations the sampler may visit, so a dense model can be used with a gauge invariant basis without learning a separate Gauss-law mask.

Increasing widths adds variational freedom and evaluation cost. It does not establish convergence by itself. Compare the observables and optimisation behaviour at the capacities you need, using the same physical setup and appropriate sampling diagnostics.

If you need an exact symmetry restriction of amplitudes, use a [projector](../symmetries.md) or a deliberately invariant architecture. A network that merely receives the graph's labels has no automatic knowledge of that symmetry.

## Reuse the first layer

For a connected configuration $x+\delta$, the first affine field changes by $\delta W_0/s$. The MLP can reuse the source field and recompute every later layer exactly. The ordinary and connected routes return the same mathematical log amplitude, including its parameter derivatives.

The default automatic policy uses this route only for sufficiently large input maps and first hidden layers. Small dense contractions can be cheaper than discovering and gathering the changed coordinates. Set a [`ConnectionPolicy`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.ConnectionPolicy) when profiling this choice, as described in the [built-in overview](../models.md#exact-reuse-during-operator-evaluation).

For a different dense construction with a prescribed hidden reduction, continue with the [RBM](rbm.md). For an explicitly pairwise log amplitude, see [Jastrow](jastrow.md).
