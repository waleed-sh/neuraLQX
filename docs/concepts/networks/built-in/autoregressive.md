---
title: Autoregressive networks
---

# Autoregressive networks

[`AutoregressiveRNN`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.AutoregressiveRNN) describes a configuration through a sequence of conditional distributions. Each coordinate is predicted from those that precede it. This gives a normalised Born distribution on a specified product alphabet and permits direct independent sampling of that distribution.

The normalisation statement applies to the full Cartesian product of the supplied labels. A gauge invariant Hilbert space, a heterogeneous set of local domains, or a [projector](../symmetries.md) can restrict that product. The direct sampler does not enforce those restrictions.

## Conditional probabilities and phases

For a configuration $x=(x_1,\ldots,x_N)$, the model represents

$$
f_\theta(x)
=\frac12\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
+i\sum_{i=1}^{N}\phi_\theta(x_i\mid x_{<i}).
$$

The recurrent hidden state initially vanishes. At each position, a linear map of the current hidden state produces categorical logits, normalised with `log_softmax`. An optional second map supplies a phase for each category. Only after choosing or evaluating $x_i$ is its learned embedding used to update the hidden state.

The factor $1/2$ ensures

$$
|\psi_\theta(x)|^2
=\prod_{i=1}^{N}p_\theta(x_i\mid x_{<i}),
\qquad
\sum_{x\in\mathcal A^N}|\psi_\theta(x)|^2=1.
$$

Here $\mathcal A$ is `local_states`. The phase does not change this probability distribution. Setting `complex_output=False` removes the phase map.

## Specify the alphabet

`local_states` must contain at least two distinct finite real labels. It is sorted and stored as non-trainable model state. The model maps labels to categorical indices internally, so an alphabet such as `(-1, 0, 1)` does not require the caller to shift charges into embedding indices.

The same alphabet is used at every coordinate. A configuration containing an unsupported label receives log amplitude with real part `-inf`. This is exact zero amplitude, not a clipping operation.

Unlike the other built-in families, the autoregressive network requires real computation and parameter dtypes because its logits describe real probabilities. A complex wavefunction is obtained through the phase map. `hidden_features=32` is the default recurrent width.

## Check normalisation and draw independent samples

```python
from itertools import product

import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

network = nqx.nn.AutoregressiveRNN(
    4,
    hidden_features=12,
    local_states=(-1, 0, 1),
    dtype=jnp.float64,
    param_dtype=jnp.float64,
    rngs=8,
)
basis = jnp.asarray(list(product((-1, 0, 1), repeat=4)))
probabilities = jnp.exp(2 * network(basis).real)
np.testing.assert_allclose(jnp.sum(probabilities), 1.0, atol=1e-12)

samples = network.sample(jax.random.key(9), n_samples=16)
print(round(float(jnp.sum(probabilities)), 6))
print(samples.shape)
print(bool(jnp.all(jnp.isin(samples, jnp.array([-1, 0, 1])))))
```

```text
1.0
(16, 4)
True
```

`network.sample` draws from the product-space Born law with exponent two. It is separate from the Metropolis sampler managed by a variational state. Passing this network to `VariationalState` does not automatically replace that sampler with direct autoregressive sampling.

## Restricted spaces need their own sampling

Suppose the physical basis is a subset $\mathcal S\subset\mathcal A^N$. The restricted wavefunction has norm

$$
Z_{\mathcal S}
=\sum_{x\in\mathcal S}\prod_i p_\theta(x_i\mid x_{<i}),
$$

which need not equal one. A constrained sampler can target the same network on $\mathcal S$, with the usual normalisation implicit in its sampling. For U(1) and SU(2) gauge invariant spaces, use the [corresponding sampling rules](../../samplers/index.md).

Calling the model's direct `sample` method still returns unconstrained product configurations. Likewise, a projection changes the target amplitudes and does not inherit the original model's direct sampling law. Conditional sampling that enforces a global constraint would require an additional construction.

Coordinate order is part of the ansatz. Reordering labels changes which information is available at each conditional step. The model's normalisation does not imply invariance under permutations of that order.

## Reuse a causal prefix

If a connected configuration first differs at position $k$, the recurrent computation before $k$ remains unchanged. The model can reuse that prefix and recompute every conditional contribution from $k$ onward.

Later changes therefore offer more reuse than changes near the beginning. The implementation selects static prefix boundaries and retains full evaluation when a useful prefix is unavailable. The result and its derivatives still include the complete suffix. See the [built-in overview](../models.md#exact-reuse-during-operator-evaluation) for the shared policy.
