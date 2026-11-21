---
title: Jastrow wavefunctions
---

# Jastrow wavefunctions

[`Jastrow`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.Jastrow) uses linear and pairwise contributions to the log wavefunction. It is a compact way to study a variational family whose correlations are explicit in the configuration coordinates, and a useful reference when comparing more expressive nonlinear networks.

The model works on the complete flattened label vector. Its pairwise terms can connect any two coordinates. They are not restricted to neighbouring edges or to coordinates on the same graph vertex.

## A quadratic log amplitude

For one channel,

$$
\ell(x)=a^{\mathsf T}x+\frac12x^{\mathsf T}Jx,
\qquad J^{\mathsf T}=J.
$$

The amplitude is an exponential of that quadratic expression. With real parameters and `complex_output=True`, independent quadratic channels represent the log modulus and phase. With one real channel, the wavefunction is positive.

The stored `coupling` tensor has shape `(N, N, channels)`, and `linear` has shape `(N, channels)`. The forward calculation symmetrises the first two axes,

$$
J=\frac12(J_{\mathrm{raw}}+J_{\mathrm{raw}}^{\mathsf T}).
$$

This is a transpose, not a conjugate transpose. For complex parameters the model remains a quadratic polynomial in those parameters. It does not construct a Hermitian coupling matrix.

## Inspect the represented function

The following example checks the network against its explicit quadratic expression.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

network = nqx.nn.Jastrow(
    4, complex_output=False,
    dtype=jnp.float64, param_dtype=jnp.float64, rngs=3
)
configurations = jnp.array([[1, 0, -1, 1], [0, 1, 1, -1]])
raw = np.asarray(network.coupling[..., 0])
coupling = 0.5 * (raw + raw.T)
linear = np.asarray(network.linear[..., 0])
x = np.asarray(configurations, dtype=float)
reference = x @ linear + 0.5 * np.einsum("bi,ij,bj->b", x, coupling, x)

np.testing.assert_allclose(network(configurations), reference, atol=1e-12)
print(network.coupling.shape)
print(network(configurations).shape)
```

```text
(4, 4, 1)
(2,)
```

Unlike the MLP and RBM, this constructor has no `input_scale` argument. If you need rescaled or transformed coordinates, implement that transformation in a [custom wavefunction](../custom-models.md).

## What pairwise structure allows

Jastrow can represent interactions between every pair of stored labels, but its log amplitude is restricted to degree two in those labels. A larger graph increases the number of pairwise coefficients without introducing higher-order nonlinear terms. On a constrained basis, the restriction to admissible configurations can still produce nontrivial correlations.

The implementation stores a full square coupling tensor. Its parameter storage therefore grows quadratically with the input size. Antisymmetric changes to the raw matrix leave the represented function unchanged because of the explicit symmetrisation. Such redundant directions can contribute to null directions in the [quantum geometric tensor](../../optimisers/qgt.md).

Diagonal terms are retained. For an alphabet consisting only of $-1$ and $+1$, each $x_i^2$ is constant, so some diagonal contributions affect only an overall normalisation or phase. For multivalued charges or spins, those terms can depend on the configuration.

## Exact connected-state differences

If an operator changes the configuration by $\delta$, then

$$
\ell(x+\delta)-\ell(x)
=(Jx+a)^{\mathsf T}\delta+
\frac12\delta^{\mathsf T}J\delta.
$$

The second term includes cross terms between changed coordinates. Keeping only the linear contribution would generally give the wrong target amplitude.

The supplied connected evaluator uses the complete expression. It can reuse the source field $Jx+a$ and evaluate the relevant coupling entries for a sparse change. The [connection policy](../models.md#exact-reuse-during-operator-evaluation) falls back to the ordinary quadratic calculation when reuse is unsuitable.

For complex parameters, this polynomial structure also provides a straightforward [holomorphic model](../complex.md), provided the surrounding model and any transformations preserve that property.
