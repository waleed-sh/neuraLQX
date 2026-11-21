---
title: Restricted Boltzmann machines
---

# Restricted Boltzmann machines

[`RBM`](../../../api/public/neuralqx/nn/index.md#neuralqx.nn.RBM) represents the log wavefunction through visible linear terms and a sum of hidden log-cosh contributions. The hidden units couple to the full configuration vector, so the model can represent correlations between labels that are far apart in the graph.

The constructor takes the number of visible coordinates and an explicit hidden width. It accepts the numerical labels supplied by the Hilbert space, including multivalued U(1) charges and SU(2) labels. There is no automatic conversion to binary visible variables.

## The RBM convention used here

For one output channel, let $u=x/s$, where $s$ is `input_scale`. The model evaluates

$$
\ell(u)
=a\cdot u+c+
\frac{1}{\sqrt M}
\sum_{\mu=1}^{M}
\log\cosh\!\left(b_\mu+\sum_i W_{i\mu}u_i\right).
$$

Here $M$ is `hidden_features`. With real parameters and the default complex output, two independent channels give

$$
f_\theta(x)=\ell_0(x)+i\ell_1(x).
$$

The $1/\sqrt M$ factor is part of this implementation. The familiar constant $\log 2$ in a hidden-spin summation is omitted because it changes only an overall normalisation. When comparing parameters or initialisation with another RBM implementation, account for these conventions.

`hidden_features=128` is the default number of hidden units **per output channel**. It is a count, not a density parameter. If you want a width proportional to the visible size, pass a value such as `hidden_features=2 * hilbert.size` explicitly.

## Construct a wavefunction

```python
import jax.numpy as jnp
import neuralqx as nqx

network = nqx.nn.RBM(
    6,
    hidden_features=12,
    input_scale=2.0,
    complex_output=True,
    rngs=2,
)
configurations = jnp.array([
    [0, 1, -1, 0, 2, -2],
    [1, 0, 1, -1, 0, 2],
], dtype=jnp.int32)

print(network(configurations).shape)
print(network.hidden.kernel.shape)
print(network.visible.kernel.shape)
```

```text
(2,)
(6, 24)
(6, 2)
```

The hidden map contains twelve fields for each of the two channels. The visible map contributes the corresponding linear terms and offsets. A single real log-amplitude channel is selected with `complex_output=False` and real dtypes.

Use `dtype` and `param_dtype` to choose computation and parameter precision. With complex parameters, the channels themselves can be complex, so they no longer separately identify a real log modulus and phase. The returned combination remains $f=h_0+i h_1$.

## Log-cosh and complex parameters

The supplied [`logcosh`](../../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.logcosh) uses a stable expression to avoid overflow from large real parts. On complex inputs it returns the principal complex logarithm of $\cosh z$. Zeros of $\cosh z$ and the logarithm's branch structure still matter.

This means that complex support is not a global analyticity guarantee. Away from the relevant singularities and branch cuts, the underlying analytic expression has the expected complex derivatives. A simulation can nevertheless encounter regions where its logarithmic representation is ill conditioned or discontinuous. The [holomorphic guide](../complex.md) explains how to choose a differentiation mode without treating complex dtype as sufficient evidence.

The standard RBM with real parameters and two real channels is a general complex wavefunction in real parameter coordinates. It does not require a holomorphic declaration.

## Exact updates of hidden fields

Both the hidden fields and the visible linear terms admit exact sparse changes,

$$
\begin{aligned}
b_\mu+W_\mu\cdot(u+\delta u)
&=(b_\mu+W_\mu\cdot u)+W_\mu\cdot\delta u,\\
a\cdot(u+\delta u)+c
&=(a\cdot u+c)+a\cdot\delta u.
\end{aligned}
$$

The connected evaluator updates these fields and performs the same log-cosh reduction used by ordinary evaluation. Automatic reuse is conservative for small visible or hidden dimensions. The [connection policy](../models.md#exact-reuse-during-operator-evaluation) controls this computational choice without changing the ansatz.

The RBM has no built-in graph symmetry. If a known sector should be imposed exactly, see [projectors](../symmetries.md). If you want several successive nonlinear hidden layers, the [MLP](mlp.md) supplies that structure.
