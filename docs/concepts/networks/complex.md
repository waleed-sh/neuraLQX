---
title: Complex and holomorphic models
---

# Complex and holomorphic models

A complex wavefunction can represent both the magnitude and the phase of each basis amplitude. That does not require complex parameters, and complex parameters do not automatically make a network holomorphic. These are separate choices about the variational family and how its derivatives are represented.

This page connects those choices to the [supplied models](models.md), their [activations](layers.md#select-an-activation), and the [quantum geometric tensor](../optimisers/qgt.md). The same reasoning applies to a custom Flax or JAX model.

## Complex output with real parameters

Write the log amplitude as

$$
f_\theta(\sigma)
=a_\theta(\sigma)+i\phi_\theta(\sigma).
$$

Both $a_\theta$ and $\phi_\theta$ can be ordinary real functions of real parameters. The default built-in models use this construction. Their parameters are real `float32` arrays, while `complex_output=True` combines amplitude and phase contributions into one complex scalar.

This is enough to represent sign changes and general phases. For example, a phase difference of $\pi$ changes the sign of an amplitude without requiring complex weights. The Born distribution depends on $a_\theta$, while off-diagonal observables and the wavefunction geometry also depend on $\phi_\theta$.

For a real parameter $\theta_k$,

$$
\partial_{\theta_k}f_\theta
=\partial_{\theta_k}a_\theta
+i\partial_{\theta_k}\phi_\theta.
$$

Both channels contribute to the general complex differentiation mode. Calling this model holomorphic would not describe the real parameter coordinates being optimised.

## Choose parameter and computation dtypes

`param_dtype` controls parameter storage, while `dtype` controls the computation. The dense, RBM, Jastrow, CNN, and GNN families accept complex parameters. Their computation dtype must also be complex. `dtype=None` follows the parameter dtype.

```python
import jax.numpy as jnp
import neuralqx as nqx

real_parameters = nqx.nn.MLP(
    4, hidden_features=(8,), complex_output=True, rngs=0
)
complex_parameters = nqx.nn.MLP(
    4, hidden_features=(8,),
    activation="tanh",
    complex_output=False,
    param_dtype=jnp.complex128,
    dtype=None,
    rngs=1,
)
configurations = jnp.zeros((2, 4))

print(real_parameters.hidden[0].kernel.dtype)
print(complex_parameters.hidden[0].kernel.dtype)
print(bool(jnp.iscomplexobj(real_parameters(configurations))))
print(bool(jnp.iscomplexobj(complex_parameters(configurations))))
```

```text
float32
complex128
True
True
```

Both outputs are complex, but the parameter spaces differ. For these models, `complex_output=False` selects one output channel. It does not discard the imaginary part when that channel is already complex. Conversely, two complex channels are combined as $h_0+i h_1$, so they are no longer individually real log-modulus and phase functions.

Choose precision explicitly when it matters. Enabling JAX's 64-bit support does not change constructors that explicitly default to `float32`. The [autoregressive model](built-in/autoregressive.md) requires real computation and parameters for its categorical probabilities, with a separate real phase contribution when requested.

## What holomorphic means here

Holomorphicity concerns the dependence on the **trainable parameters**. The input configuration labels can remain discrete. For a complex parameter $z=u+iv$ and log amplitude $f=a+i\phi$, the condition is

$$
\frac{\partial f}{\partial\overline z}
=\frac12\left(
\frac{\partial f}{\partial u}
+i\frac{\partial f}{\partial v}
\right)=0.
$$

Equivalently, the Cauchy–Riemann relations hold,

$$
\frac{\partial a}{\partial u}
=\frac{\partial\phi}{\partial v},
\qquad
\frac{\partial a}{\partial v}
=-\frac{\partial\phi}{\partial u}.
$$

A network made from complex affine maps and holomorphic activations preserves this property wherever all its operations are analytic. A [Jastrow model](built-in/jastrow.md) with complex coefficients is a particularly simple example, since its log amplitude is polynomial in those coefficients.

Operations involving conjugation, a modulus, or a real part generally introduce an independent dependence on $\overline z$. Complex feature normalisation can do the same. A well-defined non-holomorphic model is still a valid variational wavefunction. It simply requires derivatives with respect to both real parameter components.

The condition applies to the model, not to the real expectation value being minimised. A Born probability or a real energy depends on both the wavefunction and its conjugate even when the log wavefunction is holomorphic.

## Complex activations are not all analytic continuations

A registered activation's `supports_complex=True` describes its accepted input domain. It does not certify holomorphicity.

The native [`tanh`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.tanh) uses the complex hyperbolic tangent. By contrast, [`reim("tanh")`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.reim) applies real hyperbolic tangents independently,

$$
g(u+iv)=\tanh(u)+i\tanh(v).
$$

The two diagonal entries of its real-coordinate Jacobian are generally different, so the Cauchy–Riemann condition fails. This extension is useful when it is the activation you intend, but it requires the general differentiation mode.

One local diagnostic is to compare directional derivatives. A holomorphic derivative is complex-linear, so $Df(z)[i\delta]=iDf(z)[\delta]$.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

point = jnp.array(0.2 + 0.4j)

def complex_linear_at_point(function):
    _, along_real = jax.jvp(
        function, (point,), (jnp.ones_like(point),)
    )
    _, along_imaginary = jax.jvp(
        function, (point,), (1j * jnp.ones_like(point),)
    )
    return bool(jnp.allclose(along_imaginary, 1j * along_real))

print(complex_linear_at_point(nqx.nn.activations.tanh))
print(complex_linear_at_point(nqx.nn.reim("tanh")))
```

```text
True
False
```

A check at one point is a diagnostic, not a proof for an entire model or parameter domain. Analytic complex activations can also have singularities or branch boundaries. Complex `tanh` has poles, and [`logcosh`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.logcosh) uses a logarithm with its associated zeros and branch choices. Numerical stability does not remove those mathematical restrictions. The [JAX complex differentiation guide](https://docs.jax.dev/en/latest/complex-differentiation.html) explains the derivative conventions and the meaning of a holomorphic declaration.

## Select the QGT mode from the actual model

The [QGT constructors](../optimisers/qgt.md#real-and-complex-wavefunctions) support three modes.

| Mode | Appropriate interpretation |
| --- | --- |
| `mode="real"` | Differentiate the real log amplitude in real parameter coordinates. |
| `mode="complex"` | Differentiate both log-amplitude channels in real parameter coordinates. Complex parameters are split into real and imaginary parts. |
| `mode="holomorphic"` | Use complex parameter coordinates and holomorphic derivatives of a complex log amplitude. |

Use the general complex mode for the default real-parameter, complex-output networks and for non-holomorphic complex-parameter models. A holomorphic calculation should use complex trainable parameters throughout and a complex output whose dependence on those parameters satisfies the analytic condition.

The following example compares coordinate counts for a one-edge complex Jastrow model. It has one linear coefficient and one quadratic coefficient.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.Jastrow(
    hilbert.size,
    complex_output=False,
    dtype=jnp.complex128,
    param_dtype=jnp.complex128,
    rngs=0,
)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=64, sampler_seed=4
)

holomorphic = state.qgt(mode="holomorphic", diag_shift=1e-3)
general = state.qgt(mode="complex", diag_shift=1e-3)
matrix = np.asarray(holomorphic.to_dense())
np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)

print(holomorphic.mode, matrix.shape)
print(general.mode, general.to_dense().shape)
```

```text
holomorphic (2, 2)
complex (4, 4)
```

The first matrix acts on two complex coordinates. The second acts on four real coordinates. Their entries and update conventions should not be compared as though they were matrices in the same coordinate system. The [SR guide](../optimisers/sr.md) and [QGT guide](../optimisers/qgt.md) explain how the metric and gradient are used together.

Without an explicit declaration, output dtype selects the real or general complex mode. Complex parameters and output do not trigger automatic detection of holomorphicity. An undeclared complex-parameter, complex-output model also produces a warning about the choice. Pass either `mode` or `holomorphic`, not both.

`holomorphic=True` is a promise about the model. Dtype validation cannot prove it. The argument belongs to the differentiation or QGT interface, rather than to the `VariationalState` constructor.

## Projection and optimisation

A fixed, parameter-independent linear [projector](symmetries.md) preserves holomorphic dependence of the amplitude. The logarithm of the projected amplitude is locally holomorphic only away from zeros and the chosen branch boundary. A coherent projection can create cancellation nodes even when the base model has no nodes. A support mask also introduces configurations where the logarithmic derivative is undefined.

For a real objective $E$ and a complex parameter $z=u+iv$, neuraLQX's [`expect_and_grad`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.expect_and_grad) uses the optimiser convention

$$
g_z=\frac{\partial E}{\partial u}
+i\frac{\partial E}{\partial v}.
$$

The built-in variational and SR interfaces handle their corresponding derivative conventions. Do not add a conjugation merely because the parameters are complex. If you implement an independent gradient calculation, match its convention to the [optimiser and preconditioner](../optimisers/index.md) that will consume it.
