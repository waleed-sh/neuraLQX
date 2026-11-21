---
title: Estimator kernels
---

# Estimator kernels

A registered rule is a small Python entry point. The numerical work can live in a pure JAX function that receives arrays and explicit static settings. This separation lets dispatch choose the algorithm while JAX compiles its arithmetic. It also makes it possible to compare the kernel with a direct calculation before involving a full optimisation run.

For ordinary discrete operators, begin with the supplied connectivity estimator. A custom kernel is useful when it removes work that would otherwise be repeated or avoids constructing an unnecessarily large intermediate. It should represent the same matrix action as the operator's [connectivity contract](../operators/connectivity.md).

## Evaluate the model through the variational state

The variational state exposes `apply_variables_kernel` and `variables`. Together they provide the model's complete evaluation interface, including fixed collections as well as trainable parameters. Passing only `state.parameters` to an arbitrary model is insufficient when it also requires model state or a framework-specific variable structure.

[`apply_batched`](../../api/public/neuralqx/jax/streaming/index.md#neuralqx.jax.streaming.apply_batched) evaluates this interface in optional chunks. [`apply_connected`](../../api/public/neuralqx/jax/streaming/index.md#neuralqx.jax.streaming.apply_connected) additionally accepts source configurations and an owner index for each target. When the model supplies an exact connected-state evaluation capability, this helper can use it. Otherwise it performs ordinary batched evaluation. Both routes must return the same log amplitudes.

Flatten the leading sample dimensions for the numerical evaluation, then restore them before returning local values. The variational state itself does not need to become an argument to a jitted function.

## An estimator for a binary flip

On one binary site, the flip operator has action $X|0\rangle=|1\rangle$ and $X|1\rangle=|0\rangle$. If $f(x)$ flips that coordinate, its local value is

$$
X_{\mathrm{loc},\theta}(x)
=\exp\bigl(\log\psi_\theta(f(x))-\log\psi_\theta(x)\bigr).
$$

The example defines both the connectivity and a specialised kernel. It then checks the registered result against the direct wavefunction ratio, including a chunk size smaller than the batch.

```python
from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.jax.streaming import apply_batched, apply_connected
from neuralqx.utils.struct import field
from neuralqx.vqs import LocalEstimatorOptions, LocalEstimators

class BinaryFlip(nqx.operator.ComputationalOperator):
    site: int = field(static=True, default=0)

    def __post_init__(self):
        super().__post_init__()
        if self.hilbert.constrained:
            raise ValueError("This example requires an unconstrained space")
        if not 0 <= self.site < self.hilbert.size:
            raise ValueError("site is outside the configuration")
        values = np.asarray(self.hilbert.local_spaces[self.site].all_values())
        if not np.array_equal(values, [0, 1]):
            raise ValueError("The selected site must have values 0 and 1")

    @property
    def dtype(self):
        return jnp.dtype("float64")

    @property
    def is_hermitian(self):
        return True

    @property
    def adjoint(self):
        return self

    @property
    def max_conn_size(self):
        return 1

    def _get_conn_padded_kernel(self, configuration):
        target = configuration.at[self.site].set(1 - configuration[self.site])
        return target[None, :], jnp.ones((1,), dtype=self.dtype)

@partial(jax.jit, static_argnames=("apply_fn", "site", "chunk_size"))
def flip_values(apply_fn, variables, samples, *, site, chunk_size):
    flat = samples.reshape(-1, samples.shape[-1])
    targets = flat.at[:, site].set(1 - flat[:, site])
    sources = apply_batched(apply_fn, variables, flat, chunk_size=chunk_size)
    connected = apply_connected(
        apply_fn, variables, flat, targets,
        jnp.arange(flat.shape[0]), chunk_size=chunk_size,
    )
    return jnp.exp(connected - sources).reshape(samples.shape[:-1])

@nqx.vqs.register_local_estimator
def binary_flip_estimator(
    vstate: Any,
    operator: BinaryFlip,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    values = flip_values(
        vstate.apply_variables_kernel, vstate.variables, jnp.asarray(samples),
        site=operator.site, chunk_size=options.chunk_size,
    )
    return LocalEstimators(values)

def log_value(parameters, configurations):
    return parameters["theta"] * jnp.sum(configurations, axis=-1)

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
state = nqx.vqs.VariationalState(
    hilbert=hilbert, apply_fn=log_value,
    parameters={"theta": jnp.array(0.2)}, seed=0,
)
samples = jnp.array([[[0], [1]], [[1], [0]]], dtype=hilbert.dtype)
values = nqx.vqs.local_estimators(
    state, BinaryFlip(hilbert), samples=samples, chunk_size=1,
).data
expected = jnp.exp(state.log_value(1 - samples) - state.log_value(samples))
np.testing.assert_allclose(values, expected)
print(values.shape)
print(np.round(np.asarray(values[0]), 6).tolist())
```

```text
(2, 2)
[1.221403, 0.818731]
```

This kernel honours the model chunk size. It constructs one target per source, so it has no multi-connection operator tensor to stream and no duplicate connections within a row. A more general kernel must explicitly manage those additional costs. The [standard batching controls](../../concepts/vqs/batching.md) describe what each budget limits.

The constructor's value check runs before numerical transformations. The compiled kernel performs no NumPy conversion or host inspection of traced values. Keep that distinction when adding more involved domain checks or precomputed graph metadata.

## Padded connections must be safe to evaluate

A general connectivity kernel returns $K$ slots per source, including unused slots with zero matrix element. The estimator computes

$$
a(x)=\sum_{k=0}^{K-1}m_k(x)
\exp\bigl(\ell(y_k(x))-\ell(x)\bigr).
$$

An unused slot must contribute exactly zero. Multiplying an invalid network value by zero at the end is insufficient, since $0\times\mathrm{NaN}$ is still undefined. Before evaluating the network, replace inactive targets with a valid configuration such as their source. Mask the log ratio before exponentiation and mask the resulting contribution as well. The standard discrete kernel follows this pattern.

Repeated targets are legitimate. Their matrix elements add, including cancellation. If your specialised kernel deduplicates them, it must sum their amplitudes rather than keep only one occurrence. Compare with [`get_conn_coalesced`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.get_conn_coalesced) on a small batch when checking that optimisation.

## Numerical and differentiation checks

Evaluate ratios as differences of log amplitudes instead of forming two potentially extreme amplitudes separately. This removes a common source of overflow but does not make every ratio well conditioned. A source with zero wavefunction amplitude lies outside the ordinary sampled support, and [squared estimators and residuals](../../concepts/operators/restricted-spaces.md) can require additional support handling.

Keep parameter-dependent computations in JAX if they must be differentiated. A custom JVP or VJP is an additional numerical commitment, not a requirement for an ordinary composition of differentiable JAX operations. The [JAX custom derivative guide](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html) describes when those rules are useful. The [next page](gradients.md) explains the separate question of how differentiation of an expectation accounts for the changing sampling distribution.

Check single configurations, multiple leading batch axes, real and complex amplitudes, and several chunk sizes. For an optimised discrete kernel, compare with a dense calculation on a small basis and with its own connectivity. Agreement of the forward values alone does not establish that the chosen gradient path is valid.
