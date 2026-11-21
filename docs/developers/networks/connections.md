---
title: Connected-state evaluation
---

# Connected-state evaluation

An operator often connects a sampled configuration to many configurations that differ at only a few sites. Evaluating a neural network from the beginning on every target can repeat substantial work. [`LogWavefunction`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.LogWavefunction) provides an optional interface for reusing intermediate source features while returning exactly the same log amplitudes as ordinary forward evaluation.

This is an optimisation of model evaluation. It does not change the [operator connections](../operators/connectivity.md), discard matrix elements, or approximate the wavefunction. A model without this capability continues to use ordinary evaluation.

## Sources, targets, and owners

`connected_log_value(sources, targets, owners, *, chunk_size=None)` takes source configurations of shape `(B, N)`, full target configurations of shape `(C, N)`, and an integer owner array of shape `(C,)`. For target row $r$, `owners[r]` identifies the source from which that target was generated. The result has shape `(C,)` and must equal `model(targets)`.

Owners may repeat and need not be sorted. Their values must lie in the valid source range. The interface checks shapes and integer dtype, but validating owner bounds remains the caller's responsibility. Returning targets in a different order would misalign their amplitudes with the operator's matrix elements.

To implement reuse, provide `prepare_connections(sources)` and `apply_connections(cache, owners, sites, differences)`. The first constructs a differentiable cache for this call. The second evaluates targets from that cache and a bounded sparse description of changed coordinates. `sites` and `differences` have shape `(C_chunk, K)`. Differences mean target value minus source value. Unused entries carry zero differences, and their site indices must be safe to gather.

## Exact updates of a first layer

Consider a first layer $h_\theta(x)=xW+b$. If target $y_r$ differs from source $x_{o_r}$ at coordinates $i_{rk}$ by amounts $\delta_{rk}$, then

$$
h_\theta(y_r)
=h_\theta(x_{o_r})+\sum_k\delta_{rk}W_{i_{rk},:}.
$$

Any subsequent deterministic layers can be evaluated on this updated preactivation. The identity holds exactly in real arithmetic. Floating-point addition order can introduce small differences, so numerical comparisons use a suitable tolerance.

The following implementation caches the first-layer preactivations of a small MLP. It verifies ordinary and reused values, compares their parameter gradients through the framework adapter, and exercises a target that exceeds the sparse-change limit.

```python
from flax import nnx
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.frameworks import as_framework

class CachedMLP(nqx.nn.LogWavefunction):
    def __init__(self, in_features, hidden_features, *, rngs):
        self.dtype = jnp.float64
        self.connection_policy = nqx.nn.ConnectionPolicy(
            mode="on", max_changes=1, min_features=1
        )
        self.weights = nnx.Param(
            0.1 * jax.random.normal(
                rngs.params(), (in_features, hidden_features), dtype=self.dtype
            )
        )
        self.bias = nnx.Param(jnp.zeros(hidden_features, dtype=self.dtype))
        self.output = nnx.Param(
            0.1 * jax.random.normal(rngs.params(), (hidden_features,), dtype=self.dtype)
        )

    def __call__(self, states):
        hidden = states.astype(self.dtype) @ self.weights[...] + self.bias[...]
        return jnp.tanh(hidden) @ self.output[...]

    def prepare_connections(self, sources):
        return sources.astype(self.dtype) @ self.weights[...] + self.bias[...]

    def apply_connections(self, cache, owners, sites, differences):
        updates = jnp.einsum("ck,ckh->ch", differences, self.weights[...][sites])
        hidden = cache[owners] + updates
        return jnp.tanh(hidden) @ self.output[...]

model = CachedMLP(3, 5, rngs=nnx.Rngs(9))
sources = jnp.array([[0, 1, 0], [1, 0, 1]], dtype=jnp.int32)
owners = jnp.array([1, 0, 1, 0, 0], dtype=jnp.int32)
targets = sources[owners]
targets = targets.at[0, 0].set(0)
targets = targets.at[1, 2].set(1)
# Row two is unchanged, which exercises zero-padded differences.
targets = targets.at[3, 1].set(0)
# The final row changes two sites and must use ordinary evaluation.
targets = targets.at[4, 0].set(1).at[4, 2].set(1)

ordinary = model(targets)
for chunk_size in (None, 1, 2, 4):
    reused = model.connected_log_value(sources, targets, owners, chunk_size=chunk_size)
    np.testing.assert_allclose(reused, ordinary, rtol=1e-12, atol=1e-12)

framework, variables = as_framework(model)
parameters, fixed = framework.split_variables(variables)

def evaluate(parameters, reuse):
    current = framework.merge_variables(parameters, fixed)
    if reuse:
        return framework.apply_connected(
            current, sources, targets, owners, chunk_size=2
        )
    return framework.apply_variables(current, targets)

def objective(parameters, reuse):
    return jnp.sum(jnp.square(evaluate(parameters, reuse)))

compiled = jax.jit(lambda p: evaluate(p, True))(parameters)
np.testing.assert_allclose(compiled, ordinary, rtol=1e-12, atol=1e-12)
reference_grad = jax.grad(lambda p: objective(p, False))(parameters)
reused_grad = jax.jit(jax.grad(lambda p: objective(p, True)))(parameters)
for reference, reused in zip(jax.tree.leaves(reference_grad), jax.tree.leaves(reused_grad)):
    np.testing.assert_allclose(reused, reference, rtol=1e-11, atol=1e-12)
print(compiled.shape)
```

```text
(5,)
```

The change description is built from full source and target states. A custom model does not need to infer which operator produced a target. It only needs an exact way to update its own features for the supplied label differences.

## Policy and fallback behaviour

[`ConnectionPolicy`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.ConnectionPolicy) controls when reuse is attempted. `mode="off"` uses ordinary evaluation. `mode="auto"` requires the feature count to reach `min_features`. `mode="on"` bypasses that feature threshold for explicit testing or measurement. Both enabled modes also require more targets than sources, since otherwise the cache is not selected by the standard policy.

`max_changes` bounds the sparse update width. If any target in a chunk exceeds that width, the whole chunk falls back to ordinary forward evaluation. Changes are never truncated. This is why the example remains correct when its final target changes two coordinates while `max_changes=1`.

Chunking bounds target-side temporary activations. The source cache still covers the source batch, so it has its own memory cost. A final short chunk can be padded by repeating its last target and owner, with padded results discarded afterward. Batch independence is therefore necessary for this path as well as ordinary evaluation.

The built-in change extraction subtracts labels in an inexact dtype, or in the model's declared `dtype`. This avoids wrapping a negative difference in an unsigned configuration dtype. If a custom model declares `dtype`, choose a floating or complex type that represents the differences needed by its update identity.

## Cache differentiation and lifetime

The cache is created inside each connected evaluation at the current parameter values. Do not persist it across optimiser steps or apply `stop_gradient` to it merely because it is called a cache. For the first-layer identity,

$$
\partial_\theta h_\theta(y_r)
=\partial_\theta h_\theta(x_{o_r})
+\partial_\theta\sum_k\delta_{rk}W_{i_{rk},:}.
$$

Dropping the first term can leave the forward values unchanged while producing incorrect gradients. That error can affect an [estimator that differentiates local values](../estimators/gradients.md), including squared operators. The example compares the entire trainable tree through the same framework boundary used by the variational state.

If a model uses a more elaborate update, test repeated owners, unsorted owners, unchanged targets, the maximum supported number of changes, and overflow. Include the dtypes and complex parameterisation that the model actually supports. The real example above does not establish correctness for a different complex implementation.

## Framework integration and performance checks

The NNX and Linen adapters recognise an explicit `connected_log_value` method. [`to_linen`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.to_linen) preserves this capability for `LogWavefunction` subclasses. A callable can expose a `connected_log_value` attribute whose first argument follows its parameter or complete-variable convention. A custom adapter can override `apply_connected` directly.

The supplied adapters fall back to ordinary evaluation when nonempty `training_kwargs` are present, so that call options are preserved. Do not assume that a model's reuse hook is active merely because the method exists. Confirm the actual path and compare it with the policy disabled.

Measure after compilation, synchronise device results before timing, and include gradients when that is the intended workload. Reuse trades source-cache storage and sparse-update work against repeated full evaluation. Whether it is faster depends on architecture, source and target counts, changed sites, and hardware. The [profiling guides](../profiling/index.md) describe how to inspect that behaviour in a full simulation.
