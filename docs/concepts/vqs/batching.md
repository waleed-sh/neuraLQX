---
title: Batching and chunking
---

# Batching and chunking

A variational calculation uses several kinds of batches. Chains generate configurations, operators connect those configurations to others, and networks evaluate amplitudes on both. Keeping these axes distinct makes it easier to control memory without changing the statistical calculation.

`n_samples` determines how many retained configurations you request. `chunk_size` determines how many configurations are evaluated together by the model. Operator streaming separately limits batches passed to an operator kernel. Reducing a computational chunk does not collect fewer physical samples.

## Chain, sample, and configuration axes

For $C$ chains, retained length $L$, and configuration width $N=\texttt{hilbert.size}$, sampler output has shape

$$
(C,L,N),
\qquad M=C L.
$$

The network's log amplitudes and an ordinary scalar observable's local values both have shape $(C,L)$. A batch of connected configurations adds an operator connection axis, so its typical padded shape is $(C,L,K,N)$.

`n_samples` is a total across chains. It is adjusted to the nearest positive multiple of the actual chain count, with ties rounded upward. The requested value remains available as `state.n_samples`.

```python
import neuralqx as nqx

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.RBM(1, hidden_features=8, rngs=0),
    n_samples=34,
    sampler_seed=1,
)
samples = state.samples
print(state.n_samples)
print(samples.shape)
print(samples.shape[0] * samples.shape[1])
```

```text
34
(4, 9, 1)
36
```

The sampler also reports the adjustment in a warning. Choose an exact multiple when you want a predetermined retained count. On a distributed configuration, use the sampler's actual chain count, which can itself be adjusted for the sample-device layout.

Keep the chain and time axes when calculating [statistics](../samplers/diagnostics.md#preserve-the-chain-and-time-axes). Flattening them preserves an arithmetic mean but removes information needed for chain diagnostics. The VQS flattens batches internally for model calls and restores the relevant output shape.

## Model evaluation chunks

Constructing a state with `chunk_size=B` sets its default model batch limit. `log_value`, `to_array`, local estimators, gradients, and QGT construction use this setting where they evaluate the model.

For $T$ configurations to evaluate, chunking splits the computation into approximately

$$
\left\lceil\frac{T}{B}\right\rceil
$$

model calls. The final chunk is padded when necessary, and the extra outputs are removed. You do not need the batch length to be divisible by `B`. The model must assign the same amplitude to a configuration regardless of its neighbours in a batch, as explained in the [network overview](../networks/index.md#what-makes-a-model-a-valid-wavefunction).

When the constructor's `chunk_size` is `None`, there is no model chunk limit. A method's `chunk_size=None` uses the state's configured default. It does not disable a limit that was set at construction. Supply another positive integer to override that default for one call.

Model chunks are relevant during differentiation too. The chunked evaluator recomputes network intermediates during the backward pass instead of retaining the intermediates of every chunk at once. This trades additional computation for lower activation memory. It does not remove storage for the parameter tree, samples, or all operator connections.

The [JAX vectorisation guide](https://docs.jax.dev/en/latest/automatic-vectorization.html) explains evaluation over batch axes. Chunking controls how much of that batched computation is presented to the model at one time.

## Operator streaming is a separate control

For $M$ source configurations, padded row capacity $K$, and width $N$, a complete connected-state result can contain $MKN$ configuration entries and $MK$ matrix elements. A small network chunk does not by itself bound the operator's intermediate work.

Set `operator_streaming=True` and `operator_streaming_chunk_size=S` on estimator calls to evaluate operator kernels in source batches of at most `S`, with padding as needed. These are source configurations. The model may still need up to $SK$ connected configurations from such a batch.

The controls have different scopes.

| Setting | What it controls |
| --- | --- |
| `n_samples` | Requested total retained configurations |
| `n_discard_per_chain` | Points discarded before each fresh sample batch |
| `sampler.sweep_size` | Attempted proposals between stored points |
| `chunk_size` | Model evaluation batches |
| `operator_streaming_chunk_size` | Source batches evaluated by streamed operator kernels |

The streaming implementation still returns the requested connectivity arrays. It reduces intermediate work without guaranteeing that the complete connected output fits in memory. Products with large connection capacities and local operator caches can remain significant, as discussed in [matrix free operator memory](../operators/application.md#matrices-batches-and-memory).

`operator_sharding` controls the separate execution policy for distributing operator terms over a configured device mesh. Its use depends on the available terms and device layout. Operator streaming controls source batches within an evaluation. Consult [JAX parallelism](../../distributed/jax-parallelism.md) for the device-distribution setup.

## Compare values and gradients on the same samples

Chunking should preserve the represented calculation, up to floating-point differences. Compare both routes on one cached batch so sampling noise does not obscure the comparison.

```python
import jax
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), rngs=2),
    n_samples=64,
    sampler_seed=3,
)
charge = NumberOperator(hilbert, edge=0)
shift = U1HolonomyOperator(hilbert, edge=0)
observable = charge + 0.2 * (shift + shift.adjoint)
samples = state.samples

whole_stats, whole_gradient = state.expect_and_grad(
    observable, operator_streaming=False
)
options = nqx.vqs.LocalEstimatorOptions(
    chunk_size=7,
    operator_streaming=True,
    operator_streaming_chunk_size=5,
)
chunked_stats, chunked_gradient = state.expect_and_grad(
    observable, options=options
)

np.testing.assert_allclose(
    whole_stats.mean, chunked_stats.mean, rtol=1e-5, atol=1e-6
)
for whole, chunked in zip(
    jax.tree_util.tree_leaves(whole_gradient),
    jax.tree_util.tree_leaves(chunked_gradient),
):
    np.testing.assert_allclose(whole, chunked, rtol=1e-5, atol=1e-6)
np.testing.assert_array_equal(state.samples, samples)
print(samples.shape)
print("Values and gradients agree")
```

```text
(4, 16, 1)
Values and gradients agree
```

[`LocalEstimatorOptions`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.LocalEstimatorOptions) packages the same options accepted as individual keyword arguments. Pass either the options object or the individual execution options, rather than both.

The example uses an ordinary Hermitian observable. Some residual estimators make auxiliary random draws in addition to the cached source samples. Comparing those estimators also requires controlling their auxiliary random stream, rather than assuming the source batch is the only randomness.

## Choose limits for the calculation you are running

Begin with a sample count that is meaningful for the observable and optimisation problem. Then select model and operator chunks that fit the available memory. Smaller chunks can increase scheduling and recomputation costs, so compare timings after compilation with a fixed sample batch.

Chunking does not change the variational family, the physical cutoff, or the number of retained Monte Carlo points. It also does not make complete basis conversion inexpensive: [`to_array()`](arrays.md#enumeration-and-memory) still allocates the basis and its final vector.

The state's model chunk setting does not govern Metropolis proposal evaluations, which use the sampler's own batching. For memory or performance questions during sampling, inspect the sampler configuration and the [profiling guide](../../developers/profiling/for-scientists.md) rather than assuming every stage is controlled by one number.
