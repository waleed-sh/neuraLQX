---
title: Creating a state
---

# Creating a state

Creating a [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) fixes the basis, model interface, and sampling setup for a calculation. It initialises the model variables and chain state, but it does not collect a sample batch until sampling is requested.

You can supply a configured sampler and a model, as in the [overview](index.md#a-first-calculation), or supply a Hilbert space and let neuraLQX choose the default proposal rule. Explicit sampler construction is useful whenever chain counts, sweep sizes, or specialised moves are part of your study.

## Choose a sampler for the physical space

With `hilbert=...` and no `sampler`, the constructor creates a [`MetropolisSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSampler). It selects [`U1GaugeRule`](../../api/public/neuralqx/sampler/rules/index.md#neuralqx.sampler.rules.U1GaugeRule) for `U1GaugeInvariantHilbert`, [`SU2MetropolisRule`](../../api/public/neuralqx/sampler/rules/index.md#neuralqx.sampler.rules.SU2MetropolisRule) for `SU2GaugeInvariantHilbert`, and [`LocalRule`](../../api/public/neuralqx/sampler/rules/index.md#neuralqx.sampler.rules.LocalRule) for other spaces.

This convenience follows the recognised Hilbert type. It does not derive an ergodic proposal for an arbitrary custom constraint. The [sampling guides](../samplers/index.md) explain the supported spaces and move families.

```python
import jax.numpy as jnp
import neuralqx as nqx

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1GaugeInvariantHilbert(
    graph, cutoff=1, auto_constraint=True
)
network = nqx.nn.RBM(hilbert.size, hidden_features=8, rngs=3)
state = nqx.vqs.VariationalState(
    hilbert=hilbert,
    model=network,
    n_samples=64,
    n_discard_per_chain=8,
    sampler_seed=5,
)

samples = state.samples
print(state.hilbert is hilbert)
print(samples.shape[-1] == hilbert.size)
print(bool(jnp.all(jnp.isfinite(state.log_value(samples)))))
```

```text
True
True
True
```

Here `auto_constraint=True` derives the constructive charge relations from the graph. The [U(1) Hilbert guide](../hilbert/u1.md#gauss-law-on-an-oriented-graph) explains the finite charge arithmetic and how to supply your own relations.

Passing a sampler preserves its configured proposal, chain count, and sweep size. The state's Hilbert space is taken from that sampler. If you also pass `hilbert`, it must agree with `sampler.hilbert`.

## Initialisation depends on the model framework

An NNX model is already initialised when it is passed to the state. Its `rngs` argument determined the initial weights. A `seed` supplied to the VQS does not overwrite them.

A Linen module instead supplies an initialisation recipe. The state uses `seed` to create its variables unless you supply them explicitly. The following example initialises a [built-in Linen MLP](../networks/models.md#nnx-and-linen-use-the-same-implementations).

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
model = nqx.nn.models.linen.MLP(1, hidden_features=(8,))
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, model, seed=10, sampler_seed=20, n_samples=32
)
print(state.log_value(jnp.array([[-1], [0], [1]])).shape)
print(state.n_parameters > 0)
```

```text
(3,)
True
```

`seed` also seeds the sampler when `sampler_seed` is omitted. Passing both makes the two choices explicit. An omitted model seed uses zero. An explicit JAX apply function can instead receive a `parameters` tree or an `init_fn`, as shown in [custom models](../networks/custom-models.md#an-explicit-jax-apply-function).

Complete `variables` take precedence over variables extracted from an initialised model. Extracted variables in turn take precedence over the separate `parameters` and `model_state` constructor arguments. To reuse learned weights in an existing state, the explicit [transfer interface](transfer.md) performs stronger compatibility checks than an ordinary parameter assignment.

## Parameters, fixed collections, and evaluation

The three variable properties serve related but different purposes.

| Property | Meaning |
| --- | --- |
| `parameters` | Trainable arrays in the model framework's pytree structure |
| `model_state` | Non-trainable collections required for evaluation |
| `variables` | The complete framework representation of both |

`n_parameters` counts scalar entries in the trainable tree. A complex scalar counts as one entry here, even though a general complex QGT may use two real coordinates for it.

Read the evolving wavefunction through `state.log_value(...)` or its other methods. The original NNX object passed to the constructor is not the object through which a driver updates the VQS parameters.

Assign a replacement through `state.parameters`, `state.model_state`, or `state.variables` when updating the model. These setters merge the appropriate collections and invalidate sampling caches. Editing a nested dictionary in place bypasses that mechanism. Parameter assignment requires the original pytree structure, while [weight transfer](transfer.md) also checks parameter shapes and dtypes.

The forward model must remain deterministic and independent of the other configurations in its batch. `training_kwargs` supplies fixed evaluation options to both measurement and sampling. `mutable` currently accepts only `False` or `None`, so mutable training-mode statistics are not updated by the VQS.

## Samples are collected and cached deliberately

Reading `state.samples` draws a batch if none is cached. A second read returns the cached configurations. Expectations and gradients normally share that batch, which allows related quantities to be compared on the same samples.

`state.sample()` explicitly advances the chains and returns `(samples, sampler_state)`. It also replaces the cached batch. The sampler state contains the final chain positions, rather than the full history just collected.

`state.reset()` clears the batch and refreshes chain probabilities against the current model. It retains chain configurations unless the sampler was configured with `reset_chains=True`. Resetting does not rewind the random stream or by itself collect samples.

After a parameter assignment, the next sampling operation refreshes the probabilities at the retained endpoints before advancing the chains. This lets optimisation continue from nearby configurations while sampling the new wavefunction.

## Choose retained samples and discarded points

`n_samples` is the requested total across all chains. The sampler adjusts it to the nearest positive multiple of the actual chain count, choosing the larger multiple on a tie. The property retains the requested value. Inspect `state.samples.shape` for the batch actually generated.

`n_discard_per_chain` removes that many chain points before **every fresh batch**, including batches after a parameter update. Each point follows `sampler.sweep_size` attempted proposals. The discard count is therefore different from a count of single proposals, and it is not restricted to the first measurement.

Assigning a new positive `state.n_samples` clears the cached batch. Changing `state.n_discard_per_chain` affects future sampling but leaves an already cached batch in place. Choose thermalisation and retained lengths from [sampling diagnostics](../samplers/diagnostics.md), especially for SU(2) chains initialised at the vacuum.

The model-evaluation `chunk_size` controls memory rather than statistical precision. Its role, and the distinction between chain, sample, and connection axes, are developed in [batching and chunking](batching.md).
