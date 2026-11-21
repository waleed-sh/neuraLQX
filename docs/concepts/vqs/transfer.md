---
title: Transferring weights
---

# Transferring weights

[`transfer_weights`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.transfer_weights) initialises an existing variational state's network from compatible learned parameters. It is useful when starting a related calculation, for example increasing a charge cutoff while keeping the same graph, coordinate layout, and network architecture.

Construct the destination state for the new calculation first, then transfer the weights into it. The destination retains its Hilbert space, sampler, model definition, fixed collections, and sampling settings. This is a different operation from [loading a checkpoint](saving.md), which restores the saved wavefunction configuration and can continue its chains.

## What is transferred

Write the model as $f(\theta,m,\sigma)$, where $\theta$ contains trainable parameters and $m$ denotes fixed model data and construction choices. A transfer performs

$$
\theta_{\mathrm{destination}}
\leftarrow \theta_{\mathrm{source}}.
$$

Evaluation afterwards uses

$$
f_{\mathrm{destination}}
(\theta_{\mathrm{source}},m_{\mathrm{destination}},\sigma).
$$

Only the trainable parameters move. Fixed graph data, non-trainable collections, model random state, activation choices, input transformations, and projectors remain those of the destination.

Consequently, matching parameter shapes does not by itself prove that two models represent the same function. A changed activation or a different ordering of physical labels can change the wavefunction even after every parameter has been transferred successfully.

## Continue at a larger cutoff

For an ordinary dense model, increasing a U(1) cutoff can leave the number of input coordinates and the parameter layout unchanged. The following example transfers between one-edge spaces with different charge ranges.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
small_space = nqx.hilbert.U1Hilbert(graph, cutoff=1)
large_space = nqx.hilbert.U1Hilbert(graph, cutoff=2)

source = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(small_space, n_chains=2),
    nqx.nn.MLP(1, hidden_features=(8,), rngs=4),
    n_samples=16,
    sampler_seed=5,
)
destination = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(large_space, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), rngs=8),
    n_samples=64,
    n_discard_per_chain=16,
    sampler_seed=9,
)

destination.transfer_weights(source)
shared_configurations = jnp.array([[-1], [0], [1]], dtype=jnp.int32)
np.testing.assert_allclose(
    destination.log_value(shared_configurations),
    source.log_value(shared_configurations),
)
print(source.hilbert.dimension, destination.hilbert.dimension)
print(destination.n_samples)
print(destination.sampler.n_chains)
```

```text
3 5
64
4
```

The source in this self-contained example is freshly initialised. The same call accepts a source that has undergone optimisation.

Both networks use the same input scaling and architecture, so their unnormalised log amplitudes agree on the shared configurations. The destination can also evaluate the newly available charges $-2$ and $2$. Transferring parameters does not establish that their amplitudes are already suitable for the enlarged physical problem.

Normalised amplitudes on the shared configurations can change because the basis changes. For nested domains $\mathcal H_1\subset\mathcal H_2$,

$$
Z_2
=\sum_{\sigma\in\mathcal H_2}
|\psi(\sigma)|^2
$$

generally differs from $Z_1$, even when the underlying function is unchanged. Expectation values must therefore be remeasured on the destination space, and further optimisation may be required.

## Parameter compatibility is exact

Transfer checks parameter paths, container types, shapes, and dtypes before updating the destination. It does not resize tensors, rename parameter paths, truncate layers, or cast precision. Ordinary dictionaries and Flax frozen dictionaries are treated as the same mapping layout.

The source and destination must therefore expose compatible trainable trees. An MLP with a different hidden width normally fails the shape check. Changing an embedding alphabet can change the embedding table size. Changing the graph can change input dimensions or node-embedding arrays. A common model name is not sufficient evidence of compatibility.

Even a structurally accepted transfer needs consistent scientific meaning. Keep track of edge ordering, gauge-component layout, doubled-spin conventions, and any feature rescaling. The [state layout](../hilbert/state-layouts.md) and [network input conventions](../networks/index.md#inputs-follow-the-hilbert-layout) explain those choices.

When the model uses fixed graph data, that data remains with the destination. This lets you initialise a new calculation without silently replacing its topology, but you must decide whether the transferred parametrisation is meaningful for that topology.

## Transfer from a model or explicit parameter tree

A source can be another variational state, an initialised NNX model, a bound Linen model, or a supported framework adapter. For an unbound Linen module or an explicit function, supply its trained `variables` or `parameters`, since the definition alone does not contain learned weights.

| Source | Call |
| --- | --- |
| Variational state | `destination.transfer_weights(source_state)` |
| Initialised NNX model | `destination.transfer_weights(source_model)` |
| Model definition with saved variables | `destination.transfer_weights(source_model, variables=source_variables)` |
| Parameter tree in the destination's framework convention | `destination.transfer_weights(parameters=source_parameters)` |
| Complete variables in the destination's convention | `destination.transfer_weights(variables=source_variables)` |

Passing complete variables still transfers only their trainable portion. The destination's fixed collections are retained. Supply either `variables` or `parameters`, and do not supply either alongside a source variational state, which already owns its parameters.

For a model trained through a VQS, transfer from the trained state. Its current parameters can differ from those of the original NNX instance used to construct it. Cross-framework transfer requires compatible parameter representations, rather than merely identical mathematical layer names.

The method returns the destination state itself after a successful transfer. Source parameters and source sampling state are left unchanged.

## Start from a saved calculation

Load the source checkpoint into its own state, construct the new destination, then transfer. Loading the old checkpoint directly into the destination would also request restoration of the old physical-space description and is not the appropriate operation for a changed cutoff.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import neuralqx as nqx
import numpy as np

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
small_space = nqx.hilbert.U1Hilbert(graph, cutoff=1)
large_space = nqx.hilbert.U1Hilbert(graph, cutoff=2)
source = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(small_space, n_chains=2),
    nqx.nn.RBM(1, hidden_features=8, rngs=6),
    n_samples=16,
    sampler_seed=7,
)
destination = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(large_space, n_chains=4),
    nqx.nn.RBM(1, hidden_features=8, rngs=10),
    n_samples=64,
    sampler_seed=11,
)

with TemporaryDirectory() as directory:
    path = source.save(Path(directory) / "small_cutoff.nqx")
    loaded_source = nqx.vqs.VariationalState.from_file(path)
    destination.transfer_weights(loaded_source)

np.testing.assert_allclose(
    destination.log_value(small_space.all_states()),
    source.log_value(small_space.all_states()),
)
print(destination.hilbert.dimension)
print(destination.samples.shape)
```

```text
5
(4, 16, 1)
```

Use your persistent checkpoint path in a simulation. For a custom model without a reconstruction recipe, provide its implementation when loading the source, following [saving and loading](saving.md#restore-a-custom-model).

## Sampling after a transfer

Transfer clears the destination's cached samples and importance weights, then marks its chain probabilities for refresh. It retains the destination's random streams, sample count, discard count, and sampler configuration.

Existing destination chain positions follow the usual reset policy. With the normal persistent-chain setup, they are retained and their probabilities are recalculated before the next sampling operation. With `reset_chains=True`, the sampler reinitialises them. The transfer does not copy the source's endpoints or its cached configurations.

A substantial change in weights can move the important regions of configuration space. Use the destination's discard settings and [mixing diagnostics](../samplers/diagnostics.md) before interpreting new estimates.

Transferred arrays are placed on the destination parameters' devices and shardings. When arrays span multiple JAX processes, all participating processes must perform the transfer. The destination's parameter metadata is retained, and validation completes before its state is changed.
