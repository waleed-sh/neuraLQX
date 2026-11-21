---
title: Saving and loading
---

# Saving and loading

A VQS checkpoint preserves the wavefunction variables together with the sampling state needed to continue measuring it. Use [`state.save`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.save) to write a checkpoint, [`VariationalState.from_file`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.from_file) to reconstruct supported components, or [`state.load`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.load) to restore into an existing compatible state.

Saving a VQS is different from exporting its [complete amplitude vector](arrays.md). It retains the parameterised representation, so the restored state can evaluate new configurations and continue sampling without storing every amplitude.

## What a checkpoint contains

The `.nqx` file stores trainable parameters, non-trainable model collections, random streams, chain endpoints, acceptance counters, and state-level settings such as sample count, discard count, and model chunk size. The implementation also preserves auxiliary estimator randomness and the associated importance-sampling state when present.

Cached sample batches are omitted by default. `include_samples=True` includes an existing cached batch and its sampling-density data where applicable. Saving does not draw samples, refresh stale chain probabilities, or otherwise advance the simulation to manufacture a batch.

The checkpoint records software versions, exact array shapes and dtypes, and reconstruction recipes for supported models, Hilbert spaces, and samplers. Optional `metadata` supplies your own JSON-compatible annotations, such as a run label, physical parameter values, or the optimisation step you recorded.

The VQS does not own the optimiser's moments, the [driver's](../drivers/index.md) step counter, or a learning-rate schedule's progress. Its checkpoint therefore does not constitute a complete optimisation restart. Resuming the same optimisation trajectory requires saving and restoring those additional quantities separately from this file.

## Save and reconstruct a built-in state

Built-in NNX models, their supplied Linen facades, standard Hilbert spaces, and supported Metropolis configurations have reconstruction recipes. The following example saves an existing sample batch and verifies both its restoration and the next sampling step.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import neuralqx as nqx
import numpy as np
from neuralqx.utils.io.serialization import inspect_checkpoint

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), rngs=2),
    n_samples=32,
    n_discard_per_chain=4,
    sampler_seed=3,
)
saved_samples = state.samples
configurations = hilbert.all_states()
saved_values = state.log_value(configurations)

with TemporaryDirectory() as directory:
    path = state.save(
        Path(directory) / "wavefunction.nqx",
        include_samples=True,
        metadata={"label": "one-edge demonstration"},
    )
    restored = nqx.vqs.VariationalState.from_file(path)
    np.testing.assert_allclose(
        restored.log_value(configurations), saved_values
    )
    np.testing.assert_array_equal(restored.samples, saved_samples)

    next_original, _ = state.sample()
    next_restored, _ = restored.sample()
    np.testing.assert_array_equal(next_restored, next_original)

    information = inspect_checkpoint(path)
    print(path.suffix)
    print(restored.samples.shape)
    print(information["metadata"]["user"]["annotations"]["label"])
```

```text
.nqx
(4, 8, 1)
one-edge demonstration
```

The temporary directory keeps this demonstration self-contained. In a simulation, pass a persistent path such as `"runs/experiment_01/state.nqx"` instead. Parent directories are created by the writer. A missing `.nqx` suffix is added automatically, and `save` returns the resolved path.

Existing files require `overwrite=True`. Saving publishes the completed replacement atomically, so a failed replacement leaves the previous checkpoint intact. The method returns after publication, rather than leaving an unfinished background save.

The equality of the next batches above is checked within the same software and execution configuration. Restored random streams do not imply identical trajectories across hardware or software changes, since different floating-point decisions can affect Metropolis acceptance.

## Continue chains or start new ones

The two loading modes serve different purposes.

| Mode | Sampling behaviour |
| --- | --- |
| `sampler_mode="continue"` | Restore chain endpoints, random streams, and counters for a compatible sampling configuration |
| `sampler_mode="restart"` | Restore the wavefunction and initialise fresh chains with an explicit new `sampler_seed` |

Continuation is the default. It requires compatible chain counts, proposal settings, and the same number of sample partitions. Do not supply a new sampler seed in this mode.

If a cached batch was saved, an ordinary expectation can reuse that batch after loading. If no batch was saved, the first expectation samples anew. An explicit `sample()` always advances the restored chains, regardless of whether a batch was included.

Restart mode is useful when changing the sampler or the number of devices used for sample distribution. It requires `sampler_seed=...` and discards any saved sample batch. Both modes restore the checkpoint's state-level `n_samples`, `n_discard_per_chain`, `chunk_size`, and `training_kwargs`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import neuralqx as nqx
import numpy as np

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
source = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=2),
    nqx.nn.RBM(1, hidden_features=8, rngs=4),
    n_samples=16,
    n_discard_per_chain=4,
    sampler_seed=5,
)
destination = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.RBM(1, hidden_features=8, rngs=9),
    n_samples=64,
    sampler_seed=10,
)

with TemporaryDirectory() as directory:
    path = source.save(Path(directory) / "source.nqx")
    destination.load(path, sampler_mode="restart", sampler_seed=57)
    np.testing.assert_allclose(
        destination.log_value(hilbert.all_states()),
        source.log_value(hilbert.all_states()),
    )
    print(destination.sampler.n_chains)
    print(destination.n_samples)
    print(destination.samples.shape)
```

```text
4
16
(4, 4, 1)
```

The destination keeps its four-chain sampler, but the saved request for sixteen retained samples is restored. Assign `destination.n_samples` afterwards if you want a different sample count. Thermalise the new chains according to the [sampling diagnostics](../samplers/diagnostics.md), rather than assuming learned weights imply equilibrated chains.

Restarting does not permit changing the Hilbert space or replacing the model with an incompatible architecture. Use [weight transfer](transfer.md) when beginning a related physical calculation on a different compatible space.

## Restore a custom model

Checkpoints contain data and reconstruction descriptions, not executable Python definitions. A custom model's implementation must be available when loading. You can construct a compatible state yourself and call `load`, or pass the model explicitly to `from_file`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from flax import linen as nn
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

class CustomWavefunction(nn.Module):
    width: int = 8

    @nn.compact
    def __call__(self, configurations):
        x = configurations.astype(jnp.float32)
        x = jnp.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)[..., 0]

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=2)
source = nqx.vqs.VariationalState(
    sampler, CustomWavefunction(), n_samples=16, seed=6
)
target = nqx.vqs.VariationalState(
    sampler, CustomWavefunction(), n_samples=16, seed=99
)

with TemporaryDirectory() as directory:
    path = source.save(Path(directory) / "custom.nqx")
    target.load(path)
    reconstructed = nqx.vqs.VariationalState.from_file(
        path, model=CustomWavefunction()
    )
    configurations = hilbert.all_states()
    for restored in (target, reconstructed):
        np.testing.assert_allclose(
            restored.log_value(configurations),
            source.log_value(configurations),
        )
    print(target.n_parameters == source.n_parameters)
```

```text
True
```

Supply `sampler=...` when a custom sampler or Hilbert component also lacks a reconstruction recipe. Callable models may need constructor options such as `init_fn` or a compatible `variables` template. A model with a custom activation, projector, or mixed parameter dtypes can similarly need an explicit template even when its outer architecture is a built-in class.

For repeated use, [`register_codec`](../../api/public/neuralqx/utils/io/serialization/index.md#neuralqx.utils.io.serialization.register_codec) defines a versioned reconstruction recipe. The same recipe implementation must be registered before saving and loading. Ordinary scientific use can keep the explicit-model approach shown above, with deeper integration covered by the [developer model guide](../../developers/networks/index.md).

## Compatibility, inspection, and distributed files

Loading checks the model framework, physical-space description, and array layout before changing the target state. Restored arrays must match the required shapes and dtypes. The existing state's device placements are used when loading into it. For 64-bit arrays, JAX must permit 64-bit device values before restoration.

Model compatibility includes more than the number of parameters. Input ordering, fixed collections, activation choices, and projector meaning must still describe the saved wavefunction. A changed library version is recorded and can produce a warning, but it is not a guarantee of compatibility in either direction.

[`inspect_checkpoint`](../../api/public/neuralqx/utils/io/serialization/index.md#neuralqx.utils.io.serialization.inspect_checkpoint) reads the manifest without loading the tensor arrays. It exposes versions, reconstruction recipes, shapes, sampling settings, and annotations, as used in the first example.

Saving and loading use temporary staging beside the checkpoint. Its parent directory must be writable, and sufficient space is needed for the staged data as well as the archive. In multi-process JAX, every participating process must call `save`, `load`, or `from_file` with the same path on a shared filesystem. Do not restrict the collective call to process zero. Metadata inspection alone is read-only and is not collective.

Keep a persistent checkpoint when you need to return to the same wavefunction and sampling state. Use the separate [transfer guide](transfer.md) when you want compatible learned weights to initialise a new calculation.
