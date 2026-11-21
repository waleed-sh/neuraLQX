---
title: Checkpointing
---

# Checkpointing

A checkpoint lets you recover a useful wavefunction, inspect an earlier stage of optimisation, or start a related calculation. In neuraLQX, the native checkpoint interface belongs to the [variational state](../vqs/saving.md). The driver coordinates when that state should be saved.

There is an important distinction between continuing a wavefunction's Monte Carlo chains and continuing the complete optimisation algorithm. A state checkpoint can preserve the former. It does not contain the driver's optimiser state, accepted-update counter, callback state, or log history.

## Choose when to save

[`VariationalState.save`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.save) writes the current wavefunction variables and sampling state to a `.nqx` file. Call it through `driver.state.save(...)` for a single-state driver.

At `step_finished`, an accepted update has already changed the parameters. This is a useful place to save the state that will enter the next iteration. Name the snapshot using `context.driver.step_count`, which counts the accepted updates already applied.

At `update_computed`, the parameters still correspond to the newly measured loss. Save at that event if you are selecting the state associated with the best measured objective. A snapshot saved at the end of the same step usually contains different parameters. The [monitoring guide](monitoring.md#match-a-record-to-the-correct-parameters) explains this timing in detail.

Saving does not draw samples or recompute stale chain probabilities. Cached sample arrays are omitted by default. In particular, saving immediately after an accepted update need not include a batch sampled from those new parameters. Restoring the state preserves the information needed to refresh sampling when it next occurs.

## Save periodically during a run

The callback below saves every two accepted updates and also saves the final state after a normal completion or a graceful stop. It leaves earlier snapshots in place and records the update count as metadata. It is written for a single-state driver.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

class SaveStates(nqx.callbacks.Callback):
    def __init__(self, directory, every=2):
        if every < 1:
            raise ValueError("every must be positive")
        self.directory = Path(directory)
        self.every = every
        self.last_saved_step = None
        self.last_path = None

    def save_current(self, context):
        step = context.driver.step_count
        self.last_path = context.state.save(
            self.directory / f"state_{step:06d}.nqx",
            metadata={"annotations": {"accepted_updates": step}},
        )
        self.last_saved_step = step

    def step_finished(self, context):
        if context.accepted and context.driver.step_count % self.every == 0:
            self.save_current(context)

    def run_finished(self, context):
        if (context.error is None
                and self.last_saved_step != context.driver.step_count):
            self.save_current(context)

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=8),
    n_samples=64,
    n_discard_per_chain=8,
    sampler_seed=9,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.adam(0.01), variational_state=state
)

with TemporaryDirectory() as directory:
    snapshots = SaveStates(directory, every=2)
    driver.run(3, callbacks=snapshots)
    print([item.name for item in sorted(Path(directory).glob("*.nqx"))])

    restored = nqx.vqs.VariationalState.from_file(snapshots.last_path)
    restored_charge = NumberOperator(restored.hilbert, edge=0)
    new_driver = nqx.driver.VMC(
        restored_charge @ restored_charge,
        optax.adam(0.01),
        variational_state=restored,
    )
    print(new_driver.step_count)
    new_driver.run(1)
    print(new_driver.step_count)
```

```text
['state_000002.nqx', 'state_000003.nqx']
0
1
```

The temporary directory keeps this demonstration self-contained. Use a persistent directory for a simulation. Saving creates missing parent directories, while an existing checkpoint requires explicit `overwrite=True`. Choose a new run directory when starting a new driver at step zero so its filenames do not collide with earlier snapshots.

The final hook deliberately avoids saving a possibly partially failed iteration when `context.error` is set. Periodic snapshots already completed remain available. A process terminated outside normal Python cleanup cannot be expected to execute a final callback.

## What the restored run continues

In the example, `from_file` restores the saved model variables and, by default, continues its compatible sampler state. The new `VMC` nevertheless starts with fresh Adam moments and a zero driver counter. It is a new optimisation phase starting from learned parameters.

| Part of the calculation | Stored by the VQS checkpoint |
| --- | --- |
| Trainable parameters and fixed model collections | Yes |
| Sampler endpoints, random streams, counters, and settings | Yes |
| Cached samples | Only with `include_samples=True` |
| Optax moments and internal schedule counters | No |
| Driver step count and callback state | No |
| SR solver caches and previous solutions | No |
| Observable history and runtime logger contents | No |

The current drivers do not provide a public `save` or `load` method that restores this entire collection. In particular, adding an update count to checkpoint metadata does not restore the driver's counter or an Optax schedule automatically.

For application-managed checkpoint layouts, [`save_pytree`](../../api/public/neuralqx/utils/io/serialization/index.md#neuralqx.utils.io.serialization.save_pytree) and [`load_pytree`](../../api/public/neuralqx/utils/io/serialization/index.md#neuralqx.utils.io.serialization.load_pytree) can persist additional numerical trees such as optimiser state. Restoring such a tree alone is not a complete driver-resume protocol. Keep optimiser configuration, counters, scheduled functions, preconditioner state, and callback bookkeeping consistent with the saved wavefunction, and validate the resulting continuation for your application.

Within the same Python process, consecutive calls to `run` on the same driver already retain its optimisation state. That is the simplest way to split a calculation into stages without restarting the optimiser.

## Custom models and several states

Built-in models with reconstruction recipes can be restored through `from_file`. For custom networks, supply the implementation or a compatible state template as described in [saving and loading states](../vqs/saving.md#restore-a-custom-model).

For a [multi-state calculation](multiple-states.md), save each member of `driver.states` at the same accepted-update boundary. Each state has its own sampler and parameter tree, and `driver.optimizer_states` contains one optimiser state per member. A collection of independently written files is not a single atomic multi-state checkpoint, so record which complete group belongs to an iteration.

When changing a cutoff or physical setup, use [weight transfer](../vqs/transfer.md) into a newly constructed destination state. Loading a checkpoint requests restoration of the saved wavefunction configuration, which is a different operation.

## Distributed saves

State saves and loads are collective across participating JAX processes. Every process must call them with the same shared-filesystem path. Do not put a state-saving callback behind a process-zero-only condition, and do not use a primary-process-only logger as the sole place where a collective save is invoked.

Each `.nqx` save publishes a complete file atomically. Temporary staging requires writable space beside the destination. The [state checkpoint guide](../vqs/saving.md) covers storage, placement, and restoration requirements in more detail.
