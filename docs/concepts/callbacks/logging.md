---
title: Logging and monitoring
---

# Logging and monitoring

A callback decides what happens at an event. A logger retains the resulting record. Separating these roles lets you add a diagnostic once, then send it to an in-memory history, a plot, or your own output format.

The driver builds a `log_data` dictionary when a consumer needs it. Callbacks can add entries, and the logger receives the completed dictionary after `step_finished`. The measurements in that dictionary usually describe the parameters before the update, as explained in [driver monitoring](../drivers/monitoring.md#match-a-record-to-the-correct-parameters).

## Retain a runtime log

[`RuntimeLogger`](../../api/public/neuralqx/utils/io/logging/index.md#neuralqx.utils.io.logging.RuntimeLogger) stores numerical histories in memory. Each leaf is a [`HistorySeries`](../../api/public/neuralqx/utils/io/logging/index.md#neuralqx.utils.io.logging.HistorySeries) with separate `steps` and `values`. Statistics are expanded into named fields such as `Mean` and `Sigma`.

The following example keeps every second step, writes JSON, and saves a plot. It closes the Matplotlib figure after exporting it. The temporary directory makes the example safe to rerun without accumulating output files.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import pyplot as plt
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator
from neuralqx.utils.io.logging import RuntimeLogger

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=22),
    n_samples=64,
    sampler_seed=23,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)
logger = RuntimeLogger(every=2)
driver.run(4, logger=logger, store_history=False)
steps, means = logger["Energy"]["Mean"].as_arrays()
print(steps.tolist())
print(means.shape)

with TemporaryDirectory() as directory:
    directory = Path(directory)
    json_path = logger.save_json(directory / "training.json")
    figure, axes = logger.plot("Energy", show=False, title="Expected squared charge")
    image_path = directory / "training.png"
    figure.savefig(image_path)
    plt.close(figure)
    print(json_path.is_file() and image_path.is_file())
```

```text
[0, 2]
(2,)
True
```

Use persistent paths for files you want to keep. `save_json` writes the logger's current contents when called. It is not an automatic on-disk log, and the runtime logger does not save the wavefunction. Use [state checkpoints](../drivers/checkpointing.md) for that purpose.

`logger.plot("Energy")` selects the `Mean` field by default. A complex mean is plotted as separate real and imaginary curves. The plot does not automatically draw uncertainty bands. The corresponding errors are available through `logger["Energy"]["Sigma"]` when you want to construct a figure with them.

For headless batch jobs, select a non-interactive Matplotlib backend, for example by setting `MPLBACKEND=Agg` in the environment before starting Python.

## Keep logging separate from driver history

`driver.history` retains whole per-attempt dictionaries, including native statistics objects. `RuntimeLogger` converts retained values to host-side numerical data and organises them by quantity. Using both stores two representations, so `store_history=False` is often useful when the logger already contains what you need.

A runtime logger accumulates across runs when the same object is reused. `clear()` discards its retained data, while `last_step` reports the last label it accepted. Passing `logger=True` creates a new runtime logger for that call, available afterward through `driver.logger`.

`RuntimeLogger(every=k)` retains labels divisible by $k$. It does not reduce the driver's gradient or `obs` calculations, and it does not necessarily retain the final label. Rejected attempts can share a label and therefore create repeated entries. If complete attempt accounting matters, add an explicit attempt number in a callback rather than treating step labels as unique record identifiers.

Host conversion can wait for pending JAX results. The [asynchronous dispatch guide](https://docs.jax.dev/en/latest/async_dispatch.html) explains why reading an array has a different timing cost from queuing its computation. Thinning retained records reduces this conversion work for a runtime logger, while [live monitoring](built-in.md#live-monitoring) uses a bounded background queue.

## Measure an observable less often

The `obs` dictionary is evaluated on every attempted iteration. For an expensive measurement needed only occasionally, calculate it in `update_computed` when its interval is reached. At that event the parameters and sample batch still match the current loss.

This callback measures the mean charge every two accepted-update labels. It also records the norm of the preconditioned direction, which is already available and does not require a second gradient calculation.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator
from neuralqx.utils.io.logging import RuntimeLogger

class MeasureOccasionally(nqx.callbacks.Callback):
    requires_log_data = True

    def __init__(self, observable, every=2):
        if every < 1:
            raise ValueError("every must be positive")
        self.observable = observable
        self.every = every

    def update_computed(self, context):
        context.log_data["direction_norm"] = optax.global_norm(context.update)
        if context.step % self.every == 0:
            context.log_data["Charge"] = context.state.expect(self.observable)

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=24),
    n_samples=64,
    sampler_seed=25,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)
logger = RuntimeLogger()
driver.run(
    4,
    callbacks=MeasureOccasionally(charge, every=2),
    logger=logger,
    store_history=False,
)
print(logger["Charge"]["Mean"].steps)
print(logger["direction_norm"].steps)
```

```text
[0, 2]
[0, 1, 2, 3]
```

This is a single-state callback. Its extra expectation reuses the ordinary cached batch at unchanged parameters. Moving that measurement to `step_finished` would measure the updated wavefunction and normally trigger new sampling, while the loss in the same record would still describe the old parameters.

The direction norm is not necessarily the raw gradient norm. With SR it describes the SR direction, and with an adaptive optimiser it is not the norm of the final parameter increment. Give additional diagnostics names that make their interpretation clear and avoid overwriting the loss or existing observables.

## Read nested and multi-state records

For a simple statistic, `logger.series("Energy/Mean")` and `logger["Energy"]["Mean"]` access the same series. `flatten()` returns a dictionary keyed by path tuples, while `to_dict()` produces a JSON-compatible representation of the complete history.

A slash in a driver key is not automatically converted into a nested mapping. In a [multi-state run](../drivers/multiple-states.md), `"Energy/state_0"` is one key. Use `logger.series(("Energy/state_0", "Mean"))` or `logger["Energy/state_0"]["Mean"]` to access it. The equivalent plot selection is `logger.plot(("Energy/state_0", "Mean"))`.

Each series keeps its own step labels. This supports occasional observables without filling other iterations with invented values. Align series by their labels when comparing them, rather than assuming every array has the same length. Remember that repeated labels are possible after rejected updates.

## Use your own logger

The `logger` argument accepts a logger object, a callable, or an iterable of them. A plain callable can receive `(step, item)` or `(step, item, variational_state)`. It can additionally accept `driver` as a keyword argument. [`CallableLogger`](../../api/public/neuralqx/utils/io/logging/index.md#neuralqx.utils.io.logging.CallableLogger) adapts this interface, and [`LoggerList`](../../api/public/neuralqx/utils/io/logging/index.md#neuralqx.utils.io.logging.LoggerList) sends each record to several loggers.

For a persistent sink with buffering, subclass [`AbstractLogger`](../../api/public/neuralqx/utils/io/logging/index.md#neuralqx.utils.io.logging.AbstractLogger) and implement `__call__` and, when needed, `flush`. The driver flushes during finalisation, but it does not take ownership of closing every custom resource. Keep resource lifetime explicit in your application. A pathname string is not a supported substitute for a logger object.

Runtime loggers and callable adapters retain or write on JAX process zero by default. Their `log_on_all_processes=True` option changes that policy. Callbacks themselves run on every process, so use them for any collective measurement or checkpoint operation. A logger's primary-process policy is suitable for ordinary output, not for skipping the participation required by a distributed state save.
