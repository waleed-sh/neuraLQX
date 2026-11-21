---
title: Built-in callbacks
---

# Built-in callbacks

neuraLQX supplies callbacks for terminal progress and a local browser dashboard. Both observe the driver through its [event interface](index.md). They help you follow an active calculation, while a [logger](logging.md) provides a record you can retain and analyse.

| API | Purpose |
| --- | --- |
| [`ProgressBar`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.ProgressBar) | Display completed attempts and the latest objective in a terminal |
| [`LiveMonitoringCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.LiveMonitoringCallback) | Send existing scalar measurements to a local browser dashboard |
| [`LiveMonitor`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.LiveMonitor) | Operate that dashboard directly from a custom loop |
| [`FunctionCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.FunctionCallback) | Adapt a context-taking function to an end-of-step callback |

Stopping criteria and periodic state snapshots can be implemented with the examples in [controlling a run](control.md) and [checkpointing](../drivers/checkpointing.md). The supplied callback namespace does not currently include separate ready-made early-stopping or timeout classes.

## Terminal progress

For the common case, pass `show_progress=True` to `driver.run`. The driver adds a `ProgressBar` whose total is the requested number of attempted iterations. The following example updates the displayed estimate every two completed attempts.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=14),
    n_samples=32,
    sampler_seed=15,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)

driver.run(
    4,
    show_progress=True,
    progress_every=2,
    progress_min_interval=0.1,
    progress_leave=False,
)
print(driver.step_count)
```

```text
4
```

The bar advances when `step_finished` is reached, including a rejected parameter update. It therefore counts completed attempts rather than accepted parameter updates. A callback that stops an event before the bar runs can leave the display short of its requested total.

`progress_every` limits objective-text refreshes, while `progress_min_interval` sets a minimum interval in seconds for display refresh. Neither thins the gradient calculation or the driver's `obs` measurements. `progress_leave=False` removes the finished bar. Displaying statistics can synchronise device results with the host, so frequent updates are not free.

For an explicit callback list, construct `ProgressBar(total=..., every=..., min_interval=..., leave=...)`. Do not also enable `show_progress=True` unless you intend to create two bars. [`ProgressBarCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.ProgressBar) is an alias of the same class. Its priority is `10000`, so ordinary callbacks run first.

The bar uses `tqdm` and appears only on JAX process zero. If `tqdm` is unavailable, the run continues with a warning and no progress display. The bar closes during the normal callback finalisation path.

## Live monitoring

`LiveMonitoringCallback` starts a local dashboard when the run begins and prints its address. It displays the objective, scalar entries already present in `log_data`, and diagnostics such as acceptance, local-value variance, and `R_hat` when available. It does not calculate additional expectations on its own.

The example uses an automatically selected local port and a context manager, which closes the dashboard when the block ends. Increase the run length in a real calculation to inspect it while optimisation is active.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=16),
    n_samples=64,
    sampler_seed=17,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge,
    optax.sgd(0.01),
    variational_state=state,
    loss_name="ChargeSquared",
)

with nqx.callbacks.LiveMonitoringCallback(
    port=0,
    every=2,
    min_interval=0.0,
    max_points=200,
) as live:
    driver.run(
        4,
        obs={"Charge": charge},
        callbacks=live,
        store_history=False,
    )
    print(live.monitor.is_running)

print(live.monitor is None)
```

The output contains the actual dashboard URL followed by two `True` values. The first confirms that the dashboard is available before leaving the block. The second confirms that its server has been released afterward.

The default address is `127.0.0.1` with a requested port of `8050`. If that port is occupied, the monitor tries nearby ports. `port=0` lets the operating system select an available port. Read the printed URL or `live.monitor.url` rather than assuming the requested port was used. In a distributed run, the `monitor` object exists only on process zero.

## Choose the display frequency and lifetime

`every` filters intermediate updates by the driver's step label, and `min_interval` limits their frequency in seconds. The first and final completed steps are submitted even when the intermediate filter would omit them. `update_interval_ms` controls how frequently the browser requests updates, which is separate from how often the callback submits them.

Device-to-host conversion happens in a background consumer. The pending queue is bounded, and a full queue drops an older update so a slow display does not block the training hook. `queue_size` defaults to 32, while `max_points` limits retained dashboard history to 10000 points by default. Use a [runtime or file logger](logging.md) when you need a complete record.

By default, the final dashboard remains available after `run` returns. Call `live.shutdown()` when you finish inspecting it. Alternatively, set `close_on_finish=True` or use the context manager shown above. Calling `LiveMonitor.finish()` marks a run complete but keeps its HTTP server available. `shutdown()` closes the server and background threads.

The browser's stop control requests a graceful stop at the next `step_started` event. It does not interrupt a JAX calculation already in progress. Attach the callback on every participating JAX process so that the request can be broadcast consistently. Only process zero serves the interface.

## Which values are displayed

The objective label defaults to the driver's `loss_name` and can be overridden with `objective_label`. Scalar statistics supply a mean and standard error. For complex scalars, the dashboard displays the real component. Inspect complex measurements through a logger when the imaginary part matters.

Array-valued entries and lists of per-state statistics are not automatically turned into separate curves. In a [multi-state run](../drivers/multiple-states.md), the supplied per-state objective keys are already scalar statistics. For other per-state observables, a callback can add one named scalar entry per state before the live-monitoring callback runs.

`LiveMonitor` offers the same display independently of a driver. Its `log(step, objective=..., observables=...)` method accepts statistics, scalar values, or `(mean, error)` pairs. Use it as a context manager, or finish and shut it down explicitly when your own loop ends.

## Function and compatibility adapters

`FunctionCallback(function)` calls `function(context)` at `step_finished`. It does not request log data or history by default. If the function needs the driver's standard log entries and no logger or observable request already provides them, set the wrapper's `requires_log_data=True`.

For older code, [`LegacyCallbackWrapper`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.LegacyCallbackWrapper) adapts a three-argument function, and [`AbstractCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.AbstractCallback) exposes the older hook names. [`CallbackPipeline`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.CallbackPipeline), also exported as [`CallbackList`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.CallbackPipeline), orders callbacks and combines their control requests. Ordinary use only needs to pass a list to `run`, which builds this pipeline automatically.
