---
title: Callbacks
---

# Callbacks

A callback is a small piece of Python code that the driver calls at a defined point in a run. It can inspect a measurement, add a diagnostic to the record, save a state, or request that optimisation stop. The driver still owns the optimisation loop, so these additions do not require you to rewrite sampling and parameter updates.

Use [`Callback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.Callback) for new event-based callbacks. Override only the methods you need. The supplied [progress and live-monitoring callbacks](built-in.md) use the same interface.

## The context passed to a callback

Each hook receives a [`CallbackContext`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.CallbackContext). It gives access to the current driver and the data associated with that attempt.

| Context attribute | Meaning |
| --- | --- |
| `driver` | The driver running this callback |
| `state` | `driver.state`, which is one VQS for `VMC` and a tuple for a multi-state driver |
| `step` | The accepted-update counter at the start of this attempt |
| `loss` and `loss_name` | The driver's latest loss statistics and its configured label |
| `log_data` | The mutable dictionary that will be sent to the logger and optionally retained |
| `update` | The computed direction, available after gradient calculation and preconditioning |
| `accepted` | Whether this attempt applied its parameter update |
| `error` | The exception associated with a failed run, otherwise `None` |

The context object is reused during the run. If you want a lasting record, retain the values you need rather than appending the context itself to a list.

At the beginning of an attempt, the new loss has not yet been calculated. `context.loss` can still refer to the preceding calculation. Use `update_computed` or a later hook to inspect the current loss.

## Where hooks occur

The normal order is shown below. The distinction between measurements and updated parameters is also developed in the [driver monitoring guide](../drivers/monitoring.md#match-a-record-to-the-correct-parameters).

| Hook | Position in the run |
| --- | --- |
| `run_started` | Once when `run` begins |
| `step_started` | At the start of an attempt, before resetting samples or computing a loss |
| `update_started` | Immediately after `step_started`, also before the state's sample reset |
| `update_computed` | After the loss, gradient, preconditioner, and requested observables have been evaluated |
| `parameters_updating` | Immediately before an accepted update is applied |
| `step_finished` | After the attempt, including attempts whose update was skipped |
| `run_failed` | When a run-body exception is being propagated |
| `run_finished` | During finalisation after completion, a graceful stop, or failure |

After `step_finished`, the driver sends the record to its logger and appends it to history if requested. Finalisation flushes the logger before calling `run_finished`.

In the current native driver, the sample reset happens **after** `update_started`. Assigning a custom sample batch in that hook would therefore have it cleared. Do not assume that identically named hooks in another library have the same timing.

## Record a small custom diagnostic

This callback adds the attempt label, accepted-update count, and acceptance decision to each retained record. The counts make the before-and-after timing visible without adding another operator measurement.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

class RecordUpdates(nqx.callbacks.Callback):
    requires_log_data = True
    stores_history = True

    def step_finished(self, context):
        context.log_data["step"] = context.step
        context.log_data["accepted_updates"] = context.driver.step_count
        context.log_data["accepted"] = context.accepted

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=12),
    n_samples=32,
    sampler_seed=13,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)

history = driver.run(3, callbacks=RecordUpdates())
print([record["step"] for record in history])
print([record["accepted_updates"] for record in history])
print(all(record["accepted"] for record in history))
```

```text
[0, 1, 2]
[1, 2, 3]
True
```

`requires_log_data=True` asks the driver to populate the usual loss and diagnostic entries even when no other consumer needs them. `stores_history=True` additionally enables history retention by default for a single-state driver. An explicit `store_history=False` on `run` overrides that retention request.

A callback that only reads `context.loss` after computation does not need to request a log dictionary. The loss is computed for optimisation anyway. Keep both flags at their default `False` when neither is needed.

## Attach one callback or several

Pass a callback instance or a sequence through `callbacks=...`. The singular `callback=...` is also accepted, but supplying both arguments raises an error.

Callbacks are ordered by their integer `priority`, from smaller to larger values. The default is zero, and equal priorities preserve the supplied order. This lets a diagnostic callback populate `log_data` before a later monitoring callback consumes it. A stop request prevents the remaining callbacks at that event from running.

[`FunctionCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.FunctionCallback) wraps a function with signature `function(context)` and invokes it at `step_finished`. A bare callable is instead interpreted as the compatibility signature `function(step, log_data, driver)`. Wrap a one-argument function explicitly rather than passing it bare.

For the context API, hooks can return [`CallbackControl`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.CallbackControl), `None`, or a Python boolean. [Controlling a run](control.md) explains the precise stopping and rejection behaviour. Prefer explicit control objects when the distinction matters.

## Existing callback code

[`AbstractCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.AbstractCallback) preserves the older hook names such as `on_step_end(step, log_data, driver)`. Its `callback_order` supplies the priority, and it requests log data and history by default. The compatibility hooks are mapped onto the native event sequence above.

[`LegacyCallbackWrapper`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.LegacyCallbackWrapper) is applied automatically to a bare three-argument function. That function runs at the end of the step, after an accepted update. Returning the Python value `False` stops the run. Other return values are ignored by this wrapper, so use the context API for explicit control messages.

The compatibility `on_compute_update_end` hook has a separate boolean convention: `True` skips the parameter update and `False` continues. This convention does not apply to the new `update_computed(context)` hook, where `False` requests a stop.

## Continue through the guides

[Built-in callbacks](built-in.md) covers the terminal progress bar and local dashboard. [Controlling a run](control.md) explains graceful stopping and skipped updates. [Logging and monitoring](logging.md) shows how to retain, plot, and export measurements, and how to add observables without measuring them on every iteration.

For a state-saving callback, continue with [driver checkpointing](../drivers/checkpointing.md). Callbacks execute as ordinary Python around JAX work, so a host-side conversion or file write can affect iteration time. Keep that work purposeful and choose its frequency deliberately.
