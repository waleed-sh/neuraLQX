---
title: Controlling a run
---

# Controlling a run

Control callbacks make decisions at the boundaries of the [optimisation loop](../drivers/index.md). They can stop a calculation before another expensive iteration, examine a computed direction before it is applied, or skip one parameter update. Choosing the event is as important as choosing the condition.

Use [`CallbackControl`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.CallbackControl) to make the intention explicit. The driver handles its stop request as a normal end to `run`.

## Continue, stop, or skip

| Return value from a context-based hook | Effect |
| --- | --- |
| `None`, `True`, or `CallbackControl.continue_()` | Continue the event sequence |
| `False` or `CallbackControl.stop_run(reason)` | Stop the run at this event |
| `CallbackControl.skip_parameter_update(reason)` from `update_computed` | Complete the attempt without applying its update |

Use Python booleans when returning a boolean. A JAX or NumPy boolean scalar is not one of the explicitly accepted boolean objects. Returning a control object avoids that ambiguity.

[`StopRun`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.StopRun) can also be raised to request a graceful stop. Other exceptions follow the failure path, with `context.error` supplied to `run_failed`, and are propagated to the caller. The ordinary finalisation path flushes the logger and calls `run_finished`.

A reason attached to a control request is not automatically added to the history or printed. Keep it on your callback or record it explicitly if you need to inspect it later.

## Stop at the boundary you intend

At `step_started`, no new gradient has been computed. This is a suitable place to enforce a maximum accepted-update count or a wall-clock budget before beginning more work.

At `update_computed`, the current loss and direction are available, but the parameters and Optax state have not changed. A stop here retains the parameters that produced that loss. It can be useful when a residual target has been met or a numerical check fails.

At `step_finished`, an accepted update has already happened. Stopping there leaves that update in place. It also exits before the driver writes the current record to its logger or history. If you want the final completed record to be retained, store your stopping decision and return it at the next `step_started` event instead.

In all cases, a stop request ends the remaining callbacks at that event. Place essential work before the stopping callback through its `priority`, or perform suitable final work in `run_finished`. Finalisation hooks should release resources without issuing another stop request.

## Skip an update

A skip request is acted on only at `update_computed`. The driver still reaches `step_finished`, writes its log record, and consumes one attempted iteration. The parameter tree, Optax state, and accepted-update count are unchanged. Sampling and preconditioning have already run, so a skip is not a rollback of the entire attempt.

The next attempt collects a new batch and uses the same step label. There is no automatic retry loop that extends `n_iter` until a particular number of updates has been accepted. A callback with its own attempt counter is useful if you want a separate limit on repeated rejections.

This example intentionally skips the first direction so the accounting is visible, then stops before starting work beyond three accepted updates.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

class StopAfterUpdates(nqx.callbacks.Callback):
    priority = -10

    def __init__(self, target):
        self.target = target

    def step_started(self, context):
        if context.driver.step_count >= self.target:
            return nqx.callbacks.CallbackControl.stop_run("Update budget reached")

class SkipFirstUpdate(nqx.callbacks.Callback):
    def __init__(self):
        self.skipped = False

    def update_computed(self, context):
        if not self.skipped:
            self.skipped = True
            return nqx.callbacks.CallbackControl.skip_parameter_update(
                "Demonstrating a rejected attempt"
            )

    def step_finished(self, context):
        context.log_data["step"] = context.step
        context.log_data["accepted"] = context.accepted

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=18),
    n_samples=32,
    sampler_seed=19,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)

history = driver.run(
    5,
    callbacks=[StopAfterUpdates(3), SkipFirstUpdate()],
    store_history=True,
)
print(driver.step_count)
print([record["step"] for record in history])
print([record["accepted"] for record in history])
```

```text
3
[0, 0, 1, 2]
[False, True, True, True]
```

The repeated zero labels two attempts at the same parameters, not two accepted updates. A logger using `every` filters these labels, so repeated attempts can also produce repeated entries at one logged step.

Treat `context.update` as the computed preconditioned direction. It is not yet an Optax parameter increment. Assigning a different object to that context field does not replace the driver's direction. To transform updates, configure the optimiser or preconditioner rather than relying on context assignment.

## Stop when a residual is sufficiently small

A positive constraint objective often has a meaningful target scale $\varepsilon$. One possible practical rule compares the estimated mean $\widehat{\mathcal L}_t$ and its standard error $s_t$ through

$$
\operatorname{Re}\widehat{\mathcal L}_t+z s_t
\leq\varepsilon.
$$

Here $z$ controls how conservatively the reported sampling error enters the criterion. Requiring the condition on several successive estimates avoids responding to one downward fluctuation. It is still a stopping rule, not a rigorous confidence statement about the exact physical result. Correlation, inadequate equilibration, and systematic approximation errors remain relevant.

The callback below checks the current estimate before applying its update. It also stops if the mean or error is not finite, rather than continuing with an invalid objective.

```python
import math

import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

class ResidualTarget(nqx.callbacks.Callback):
    def __init__(self, target, consecutive=3, error_multiplier=2.0):
        self.target = target
        self.consecutive = consecutive
        self.error_multiplier = error_multiplier
        self.streak = 0
        self.reason = None

    def run_started(self, context):
        self.streak = 0
        self.reason = None

    def update_computed(self, context):
        mean = float(context.loss.mean.real)
        error = float(context.loss.error_of_mean)
        if not (math.isfinite(mean) and math.isfinite(error)):
            self.reason = "Non-finite objective statistics"
            return nqx.callbacks.CallbackControl.stop_run(self.reason)

        upper = mean + self.error_multiplier * error
        self.streak = self.streak + 1 if upper <= self.target else 0
        if self.streak >= self.consecutive:
            self.reason = "Residual target reached repeatedly"
            return nqx.callbacks.CallbackControl.stop_run(self.reason)

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=20),
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=21,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.02), variational_state=state
)
criterion = ResidualTarget(target=0.1)
driver.run(8, callbacks=criterion)
print(0 <= driver.step_count <= 8)
```

```text
True
```

This small demonstration may exhaust its eight-iteration budget before reaching the target. Inspect `criterion.reason` to distinguish that case from a callback stop. On a target-based stop, the current loss is available through `driver.loss_stats`, but its record has not reached the logger. A [checkpoint callback](../drivers/checkpointing.md) can save the retained state during finalisation.

For a plateau criterion, compare changes over a window against both a physically meaningful scale and the sampling error. A fixed patience count without that context can stop merely because the noise hides further progress. The [monitoring guide](../drivers/monitoring.md#read-the-diagnostics-together) explains which diagnostics to inspect alongside an objective curve.

## Time budgets and distributed decisions

A time-budget callback can store a `time.monotonic()` deadline in `run_started` and compare against it in `step_started`. Leave enough time for another complete iteration and any final checkpoint. A boundary check cannot interrupt a long kernel or a checkpoint that is already executing.

Callbacks run on every participating JAX process. A condition that changes control flow must therefore produce the same decision everywhere. A local deadline, process-specific metric, or external stop request needs coordination before some processes stop while others enter collective work. The [live-monitoring callback](built-in.md#choose-the-display-frequency-and-lifetime) handles its browser stop request by broadcasting it.

Keep collective measurements and state saves outside process-zero-only conditions. Restrict only ordinary host output, such as printing or writing a non-collective scalar log, when appropriate. The [distributed simulation guide](../../distributed/multi-node.md) provides the broader execution context.
