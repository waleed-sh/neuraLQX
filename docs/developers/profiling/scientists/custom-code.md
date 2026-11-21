---
title: Profiling your own code
---

# Profiling your own code

A simulation often includes work outside the driver, such as preparing an initial state, evaluating a separate observable, or analysing sampled configurations. Public profiling annotations let you name that work and retain it alongside neuraLQX's own regions. You do not need a separate trace for each part of the experiment.

Use [`span`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.span) for a block of Python code and [`annotate`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.annotate) for a function. Both contribute to the active session. Without an active recording interval, ordinary annotations leave the computation unrecorded.

## Name an operation and its compiled computation

There are two useful places to attach a name. A host span marks the interval during which Python enters and leaves an operation. A [`compiled_scope`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.compiled_scope) attaches a name while JAX constructs the compiled computation. That name can then appear in native traces of subsequent executions.

In this example, the function annotation identifies each call to an analysis routine. The compiled scope identifies the matrix calculation inside it. The function is warmed up before recording so that the demonstration focuses on an already compiled calculation.

```python
import jax
import jax.numpy as jnp
from neuralqx.profile import Profiler, annotate, compiled_scope, counter, event, span

@annotate(name="analysis/correlation", category="analysis")
@jax.jit
def correlation(samples):
    with compiled_scope("analysis/gram_matrix"):
        return samples.T @ samples / samples.shape[0]

samples = jnp.arange(128, dtype=jnp.float32).reshape(32, 4)
result = correlation(samples)
jax.block_until_ready(result)

with Profiler("profiles", level="trace", wait_for=lambda: result) as profiler:
    event("analysis_started", attributes={"samples": 32})
    with span("analysis", attributes={"components": 4}):
        result = correlation(samples)
        counter("analysis_size", {"samples": 32, "components": 4})

print(result.shape)
print(profiler.result.trace_path.is_file())
```

```text
(4, 4)
True
```

The annotation is outside `jax.jit`, so it runs on every Python invocation. Putting a Python span inside a jitted function would normally execute it when JAX traces that function, rather than on every cached execution. `compiled_scope` is intended for the compiled side of this boundary. Compiler fusion can combine several named operations, so a scope does not promise a distinct kernel.

The session's `wait_for` callback returns the current result at a capture boundary. This ensures that the relevant computation has completed before collection closes. It does not extend the earlier host span to cover the entire device calculation. Read the native lanes to examine that execution.

Public host spans appear on the application timeline. They do not themselves create native JAX dispatch annotations. The compiled name and the native runtime events provide the complementary execution information.

## Record context without reading device arrays

[`event`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.event) marks an instant, such as the start of an analysis phase. [`counter`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.counter) records numeric host values over time. Span and event attributes attach small pieces of context to an occurrence.

Use information already known on the host, such as an iteration number, graph size, or batch shape. Names and attributes should describe the operation rather than encode an entire configuration. Attribute values accept ordinary Python scalars, and counter values must be finite Python integers or floats.

Do not pass a JAX array as an attribute or counter value. The profiler rejects it instead of silently copying it to the host. Converting an array with `float(...)` yourself can wait for device work and change the behaviour being measured. If you intentionally record a physical quantity, account for that measurement and transfer cost in the experiment.

## Include a custom measurement in a driver capture

A [callback](../../../concepts/callbacks/index.md) runs inside the active driver session, so its own spans join the same recording. The following callback measures a charge before the parameter update and adds the result to the step record.

The surrounding `analysis/charge` span identifies the custom operation. The library's estimator annotations can appear within it at greater detail. Storing the returned mean in `context.log_data` also lets the driver's completion tracking retain that result until the appropriate capture boundary.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator
from neuralqx.profile import span

class MeasureCharge(nqx.callbacks.Callback):
    def __init__(self, operator):
        self.operator = operator

    def update_computed(self, context):
        with span("analysis/charge", category="analysis"):
            estimate = context.driver.state.expect(self.operator)
            context.log_data["ChargeMean"] = estimate.mean

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=3),
    n_samples=64,
    sampler_seed=4,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)

with nqx.config.patch(
    profile=True,
    profile_level="basic",
    profile_output_dir="profiles",
    profile_run_id=None,
):
    history = driver.run(2, callbacks=MeasureCharge(charge), store_history=True)

print(len(history))
print(all("ChargeMean" in record for record in history))
print(driver.profile_result.trace_path.is_file())
```

```text
2
True
True
```

This pattern is useful when an expensive analysis is part of the real simulation. Keep the annotation around the work that is actually performed. Adding a second evaluation solely to obtain a timing would measure a different calculation.

## Advance a schedule in your own loop

In a loop that you manage yourself, [`Profiler.step`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler.step) closes the current application step and advances the recording schedule. Pass its completed step's result tree so that the profiler can wait at a recording boundary when needed.

```python
import jax
import jax.numpy as jnp
from neuralqx.profile import Profiler, Schedule, span

@jax.jit
def advance(value):
    return jnp.tanh(value + 0.01)

value = jnp.ones(16)
jax.block_until_ready(advance(value))
schedule = Schedule(wait=1, warmup=1, active=2, repeat=1)

with Profiler("profiles", level="trace", schedule=schedule) as profiler:
    for iteration in range(4):
        with span("relaxation_step", attributes={"iteration": iteration}):
            value = advance(value)
        profiler.step(wait_for=value)

print(profiler.result.trace_path.is_file())
```

```text
True
```

The four iterations still execute. Only the final two belong to the retained recording interval. The profiler does not wait after every call merely because `step` is called. It waits where the configured capture boundaries require it.

If an iteration launches independent branches, supply all outputs whose completion matters as a tuple or another pytree. If buffers are donated or replaced, a session-level callback returning the currently valid results can be more appropriate than retaining an obsolete array. Such a callback takes precedence over result trees passed to `step` or `stop`.

When a driver is inside the explicit session, it already advances the schedule. Extra manual calls would count additional steps and change which interval is recorded.

## Choose synchronising annotations deliberately

`annotate(sync=True)` waits for the decorated function's returned arrays before ending its span, but only while recording. This is useful when the question is the elapsed latency of obtaining that result. It can also include time spent waiting for earlier dependent work.

For normal execution analysis, start with ordinary spans and boundary completion. Repeated waits can serialise an otherwise overlapping workload. The [instrumentation guide](../instrumentation.md) develops this distinction and describes threads, asynchronous functions, and annotation limits.
