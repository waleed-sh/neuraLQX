---
title: Choosing what to record
---

# Choosing what to record

A profile is most useful when the recorded information matches the question. Investigating an expensive observable does not require the same capture as investigating idle gaps between GPU kernels. neuraLQX separates the kinds of activity collected, the depth of its own annotations, and the interval retained in the result.

This page explains those choices. The final [configuration reference](../configuration.md) gives every option, its default, and its accepted values.

## Choose the observation level

`profile_level` controls which execution information is collected.

| Level | Recorded execution information |
| --- | --- |
| `basic` | Application spans, events, and counters, including the enabled neuraLQX annotations |
| `trace` | The application record plus native JAX host activity and supported device activity |
| `full` | The `trace` capture with native Python call tracing enabled by default |
| `off` | An explicitly created profiler records nothing and creates no artifacts |

Start with `basic` when you want to locate a costly phase in the Python-level calculation. Use `trace` to relate that phase to actual runtime execution. Use `full` when Python call structure itself is relevant, for example when investigating host overhead between dispatches. Python tracing adds information and overhead, so it is usually better suited to a short capture than a complete production run.

`full` does not automatically enable GPU telemetry, NVTX, or every neuraLQX annotation. Those are independent controls. Its Python tracing default can also be overridden explicitly.

For automatic driver capture, `profile=True` is the activation switch. If the configured level is `off`, the driver promotes it to `basic`. Set `profile=False` to disable automatic capture. An explicit `Profiler(level="off")` remains disabled regardless of that automatic-driver convention.

## Choose the annotation depth

`profile_detail` controls how far into neuraLQX's implementation the application record goes.

| Detail | What it helps distinguish |
| --- | --- |
| `summary` | Sampling, gradients, preconditioning, updates, observables, callbacks, and logging |
| `detailed` | Stages within those operations, including estimator work, named operator contributions, shared evaluations, and linear solves |
| `developer` | Additional preparation, dispatch, and bookkeeping regions useful when investigating the implementation |

These controls are independent. A `basic` capture with `detailed` annotations can explain which estimator stage is costly without collecting native activity. A `trace` capture with `summary` annotations can show device execution while keeping the application lanes relatively simple.

Your own [spans and annotations](custom-code.md) are not filtered by this package detail setting. Names attached to compiled JAX computations also follow a different mechanism, described in [instrumentation](../instrumentation.md).

## Record a short window

A [`Schedule`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.Schedule) divides application steps into waiting, collector warmup, and retained recording. It starts at step zero when the session opens.

During `wait`, collectors are inactive. During `warmup`, native collection can run, but that interval is discarded from the normalised recording. During `record`, application events and the relevant native interval are retained. A finite schedule returns to waiting after its final cycle, while the application itself continues running.

```python
from neuralqx.profile import Schedule

schedule = Schedule(
    skip_first=1,
    wait=1,
    warmup=1,
    active=2,
    repeat=2,
)
print([schedule.action(step).value for step in range(10)])
```

```text
['wait', 'wait', 'warmup', 'record', 'record', 'wait', 'warmup', 'record', 'record', 'wait']
```

The initial skipped step happens once. Each cycle then has one waiting step, one warmup step, and two recorded steps. Setting `repeat=0` repeats cycles until the session closes.

For automatic driver capture, use the corresponding package options. For example, `profile_wait_steps=2`, `profile_warmup_steps=1`, and `profile_active_steps=3` retain three steps after the initial waiting and warmup intervals. `profile_repeat` controls the number of cycles. The default `profile_active_steps=0` instead records continuously through the context, with no schedule.

The driver advances the profiler once per attempted iteration, including a rejected parameter update. This count can differ from `driver.step_count`, which counts accepted updates. Do not call `profiler.step()` yourself around a driver that is already advancing the session. In a custom loop, call it once after each application iteration, as shown on the [next page](custom-code.md#advance-a-schedule-in-your-own-loop).

Collector warmup and compilation warmup are related but different. The schedule discards a time interval. It cannot guarantee that all later shapes and branches have already compiled. Use the same workload during warmup and inspect the retained trace for unexpected compilation.

## Decide where completion matters

JAX normally dispatches work asynchronously. For a host function that submits device work, its observed duration is approximately

$$
T_{\mathrm{host}}
=T_{\mathrm{Python}}+T_{\mathrm{dispatch}}+T_{\mathrm{wait\ inside\ call}}.
$$

It need not include the full execution time of the submitted computation. The [JAX asynchronous dispatch guide](https://docs.jax.dev/en/latest/async_dispatch.html) explains why reading a result or explicitly waiting for it changes the measurement.

The default `profile_synchronize="boundary"` waits for explicitly supplied result arrays when recording boundaries require completion. Automatic driver capture supplies its relevant results. In a custom session, you supply them through `wait_for` or `step(wait_for=...)`. Arrays unrelated to those results are not automatically covered.

With `profile_synchronize="none"`, the profiler performs no such completion waits. That choice may help investigate an already managed pipeline, but queued work can extend beyond the captured interval. It is then your responsibility to establish which work the trace actually contains.

Ordinary spans never insert a wait. A synchronising function annotation, `annotate(sync=True)`, waits for its return value inside the recorded span. Use it for a deliberate latency experiment, since placing waits inside a loop can remove overlap that exists in the normal simulation.

## Add device measurements only when they help

GPU telemetry can provide context for sustained utilisation, memory use, temperature, and power. It is sampled on a background thread and does not identify the cost of a single kernel. Its sampling interval should be judged against the duration of the phenomenon you are investigating.

NVTX supplies ranges for an external NVIDIA profiler. Enabling it does not launch that profiler. The [backends guide](../backends.md) explains both features and their optional dependencies.

Finally, keep captures bounded. More detail, more Python tracing, and more recorded steps can all enlarge the result. Check the manifest for dropped events before interpreting a missing region as evidence that no work occurred. The [reading guide](reading-results.md) shows where that information is recorded.
