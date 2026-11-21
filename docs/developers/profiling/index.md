---
title: Profiling
---

# Profiling

Profiling records the execution of a calculation so that you can relate its cost to the work being performed. In neuraLQX, a profile can show sampling, estimator evaluation, differentiation, SR, parameter updates, observables, callbacks, and logging on one timeline. It can also include native JAX activity and GPU management measurements.

This is useful when a run slows down as the graph grows, when a change in chunk size has an unexpected effect, or when an additional operator dominates an otherwise familiar calculation. The aim is to identify the computational cause before changing the scientific setup or numerical method.

[`Profiler`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler) owns a capture session. A [VMC driver](../../concepts/drivers/index.md) can create that session automatically, or you can open one explicitly around your own code. Importing the profiling module does not start collection.

## Different observations answer different questions

Application regions identify meaningful operations such as `sampling` or `preconditioner.solve`. Their durations measure elapsed host time between entering and leaving Python code. They help explain the organisation of a calculation, including dispatch, bookkeeping, and waits that occur within a region.

Native activity comes from the JAX runtime. It supplies CPU execution and, on supported accelerator setups, device activity and dispatch relationships. It is the appropriate evidence when a Python call returns quickly but the device continues working afterward.

Telemetry samples management counters such as GPU utilisation, used memory, temperature, and power. Those measurements describe the device over their sampling intervals. They are not timestamps for individual kernels or measurements of memory owned by one neuraLQX object.

| Observation | Useful question |
| --- | --- |
| Application regions | Which scientific operation was the program performing? |
| Native host and device activity | When did the dispatched computation actually execute? |
| Telemetry | What was happening to device utilisation, memory, and operating conditions? |
| Rank metadata and clock calibration | How reliably can different processes be placed on one timeline? |

## Measure elapsed work without confusing it with dispatch

JAX computations can continue after their Python call returns. Consequently,

$$
T_{\mathrm{host\ region}}
\neq T_{\mathrm{device\ execution}}
$$

in general. A long host region can include compilation or waiting for earlier work. A short one can launch a substantial device computation. The [JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html) gives the underlying runtime perspective.

neuraLQX keeps these observations separate. Ordinary annotations do not wait for arrays. A session can wait for explicitly supplied results at capture boundaries, while an explicitly synchronising function annotation is available for deliberate latency measurements. The choice affects the experiment being performed and is explained in [choosing what to record](scientists/capture-options.md).

## A route through this section

[Profiling for scientists](for-scientists.md) starts with an existing simulation. It explains how to enable the supplied instrumentation, select a useful recording interval, include custom calculations, and interpret the result. No knowledge of the recorder or trace merger is required for that route.

The technical guides then explain [instrumentation](instrumentation.md), [session internals](internals.md), [trace representation and analysis](traces.md), [backends and telemetry](backends.md), and [distributed profiling](distributed.md). They describe the measurement model as well as the implementation, including what can and cannot be concluded from a trace.

The [configuration reference](configuration.md) comes last. It collects the options and their defaults after the concepts they control have been introduced.
