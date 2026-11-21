---
title: Profiler internals
---

# Profiler internals

The profiler separates application instrumentation from the machinery that owns collectors, buffers, files, and distributed coordination. An annotated numerical operation needs only to contribute an event to an active recording window. It does not open files, negotiate with another rank, or decide when a native collector should stop.

This separation matters in a JAX application because the lifetime of a Python call, a queued computation, and a capture session can all differ. The session must preserve those distinctions while producing artifacts that can be inspected after the workload finishes or fails.

## Session ownership and configuration

Constructing [`Profiler`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler) creates no output directories and starts no collectors. Entering its context, or calling `start()`, freezes the effective options and begins the session. Without an explicit [`ProfileOptions`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileOptions), that snapshot includes the package configuration and active thread-local overrides at entry.

An active session belongs to one process and one controlling thread. Application annotations from other threads and tasks participate through a process-wide active reference, but control operations such as `step()` and `stop()` must run on the owner thread. Non-disabled sessions cannot overlap or nest. A driver inside an explicit session therefore borrows it instead of starting another collector.

Each instance is single-use. Successful finalisation makes its result available, and a repeated `stop()` returns that result. It cannot be started again. An inherited instance also cannot be controlled from a forked child. Create a new session in the child if it needs a separate capture.

An explicit session at level `off` is a special inactive case. It claims no active recorder, creates no artifacts, and does not call the completion callback. Its result contains no artifact paths. The `profile` package option controls automatic driver capture and is independent of whether a non-disabled explicit session may be opened.

## Recording windows and collector lifetimes

A continuous session opens one recording window. A scheduled session interprets [`Schedule`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.Schedule) actions at successive application-step boundaries.

| Action | Application recorder | Native collector when requested |
| --- | --- | --- |
| `WAIT` | Does not retain annotations | Inactive |
| `WARMUP` | Does not retain annotations | Collects an interval that is discarded from normalised output |
| `RECORD` | Retains annotations | Collects the interval retained in the trace |

Each cycle owns a separate collector window, including adjacent cycles with no waiting steps between them. This bounds the native capture lifetime and keeps repeated captures distinct. The raw native artifacts can include collector warmup, while the exported rank trace retains only the recording interval.

`step()` first records the elapsed application step, increments the session's zero-based step index, and then applies the next action. A recorded `step` interval therefore ends before a completion wait performed during that transition. The wait appears separately when it occurs inside a retained window.

The schedule does not determine how many iterations the application executes. It determines which parts of those iterations are observed. A finite schedule can finish recording while the surrounding loop continues.

## Completion boundaries

With `synchronize="boundary"`, the profiler waits through `jax.block_until_ready` on a supplied result pytree. The session-level `wait_for` callback takes precedence over values supplied to `step(wait_for=...)` or `stop(wait_for=...)`. Without a callback, a new supplied value replaces the previous retained reference.

Completion can be required before entering a capture after waiting, when changing from collector warmup to recording, and when closing a window. The initial entry of a continuous session does not first drain arbitrary work that was queued before it. If that earlier work should be excluded, complete the relevant warmup results before entering the context.

This is a result-based boundary, not a global device barrier. All independent branches that must finish inside the recording should be represented by the supplied result tree. Unrelated computations and JAX effects are not implicitly included. `synchronize="none"` disables these waits and does not invoke the callback.

Donation makes the choice of retained result particularly important. If an array is later donated to another computation, an old reference may no longer be a valid completion target. A callback that returns the current live results can avoid that stale reference. It must return existing results rather than launching new work merely to obtain something to wait for.

## Integration with the driver

The automatic driver path owns a session for the bounded lifetime of `run`. It supplies current parameters, sampler state, objective statistics, derivatives, update directions, optimiser state, and relevant preconditioner information as completion values. Multiple-state drivers also expose their current energies and overlaps.

Observables and callbacks can produce result arrays independent of the current update. The driver retains unfinished values from its log records and prunes values that are already ready, using a readiness check that does not wait. This keeps an earlier outstanding measurement represented at a later boundary without retaining the entire numerical history solely for profiling.

The `vmc.step` region covers one attempted iteration. Its `step` attribute uses the accepted-update count, and its `iteration` attribute uses the attempt index within the call. The profiler advances once per attempt, including a rejected update. These counters answer different questions and should not be interchanged when analysing a scheduled capture.

A borrowed explicit session receives the same driver instrumentation and step advancement. The outer context owns finalisation and the resulting `ProfileResult`. The driver iterator participates only in an existing explicit session, avoiding an automatic collector whose lifetime could outlast an incompletely consumed generator.

## Bounded recording

The application recorder allocates thread buffers lazily. Buffers reserve event capacity in small batches, currently up to 64 slots, under a shared lock. Most appends then use the already reserved capacity rather than taking that shared reservation lock for every event. Task-to-lane bookkeeping has its own coordination, so this should not be interpreted as a universally lock-free recorder.

`max_events` bounds application and telemetry event reservations across the process and the whole session, including repeated recording windows. It is not a fresh budget per window. Unused reserved capacity in a thread can mean the number of retained events is lower than the configured maximum. `max_lanes` bounds application thread and task lanes, with bounded handling for excess participants.

When a limit is reached, the recorder counts omitted events instead of growing without bound or overwriting earlier records. Final diagnostics include retained and dropped counts, events without a lane, reserved capacity, and unfinished spans. A `profiler_dropped_events` counter is emitted when loss needs to be exposed in the trace.

These limits do not bound JAX's native collection buffers. To limit native capture size, shorten the recording windows or reduce native tracing detail. The [configuration reference](configuration.md#recording-budgets) distinguishes these budgets.

Complete span records are appended when the span exits. A nested child can therefore be written before its parent. Event file order is not a chronological ordering guarantee. Timestamps and durations carry that information, and the [trace exporter](traces.md) does not need to sort the entire capture in memory.

## Finalisation and failures

Closing a recording window first performs the configured completion wait, establishes the retained end time, and stops its collectors. Session finalisation then stops accepting application events, exports the local normalised trace, and writes rank metadata. The transport commits completed manifests and, where applicable, rank zero assembles the combined trace.

Join worker threads and await tasks before closing the context. Unfinished spans are reported as a collection failure, because the session cannot infer when their work would have ended. There is no garbage-collection or process-exit mechanism that safely substitutes for explicit distributed finalisation.

A workload exception remains the primary exception. The profiler attempts cleanup and records the exception type in the manifest. If cleanup also fails, that information is attached to the original exception. A collection failure without a workload exception raises [`ProfileError`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileError), with local artifacts retained where finalisation reached their creation.

Incorrect ownership or lifecycle use raises [`ProfileStateError`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileStateError). Distributed deadlines can raise [`ProfileTimeoutError`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileTimeoutError). These failures have different causes, so inspect the manifest and original exception before treating an absent combined trace as a failed simulation.

The [trace guide](traces.md) describes the artifacts left by this lifecycle. [Distributed profiling](distributed.md) explains how rank registration, clocks, and transfer fit around it without adding communication to individual annotations.
