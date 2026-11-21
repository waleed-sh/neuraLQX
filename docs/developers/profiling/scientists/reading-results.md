---
title: Reading a profile
---

# Reading a profile

A trace is a record of one execution under a particular collection policy. Reading it well means connecting the visible intervals to the scientific operation, the runtime that executes it, and the boundaries of the measurement. Start with the whole run, locate a representative iteration, and then inspect the phase that motivates your investigation.

The [`ProfileResult`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileResult) returned by a completed session identifies its artifacts. In automatic driver capture, it is available as `driver.profile_result`. For an explicit context, use `profiler.result` after leaving that context.

## Open the combined trace

Open `result.trace_path` in the [Perfetto trace viewer](https://ui.perfetto.dev/) using its file-opening control. The trace uses the [Chrome Trace Event format supported by Perfetto](https://perfetto.dev/docs/getting-started/other-formats), compressed as `trace.json.gz`.

On a single process, the combined file normally sits at `output_dir/run_id/trace.json.gz`. The rank directory also contains a local trace and `manifest.json`. With a distributed capture, rank zero produces the combined file when using an online collection transport. Other ranks retain their local paths. [Offline distributed capture](../distributed.md#offline-artifacts) deliberately leaves assembly for later.

Use the application lanes to find names such as `vmc.step`, `sampling`, and `gradient`. Native collection adds runtime lanes and, where supported, device activity. Telemetry adds sampled counters. The available lanes depend on the [capture options](capture-options.md), the executed code paths, and the installed runtime.

## Read one iteration in context

First establish whether the selected iteration includes compilation or other initialisation. Then examine the ordering of sampling, estimator work, differentiation, preconditioning, and the update. Observables and callbacks can add substantial work after the main gradient calculation, so include them when judging the cost of a complete iteration.

The `step` attribute on `vmc.step` follows the accepted-update count, while its `iteration` attribute identifies the attempted iteration within that driver call. A rejected update can therefore produce repeated step labels. The profiler's schedule still advances for each attempt.

A host region measures elapsed time inside its Python boundary. It can contain compilation, dispatch, bookkeeping, or waits. It is not an automatic measurement of all the device work caused by that operation. With native collection, inspect whether kernels continue after the host call returns and where the application eventually needs their results.

At capture boundaries, a `wait_for_completion` region can expose the final completion wait. Its duration can reflect work dispatched earlier in the interval. Assigning all of that cost to finalisation would obscure the computation that actually caused it.

## Avoid adding overlapping durations

Nested spans are inclusive. If `gradient` contains local estimator evaluation, summing both durations counts that interval twice. Host and device lanes can also overlap, as can execution on different ranks.

For intervals $I_1,\ldots,I_n$, their total occupied elapsed time is

$$
T_{\mathrm{occupied}}=\left|\bigcup_{k=1}^{n}I_k\right|,
$$

which generally differs from $\sum_k |I_k|$. A shorter complete iteration can result from increased overlap even when the sum of individual operation durations changes little.

Likewise, a named operator region does not necessarily contain every cost that can be associated with that operator. neuraLQX can share connectivity preparation and network evaluation across terms. Those shared regions represent work performed once, and should not be charged in full to every term. The [instrumentation guide](../instrumentation.md#shared-work-and-operator-attribution) explains this attribution.

## Check that the recording is usable

The manifest records errors, recording windows, dropped application events, and backend information. The combined trace also reports whether all expected rank artifacts completed successfully and whether bounded application recording lost events. These are separate conditions. A trace can contain every rank and still have dropped application events.

The following small example checks these fields without depending on a simulation. Loading the JSON object is reasonable for this deliberately small capture. For a large native trace, use a trace viewer or the [streaming merger](../traces.md#merging-completed-artifacts) rather than building an additional full in-memory copy.

```python
import gzip
import json
from tempfile import TemporaryDirectory

from neuralqx.profile import Profiler, span

with TemporaryDirectory() as directory:
    with Profiler(directory, level="basic") as profiler:
        with span("analysis/check"):
            total = sum(range(100))

    result = profiler.result
    manifest = json.loads(result.manifest_path.read_text())
    with gzip.open(result.trace_path, "rt") as stream:
        trace = json.load(stream)

    print(manifest["status"])
    print(trace["neuralqx"]["complete"])
    print(trace["neuralqx"]["data_loss"])
    print(any(item.get("name") == "analysis/check" for item in trace["traceEvents"]))
```

```text
ok
True
False
True
```

If a region is missing, first check whether it ran during a retained recording window and whether the chosen detail includes it. Then inspect dropped-event diagnostics and backend errors. A `basic` capture intentionally has no native kernel record. A missing GPU lane in a native capture also requires checking runtime support, rather than concluding that the device was idle.

`complete=True` and `data_loss=False` describe the collected rank artifacts and application recorder diagnostics. They do not guarantee that every possible native activity was supported or that an unrecorded part of the application contained no work.

## Make comparisons that retain scientific meaning

Compare representative repeated iterations after accounting for compilation. Keep graph topology, representation cutoffs, model structure, estimator choice, and numerical precision fixed when they are not the variables under study. Record sample counts and [chunk sizes](../../../concepts/vqs/batching.md), since they control different aspects of the calculation.

A smaller sample batch can make an iteration faster while increasing statistical uncertainty. A different sampler can change both execution cost and autocorrelation. For a scientific comparison, consider the effort needed to obtain estimates of comparable quality, using the [sampling diagnostics](../../../concepts/samplers/diagnostics.md) alongside timing information.

Use short repeated captures to distinguish a reproducible effect from incidental load. For distributed traces, also respect the reported [clock uncertainty](../distributed.md#aligning-process-clocks) before interpreting small ordering differences between ranks.

Once a specific phase is identified, increase detail only where it resolves the remaining question. The technical guides explain [how annotations are recorded](../instrumentation.md), [how traces are assembled](../traces.md), and [what each backend measures](../backends.md).
