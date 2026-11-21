---
title: Traces and analysis
---

# Traces and analysis

A profiling session produces both an event timeline and the metadata needed to interpret it. The timeline shows observed execution. The metadata explains which ranks and windows it contains, how clocks were aligned, whether events were dropped, and whether collection completed successfully.

Keeping these together matters when a trace is copied from a compute node, combined after a run, or revisited during a later performance investigation. A visually plausible timeline is not enough to establish that its clocks and boundaries are meaningful.

## Artifact layout

For a single rank using native collection, the relevant layout is:

```text
output_dir/
  run_id/
    trace.json.gz
    rank-00000/
      manifest.json
      trace.json.gz
      native/
        capture-00000/
          ... native JAX artifacts ...
```

The root `trace.json.gz` is the combined trace. The rank-local `trace.json.gz` contains the normalised application, native, and telemetry events for that rank. `manifest.json` describes that capture and its clock mapping. Native captures retain their original JAX files beneath a separate directory for each collector window. A `basic` capture does not create those native artifacts.

[`ProfileResult`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileResult) exposes these locations as `trace_path`, `rank_trace`, `manifest_path`, and `run_dir`. Only rank zero receives a combined `trace_path` for an online distributed capture. Offline multi-process sessions retain local artifacts for later assembly.

A transient `manifest.pending.json` can appear during finalisation. It is not a substitute for the committed `manifest.json` when merging. Preserve the completed manifest and its adjacent rank trace as a pair if you move the files.

## Event representation and units

The portable output is a Chrome Trace Event JSON object with a `traceEvents` array and a `neuralqx` metadata object. Gzip compression is selected by the `.gz` filename suffix.

| Event form | Meaning |
| --- | --- |
| `ph="X"`, with `ts` and `dur` | A complete elapsed interval, such as a host span |
| `ph="i"` | An instantaneous application event |
| `ph="C"` | Numeric counters sampled at a timestamp |
| `ph="M"` | Lane and process metadata |
| Native flow and asynchronous records | Relationships supplied by the JAX runtime |

Timestamps `ts` and durations `dur` are expressed in **microseconds**. The top-level `displayTimeUnit="ms"` is a display preference and does not change those stored units. Host clock origins, calibration samples, and clock uncertainties in neuraLQX metadata use **nanoseconds**.

Application events originate from a monotonic integer clock. If the session origin is $o$ nanoseconds and an event occurs at $t$ nanoseconds, its local trace timestamp is

$$
u=\frac{t-o}{1000}.
$$

The origin remains part of the rank metadata so that a later merger can map local relative timestamps onto a common timeline. Wall-clock dates are not used as a substitute for this calibration.

## Aligning native and application events

The native JAX trace and the application's monotonic clock need an explicit relationship. The collector inserts uniquely named clock markers into native collection while measuring the corresponding host interval. Marker timing supplies an anchor and an uncertainty estimate derived from that host bracket.

If a native marker has timestamp $u_a$ in microseconds and the corresponding host anchor is $t_a$ nanoseconds, a native timestamp $u$ maps into the application's local trace through

$$
u_{\mathrm{local}}
=u-u_a+\frac{t_a-o}{1000}.
$$

The implementation looks for its own markers and selects an available anchor with the lowest measured uncertainty. This is a local offset alignment. The separate [distributed clock mapping](distributed.md#aligning-process-clocks) accounts for the relationship between processes.

If the required native anchor is absent, export reports a failure instead of silently combining unrelated clocks. This is why native host tracer level zero is not supported by the package's capture options.

The exporter clips complete native intervals to the retained recording window and marks affected records with `neuralqx_clipped_at_boundary`. Raw native files can still contain collector warmup or activity outside that retained interval. Those raw files and the normalised rank trace therefore answer slightly different inspection questions.

## Preserve relationships without inventing them

Native traces contain process identifiers, thread or stream identifiers, and flow identifiers. Their numerical values are local to the producer and can be reused when a collector restarts. The exporter scopes flow identifiers to their capture window. The merger then remaps process, lane, and flow identifiers across ranks without collisions.

This preserves supplied host-to-device relationships within their proper scope. It does not manufacture a cross-rank communication edge merely because two events occur at similar times. Process labels include the rank and hostname so that the combined view remains interpretable.

Complete spans can be written after their nested children, and different rank streams are merged without a global in-memory sort. Use timestamps and relationships rather than file position to determine ordering.

## Manifests and evidence of completeness

The rank manifest uses schema `neuralqx.profile.rank.v1`. It includes the run and process identity, origin and clock calibration, snapshotted options, recording statistics, window metadata, telemetry status, collection errors, and the type of any workload exception.

The combined trace uses schema `neuralqx.profile.trace.v1`. Its metadata embeds the contributing rank manifests and summarises the result.

| Metadata field | Interpretation |
| --- | --- |
| `world_size` | Expected number of processes |
| `missing_ranks` | Expected rank indices absent from the merge |
| `complete` | Every expected rank is present and reports successful completion |
| `data_loss` | At least one rank reports dropped application or telemetry events |
| `clock_origin_ns` | Common origin used for merged timestamps |
| `clock_uncertainty_ns` | Reported clock uncertainty for each rank, or unknown for a manual offset |
| `ranks` | Full contributing rank metadata |

`complete` and `data_loss` are independent. All ranks can finish successfully while a bounded recorder drops events. Conversely, a partial trace may retain all application events from the ranks that did finish.

These fields describe the captured evidence. They do not establish complete native backend coverage, and they do not imply that collection continued outside the requested windows. Inspect the options and individual windows alongside the summary flags.

## Merging completed artifacts

[`merge_traces`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.merge_traces) streams events from completed rank artifacts into a new JSON or compressed JSON file. The associated rank trace must remain beside each supplied manifest. The output is written through a temporary file and committed atomically, so an unfinished merge is not presented as the final file.

This runnable example re-merges a small single-rank capture into a separate path. A multi-rank merge uses the same call with the completed manifest from every rank in that run.

```python
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from neuralqx.profile import Profiler, event, merge_traces

with TemporaryDirectory() as directory:
    with Profiler(directory, level="basic") as profiler:
        event("analysis_complete", attributes={"configurations": 100})

    merged = merge_traces(
        [profiler.result.manifest_path],
        Path(directory) / "assembled.json.gz",
    )
    with gzip.open(merged, "rt") as stream:
        metadata = json.load(stream)["neuralqx"]

    print(metadata["world_size"])
    print(metadata["complete"])
    print(metadata["missing_ranks"])
```

```text
1
True
[]
```

The merger rejects duplicate ranks, incompatible run identifiers or world sizes, unsupported schemas, invalid durations, non-finite timestamps, and output paths that would replace an input rank artifact. Missing ranks are rejected by default. `allow_partial=True` permits an intentionally incomplete assembly and records which ranks are missing.

The command-line interface exposes the same operation through `python -m neuralqx.profile merge`, followed by manifest paths and `--output`. `--allow-partial` permits missing ranks. Repeated `--clock-offset RANK=NANOSECONDS` arguments provide externally measured clock offsets where needed. Use `python -m neuralqx.profile merge --help` for the complete invocation syntax.

Offline rank traces have no automatically established common clock. They require real externally measured offsets through `clock_offsets_ns` or the command-line equivalent. Partial merging does not remove that requirement, and arbitrary zero offsets would give an unjustified cross-process ordering.

Continue with [backends and telemetry](backends.md) to understand the measurements entering these artifacts, and [distributed profiling](distributed.md) for transport and clock calibration.
