---
title: Backends and telemetry
---

# Backends and telemetry

neuraLQX combines measurements from different sources because no single source describes the whole calculation. The application recorder supplies scientific and algorithmic context. Native JAX collection supplies runtime activity. NVTX can expose application ranges to an external NVIDIA tool. GPU telemetry samples device management counters independently of individual operations.

These sources have different meanings and requirements. Increasing the profiling level does not automatically enable all of them, and the absence of one source does not invalidate observations made by another.

## Native JAX collection

The `trace` and `full` levels use JAX's public profiling interface to start and stop a native collector for each capture window. The default host tracer level is one. Levels two and three request additional native host detail. Python call tracing defaults to zero for `trace` and one for `full`, with an explicit option to override that choice.

CPU activity is useful even without an accelerator. Device activity depends on the installed JAX runtime, backend, and hardware support. Follow the [JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html) for backend-specific collection requirements, and the [single-GPU guide](../../distributed/single-gpu.md) for the package's hardware setup route.

The collector retains JAX's original artifacts, including native trace files and XPlane files when produced, under each rank's `native/capture-...` directory. It also extracts the events used in the [portable combined trace](traces.md). The package does not request an automatic Perfetto sharing link or open a browser as part of capture.

neuraLQX owns only the collector it successfully starts. If another native JAX profiler is already active, starting a conflicting collection fails. Cleanup does not stop that unrelated collector. Avoid overlapping explicit `jax.profiler` captures with a neuraLQX native session.

The host tracer must retain the markers needed for [native clock alignment](traces.md#aligning-native-and-application-events). Host tracer level zero is therefore rejected rather than producing a trace whose application and native timestamps cannot be related.

## NVTX ranges for external tools

`nvtx=True` emits application ranges through the optional NVTX Python package. The ranges use the `neuralqx` domain and process range handles, allowing their lifetimes to overlap across threads or asynchronous tasks.

NVTX is an annotation mechanism. Enabling it does not launch Nsight, start an external system trace, or create an Nsight report. An external NVIDIA profiling tool must be recording the application to collect those ranges. The native JAX collector and NVTX can be used together, but they remain separate observation mechanisms.

The optional NVIDIA profiling dependencies can be installed with the package's profiling extra:

```sh
pip install 'neuralqx[profile-nvidia]'
```

This extra provides the NVTX package and NVIDIA management bindings. It does not install a GPU driver, configure JAX for CUDA, or install Nsight. Use the hardware-specific [simulation guides](../../distributed/index.md) for execution setup before enabling these additional measurements.

## Selecting a telemetry provider

Telemetry is disabled by default. The `telemetry` option chooses how GPU management data is obtained.

| Setting | Behaviour |
| --- | --- |
| `off` | No management queries |
| `auto` | Try NVML, then `nvidia-smi`, recording why unavailable providers failed |
| `nvml` | Require the NVIDIA management library and Python bindings to initialise |
| `nvidia-smi` | Require the command-line tool to answer read-only CSV queries |

An explicit provider must work. Initialisation failure raises a profiling error. Later query or shutdown failures are retained in metadata and reported during finalisation for an explicit provider. With `auto`, unavailable telemetry is recorded without making the application capture fail solely for that reason.

Automatic selection happens when the collector window starts. It is not a continuous failover mechanism. If a selected provider later fails during sampling, that failure is recorded rather than silently replacing the provider halfway through the window.

This example requests optional telemetry around a small application capture. It runs on a CPU-only machine as well, since `auto` records provider unavailability. The selected provider and any failures can be inspected in the manifest.

```python
import json
from tempfile import TemporaryDirectory

from neuralqx.profile import ProfileOptions, Profiler, event

options = ProfileOptions(level="basic", telemetry="auto")
with TemporaryDirectory() as directory:
    with Profiler(directory, options=options) as profiler:
        event("analysis_started")

    manifest = json.loads(profiler.result.manifest_path.read_text())
    telemetry = manifest["telemetry"][0]
    print(telemetry["requested"])
    print(telemetry["provider"] in (None, "nvml", "nvidia-smi"))
    print(isinstance(telemetry["failures"], list))
```

```text
auto
True
True
```

This short example demonstrates configuration and metadata, not a useful utilisation measurement. A real sampling window must be long enough to contain observations at the chosen interval.

## What management counters describe

Available values depend on the device and provider. Unsupported or inaccessible quantities are omitted instead of being reported as artificial zeros.

| Trace counter | Unit or interpretation |
| --- | --- |
| `utilization_gpu_percent` | Management-reported GPU utilisation |
| `utilization_memory_percent` | Management-reported memory activity, not the fraction of memory allocated |
| `memory_used_bytes`, `memory_total_bytes` | Device memory usage and capacity in bytes |
| `temperature_celsius` | Device temperature in degrees Celsius |
| `power_watts` | Device power in watts |
| `clock_sm_mhz`, `clock_memory_mhz` | Reported clock frequencies in MHz |
| `query_duration_ns` | Time spent obtaining the management sample |

The `nvidia-smi` provider converts reported MiB memory values into bytes. Provider measurements can have their own averaging intervals, so a timestamp in the trace does not imply an instantaneous physical measurement of every quantity.

In particular, management memory usage is not an allocation profile of a single variational state. It can include other processes and runtime reservations. GPU utilisation is not the occupancy of one kernel, and memory utilisation is not memory capacity consumed. Use native execution information and the scientific workload to interpret these counters together.

## Sampling time and resolution

A background thread performs the queries. If a query starts at $t_a$ and returns at $t_b$, the sample is placed at

$$
t_{\mathrm{sample}}=\frac{t_a+t_b}{2},
\qquad
\Delta t_{\mathrm{query}}=t_b-t_a.
$$

The query duration is retained as a diagnostic. The configured `telemetry_interval_s` is the delay after a query finishes. The effective interval between query starts therefore includes both that delay and the query time. It is not an exact periodic clock and is not tied to driver steps.

Telemetry from collector warmup is not retained in the normalised recording. Samples that finish after collection has been asked to stop can also be omitted. A very short window may therefore contain no retained samples even when provider initialisation succeeded.

Reducing the interval can reveal shorter changes, but it also increases management traffic and trace volume. It cannot resolve kernel-scale behaviour when the provider itself reports averaged quantities. Choose the interval for the timescale of interest, such as sustained memory growth or temperature changes during a longer calculation.

## Device selection and ownership

`telemetry_devices` accepts management device UUIDs or indices as strings. An empty selection means all devices visible to the management provider. This is independent of JAX's logical device numbering and is not automatically restricted by `CUDA_VISIBLE_DEVICES`. Explicit UUID selection is useful when the job's allocation and the management view differ.

With `telemetry_scope="host"`, online distributed coordination elects the lowest rank on each hostname to sample management data. That avoids multiple ranks recording the same host devices independently. With `telemetry_scope="rank"`, each rank samples its own management view, which is useful for isolated containers but can produce duplicate observations of the same physical GPU.

Offline multi-rank capture cannot perform host election, so enabled telemetry must use rank scope there. Device UUIDs are retained in the trace lanes to distinguish the measured devices. Continue with [distributed profiling](distributed.md) for the coordination model and the final [configuration reference](configuration.md) for all related controls.
