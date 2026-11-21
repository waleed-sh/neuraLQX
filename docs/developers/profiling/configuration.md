---
title: Profiling configuration
---

# Profiling configuration

This page collects the controls introduced throughout the profiling guides. It covers profiling specifically, including automatic driver activation, explicit session settings, recording windows, native collection, telemetry, and distributed transport. Begin with [profiling for scientists](for-scientists.md) if you have not yet made a capture.

The [complete configuration reference](../../start/configuration.md#profiling-activation-and-output) lists every package option with its Python assignment, environment binding, default, and accepted values.

## Package options and explicit sessions

The package option `profile`, defaulting to `False`, enables automatic capture for bounded driver `run` calls. Every field of [`ProfileOptions`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileOptions) also has a package option named `profile_<field>`. For example, `detail` corresponds to `nqx.config.profile_detail`.

An explicit [`Profiler`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler) with no `options=` argument snapshots the effective package configuration when it starts, not when it is constructed. This includes thread-local overrides active on entry. Later configuration changes affect later sessions, rather than changing an existing recording.

Constructing `ProfileOptions(...)` directly uses its dataclass defaults for unspecified fields. It does not inherit the current package configuration. Use `ProfileOptions.from_config()` when you want a validated immutable snapshot of those effective values.

```python
import neuralqx as nqx
from neuralqx.profile import ProfileOptions

before = nqx.config.profile_detail
with nqx.config.patch(profile_level="trace", profile_detail="detailed"):
    snapshot = ProfileOptions.from_config()
    independent = ProfileOptions()
    print(snapshot.level, snapshot.detail)
    print(independent.level, independent.detail)

print(nqx.config.profile_detail == before)
```

```text
trace detailed
basic summary
True
```

`nqx.config.patch(...)` restores the previous settings on exit. Use `thread_local=True` when the override should apply only to the current thread's configuration view. This does not create a separate profiler per thread. A session still owns the process-wide application recording described in [profiler internals](internals.md).

Explicit `output_dir`, `level`, and `run_id` arguments to `Profiler` override the corresponding fields of its configuration snapshot or supplied `options`. An explicit `schedule=` overrides the schedule derived from option fields. Record that explicit schedule with your experiment description, since the manifest's `options` object records the option fields rather than rewriting them from a separately supplied `Schedule`.

All profiling package options are runtime settings excluded from the package's compilation fingerprint. Changing the observation policy does not itself change that fingerprint. Instrumentation overhead and explicit synchronisation can still change execution timing.

## Environment defaults

The activation option reads the environment default `NEURALQX_PROFILE`. A field named `field` reads `NEURALQX_PROFILE_FIELD`, with the field name uppercased. Thus `profile_level`, `profile_max_events`, and `profile_telemetry_interval_s` correspond to `NEURALQX_PROFILE_LEVEL`, `NEURALQX_PROFILE_MAX_EVENTS`, and `NEURALQX_PROFILE_TELEMETRY_INTERVAL_S`.

Set these variables before importing neuraLQX. Environment defaults are read when the configuration manager is initialised, and profiling does not declare forced environment overrides. Explicit global settings take precedence over those defaults, while an active thread-local override takes precedence over the global setting.

Environment defaults affect sessions that read package configuration. A directly constructed `ProfileOptions()` continues to use its own declared defaults. This distinction is useful when writing a reusable script that should either follow the surrounding job configuration or establish an independent capture policy.

## Activation, detail, and output

Except for `profile` itself, the first column below gives the `ProfileOptions` field name. Add the `profile_` prefix when setting the corresponding package option.

| Field or activation option | Default | Accepted values and effect |
| --- | --- | --- |
| `profile` | `False` | Package-only boolean enabling automatic driver capture |
| `level` | `"basic"` | `"off"`, `"basic"`, `"trace"`, or `"full"` |
| `detail` | `"summary"` | `"summary"`, `"detailed"`, or `"developer"` for package annotation depth |
| `output_dir` | `".neuralqx/profiles"` | Nonempty string naming the artifact root |
| `run_id` | `None` | Optional unique directory component shared by the participating ranks |
| `synchronize` | `"boundary"` | `"boundary"` waits for supplied results at capture boundaries, while `"none"` introduces no such waits |

`profile=False` disables automatic capture but does not disable an explicit profiler. Conversely, automatic `profile=True` with `level="off"` uses `basic` capture. To disable automatic driver profiling, change the activation option. An explicit `Profiler(level="off")` creates no artifacts.

A run identifier is at most 128 characters, begins with a letter or digit, and then contains only letters, digits, dots, underscores, or hyphens. It is one directory component. Single-process sessions generate an identifier when none is supplied. Multi-process sessions require the same explicit identifier on every rank, and an existing rank directory is never reused.

`Profiler(output_dir=...)` also accepts a `pathlib.Path` override. In a directly constructed `ProfileOptions`, `output_dir` is a string.

## Recording schedules

| Field | Default | Meaning |
| --- | --- | --- |
| `wait_steps` | `0` | Unprofiled steps at the start of each cycle |
| `warmup_steps` | `0` | Collector warmup steps excluded from the retained recording |
| `active_steps` | `0` | Recorded steps per cycle, with zero selecting continuous capture |
| `repeat` | `1` | Number of cycles, with zero repeating until context exit |
| `skip_first` | `0` | Unprofiled steps before the first cycle |

These counts are nonnegative integers. A scheduled capture requires positive `active_steps`. Nonzero waiting, warmup, or initial skip counts with `active_steps=0` are rejected. Without a schedule, `repeat` has no cycling effect.

The separately constructed [`Schedule`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.Schedule) uses argument names `wait`, `warmup`, and `active`. Its defaults are `wait=0`, `warmup=1`, `active=5`, `repeat=1`, and `skip_first=0`. Unlike the option fields, `Schedule.active` must always be positive.

This example converts an option-based policy into its schedule and inspects the first steps.

```python
from neuralqx.profile import ProfileOptions

options = ProfileOptions(wait_steps=2, warmup_steps=1, active_steps=3)
schedule = options.make_schedule()
print([schedule.action(step).value for step in range(7)])
print(ProfileOptions().make_schedule() is None)
```

```text
['wait', 'wait', 'warmup', 'record', 'record', 'record', 'wait']
True
```

The driver advances once per attempted iteration. Custom loops must call `step()` themselves. See [choosing a recording window](scientists/capture-options.md#record-a-short-window) for the interpretation of warmup and rejected updates.

## Native collection and external ranges

| Field | Default | Accepted values and effect |
| --- | --- | --- |
| `jax_host_tracer_level` | `1` | Integer `1`, `2`, or `3` controlling native host detail |
| `jax_python_tracer_level` | `None` | `None`, `0`, or `1`, with `None` choosing one for `full` and zero otherwise |
| `nvtx` | `False` | Boolean enabling ranges for an external NVIDIA profiler |

The JAX tracing settings matter when `level` requests native collection. Host level zero is invalid because native clock alignment requires host markers. NVTX requires the optional package and does not launch an external profiler. See [backends and telemetry](backends.md) for installation and the distinction between these sources.

## Recording budgets

| Field | Default | Scope |
| --- | --- | --- |
| `max_events` | `100_000` | Positive integer budget for application and telemetry event reservations per process |
| `max_lanes` | `1024` | Positive integer limit for application thread and asynchronous task lanes |
| `max_trace_bytes` | `4_294_967_296` | Positive integer maximum compressed rank trace size accepted by TCP transfer |

The event budget is shared across recording windows, and reservation granularity can leave fewer retained events than the nominal limit. Dropped-event and lane diagnostics are retained in the artifacts. These limits do not bound JAX native collection buffers. Shorter windows are the primary way to bound the duration of native collection.

The TCP byte limit is a transfer limit, not an event count or a maximum combined-trace size. It applies to the compressed file from one rank.

## GPU telemetry

| Field | Default | Accepted values and effect |
| --- | --- | --- |
| `telemetry` | `"off"` | `"off"`, `"auto"`, `"nvml"`, or `"nvidia-smi"` |
| `telemetry_interval_s` | `1.0` | Positive finite delay after each management query |
| `telemetry_scope` | `"host"` | `"host"` elects one sampler per hostname, while `"rank"` samples independently |
| `telemetry_devices` | `()` | Tuple of management GPU UUIDs or index strings, with an empty tuple selecting all management-visible devices |
| `nvidia_smi_timeout_s` | `2.0` | Positive finite timeout for a read-only `nvidia-smi` query |

For direct options, a device selection can be `telemetry_devices=("0", "1")`. Identifiers must be nonempty strings of at most 256 characters. The package configuration parser also accepts a comma-separated selection such as `"0,1"`, which is useful for `NEURALQX_PROFILE_TELEMETRY_DEVICES`.

Device selection is independent of JAX logical device numbering and is not automatically filtered by `CUDA_VISIBLE_DEVICES`. An explicit provider must work, while `auto` preserves provider failures in metadata and permits unavailable telemetry. Offline multi-rank sessions with enabled telemetry require rank scope. The [telemetry discussion](backends.md#sampling-time-and-resolution) explains query timing and physical interpretation.

## Distributed coordination

| Field | Default | Accepted values and effect |
| --- | --- | --- |
| `transport` | `"shared"` | `"shared"`, `"tcp"`, or `"offline"` |
| `coordinator_address` | `None` | Reachable rank-zero `host:port` for TCP coordination |
| `timeout_s` | `60.0` | Positive finite deadline for each coordination or transfer phase |
| `clock_samples` | `5` | Positive integer number of round trips at each clock-calibration boundary |

Shared transport needs common writable output storage. TCP supports rank-local storage and requires its own reachable coordinator address for a multi-process capture. Offline mode performs no clock exchange. Rank identity normally comes from the initialised JAX runtime, with explicit `rank` and `world_size` overrides available together for other workers.

These settings configure profiling, not the numerical runtime. Read [distributed profiling](distributed.md) for the transport and clock model, and [JAX parallelism](../../distributed/jax-parallelism.md) for the separate execution setup.

## Validate the policy before a longer capture

Constructing `ProfileOptions` validates choices, finite positive intervals, integer counts, device identifiers, and the relationships between schedule fields. Direct integer fields reject booleans. Session entry then validates conditions that depend on the runtime, such as rank identity, distributed run identifiers, existing output directories, and collector availability.

Use a short representative capture before committing to a long recording interval. Inspect its manifest, retained windows, and dropped-event diagnostics, then adjust only the controls relevant to the question. The [scientist reading guide](scientists/reading-results.md) provides the route from those artifacts back to the calculation.
