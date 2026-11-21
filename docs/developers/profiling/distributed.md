---
title: Distributed profiling
---

# Distributed profiling

A distributed simulation has several timelines. Each process observes its own host work, may submit work to several devices, and reads a monotonic clock whose numerical origin need not match another process's clock. A useful combined profile must preserve rank identity and establish a defensible relationship between those clocks.

neuraLQX separates profiling coordination from the numerical computation. The profiler does not initialise JAX distributed execution, add JAX or MPI collectives to annotations, or decide how the variational state is sharded. Configure the simulation first through the [distributed simulation guides](../../distributed/multi-node.md), then let profiling observe that execution.

## Establish the process identity

By default, [`Profiler`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler) obtains the process index and process count from the initialised JAX runtime. Every participating process must enter a session with the same explicit `run_id` and the actual world size. Use the same intended capture schedule across the job so that the retained windows describe comparable phases.

For non-JAX worker processes, `rank` and `world_size` can be supplied explicitly. They must be supplied together, with `0 <= rank < world_size`. These overrides identify profiling participants. They do not create a JAX distributed runtime or distribute arrays.

A multi-process capture requires an explicit shared run identifier because independently generated identifiers would describe unrelated sessions. Choose a new identifier for each experiment, such as one derived from the job identifier and capture number. Existing rank directories are not reused.

Only rank zero writes the combined trace for an online capture. Every rank retains its own completed trace and manifest. Retrieve the combined path from rank zero's `ProfileResult`, and preserve local artifacts when diagnosing a failed collection.

## Choose a collection transport

| Transport | Storage and connectivity | Result |
| --- | --- | --- |
| `shared` | All ranks can read and write the same output directory | Rank zero reads completed rank artifacts and merges them |
| `tcp` | Workers can reach a profiling coordinator on rank zero | Rank-local captures are transferred and merged on rank zero |
| `offline` | Each rank can write local artifacts | No cross-rank coordination or automatic multi-rank merge |

The transport handles rank registration, boundary clock exchanges, and completed artifact collection. Individual spans never send network messages or perform filesystem coordination. Capture startup and finalisation can still wait for other ranks, so include those phases when planning the job's lifetime.

### Shared storage

With `transport="shared"`, all ranks must see the same writable `output_dir/run_id`. Registration and clock exchange use files in that shared location. Rank zero waits for completed rank manifests before assembling the result.

This is convenient on a cluster with suitable shared storage. Filesystem latency also enters the clock exchange, so a slow shared filesystem can increase the uncertainty of cross-rank alignment. It does not directly add that latency to each annotated operation.

The following complete example launches two ordinary Python workers and collects their application events through shared storage. It demonstrates the profiling transport without claiming to initialise a multi-device numerical calculation.

```python
import gzip
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

worker = """
import sys
from neuralqx.profile import ProfileOptions, Profiler, span

rank = int(sys.argv[1])
options = ProfileOptions(level="basic", transport="shared", timeout_s=20)
with Profiler(
    sys.argv[2],
    options=options,
    run_id="shared-example",
    rank=rank,
    world_size=2,
):
    with span("local_analysis", attributes={"rank": rank}):
        total = sum(range(1000))
"""

with TemporaryDirectory() as directory:
    processes = []
    try:
        for rank in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, "-c", worker, str(rank), directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ))
        for process in processes:
            stdout, stderr = process.communicate(timeout=45)
            if process.returncode != 0:
                raise RuntimeError(stdout + stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

    trace_path = Path(directory) / "shared-example" / "trace.json.gz"
    with gzip.open(trace_path, "rt") as stream:
        metadata = json.load(stream)["neuralqx"]
    print(metadata["world_size"])
    print(metadata["complete"])
    print(sorted(metadata["clock_uncertainty_ns"]))
```

```text
2
True
['0', '1']
```

For an initialised distributed JAX job, omit the explicit rank overrides and use the job's real process identity. The profiling mechanism is the same, while array placement and numerical communication remain the responsibility of JAX and the simulation.

### TCP collection

With `transport="tcp"`, `coordinator_address` identifies a reachable `host:port` on rank zero. This is the profiling coordinator address, separate from any address used to initialise JAX distributed execution. Rank-local output directories can differ across nodes.

Workers register with the coordinator, exchange clock samples, and transfer their completed normalised trace and manifest. Original native JAX artifacts remain on the worker's storage. Copy them separately if a later investigation needs the raw XPlane or native trace files.

`max_trace_bytes` limits the compressed rank trace accepted by TCP transfer. It does not limit JAX's collection buffers or the final combined file. `timeout_s` applies to individual coordination and transfer phases, rather than imposing a time limit on the whole simulation.

The profiling service is an unauthenticated HTTP service intended for a trusted job network. A run identifier distinguishes captures but is not an authentication credential. Choose an address reachable by the participating ranks within that environment.

### Offline artifacts

With `transport="offline"`, each process writes its own artifacts without cross-rank communication. There is no automatic clock calibration and no combined trace for a multi-process session. This can be useful when online collection is unavailable or when artifacts are gathered after the job.

Later, copy each completed `manifest.json` with its adjacent `trace.json.gz` and use [`merge_traces`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.merge_traces). Every unmapped rank needs an externally measured integer clock offset in nanoseconds. The offset is added to that rank's local monotonic time to obtain the chosen common time.

Do not assume that wall-clock synchronisation makes monotonic origins equal, or provide zero merely to make a merge succeed. A manual offset is recorded with unknown uncertainty. The resulting trace should be interpreted with the external calibration method in mind.

Enabled offline telemetry must use `telemetry_scope="rank"` in a multi-process job, because host-level sampling ownership cannot be elected without coordination.

## Aligning process clocks

Online transports measure clocks through round trips at session entry and finalisation. Consider four timestamps, all expressed in nanoseconds. A worker sends at its local time $t_0$, rank zero receives at its time $t_1$, rank zero replies at $t_2$, and the worker receives at its local time $t_3$.

After subtracting time spent serving the request, the estimated transport delay is

$$
d=(t_3-t_0)-(t_2-t_1).
$$

The estimated offset to add to the worker clock is

$$
\widehat o=\frac{(t_1-t_0)+(t_2-t_3)}{2},
\qquad
\widehat t=\frac{t_0+t_3}{2}.
$$

The sample is associated with worker midpoint $\widehat t$. Its uncertainty is half the transport delay, rounded upward to integer nanoseconds. This accounts for unknown delay asymmetry under the round-trip model. Invalid negative timing intervals are rejected.

The profiler takes `clock_samples` exchanges at each boundary and selects the sample with the smallest uncertainty. This favours the exchange least affected by incidental transport delay without pretending that its remaining uncertainty is zero.

### Resolving drift

Let the selected entry and exit samples have worker times $a$ and $b$, offsets $o_a$ and $o_b$, and uncertainties $u_a$ and $u_b$. A change in clock offset is treated as resolved only when

$$
|o_b-o_a|>u_a+u_b.
$$

When it is resolved and $b>a$, the fitted slope is

$$
s=\frac{o_b-o_a}{b-a},
\qquad
t_{\mathrm{root}}=t+o_a+s(t-a).
$$

Otherwise, the implementation uses the entry offset with $s=0$, avoiding a drift fit dominated by transport jitter. Event durations are scaled by $1+s$ under the affine mapping. A fitted slope with magnitude greater than one percent is rejected as an unreliable alignment.

The reported uncertainty uses the larger endpoint uncertainty and also includes unresolved offset change when a constant mapping is used. Nonlinear drift between the endpoints is not measured. The merged trace therefore carries calibration evidence, rather than claiming exact ordering at arbitrary resolution.

## Interpret rank differences within that resolution

A long gap on one rank may indicate load imbalance, a wait for another rank, or work outside the retained capture. Use native communication and dispatch relationships where available, together with the application phase names. Similar timestamps alone do not establish a causal relationship.

If the apparent ordering difference is comparable to the clock uncertainty, the trace does not resolve which rank acted first. Larger persistent gaps across representative iterations are more useful evidence for investigating imbalance. The [reading guide](scientists/reading-results.md) explains why durations across overlapping ranks should not simply be added.

If a rank never registers or never supplies its completed artifacts, online coordination eventually reaches its configured deadline. Inspect available local artifacts and the original failure. An intentional partial merge is possible with `allow_partial=True`, but it remains incomplete and still requires valid clock mappings for every included rank.

The [configuration reference](configuration.md) collects the transport, timeout, sampling, and storage options after these mechanisms have been introduced.
