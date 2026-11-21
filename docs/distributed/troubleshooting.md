---
title: Troubleshooting
---

# Troubleshooting

A useful first distinction is whether the problem occurs before the devices are available, while creating the arrays, or during a numerical step. The checks below follow that order. Use the [CPU](cpu.md), [single GPU](single-gpu.md), [multiple GPU](multi-gpu.md), and [multiple node](multi-node.md) launch commands as reference configurations before adding application-specific settings.

## Establish the environment actually running

Check the Python environment inside the allocation. A login shell, notebook, and batch job can select different interpreters or libraries. `python -m pip check` checks declared package dependencies. The following diagnostic reports the package and device configuration after neuraLQX's configured startup has taken place:

```python
import jax
import neuralqx as nqx

print("neuraLQX:", nqx.__version__)
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Process:", jax.process_index(), "of", jax.process_count())
print("Local devices:", jax.local_devices())
print("Global device count:", jax.device_count())
print("Sample sharding:", nqx.config.sample_sharding_enabled)
print("Operator sharding:", nqx.config.operator_sharding_enabled)
```

In a multi-process job, run this on every process with the intended distributed environment, just as you would run the simulation. Running it separately without joining the job only reports that separate process's devices. Keep the outputs when reporting an issue, together with the launch command and relevant configuration values.

## A GPU job reports a CPU backend

Install the GPU backend in the interpreter actually used by the job. The base package installation alone does not establish CUDA support. Follow [single GPU installation](single-gpu.md#installation-and-environment), then launch with `JAX_PLATFORMS=cuda` so a missing backend is reported as an error.

If `nvidia-smi` succeeds but JAX cannot initialise CUDA, check the selected JAX extra, driver compatibility, and CUDA library paths. A cluster module may expose different libraries from those installed by pip. The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) is the reference for that compatibility. neuraLQX's `backend_target` option does not select JAX's backend in this release.

## Fewer devices appear than were allocated

Compare the requested resources, scheduler task layout, CUDA visibility, and JAX local/global counts. With one process controlling four GPUs on one host, both local and global device counts should be four. With two coordinated processes controlling four GPUs each, each process should report four local devices and eight global devices.

If each process reports only its own devices as the global set, distributed initialization has not established the intended runtime. If a one-process-per-node Slurm job sees only one GPU locally, check `NEURALQX_DISTRIBUTED_LOCAL_DEVICE_IDS`. JAX's automatic GPU selection under Slurm can default to one device per process. The [explicit job examples](multi-node.md#a-two-node-slurm-job) avoid relying on that default.

On a CPU, one JAX device does not mean only one host core can execute numerical work. Set `JAX_NUM_CPU_DEVICES` before startup only when you intend to create several logical CPU devices. It is not a general CPU-thread tuning control.

## Initialization waits or times out

The initialization call needs every configured process to join. Check that the launcher started the stated number of processes and that each has a unique ID. In the two-node example there are two processes and eight GPUs, so `distributed_num_processes` must be two. Supplying eight leaves the runtime waiting for six nonexistent processes.

Next, check that all processes use the same coordinator address, that it belongs to process zero, and that its port is reachable from the other nodes. A per-host `localhost` address cannot join different hosts. Preserve output from every task, since one early import or CUDA failure may leave the remaining tasks waiting.

Initialize before calling `jax.devices()` or constructing any device arrays. In a notebook where those operations have already run, restart the kernel with the correct launch environment. A later change to `os.environ` does not undo backend initialization.

Cluster proxy settings can also interfere with runtime communication. JAX's [`initialize` reference](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html) describes this failure mode. Follow your site's guidance for internal traffic and proxy exclusions. Increase `NEURALQX_DISTRIBUTED_INITIALIZATION_TIMEOUT` only when the configuration is correct and startup is legitimately slow.

## Shape or mesh errors

Start with the default mesh sizes. The standard sampler and estimator planners choose phase-specific layouts, while generic mesh helpers require their shape product to match the device count. Manually setting one mesh axis to a fraction of the available devices can conflict with a sample-only helper. See [mesh names and sizes](configuration.md#mesh-names-and-sizes).

Inspect the resolved chain count and `state.samples.shape`. The sampler rounds chain counts upward to device divisibility, and the requested sample count is adjusted to a multiple of that resolved chain count. An input shape copied from a different device count may no longer describe the actual batch.

For externally supplied batches, a placement helper can leave an indivisible batch unchanged rather than shard it. Do not infer placement only from an enabled flag. Inspect `array.sharding` and each addressable shard's index and shape. Keep the Hilbert configuration axis complete unless the particular operator explicitly implements another layout.

## A job stalls during an expectation or save

Check control flow before increasing timeouts. Every process must participate in a collective expectation, gradient, or checkpoint operation. A process-zero condition around the whole operation can leave the remaining processes outside a required collective.

Also check callbacks that stop early. A decision based on process-local state must not cause one process to leave the optimisation loop while others continue. Use the coordinated semantics described in [callback control](../concepts/callbacks/control.md), and keep [distributed checkpoint calls](../concepts/vqs/saving.md) on every process.

A dense array conversion can be another source of trouble. A global array spanning hosts is not automatically available as a local NumPy array. Inspect `addressable_shards` for local data, or use an explicitly collective gather only when the complete result is actually needed.

## Out-of-memory errors

Identify which allocation grows before changing the execution policy. Model parameter memory, optimiser state, samples, operator metadata, connected configurations, and differentiation intermediates have different scaling.

Sample sharding reduces the number of chains stored on each device. It does not partition the network weights. Operator-only sharding may replicate the sample batch across devices. Enabling it can therefore increase sample storage per device even while reducing the operator work assigned to that device. Neural network TP and PP are not available in this release.

Use a smaller `chunk_size` for model-evaluation intermediates. Use operator streaming for connection-generation workspace. Neither promises that the complete connected output will fit, and neither changes the number of retained samples. If parameter memory itself is too large, reducing a sample chunk cannot fix it. The [VQS batching guide](../concepts/vqs/batching.md) explains these separate controls.

An error immediately after startup may instead involve GPU preallocation or several processes claiming the same device. Check GPU binding and JAX's [allocation settings](https://docs.jax.dev/en/latest/gpu_memory_allocation.html). Changing the allocator can help with sharing, but it cannot make an oversized live set of arrays fit.

## Measure completed work

JAX can return before device work is complete. A Python timer around an asynchronous call can therefore measure dispatch rather than numerical execution. Warm up the same operation first, then wait for the timed result. The [JAX asynchronous-dispatch guide](https://docs.jax.dev/en/latest/async_dispatch.html) explains this distinction.

For example, with [simulation.py](examples/simulation.py) in the working directory:

```python
from time import perf_counter

import jax
from simulation import build_calculation

state, objective, _ = build_calculation()
samples = state.samples
warmup = state.local_estimators(objective, samples=samples)
jax.block_until_ready(warmup.data)

start = perf_counter()
result = state.local_estimators(objective, samples=samples)
jax.block_until_ready(result.data)
elapsed = perf_counter() - start
if jax.process_index() == 0:
    print(f"Local-estimator evaluation: {elapsed:.6f} seconds")
```

This measures repeated local-estimator evaluation for a fixed sample batch. It excludes sampling, optimisation, and the first compilation, so it must not be reported as the time for a complete VMC step. For an end-to-end run, use the [profiling guides](../developers/profiling/for-scientists.md), and compare equivalent global workloads and precision settings. On multiple hosts, account for the slowest participating process.

## Different hardware gives different numerical results

Changing a device count can change the sampler's random stream partitioning. Floating-point summation also depends on reduction order, and different backends may select different kernels. A fixed seed is useful for reproducibility within a given setup, but it is not a guarantee of identical trajectories across device layouts.

To compare estimator implementations, reuse the same sample batch as in the [operator-sharding comparison](sharding.md#inspect-and-compare-operator-execution). To compare full Monte Carlo calculations, examine uncertainties, autocorrelation, and repeated runs with appropriate numerical tolerances. Record precision, global sample counts, resolved chains, package versions, and the topology.

If the discrepancy remains, [open an issue](https://github.com/waleed-sh/neuraLQX/issues) with a small reproducer and the diagnostics from this page. A complete launch command and the first failing operation are more useful than a long scheduler log without the configuration that produced it.
