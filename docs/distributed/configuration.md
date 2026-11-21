---
title: Parallelism configuration
---

# Parallelism configuration

There are three separate choices in a parallel launch. JAX selects and initialises a numerical backend, the launcher starts the Python processes, and neuraLQX chooses how supported operations use the visible devices. Keeping those choices separate makes it easier to interpret a configuration and diagnose a run.

This page covers settings relevant to that execution path. It builds on [JAX parallelism](jax-parallelism.md), [neuraLQX's sharding policies](sharding.md), and the [hardware examples](cpu.md). The [complete configuration reference](../start/configuration.md) lists every package setting, including its Python and environment forms. Profiling options also have a [dedicated guide](../developers/profiling/configuration.md).

## Set topology before starting the calculation

Environment variables are the most direct way to establish a reproducible launch. Set them before importing neuraLQX or creating JAX arrays. For example:

```bash
NEURALQX_SHARDING=samples,operators \
JAX_PLATFORMS=cuda python simulation.py
```

neuraLQX reads the environment when its configuration manager is created during import. Changing `os.environ` later does not update an existing manager. Mesh and topology options are classified as startup settings. Once `config.lock_runtime()` has been called they reject further changes, but even before that lock, changing a value does not rebuild existing states or invalidate every compiled computation. Use a fresh process for a different device topology or global sharding policy.

The public [`config`](../api/public/neuralqx/config/index.md) object supports attribute access, `get`, `set`, and metadata inspection. This complete example does not start a distributed run:

```python
import neuralqx as nqx

for name in (
    "distributed",
    "sharding_axes",
    "sample_sharding_enabled",
    "parameter_sharding_enabled",
    "operator_sharding_enabled",
):
    metadata = nqx.config.describe_option(name)
    print(name, metadata["effective_value"], metadata["env_default"])
```

`nqx.config.show(include_env=True)` prints the full package configuration. `describe_option` also reports the declared default and mutability. For options with a `NEURALQX_FORCE_*` environment binding, a forced value takes precedence and prevents a conflicting Python update. Not every option has such a binding.

## Sharding policy

| Python option | Environment variable | Default and effect |
| --- | --- | --- |
| `sharding_axes` | `NEURALQX_SHARDING` | `samples`, used to initialise the individual flags |
| `sample_sharding_enabled` | `NEURALQX_SAMPLE_SHARDING_ENABLED` | Enabled by the default policy, distributes supported sample and chain batches |
| `operator_sharding_enabled` | `NEURALQX_OPERATOR_SHARDING_ENABLED` | Disabled by the default policy, enables independent term dispatch |
| `parameter_sharding_enabled` | `NEURALQX_PARAMETER_SHARDING_ENABLED` | Disabled by the default policy, currently selects replicated parameter placement |

`NEURALQX_SHARDING_AXES` is an alias for the high-level environment setting. If both it and `NEURALQX_SHARDING` are present, `NEURALQX_SHARDING` is read first. The high-level policy accepts comma-separated axes, and individual environment flags can override the resulting defaults.

| High-level value | Resulting default flags |
| --- | --- |
| `samples` | Sample sharding only |
| `samples,operators` or `operators` | Sample and operator sharding |
| `samples,parameters` or `parameters` | Sample sharding and replicated parameter placement |
| `all` | All three flags enabled |
| `none` | All three flags disabled |

An operator-only or parameter-only high-level value adds to the default sample policy. To disable sample sharding while retaining operator sharding, explicitly set `NEURALQX_SAMPLE_SHARDING_ENABLED=0` and `NEURALQX_OPERATOR_SHARDING_ENABLED=1`. In that case the standard operator planner cannot choose the combined sample/operator layout. Turning off sharding also does not force CPU execution or hide extra devices.

**The high-level option is resolved into individual defaults during configuration construction.** Assigning `nqx.config.sharding_axes = ("samples", "operators")` after import does not update the three enabled flags. For a Python configuration performed before creating a state, update the flags directly:

```python
import neuralqx as nqx

nqx.config.set_many({
    "sample_sharding_enabled": True,
    "operator_sharding_enabled": True,
    "parameter_sharding_enabled": False,
})
assert nqx.config.operator_sharding_enabled
```

Prefer the environment-based launch for batch jobs. All processes must use compatible values. Neither `parameters` nor `all` enables neural network TP or PP. Those strategies are under development and are not part of this release, as explained in [parameter placement](sharding.md#parameters-and-neural-network-parallelism).

## Distributed runtime

These settings establish cooperation between Python processes. A single process using several GPUs does not require them.

| Python option | Environment variable | Default |
| --- | --- | --- |
| `distributed` | `NEURALQX_DISTRIBUTED` | `False` |
| `distributed_auto_initialize` | `NEURALQX_DISTRIBUTED_AUTO_INITIALIZE` | `True` |
| `distributed_coordinator_address` | `NEURALQX_DISTRIBUTED_COORDINATOR_ADDRESS` | `None` |
| `distributed_num_processes` | `NEURALQX_DISTRIBUTED_NUM_PROCESSES` | `None` |
| `distributed_process_id` | `NEURALQX_DISTRIBUTED_PROCESS_ID` | `None` |
| `distributed_local_device_ids` | `NEURALQX_DISTRIBUTED_LOCAL_DEVICE_IDS` | `None` |
| `distributed_initialization_timeout` | `NEURALQX_DISTRIBUTED_INITIALIZATION_TIMEOUT` | `300` seconds |

With `distributed=True` and `distributed_auto_initialize=True`, package startup calls `jax.distributed.initialize()` unless JAX is already initialized. This does not launch any additional processes. Slurm, another scheduler, or a manual launcher must start every process.

The coordinator is the reachable `host:port` of process zero. The process count is the number of Python interpreters, and process IDs must be unique and contiguous from zero. The configuration requires coordinator address, process count, and process ID together when any of those three is supplied explicitly. Local device IDs are an optional comma-separated list of ordinals visible inside that process, such as `0,1,2,3`.

When the explicit arguments are omitted, JAX attempts supported environment detection. Its GPU defaults under Slurm can select one local device per process. The [two-node job](multi-node.md#a-two-node-slurm-job) therefore provides the IDs explicitly. Increase the timeout only after checking that the topology and launch are correct. Waiting longer cannot repair a missing process or an unreachable coordinator.

`NEURALQX_DISTRIBUTED_ENABLED` aliases `NEURALQX_DISTRIBUTED`. Forced equivalents are `NEURALQX_FORCE_DISTRIBUTED` and `NEURALQX_FORCE_DISTRIBUTED_ENABLED`. Use these only when an environment intentionally needs to prevent the application from changing that flag.

## Mesh names and sizes

| Logical axis | Name option and default | Size option and default |
| --- | --- | --- |
| Samples | `mesh_sample_axis_name="samples"` | `mesh_sample_axis_size=None` |
| Parameters | `mesh_parameter_axis_name="parameters"` | `mesh_parameter_axis_size=None` |
| Operators | `mesh_operator_axis_name="operators"` | `mesh_operator_axis_size=None` |

Each option has the environment name formed by adding `NEURALQX_` and converting it to uppercase, for example `NEURALQX_MESH_OPERATOR_AXIS_SIZE`. Names must be nonempty and distinct. Size values, when provided, must be positive integers.

The generic [`create_device_mesh`](../api/public/neuralqx/jax/mesh/index.md#neuralqx.jax.mesh.create_device_mesh) helper consumes all supplied devices. Explicit axis sizes must be compatible with that count, and unspecified sizes are inferred. If the operator axis is one of several unspecified axes, the generic helper first assigns it a factor near the square root of the remaining device count, then assigns the remainder to samples when possible. A fully specified mesh must have a size product equal to the available device count.

The **execution planners do not simply reuse this generic mesh for every phase**. Sampling and gradient-force planning request sample partitions over the device set. Operator planning derives term and sample partition counts from the workload, overriding generic axis-size choices for that phase. Some placement helpers also construct their own sample-only meshes. Leave all three size options at `None` for the standard guides. Setting `mesh_sample_axis_size=2` on an eight-device allocation is not a supported way to limit the standard sampler to two GPUs. Select the visible devices before startup instead.

A custom mesh passed directly to a lower-level JAX helper is an advanced interface choice. It should be tested with the consumers that use it, rather than inferred from a configuration name.

## Model chunks and operator streaming

| Control | Default | Meaning |
| --- | --- | --- |
| `VariationalState(..., chunk_size=...)` | `None` | Model evaluation batch limit |
| `operator_streaming_enabled` | `False` | Enables chunked connection generation where supported |
| `operator_streaming_chunk_size` | `1024` | Flattened input configurations per operator-generation chunk |

The operator environment names are `NEURALQX_OPERATOR_STREAMING_ENABLED` and `NEURALQX_OPERATOR_STREAMING_CHUNK_SIZE`. These two options are runtime-mutable. Explicit estimator arguments `operator_streaming` and `operator_streaming_chunk_size` take precedence for that call. Model chunk size belongs to the variational state or the individual evaluation, not to a global `NEURALQX_CHUNK_SIZE` variable.

Streaming and chunking do not change the number of samples or divide model weights. The [sharding guide](sharding.md#chunking-and-streaming) explains which allocations they affect. Choose positive chunk sizes and test them with the actual array layout and operator.

## Backend selection and JAX settings

`JAX_PLATFORMS=cpu` and `JAX_PLATFORMS=cuda` are JAX settings used by the launch commands in these guides. `CUDA_VISIBLE_DEVICES` controls CUDA visibility before runtime initialization. `JAX_NUM_CPU_DEVICES` creates a chosen number of logical CPU devices for inspection and testing. These are not neuraLQX configuration options.

The neuraLQX option `backend_target`, defaulting to `auto`, is currently stored configuration metadata. It does not update JAX's backend selection in this release. Use `JAX_PLATFORMS` to select the backend. Similarly, the registered `jit_enabled`, `compile_cache_enabled`, and `compile_cache_dir` options do not currently configure JAX's compilation controls. Use JAX's own settings for those behaviours rather than relying on these names.

For persistent compilation caching, set `JAX_COMPILATION_CACHE_DIR` to an appropriate cache location before execution and follow the [JAX cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html). Cache reuse depends on the computation and execution environment. It does not guarantee that a changed shape or device topology avoids compilation.

## Precision and memory

neuraLQX defaults to `enable_x64=True`, `dtype_real="float64"`, and `dtype_complex="complex128"`. Their environment names are `NEURALQX_ENABLE_X64`, `NEURALQX_DTYPE_REAL`, and `NEURALQX_DTYPE_COMPLEX`. The first has a hook that updates JAX's X64 setting. The dtype settings are defaults used by package components, not a universal cast of every user array or every operator's implementation.

A lower-precision experiment should set the precision policy before creating models and should compare observables and gradients against a suitable reference. Explicit dtype arguments and fixed operator dtypes can still matter. Do not treat a lower precision as a purely computational change when the physical calculation is sensitive to it.

For GPU allocation behaviour, JAX supplies `XLA_PYTHON_CLIENT_PREALLOCATE` and memory-fraction controls described in its [GPU memory guide](https://docs.jax.dev/en/latest/gpu_memory_allocation.html). Those settings affect the allocator. They do not change the model, the sample count, or the sharding policy. Use [memory troubleshooting](troubleshooting.md#out-of-memory-errors) to identify the actual buffer that needs attention.
