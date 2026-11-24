---
title: Configuration options
---

# Configuration options

neuraLQX keeps package settings in [`nqx.config`](../api/public/neuralqx/config/index.md). These settings cover numerical precision, parallel execution, operator evaluation, logging, and profiling. This reference lists every registered option, grouped by its purpose, with the default value and the effect of each accepted choice.

Most calculations can start with the defaults. Change a setting when you need a particular execution layout, a diagnostic recording, or an explicit numerical policy. Settings such as the number of samples, network width, learning rate, and model evaluation chunk size belong to the objects you construct. Their arguments are explained in the [core concepts](../concepts/index.md), rather than being global configuration variables.

The [symbolic operator compiler](../concepts/operators/symbolic/index.md) has a separate options object and configuration namespace, `neuralqx.operator.symbolic.config`. Its [configuration reference](../developers/symbolic/configuration.md) lists every symbolic compiler option and debug setting, including their defaults and environment bindings.

## Reading and changing settings

### Python configuration

Import neuraLQX and use attribute access or the configuration manager's methods. These two assignments have the same effect:

```python
import neuralqx as nqx

nqx.config.log_level = "info"
nqx.config.set("log_level", "info")

print(nqx.config.log_level)
print(nqx.config.get("log_level"))
```

To change several options, pass a mapping to `set_many`. To make a temporary change, use `patch`, which restores the previous settings when the context exits. For example, this inspects a profiling policy without starting a capture:

```python
import neuralqx as nqx
from neuralqx.profile import ProfileOptions

before = nqx.config.profile_detail
with nqx.config.patch(profile_level="trace", profile_detail="detailed"):
    options = ProfileOptions.from_config()
    print(options.level, options.detail)

assert nqx.config.profile_detail == before

nqx.config.set_many({
    "operator_streaming_enabled": True,
    "operator_streaming_chunk_size": 256,
})
```

Runtime settings also support `patch(..., thread_local=True)`, which changes the configuration view of the current thread. This does not isolate process-wide side effects such as the Python logger, and it does not create an independent JAX runtime.

### Environment configuration

Set environment variables **before importing neuraLQX**. This is particularly useful in batch jobs, where the same script can run with different settings:

```sh
export NEURALQX_LOG_LEVEL=info
export NEURALQX_PROFILE_DETAIL=detailed
python -c 'import neuralqx as nqx
print(nqx.config.log_level, nqx.config.profile_detail)'
```

You can also establish the environment at the start of a Python script, before any import that might load neuraLQX:

```python
import os

os.environ["NEURALQX_LOG_LEVEL"] = "info"
os.environ["NEURALQX_PROFILE_DETAIL"] = "detailed"

import neuralqx as nqx

print(nqx.config.log_level, nqx.config.profile_detail)
```

Changing `os.environ` after the configuration manager has been created does not update its stored settings. Use `nqx.config` for later changes that the option permits. For [distributed initialisation](#distributed-runtime), establish the launch environment before importing libraries that initialise JAX devices.

The Python and environment assignments in each entry below are **alternative examples**, not a sequence to execute together. Python assignments assume `import neuralqx as nqx`. An environment assignment can be prefixed with `export` in a shell or placed in `os.environ` before import.

### Value syntax and precedence

Booleans accept Python `True` and `False`. Environment values are case insensitive and accept `1`, `true`, `t`, `yes`, `y`, or `on` for true, and `0`, `false`, `f`, `no`, `n`, or `off` for false. The examples use `1` and `0`. Named choices such as `"warning"` and `"trace"` are also case insensitive and have surrounding whitespace removed.

Integer settings accept integers or integer strings. Positive counts start at one, while nonnegative counts include zero. Optional integer and string settings accept `None` in Python, or an empty value, `none`, or `null` in the environment. Tuple settings use Python tuples or comma-separated environment values. An empty tuple is distinct from `None` where both concepts occur, as explained in the relevant entry.

The effective value is selected in the following order:

| Priority | Source | Effect |
| --- | --- | --- |
| Highest | Forced environment value | Prevents Python changes to that option |
| | Active thread-local override | Applies to the current thread's configuration view |
| | Explicit Python setting | Overrides the environment default |
| | Environment default | Overrides the registered default |
| Lowest | Registered default | Used when no override is present |

Only the options that explicitly list a `NEURALQX_FORCE_*` binding support forced environment values. Adding `FORCE` to another variable name does not create such a binding. Where aliases are listed, the first environment name present takes precedence, even when its value represents `False` or `None`.

### When changes take effect

Each entry is marked **startup** or **runtime**. Configure startup options before constructing arrays, meshes, states, or compiled functions that depend on them. The manager rejects further startup changes once `nqx.config.lock_runtime()` has been called. A Python assignment before that lock is permitted, but does not rebuild existing simulation objects or undo JAX initialisation. A fresh process is the reliable way to change a device topology.

Runtime options remain mutable after the lock. Their consumers may still read them only at a particular boundary. In particular, profiling options are sampled when a session starts, so changing them does not alter an ongoing recording. A traced computation can also retain decisions made while it was compiled. Set an evaluation policy before the first call that uses it.

### Inspecting the effective configuration

This example reports a value together with its source and metadata:

```python
import neuralqx as nqx

name = "operator_streaming_chunk_size"
metadata = nqx.config.describe_option(name)
print("Value:", metadata["effective_value"])
print("Default:", metadata["default"])
print("Environment:", metadata["env_default"])
print("Source:", nqx.config.value_source(name).value)
print("Registered options:", len(nqx.config.list_options()))

settings = nqx.config.snapshot()
assert settings[name] == nqx.config.get(name)
```

Use `nqx.config.show(include_env=True)` to print the complete configuration table. A snapshot is useful alongside a simulation's input parameters and [saved state](../concepts/vqs/saving.md). The [configuration API](../api/public/neuralqx/config/index.md#neuralqx.config.ConfigManager) documents inspection, overrides, hooks, and runtime locking.

## Numerical precision

`enable_x64` directly configures JAX. The three `dtype_*` options record a preferred dtype, but the built-in constructors in this release do not read them as universal defaults. Set the `dtype` and `param_dtype` arguments supported by the [network](../concepts/networks/models.md), [Hilbert space](../concepts/hilbert/index.md), or other object you construct. Changing a configuration string does not cast existing arrays.

### `enable_x64`

**Default:** `True`. **When:** startup. **Values:** `True` or `False`.

`True` enables JAX's 64-bit mode through `jax.config.update("jax_enable_x64", True)`. `False` disables it, which prevents ordinary JAX array construction from retaining requested 64-bit dtypes. This affects whether `float64`, `complex128`, and `int64` can be used, rather than forcing every calculation to use those types. neuraLQX applies this setting during configuration initialisation, so use it consistently with explicit constructor dtypes. See [precision and memory](../distributed/configuration.md#precision-and-memory).

**Python:** `nqx.config.enable_x64 = False`  
**Environment:** `NEURALQX_ENABLE_X64=0`

### `dtype_real`

**Default:** `"float64"`. **When:** startup.

**Values:** `"float16"` selects a 16-bit floating-point preference, `"bfloat16"` a 16-bit format with a wider exponent range and fewer precision bits, `"float32"` single precision, and `"float64"` double precision. The value is stored as a string. In this release it does not change built-in model or graph dtypes automatically. Use explicit constructor arguments for the numerical calculation.

**Python:** `nqx.config.dtype_real = "float32"`  
**Environment:** `NEURALQX_DTYPE_REAL=float32`

### `dtype_complex`

**Default:** `"complex128"`. **When:** startup.

**Values:** `"complex64"` records a preference for complex numbers with two 32-bit components, and `"complex128"` for two 64-bit components. This setting does not automatically change wavefunction outputs or operator matrix elements in this release. Choose the relevant object's dtype explicitly, with `enable_x64=True` when using `complex128`. The [complex network guide](../concepts/networks/complex.md) explains how complex parameters and outputs enter a variational model.

**Python:** `nqx.config.dtype_complex = "complex64"`  
**Environment:** `NEURALQX_DTYPE_COMPLEX=complex64`

### `dtype_index`

**Default:** `"int32"`. **When:** startup.

**Values:** `"int32"` records a 32-bit signed integer preference, and `"int64"` a 64-bit signed integer preference. The built-in graph and indexing code does not universally consume this setting. Changing it does not enlarge an enumerator's supported state count or change existing connectivity arrays. See [state layouts](../concepts/hilbert/state-layouts.md) for the distinction between configuration coordinates and state indices.

**Python:** `nqx.config.dtype_index = "int64"`  
**Environment:** `NEURALQX_DTYPE_INDEX=int64`

## Execution, compilation, and logging

Several registered execution options are reserved configuration values in this release. Their entries distinguish those values from controls that currently change execution. The hardware guides explain [CPU](../distributed/cpu.md) and [GPU](../distributed/single-gpu.md) installation and selection.

### `backend_target`

**Default:** `"auto"`. **When:** startup.

**Values:** `"auto"`, `"cpu"`, `"gpu"`, or `"tpu"` record a preference for automatic selection, CPU, GPU, or TPU execution. **This option does not select the JAX backend in this release.** Use `JAX_PLATFORMS` and the hardware-specific launch instructions to select actual devices. Accepting `"tpu"` here is not a compatibility guarantee for every neuraLQX operation.

**Python:** `nqx.config.backend_target = "cpu"`  
**Environment:** `NEURALQX_BACKEND_TARGET=cpu`  
**Forced environment:** `NEURALQX_FORCE_BACKEND_TARGET=cpu`

### `jit_enabled`

**Default:** `True`. **When:** runtime. **Values:** `True` or `False`.

Records whether compiled execution is requested. Neither value currently changes JAX compilation behaviour through this option. To disable JIT for debugging, use JAX's `jax_disable_jit` setting or `JAX_DISABLE_JIT=1`, as described in the [JAX configuration reference](https://docs.jax.dev/en/latest/config_options.html#disable-jit).

**Python:** `nqx.config.jit_enabled = False`  
**Environment:** `NEURALQX_JIT_ENABLED=0`

### `compile_cache_enabled`

**Default:** `True`. **When:** runtime. **Values:** `True` or `False`.

Records an enabled or disabled compilation-cache preference. This option is not connected to JAX's memory or disk caches in this release. `False` does not clear cached executables, and `True` does not create a persistent cache. Configure caching with JAX's own settings and the [persistent compilation cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

**Python:** `nqx.config.compile_cache_enabled = False`  
**Environment:** `NEURALQX_COMPILE_CACHE_ENABLED=0`  
**Forced environment:** `NEURALQX_FORCE_COMPILE_CACHE_ENABLED=0`

### `compile_cache_dir`

**Default:** `None`. **When:** startup. **Values:** `None` or a directory path string.

`None` leaves the recorded cache location unspecified. A string records a proposed location, but does not configure or create a JAX cache in this release. For an actual persistent cache, set `JAX_COMPILATION_CACHE_DIR` before the first compilation, following the [parallelism configuration guide](../distributed/configuration.md#backend-selection-and-jax-settings).

**Python:** `nqx.config.compile_cache_dir = ".cache/neuralqx"`  
**Environment:** `NEURALQX_COMPILE_CACHE_DIR=.cache/neuralqx`

### `log_level`

**Default:** `"warning"`. **When:** runtime.

Sets the level of Python's `neuralqx` logger. `"critical"` admits critical messages, `"error"` also admits errors, `"warning"` also admits warnings, `"info"` adds informational messages, and `"debug"` includes debugging messages. Handler configuration still determines where admitted messages are written. This is separate from a driver's progress display, metric loggers, and Python's warnings filters. See [driver output](../concepts/drivers/monitoring.md) for simulation results and progress reporting.

**Python:** `nqx.config.log_level = "info"`  
**Environment:** `NEURALQX_LOG_LEVEL=info`

## Distributed runtime

These options configure cooperation between Python processes. A single process using several devices does not require `distributed=True`. For a complete launch example, follow [multiple nodes and GPUs](../distributed/multi-node.md).

Automatic initialisation runs when both `distributed` and `distributed_auto_initialize` are true, unless JAX is already initialised. Set the address, process count, rank, and local device selection **before enabling that hook**. Environment configuration before import is the recommended batch-job setup. If configuring in Python, first keep `distributed=False`, assign the other fields, then enable it before JAX device initialisation. Changing fields after initialisation does not reconfigure the running cluster.

### `distributed`

**Default:** `False`. **When:** startup. **Values:** `True` or `False`.

`True` requests JAX distributed initialisation through the automatic hook. `False` leaves initialisation to the application or keeps ordinary single-process execution. It does not shut down an already initialised runtime. Neither value launches additional Python processes. Your scheduler or launcher must start them.

**Python:** `nqx.config.distributed = True`  
**Environment:** `NEURALQX_DISTRIBUTED=1`  
**Environment alias:** `NEURALQX_DISTRIBUTED_ENABLED=1`  
**Forced environment:** `NEURALQX_FORCE_DISTRIBUTED=1`, with alias `NEURALQX_FORCE_DISTRIBUTED_ENABLED=1`

### `distributed_auto_initialize`

**Default:** `True`. **When:** startup. **Values:** `True` or `False`.

`True` allows neuraLQX to call `jax.distributed.initialize()` when distributed mode is enabled. `False` suppresses that automatic call so your application can initialise JAX explicitly. Set this to false in the environment before import when your own launcher needs to control initialisation. Enabling it later while `distributed=True` can immediately invoke the hook.

**Python:** `nqx.config.distributed_auto_initialize = False`  
**Environment:** `NEURALQX_DISTRIBUTED_AUTO_INITIALIZE=0`

### `distributed_coordinator_address`

**Default:** `None`. **When:** startup. **Values:** `None` or a reachable `"host:port"` string.

`None` leaves coordinator discovery to JAX's supported launch-environment detection. An explicit address identifies the coordinator used by every participating process. When supplying any of the coordinator address, process count, or process ID in the environment, supply all three. neuraLQX checks this combination during configuration construction. The profiling coordinator has a [separate setting](#profile_coordinator_address).

**Python:** `nqx.config.distributed_coordinator_address = "node0:12345"`  
**Environment:** `NEURALQX_DISTRIBUTED_COORDINATOR_ADDRESS=node0:12345`

### `distributed_num_processes`

**Default:** `None`. **When:** startup. **Values:** `None` or a positive integer.

`None` leaves process-count discovery to JAX. A positive value is the total number of cooperating Python processes, not the number of devices or nodes. Every process must agree on this count. Supply it together with the coordinator address and process ID for explicit initialisation.

**Python:** `nqx.config.distributed_num_processes = 2`  
**Environment:** `NEURALQX_DISTRIBUTED_NUM_PROCESSES=2`

### `distributed_process_id`

**Default:** `None`. **When:** startup. **Values:** `None` or a nonnegative integer smaller than `distributed_num_processes`.

`None` leaves rank discovery to JAX. Explicit IDs identify individual processes and must be distinct and contiguous from zero. Process zero is the coordinator process. This is the setting that normally differs between workers in an otherwise shared configuration.

**Python:** `nqx.config.distributed_process_id = 0`  
**Environment:** `NEURALQX_DISTRIBUTED_PROCESS_ID=0`

### `distributed_local_device_ids`

**Default:** `None`. **When:** startup. **Values:** `None` or a tuple of nonnegative local device IDs.

`None` uses JAX's distributed device-selection behaviour, which can depend on the launcher. A tuple explicitly selects ordinals visible inside that process, such as `(0, 1)`. A single integer is also accepted and becomes a one-element tuple. Use valid, distinct IDs for the actual allocation. An empty tuple is not the automatic-selection value, so use `None` when no explicit list is intended. See the [Slurm device assignment examples](../distributed/multi-node.md).

**Python:** `nqx.config.distributed_local_device_ids = (0, 1)`  
**Environment:** `NEURALQX_DISTRIBUTED_LOCAL_DEVICE_IDS=0,1`

### `distributed_initialization_timeout`

**Default:** `300` seconds. **When:** startup. **Values:** positive integers.

Passes the initialisation deadline to JAX. A larger value allows more time for all workers to join. It does not change simulation step timeouts or repair an incorrect rank count or unreachable coordinator. See [distributed troubleshooting](../distributed/troubleshooting.md) before increasing it for a stalled launch.

**Python:** `nqx.config.distributed_initialization_timeout = 600`  
**Environment:** `NEURALQX_DISTRIBUTED_INITIALIZATION_TIMEOUT=600`

## Sharding and device meshes

The [sharding guide](../distributed/sharding.md) explains how sample batches, operator terms, and neural-network tensors are distributed. The high-level policy establishes initial defaults for three enable flags. Explicit flag environment values override those defaults.

### `sharding_axes`

**Default:** `("samples",)`. **When:** startup. **Values:** tuples of logical axes or the string forms below.

| Value | Initial policy |
| --- | --- |
| `"samples"` or `"default"` | Sample sharding |
| `"operators"` or `"samples,operators"` | Sample and operator sharding |
| `"tensor"` | Tensor parallelism alone |
| `"samples,tensor"` | Independent samples with cooperating tensor workers |
| `"operators,tensor"` | Independent operator terms with cooperating tensor workers, no sample sharding |
| `"all"`, `"full"`, or `"samples,operators,tensor"` | All three modes |
| `"none"`, `"off"`, `"false"`, `"replicated"`, `""`, or `()` | All three flags disabled |
| Python `None` | The default sample policy |

Comma-separated, plus-separated, and space-separated axis names are accepted. Sample aliases are `sample`, `samples`, `chain`, `chains`, and `data`. Operator aliases are `operator`, `operators`, `op`, and `ops`. Tensor aliases are `tensor`, `tp`, `network`, and `networks`. [`parse_sharding_axes`](../api/public/neuralqx/config/index.md#neuralqx.config.parse_sharding_axes) removes duplicates and returns the order samples, operators, tensor.

Tensor policies enable exactly the work axes listed. In particular, `tensor` does not add samples, and `operators,tensor` does not add samples. The `operators` shorthand without tensor mode enables samples as well. To request operator-only execution without tensor mode, explicitly disable `sample_sharding_enabled`.

The individual defaults are derived **once, during configuration construction**. Assigning `sharding_axes` in Python afterwards updates the policy value but does not recompute the enable flags. Set those flags directly before constructing the simulation, or select the policy through the environment. See the [Python setup example](../distributed/configuration.md#sharding-policy).

**Python:** `nqx.config.sharding_axes = ("samples", "operators", "tensor")`

**Environment:** `NEURALQX_SHARDING=samples,operators,tensor`

**Environment alias:** `NEURALQX_SHARDING_AXES=samples,operators,tensor`

### `sample_sharding_enabled`

**Default:** `True` under the default policy, otherwise whether `"samples"` is in the initial `sharding_axes`. **When:** startup. **Values:** `True` or `False`.

`True` distributes supported sample batches and Markov chains. With tensor mode enabled, independent chains use the sample and operator mesh coordinates together, while each chain is replicated inside its cooperating tensor group. `False` disables independent chain partitioning. It does not hide devices or select a backend. See [sample placement](../distributed/sharding.md#samples-and-chains).

**Python:** `nqx.config.sample_sharding_enabled = False`

**Environment:** `NEURALQX_SAMPLE_SHARDING_ENABLED=0`

### `operator_sharding_enabled`

**Default:** `False` under the default policy, otherwise whether `"operators"` is in the initial `sharding_axes`. **When:** startup. **Values:** `True` or `False`.

`True` distributes supported independent operator terms, including operator lists and [`SumOperator`](../api/public/neuralqx/operator/index.md#neuralqx.operator.SumOperator). `False` disables that term-distribution path. Without tensor mode, the planner chooses operator and sample partition counts from the workload. With tensor mode, it uses the operator coordinate of the persistent tensor mesh. An operator axis of size one has no cross-group term dispatch. See [combined layouts](../distributed/sharding.md#combined-operator-and-sample-layouts).

**Python:** `nqx.config.operator_sharding_enabled = True`

**Environment:** `NEURALQX_OPERATOR_SHARDING_ENABLED=1`

### `network_sharding_enabled`

**Default:** `False` under the default policy, otherwise whether `"tensor"` is in the initial `sharding_axes`. **When:** startup. **Values:** `True` or `False`.

`True` enables tensor partition planning, weight placement, and compiler-managed network contractions over the tensor axis. It also makes sampling and estimator planners retain the same tensor groups across execution phases. `False` uses ordinary framework placement without a tensor adapter. The flag does not create devices or start additional processes.

Built-in architectures and supported layers supply default partition rules. Custom models can provide [explicit layouts](../distributed/tensor-parallelism/custom-models.md). A multi-worker plan must partition at least one trainable array, with exactly divisible dimensions. See [tensor parallelism](../distributed/tensor-parallelism/index.md) for its mathematical meaning and [direct distributed initialization](../distributed/tensor-parallelism/models.md#initialize-directly-into-shards) for large networks.

**Python:** `nqx.config.network_sharding_enabled = True`

**Environment:** `NEURALQX_NETWORK_SHARDING_ENABLED=1`

**Environment alias:** `NEURALQX_TENSOR_PARALLEL_ENABLED=1`

### `mesh_sample_axis_name`

**Default:** `"samples"`. **When:** startup. **Values:** nonempty strings distinct from the other two mesh axis names.

Names the physical mesh axis corresponding to logical samples. Custom specifications that refer to this axis must use its configured name. This changes the label, not the device count.

**Python:** `nqx.config.mesh_sample_axis_name = "chains"`

**Environment:** `NEURALQX_MESH_SAMPLE_AXIS_NAME=chains`

### `mesh_operator_axis_name`

**Default:** `"operators"`. **When:** startup. **Values:** nonempty strings distinct from the other two mesh axis names.

Names the physical mesh axis corresponding to independent operator work. Use a consistent name in custom mesh specifications.

**Python:** `nqx.config.mesh_operator_axis_name = "terms"`

**Environment:** `NEURALQX_MESH_OPERATOR_AXIS_NAME=terms`

### `mesh_tensor_axis_name`

**Default:** `"tensor"`. **When:** startup. **Values:** nonempty strings distinct from the other two mesh axis names.

Names the cooperating network axis. Explicit Flax metadata should use this physical name. The logical `"tensor"` entry in `TensorParallel` rules is translated to it by the planner. Renaming an axis does not change its size or enable the corresponding execution mode.

**Python:** `nqx.config.mesh_tensor_axis_name = "features"`

**Environment:** `NEURALQX_MESH_TENSOR_AXIS_NAME=features`

### `mesh_sample_axis_size`

**Default:** `None`. **When:** startup. **Values:** `None` or a positive integer.

In tensor mode, `None` lets the persistent mesh infer the sample dimension and a positive integer fixes it. A disabled sample axis must have size one or be left unset. The resolved sample, operator, and tensor sizes must multiply to the global device count.

Without tensor mode, standard sampling and force planners request their own sample layouts and can override generic mesh choices. This is not a global limit on visible devices. See [mesh names and sizes](../distributed/configuration.md#mesh-names-and-sizes).

**Python:** `nqx.config.mesh_sample_axis_size = 2`

**Environment:** `NEURALQX_MESH_SAMPLE_AXIS_SIZE=2`

### `mesh_operator_axis_size`

**Default:** `None`. **When:** startup. **Values:** `None` or a positive integer.

In tensor mode, fixes or infers the number of operator groups in the shared mesh. A disabled operator axis must have size one or be left unset. Fewer groups than terms assign several terms to each group. More groups than terms can introduce idle scheduled slots.

Without tensor mode, standard operator planning chooses its phase layout from the workload and can override this generic setting. See [operator placement](../distributed/sharding.md#combined-operator-and-sample-layouts).

**Python:** `nqx.config.mesh_operator_axis_size = 2`

**Environment:** `NEURALQX_MESH_OPERATOR_AXIS_SIZE=2`

### `mesh_tensor_axis_size`

**Default:** `None`. **When:** startup. **Values:** `None` or a positive integer.

Controls the number of cooperating devices per network evaluation when tensor mode is enabled. `None` infers a divisor of the available devices while respecting explicit sample and operator sizes. A positive integer requests that exact tensor group size. One retains the tensor execution path without dividing arrays between tensor workers.

The full mesh must use every visible device. A tensor-only launch on eight devices therefore needs a tensor size of eight. To choose a smaller group, enable an independent sample or operator axis for the remaining devices. Weight dimensions selected by the plan must divide the tensor size exactly. The planner does not resize the model.

**Python:** `nqx.config.mesh_tensor_axis_size = 2`

**Environment:** `NEURALQX_MESH_TENSOR_AXIS_SIZE=2`

## Operator evaluation

These controls concern connection generation within an operator. They are independent of distributing separate operator terms and of the variational state's model evaluation chunks. See [chunking and streaming](../distributed/sharding.md#chunking-and-streaming) and [VQS batching](../concepts/vqs/batching.md).

### `operator_streaming_enabled`

**Default:** `False`. **When:** runtime. **Values:** `True` or `False`.

`True` enables chunked connection generation in supported evaluation paths. `False` uses the ordinary unstreamed path. Streaming can reduce intermediate memory use, but the complete output may still have to be assembled. It does not guarantee that an arbitrary operator fits in memory. An explicit estimator argument `operator_streaming` overrides the global default for that call.

**Python:** `nqx.config.operator_streaming_enabled = True`  
**Environment:** `NEURALQX_OPERATOR_STREAMING_ENABLED=1`

### `operator_streaming_chunk_size`

**Default:** `1024`. **When:** runtime. **Values:** positive integers.

Sets the number of **flattened input configurations** processed per operator-generation chunk when streaming is enabled. This is not a number of terms, connected states, or Monte Carlo steps. Smaller values can lower temporary allocation sizes at the cost of more chunks. An explicit estimator argument with the same name overrides this default.

**Python:** `nqx.config.operator_streaming_chunk_size = 256`  
**Environment:** `NEURALQX_OPERATOR_STREAMING_CHUNK_SIZE=256`

## Profiling activation and output

The following profiling groups are all **runtime** settings. They are excluded from the configuration manager's compilation fingerprint. Start with [profiling for scientists](../developers/profiling/for-scientists.md) for a complete capture and an explanation of its results.

An explicit [`Profiler`](../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler) without an `options` argument snapshots the effective configuration on entry. A directly constructed [`ProfileOptions()`](../api/public/neuralqx/profile/index.md#neuralqx.profile.ProfileOptions) uses its own dataclass defaults instead. Use `ProfileOptions.from_config()` to capture the current package policy. Explicit profiler arguments can override that snapshot, as explained in [profiling configuration](../developers/profiling/configuration.md#package-options-and-explicit-sessions).

### `profile`

**Default:** `False`. **When:** runtime. **Values:** `True` or `False`.

`True` enables automatic profiling of supported bounded driver `run` calls. `False` disables that automatic capture, while leaving explicitly created profiler sessions available. The default `profile_level="basic"` therefore does not create recordings by itself. If automatic capture is enabled with `profile_level="off"`, the driver uses `"basic"`. Set `profile=False` to disable automatic driver profiling.

**Python:** `nqx.config.profile = True`  
**Environment:** `NEURALQX_PROFILE=1`

### `profile_level`

**Default:** `"basic"`. **When:** runtime.

| Value | Collection behaviour |
| --- | --- |
| `"off"` | An explicit profiler creates no recording artifacts |
| `"basic"` | Records application regions and events on the host |
| `"trace"` | Also requests native CPU and accelerator activity from JAX |
| `"full"` | Includes native activity and enables Python call tracing by default |

Native collection depends on the available profiling backend. `profile_jax_python_tracer_level` can override the Python tracing default. Annotation depth is controlled separately by `profile_detail`. See [capture options](../developers/profiling/scientists/capture-options.md) for choosing the information needed for a measurement.

**Python:** `nqx.config.profile_level = "trace"`  
**Environment:** `NEURALQX_PROFILE_LEVEL=trace`

### `profile_detail`

**Default:** `"summary"`. **When:** runtime.

`"summary"` retains major simulation phases. `"detailed"` also includes named operator terms and estimator stages. `"developer"` adds internal dispatch boundaries. User-defined spans are retained at every detail level. This controls neuraLQX annotations and does not enable tracing of every Python call. See [instrumenting custom code](../developers/profiling/scientists/custom-code.md).

**Python:** `nqx.config.profile_detail = "detailed"`  
**Environment:** `NEURALQX_PROFILE_DETAIL=detailed`

### `profile_output_dir`

**Default:** `".neuralqx/profiles"`. **When:** runtime. **Values:** nonempty directory path strings.

Sets the root directory for run artifacts. Relative paths are interpreted from the process's working directory. Each recording uses a run directory beneath this root, with rank-specific output and any collected trace. Shared transport requires storage that every participating process can access. See [trace files](../developers/profiling/traces.md).

**Python:** `nqx.config.profile_output_dir = "results/profiles"`  
**Environment:** `NEURALQX_PROFILE_OUTPUT_DIR=results/profiles`

### `profile_run_id`

**Default:** `None`. **When:** runtime. **Values:** `None` or a unique run identifier string.

`None` generates an identifier for a single-process recording. Multi-process sessions require the same explicit identifier on every rank. A valid identifier contains at most 128 characters, starts with a letter or digit, and then contains only letters, digits, dots, underscores, or hyphens. It names one directory component. These restrictions are checked when profiling options are constructed, and existing rank directories are not reused. Choose a fresh identifier for each recording.

**Python:** `nqx.config.profile_run_id = "volume-check-001"`  
**Environment:** `NEURALQX_PROFILE_RUN_ID=volume-check-001`

### `profile_synchronize`

**Default:** `"boundary"`. **When:** runtime. **Values:** `"boundary"` or `"none"`.

`"boundary"` waits for explicitly supplied results or a `wait_for` callback at capture boundaries. `"none"` adds no such completion waits. Boundary synchronisation is not a device-wide barrier and does not discover arbitrary outstanding arrays. It also does not insert a wait around every span. The [custom-code guide](../developers/profiling/scientists/custom-code.md) explains how to supply results so asynchronous work is included in the intended window.

**Python:** `nqx.config.profile_synchronize = "none"`  
**Environment:** `NEURALQX_PROFILE_SYNCHRONIZE=none`

## Profiling recording windows

These options describe a repeating sequence of waiting, warmup, and recorded steps. The driver advances the schedule once per attempted iteration. Custom loops advance it with the profiler's `step()` method. Read [recording windows](../developers/profiling/scientists/capture-options.md#record-a-short-window) before interpreting the first compiled iteration as steady-state performance.

For scheduled recording, `profile_active_steps` must be positive. Nonzero waiting, warmup, or initial skip counts with `profile_active_steps=0` are rejected when `ProfileOptions` is constructed. Configure the fields together before starting a session:

```python
import neuralqx as nqx
from neuralqx.profile import ProfileOptions

with nqx.config.patch(
    profile_wait_steps=2,
    profile_warmup_steps=1,
    profile_active_steps=3,
    profile_repeat=2,
):
    schedule = ProfileOptions.from_config().make_schedule()
    print([schedule.action(step).value for step in range(6)])
```

```text
['wait', 'wait', 'warmup', 'record', 'record', 'record']
```

### `profile_wait_steps`

**Default:** `0`. **When:** runtime. **Values:** nonnegative integers.

Sets the number of unprofiled steps at the beginning of each scheduled cycle. Zero skips this phase. A positive value requires positive `profile_active_steps`.

**Python:** `nqx.config.profile_wait_steps = 2`  
**Environment:** `NEURALQX_PROFILE_WAIT_STEPS=2`

### `profile_warmup_steps`

**Default:** `0`. **When:** runtime. **Values:** nonnegative integers.

Sets the collector warmup steps before the retained part of each cycle. Warmup activity is discarded from the exported recording. Zero begins recording without this phase. A positive value requires positive `profile_active_steps`. This warms the collector and does not by itself guarantee that every simulation kernel has already compiled.

**Python:** `nqx.config.profile_warmup_steps = 1`  
**Environment:** `NEURALQX_PROFILE_WARMUP_STEPS=1`

### `profile_active_steps`

**Default:** `0`. **When:** runtime. **Values:** nonnegative integers.

Zero selects continuous recording over the context. A positive integer records that many steps in each scheduled cycle. The wait, warmup, and initial skip counts must all remain zero for continuous recording.

**Python:** `nqx.config.profile_active_steps = 3`  
**Environment:** `NEURALQX_PROFILE_ACTIVE_STEPS=3`

### `profile_repeat`

**Default:** `1`. **When:** runtime. **Values:** nonnegative integers.

A positive value limits the number of scheduled capture cycles. Zero repeats until the surrounding profiler context ends. This field has no cycling effect when `profile_active_steps=0` selects continuous recording.

**Python:** `nqx.config.profile_repeat = 2`  
**Environment:** `NEURALQX_PROFILE_REPEAT=2`

### `profile_skip_first`

**Default:** `0`. **When:** runtime. **Values:** nonnegative integers.

Sets an initial unprofiled prefix before the first cycle. Unlike `profile_wait_steps`, this prefix is not repeated. Zero starts the schedule immediately. A positive value requires positive `profile_active_steps`.

**Python:** `nqx.config.profile_skip_first = 10`  
**Environment:** `NEURALQX_PROFILE_SKIP_FIRST=10`

## Profiling backends and recording limits

Native tracing and external instrumentation complement the package's own application events. The [backend guide](../developers/profiling/backends.md) explains installation, device activity, and the distinction between recorded execution and sampled counters.

### `profile_jax_host_tracer_level`

**Default:** `1`. **When:** runtime. **Values:** `1`, `2`, or `3`.

With native collection enabled, `1` retains user-instrumented host events, `2` adds higher-level runtime activity such as expensive XLA operations, and `3` adds finer execution detail. neuraLQX rejects zero because its clock alignment needs host markers. This setting has no native-tracing effect in `"basic"` mode. The meanings of these levels follow the [JAX profiler options](https://docs.jax.dev/en/latest/profiling.html#general-options).

**Python:** `nqx.config.profile_jax_host_tracer_level = 2`  
**Environment:** `NEURALQX_PROFILE_JAX_HOST_TRACER_LEVEL=2`

### `profile_jax_python_tracer_level`

**Default:** `None`. **When:** runtime. **Values:** `None`, `0`, or `1`.

`None` selects Python tracing level one for `profile_level="full"` and zero otherwise. Explicit `0` disables native Python call tracing, and `1` enables it when native collection is active. This is separate from neuraLQX's named spans and can produce substantially more events.

**Python:** `nqx.config.profile_jax_python_tracer_level = 1`  
**Environment:** `NEURALQX_PROFILE_JAX_PYTHON_TRACER_LEVEL=1`

### `profile_nvtx`

**Default:** `False`. **When:** runtime. **Values:** `True` or `False`.

`True` emits NVTX ranges for external NVIDIA profiling tools through the optional `nvtx` package. `False` leaves those ranges disabled. It does not install the dependency, start Nsight, or enable GPU counter sampling. Follow the [NVTX backend instructions](../developers/profiling/backends.md) before enabling it.

**Python:** `nqx.config.profile_nvtx = True`  
**Environment:** `NEURALQX_PROFILE_NVTX=1`

### `profile_max_events`

**Default:** `100_000`. **When:** runtime. **Values:** positive integers.

Limits application and telemetry event reservations per process across the recording windows. It does not bound JAX's native buffers. Reservation granularity can leave fewer retained events than the configured number, and dropped-event diagnostics are written to the artifacts. For large native traces, shorten the recording window as well.

**Python:** `nqx.config.profile_max_events = 200_000`  
**Environment:** `NEURALQX_PROFILE_MAX_EVENTS=200000`

### `profile_max_lanes`

**Default:** `1024`. **When:** runtime. **Values:** positive integers.

Limits the number of application thread and asynchronous-task lanes per process. This is a recording limit, not a thread-pool size, GPU stream count, or simulation concurrency setting. Lane-limit diagnostics remain available in the profiling artifacts.

**Python:** `nqx.config.profile_max_lanes = 2048`  
**Environment:** `NEURALQX_PROFILE_MAX_LANES=2048`

### `profile_max_trace_bytes`

**Default:** `4_294_967_296` bytes, or 4 GiB. **When:** runtime. **Values:** positive integers.

Sets the maximum compressed trace size accepted from one rank by TCP collection. It does not cap a combined trace, local native-tracer buffers, or total profiling memory. Choose it with the expected trace size and coordinator resources in mind.

**Python:** `nqx.config.profile_max_trace_bytes = 1_073_741_824`  
**Environment:** `NEURALQX_PROFILE_MAX_TRACE_BYTES=1073741824`

## GPU telemetry

Telemetry samples device-management counters alongside a capture. It does not resolve individual kernel timings. The [telemetry guide](../developers/profiling/backends.md#sampling-time-and-resolution) explains the sampling interval and how to interpret these measurements.

### `profile_telemetry`

**Default:** `"off"`. **When:** runtime.

`"off"` disables GPU management sampling. `"auto"` tries NVML, then `nvidia-smi`, and records provider failures or unavailability without requiring either provider to work. `"nvml"` explicitly requests NVML, while `"nvidia-smi"` requests subprocess queries through that command. An explicitly chosen provider must be available and work. Enabling telemetry does not enable native JAX device tracing.

**Python:** `nqx.config.profile_telemetry = "auto"`  
**Environment:** `NEURALQX_PROFILE_TELEMETRY=auto`

### `profile_telemetry_interval_s`

**Default:** `1.0` seconds. **When:** runtime. **Values:** positive finite numbers.

Sets the delay after each background management query. The time spent making the query contributes to the observed interval between samples, so this is not an exact sampling frequency. Smaller values increase query frequency and overhead, without converting management counters into per-kernel measurements.

**Python:** `nqx.config.profile_telemetry_interval_s = 0.5`  
**Environment:** `NEURALQX_PROFILE_TELEMETRY_INTERVAL_S=0.5`

### `profile_telemetry_scope`

**Default:** `"host"`. **When:** runtime. **Values:** `"host"` or `"rank"`.

`"host"` elects the lowest rank on each hostname to sample management-visible GPUs once for that host. `"rank"` samples independently in every rank, which can be useful when containers expose different devices. Offline multi-process captures with telemetry enabled require `"rank"` because they cannot coordinate a host-level election.

**Python:** `nqx.config.profile_telemetry_scope = "rank"`  
**Environment:** `NEURALQX_PROFILE_TELEMETRY_SCOPE=rank`

### `profile_telemetry_devices`

**Default:** `()`. **When:** runtime. **Values:** a tuple of GPU management UUIDs or index strings.

The empty tuple selects all GPUs visible to the management provider on a sampled host. A nonempty tuple selects the listed devices. Identifiers must be nonempty strings of at most 256 characters when validated as profiling options. These are management identifiers, not JAX logical device IDs, and selection is not automatically filtered by `CUDA_VISIBLE_DEVICES`. The comma-separated parser removes surrounding whitespace and empty entries. Use an empty environment value to restore `()`.

**Python:** `nqx.config.profile_telemetry_devices = ("0", "1")`  
**Environment:** `NEURALQX_PROFILE_TELEMETRY_DEVICES=0,1`

### `profile_nvidia_smi_timeout_s`

**Default:** `2.0` seconds. **When:** runtime. **Values:** positive finite numbers.

Sets the timeout for an individual read-only `nvidia-smi` query. It applies when that provider is used, including an automatic fallback to it. This does not set the overall profiler timeout or the JAX distributed initialisation timeout.

**Python:** `nqx.config.profile_nvidia_smi_timeout_s = 5.0`  
**Environment:** `NEURALQX_PROFILE_NVIDIA_SMI_TIMEOUT_S=5.0`

## Profiling across processes

These options concern trace coordination and collection. They do not initialise the numerical JAX runtime. Read [distributed profiling](../developers/profiling/distributed.md) for complete shared-storage, TCP, and offline examples.

### `profile_transport`

**Default:** `"shared"`. **When:** runtime.

`"shared"` collects rank artifacts through common writable storage. `"tcp"` transfers completed rank traces to rank zero and supports rank-local output directories. `"offline"` writes independent rank artifacts without clock exchange or live collection. For TCP capture across processes, provide `profile_coordinator_address`. Multi-process sessions also require an explicit shared `profile_run_id`.

**Python:** `nqx.config.profile_transport = "tcp"`  
**Environment:** `NEURALQX_PROFILE_TRANSPORT=tcp`

### `profile_coordinator_address`

**Default:** `None`. **When:** runtime. **Values:** `None` or a reachable rank-zero `"host:port"` string.

`None` leaves the profiler's TCP endpoint unspecified. A multi-process TCP capture requires an explicit endpoint reachable by every rank. Choose a port distinct from the JAX distributed coordinator. Shared and offline transports do not use this setting as their transport endpoint.

**Python:** `nqx.config.profile_coordinator_address = "node0:12346"`  
**Environment:** `NEURALQX_PROFILE_COORDINATOR_ADDRESS=node0:12346`

### `profile_timeout_s`

**Default:** `60.0` seconds. **When:** runtime. **Values:** positive finite numbers.

Sets the deadline for each distributed profiling coordination or transfer phase. It is not a duration limit for the complete simulation or capture. Larger values allow slower coordination and transfer, while failures still need to be checked for missing ranks or inaccessible storage and endpoints.

**Python:** `nqx.config.profile_timeout_s = 120.0`  
**Environment:** `NEURALQX_PROFILE_TIMEOUT_S=120.0`

### `profile_clock_samples`

**Default:** `5`. **When:** runtime. **Values:** positive integers.

Sets the number of round trips used at each clock-calibration boundary to estimate inter-process clock offsets. Increasing it gives calibration more observations and adds coordination work. It is separate from telemetry frequency and Monte Carlo sampling. Offline transport does not perform this exchange.

**Python:** `nqx.config.profile_clock_samples = 9`  
**Environment:** `NEURALQX_PROFILE_CLOCK_SAMPLES=9`

## Experimental behaviour and native extensions

These options distinguish registered extension policies from the implemented connection-deduplication path. A declared configuration flag does not imply that a corresponding extension is installed or enabled elsewhere.

### `experimental`

**Default:** `False`. **When:** runtime. **Values:** `True` or `False`.

Records the general experimental opt-in policy. Changing it to `True` emits an experimental-mode warning. In this release it is not a universal gate around every experimental API, and it does not automatically enable connection deduplication. `False` restores the recorded policy without unloading imported code. Consult the [experimental API](../api/experimental/index.md) for the interfaces actually provided.

**Python:** `nqx.config.experimental = True`  
**Environment:** `NEURALQX_EXPERIMENTAL=1`  
**Environment alias:** `NEURALQX_EXPERIMENTAL_ENABLED=1`  
**Forced environment:** `NEURALQX_FORCE_EXPERIMENTAL=1`

### `experimental_features`

**Default:** `()`. **When:** runtime. **Values:** a tuple of feature-name strings or a comma-separated string.

The empty tuple records no named opt-ins. A nonempty tuple records the supplied names, with surrounding whitespace and empty entries removed. There is no registry validation of feature names and no built-in consumer of these named gates in this release. An arbitrary name being accepted does not enable a feature. Names used by a custom extension must be defined and read by that extension. The example below illustrates the syntax for such an application-defined name.

**Python:** `nqx.config.experimental_features = ("custom_estimator",)`  
**Environment:** `NEURALQX_EXPERIMENTAL_FEATURES=custom_estimator`

### `experimental_connection_deduplication`

**Default:** `False`. **When:** runtime. **Values:** `True` or `False`.

`True` opts supported VQS estimator paths into deduplicating connected configurations and sharing target evaluations. `False` uses the ordinary evaluation path. Sorting, preparation, synchronisation, and compilation overhead can outweigh the saved network evaluations, so enabling it emits a warning and is not a promise of improved runtime. It is independent of `experimental`. Explicit estimator options can override this global default. Compare complete workloads using the [profiling guides](../developers/profiling/for-scientists.md).

**Python:** `nqx.config.experimental_connection_deduplication = True`  
**Environment:** `NEURALQX_EXPERIMENTAL_CONNECTION_DEDUPLICATION=1`

### `allow_ffi_kernels`

**Default:** `False`. **When:** startup. **Values:** `True` or `False`.

Records whether optional native FFI kernels are permitted by an extension policy. There is no built-in loader consuming this flag in this release, so `True` does not load, install, or select a native kernel, and `False` is not a restriction on arbitrary extension code. Follow an extension's own setup instructions if it explicitly uses this policy.

**Python:** `nqx.config.allow_ffi_kernels = True`  
**Environment:** `NEURALQX_ALLOW_FFI_KERNELS=1`

## Settings owned by JAX and the execution environment

`JAX_PLATFORMS`, `JAX_NUM_CPU_DEVICES`, `JAX_COMPILATION_CACHE_DIR`, and `CUDA_VISIBLE_DEVICES` are not entries in `nqx.config`. They control JAX or device visibility directly. GPU allocation and memory controls are covered in the [single-GPU guide](../distributed/single-gpu.md), while device layouts and launch-time settings are collected in [parallelism configuration](../distributed/configuration.md).

Use the [JAX configuration reference](https://docs.jax.dev/en/latest/config_options.html) for additional JAX options. For neuraLQX settings, `describe_option` and `snapshot` let you check what the current process actually read before beginning a longer calculation.
