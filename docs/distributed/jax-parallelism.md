---
title: JAX parallelism
---

# JAX parallelism

Parallel execution changes where the arrays and numerical work live. A calculation still consists of a graph, a Hilbert space, a wavefunction, a sampler, and an objective. JAX supplies the machinery that places these arrays on devices and communicates between them. neuraLQX uses that machinery to distribute chains, evaluate operator terms, and combine the resulting estimates and gradients.

This page develops the JAX concepts needed to understand those choices. For a wider treatment of training loops and device placement, see the [JAX Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html), especially its discussion of meshes and shardings. The following page explains [which sharding strategies neuraLQX implements](sharding.md). The hardware guides then carry [one calculation](cpu.md#the-calculation-used-throughout-these-guides) from a CPU to multiple GPU nodes.

## Hosts, processes, and devices

A **host** is a machine on which Python runs. A **process** is one running Python interpreter. A **device** is a numerical execution resource known to JAX, such as a GPU or a CPU device. One process can control several devices. On a cluster, several processes cooperate in the same calculation, with each process controlling its own local devices.

These counts answer different questions. `jax.process_count()` describes how many Python processes joined the calculation. `jax.local_device_count()` counts the devices controlled by the current process. `jax.device_count()` counts all devices in the initialized JAX computation. A workstation with four GPUs can run one process controlling four devices. A two-node job with four GPUs per node can run two processes controlling eight devices in total.

![Two hosts each run one Python process and control four local GPUs. Both processes share a view of eight global devices.](../assets/images/distributed/processes-and-devices.svg)

*A possible two-node layout. Device numbers identify positions in the illustration, rather than prescribing how a backend will number them.*

The cluster arrangement needs a startup step, described in [multiple nodes and GPUs](multi-node.md). Merely launching the same script on two machines creates two independent calculations unless the processes join the same distributed runtime. JAX's [multi-controller guide](https://docs.jax.dev/en/latest/multi_process.html) explains the process model in more detail.

## One array can occupy several devices

A JAX array has a global shape and a placement. For example, an array of configurations with shape `(8, 4)` contains eight configurations, each with four labels. We can put the whole array on one device, replicate it on several devices, or split the eight rows among devices. Its global shape remains `(8, 4)` in all three cases.

The piece stored on one device is a **shard**. With four equal partitions of the first dimension, each shard has shape `(2, 4)`. Every shard contains complete configurations. This is different from dividing the four labels of each configuration among devices, which most neuraLQX operator kernels do not support.

![An eight-row configuration array is divided into four two-row shards. Each device also holds a complete copy of the model parameters.](../assets/images/distributed/sample-sharding.svg)

*The sample axis is partitioned, while each configuration and each parameter array remain complete on a participating device. The global array is the collection of these shards.*

Replication is an intentional placement. A small parameter array may be needed by every device evaluating a different subset of samples. Replicating it avoids fetching a remote parameter for each model evaluation, but it also means that adding devices does not divide the parameter memory requirement.

## Meshes and partition specifications

A [`Mesh`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh) gives names to axes of an array of devices. A one-dimensional mesh might have an axis called `samples` with size four. A two-dimensional mesh might have `operators` of size four and `samples` of size two, using eight devices in total. The names describe how we intend to divide work. They are not physical GPU features.

A [`PartitionSpec`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec) maps array dimensions to these named mesh axes. `NamedSharding` combines the mesh and the specification.

| Specification for an array with shape `(batch, width)` | Placement |
| --- | --- |
| `P("samples", None)` | Divide rows over `samples`, keeping each row complete |
| `P(None, "features")` | Divide columns over a mesh axis named `features` |
| `P()` | Replicate the complete array across the mesh |
| `P("operators", "samples")` | Divide the first dimension over operators and the second over samples |

An unused mesh axis means replication along that axis. For example, on a mesh containing both `operators` and `samples`, `P("samples", None)` distributes rows over the sample axis and repeats each row shard for all operator partitions. Equal partitions require the divided dimension to be divisible by the corresponding mesh-axis size.

The following complete example uses a single Python process and any number of visible devices. It creates two configurations per device, places a small weight matrix on every device, and evaluates the same matrix operation on the distributed batch.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices()
mesh = Mesh(np.asarray(devices), ("samples",))
host_samples = np.arange(8 * len(devices), dtype=np.float32).reshape(-1, 4)
samples = jax.device_put(host_samples, NamedSharding(mesh, P("samples", None)))
weights = jax.device_put(
    np.full((4, 2), 0.01, dtype=np.float32), NamedSharding(mesh, P())
)

@jax.jit
def evaluate(batch, parameters):
    return jnp.tanh(batch @ parameters)

outputs = evaluate(samples, weights)
outputs.block_until_ready()
print("Global input:", samples.shape)
print("Global output:", outputs.shape)
for shard in samples.addressable_shards:
    print(shard.device, shard.index, shard.data.shape)
np.testing.assert_allclose(
    np.asarray(outputs), np.tanh(host_samples @ np.asarray(weights)), rtol=1e-6
)
```

Save the example to a file and start it with `JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python filename.py` to reproduce four partitions on a CPU. With four devices, the global input is `(8, 4)` and every input shard is `(2, 4)`. The example uses a host NumPy array containing the full input. In a large multi-process calculation, each process should ordinarily construct only the data it can address, as explained in [distributed array handling](multi-node.md#global-arrays-and-process-local-inspection).

## Compilation, vectorisation, and device parallelism

[`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) compiles a numerical function for the input shapes, dtypes, and execution configuration. Compilation does not mean that every available device is automatically useful to the computation. Array placement and the compiled program determine where the work runs.

`jax.vmap` adds a batch dimension to a function. It is useful when expressing the same operation over many configurations, including on one device. A vectorised batch can also be sharded, but vectorisation alone does not select a multi-device placement. Similarly, a Python loop over devices is not what makes the example above parallel. JAX evaluates the global array operation using its device placement.

JAX supports several levels of control. In automatic sharding mode, the compiler propagates placements and chooses internal layouts around constraints. In explicit sharding mode, sharding information is part of traced array types, and ambiguous operations may require an output annotation. Both use global array operations. With [`jax.shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html), the function instead sees a local block, and the author specifies communication when it is needed. The [JAX parallelism guide](https://docs.jax.dev/en/latest/parallel.html) develops these modes and how they compose.

neuraLQX combines global arrays with explicit per-device kernels where useful. Its sampling path, for example, evolves local chains inside a mapped computation. A user of the standard VQS and driver interfaces does not need to write these mapped kernels.

## Communication preserves the global calculation

Suppose device $d$ owns $n_d$ local estimator values, with local sum $s_d$. The overall mean is

$$
\overline A=\frac{\sum_d s_d}{\sum_d n_d}.
$$

If all $n_d$ are equal, averaging device means gives the same result. If the counts differ, an unweighted mean of device means is wrong. This is why global shapes, valid sample counts, and reduction semantics matter as much as distributing the expensive model evaluations.

A **collective** is an operation in which devices exchange information. A sum reduction combines contributions, an all-gather assembles pieces, and a redistribution moves an array between layouts. For sample parallel VMC, local model evaluations can proceed independently, while expectations and parameter gradients need global information. For operator parallel evaluation, contributions from different terms must be combined for each sample.

More devices can therefore reduce local work while introducing communication and additional compilation. The useful balance depends on the network size, number of samples, operator connectivity, and device interconnect. A small example may run faster on one device. Use the [profiling guides](../developers/profiling/for-scientists.md) to determine what dominates a real calculation.

## Global shape and addressable shards

`array.shape` describes the whole array. `array.addressable_shards` lists only the shards controlled by the current process. On a single host process these may include the entire array. Across hosts, some shards belong to another process and cannot be read directly as local NumPy data.

The standard simulation prints each addressable shard's index and shape without gathering its contents. This is a practical first check of placement. All processes must still execute collective numerical operations in the same order. Restrict ordinary printing to process zero when useful, but do not put a collective expectation or checkpoint inside that condition. The [multiple-node guide](multi-node.md#collective-control-flow-and-output) shows the distinction.

Continue with [sharding in neuraLQX](sharding.md) for the library's supported layouts, or with [CPU installation and the shared simulation](cpu.md) to start running the example.
