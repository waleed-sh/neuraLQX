---
title: Built-in networks and initialization
---

# Built-in networks and initialization

The built-in networks supply architecture-specific rules for tensor placement. You enable tensor parallelism through the [startup configuration](../configuration.md#sharding-policy), then construct a network and variational state in the usual way. The rules preserve the network's global shapes and numerical definition. They select which weight dimension each tensor worker owns.

This page assumes you have read [what tensor parallelism does](index.md). The model guides under [neural networks](../../concepts/networks/models.md) explain the scientific form of each wavefunction.

## Inspect a plan before allocating a large network

[`tensor_parallel_plan`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.tensor_parallel_plan) accepts an initialized NNX model or an abstract model produced by `nnx.eval_shape`. An abstract model contains shapes and dtypes without allocating the full weight arrays.

Run the following in a fresh process with `NEURALQX_SHARDING=samples,tensor`, `NEURALQX_MESH_TENSOR_AXIS_SIZE=2`, and four visible devices. On a CPU, also set `JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4`.

```python
from flax import nnx
import jax
import jax.numpy as jnp
import neuralqx as nqx
from neuralqx.jax.tensor_parallel import tensor_mesh

def factory():
    return nqx.nn.MLP(
        4, hidden_features=(32, 32),
        complex_output=False,
        dtype=jnp.float64, param_dtype=jnp.float64, rngs=17,
    )

abstract_model = nnx.eval_shape(factory)
plan = nqx.nn.tensor_parallel_plan(abstract_model)
print("Mesh:", dict(tensor_mesh().shape))
print("Tensor workers:", plan.partitions)
for entry in plan.parameters:
    print(entry.path, entry.shape, entry.spec)
print("Partitioned scalar weights:", plan.sharded_parameter_count)
```

For this two-worker plan, the important choices are:

| Weight path | Global shape | Specification | Shape on one tensor worker |
| --- | --- | --- | --- |
| `hidden/0/kernel` | `(4, 32)` | `P(None, "tensor")` | `(4, 16)` |
| `hidden/0/bias` | `(32,)` | `P("tensor")` | `(16,)` |
| `hidden/1/kernel` | `(32, 32)` | `P("tensor", None)` | `(16, 32)` |
| `hidden/1/bias` | `(32,)` | `P()` | `(32,)` |
| `head/kernel` | `(32, 1)` | `P()` | `(32, 1)` |
| `head/bias` | `(1,)` | `P()` | `(1,)` |

The first two layers form the output/input partition pairing described in the [contraction example](index.md#splitting-a-dense-layer). The small trailing head is replicated. `plan.sharded_parameter_count` is the number of scalar weights in partitioned arrays, not the number of bytes saved or the number of leaves in the parameter tree.

The returned [`TensorParallelPlan`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.TensorParallelPlan) also exposes a `specs` pytree and `place(parameters)` for a matching parameter tree. Inspecting a plan does not place the model or evaluate it. The global [tensor mesh](../../api/public/neuralqx/jax/tensor_parallel/index.md#neuralqx.jax.tensor_parallel.tensor_mesh) depends on the actual visible devices and configuration, so use the same launch environment as the intended calculation.

## Initialize directly into shards

An ordinary initialized NNX model can be passed directly to [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState). With tensor parallelism enabled, the framework places its trainable arrays according to the selected plan. That route is convenient when the unpartitioned initialization fits.

For a larger model, construct it with [`init_tensor_parallel`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.init_tensor_parallel):

```python
model = nqx.nn.init_tensor_parallel(factory)
for entry, value in zip(
    plan.parameters, jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True,
):
    print(entry.path, value.shape, value.sharding)
    for shard in value.addressable_shards:
        print(" ", shard.device, shard.index, shard.data.shape)
```

This continues the preceding example. The helper traces the factory for shapes, chooses the layouts, and compiles initialization with those output shardings. It avoids first materializing each complete parameter array on a single device. Non-trainable model arrays are initialized with replicated placement.

The factory must return a fresh model from a fixed seed or key. It is traced twice, so create `nnx.Rngs` inside the factory if you use explicit streams. Do not capture and advance one external RNG stream, draw host random values on each call, or use side effects to decide the architecture. Integer seeds accepted by neuraLQX constructors are convenient for this purpose.

When tensor parallelism is disabled, this helper simply calls the factory. The [common simulation](../cpu.md) therefore uses the same construction on a CPU, a single GPU, and a tensor mesh.

Linen modules and callable initializers supplied to the variational state already use shape-first initialization in tensor mode. They do not need the NNX factory helper. Existing variables supplied explicitly are placed after they are received, so supplying complete arrays still incurs the cost of creating those arrays.

## Layouts supplied for each architecture

These are default rules, subject to exact divisibility. The actual plan is the authoritative description for a particular model and tensor group size.

| Architecture | Tensor layout |
| --- | --- |
| [`MLP`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.MLP) | Alternates column-parallel and row-parallel layouts through its affine sequence. A trailing unpaired head is replicated. |
| [`RBM`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.RBM) | Uses a column-parallel hidden affine map. The visible contribution is replicated. The global hidden-unit sum retains the full wavefunction definition. |
| [`CNN`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.CNN) | Partitions the first convolution's output channels, then prefers input-channel partitions in subsequent convolutions and row-parallel readout and head projections. |
| [`GNN`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.GNN) | Partitions encoder output features and divisible node embeddings. Message maps, readout, and head prefer row-parallel layouts. This does not partition the physical graph between tensor workers. |
| [`Jastrow`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.Jastrow) | Partitions the first input-coordinate dimension of coupling and linear arrays when that dimension divides the tensor group size. The full quadratic expression remains a global contraction. |
| [`AutoregressiveRNN`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.AutoregressiveRNN) | Prefers row-parallel recurrent, probability, and optional phase projections, with the embedding layer partitioned across divisible feature dimensions. Sequential conditional sampling retains its order. Tensor workers cooperate on each conditional evaluation. |

The native [`models.linen`](../../api/public/neuralqx/nn/models/linen/index.md) facades recover these same architecture rules through their NNX definitions. Arbitrary Linen models use a general shape-based rule unless you supply [explicit parameter layouts](custom-models.md#linen-and-pure-jax-models).

For an affine map whose preferred dimension is not divisible, the planner can choose a different sufficiently large divisible input or output dimension. If none is suitable, that map is replicated. A bias follows an output-feature partition but remains replicated for an input-feature partition. These choices preserve every feature. They can change the communication pattern, so inspect the resulting plan rather than inferring it only from the model name.

## Connected evaluations and gradients

The tensor framework applies the same weight plan to ordinary model calls and to [connected-state evaluation](../../developers/networks/connections.md). A native network's exact reuse of unchanged features remains an evaluation strategy for the same wavefunction. It does not create a second independently partitioned model.

The variational state owns the arrays used by subsequent sampling and differentiation. Inspect `state.parameters` after constructing it, especially when you began with an ordinary initialized model. Editing the original NNX object afterwards does not update the state's variables. The [framework guide](../../developers/networks/parameters.md#ownership-and-updates) explains that ownership.

Optax transformations operate on the global parameter and gradient trees. Tensor parallelism does not require a new learning-rate scaling rule. Algorithms that materialize large additional arrays, such as a dense quantum geometric tensor, still need their own memory analysis. Weight partitioning alone is not a guarantee that every [SR representation](../../concepts/optimisers/sr.md) or solver workspace is divided in the same way.

## Save and restore a calculation

Use the ordinary [state saving interface](../../concepts/vqs/saving.md). Automatic tensor placement is an execution policy, separate from the stored scientific model and its global parameter values. Restoration uses the target model and current configuration to select layouts. A saved global weight matrix does not become a smaller mathematical matrix because its source run used several devices.

For a custom model, preserve its importable definition and reconstruction information as described in the saving guide. Reapply explicit layout rules on the target model when they are required. In a multi-process run, all processes must participate in checkpoint operations, even when only process zero prints a completion message.

After restoring with a different layout, compare log amplitudes on a fixed set of configurations and inspect the restored parameter shardings. Floating-point reduction order can change, so use a precision-appropriate tolerance. Continue with [custom networks and layers](custom-models.md) to define your own partitioned architecture.
