---
title: Custom networks and layers
---

# Custom networks and layers

A custom wavefunction can use tensor parallelism while retaining ordinary JAX or Flax code. Its call still accepts configurations with shape `(..., hilbert.size)` and returns one scalar log amplitude per configuration. The additional decision is how to divide its trainable arrays among the devices in a [tensor group](index.md).

If your model is composed from supplied layers, begin with their automatic rules. If you know how consecutive contractions should fit together, supply explicit layouts. A reusable architecture can also [register its own rule](../../developers/networks/tensor-rules.md). All three approaches preserve global shapes inside the model.

## Compose the supplied layers

The following NNX model uses [`Linear`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Linear), [`LayerNorm`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.LayerNorm), [`Activation`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Activation), and [`Sequential`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Sequential). Its last two real features represent log modulus and phase.

Run the Python blocks on this page in order. Use a single process with four devices and the launch settings `NEURALQX_SHARDING=samples,tensor` and `NEURALQX_MESH_TENSOR_AXIS_SIZE=2`. For a local CPU check, set `JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4` before starting Python.

```python
from flax import nnx
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from jax.sharding import PartitionSpec as P

class FeatureWavefunction(nnx.Module):
    def __init__(self, *, seed=17):
        rngs = nnx.Rngs(seed)
        options = dict(rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)
        self.body = nqx.nn.layers.Sequential(
            nqx.nn.layers.Linear(4, 32, **options),
            nqx.nn.layers.LayerNorm(32, **options),
            nqx.nn.layers.Activation("tanh"),
            nqx.nn.layers.Linear(32, 2, **options),
        )

    def __call__(self, configurations):
        values = self.body(configurations.astype(jnp.float64))
        return values[..., 0] + 1j * values[..., 1]

model = nqx.nn.init_tensor_parallel(lambda: FeatureWavefunction(seed=17))
plan = nqx.nn.tensor_parallel_plan(model)
for entry in plan.parameters:
    print(entry.path, entry.shape, entry.spec)

vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
graph = nqx.graph.Graph(
    [(vertices[i], vertices[(i + 1) % 4]) for i in range(4)],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=8)
state = nqx.vqs.VariationalState(
    sampler, model, n_samples=32, sampler_seed=23,
)
probe = jnp.array(
    [[0, 0, 0, 0], [1, -1, 0, 0], [0, 1, 2, -1], [-2, 1, 0, 1]],
    dtype=jnp.int32,
)
print(state.log_value(probe))
```

The sequential rule pairs the first affine layer's output partition with the last layer's input partition. The intervening activation has no trainable arrays. Layer normalization can partition its scale and bias, while its statistics still use all 32 global hidden features.

The native NNX layer classes are recognized through their class hierarchy, so you can mix neuraLQX wrappers with supported Flax NNX layers. An arbitrary custom `nnx.Param` has no inferred semantic role. Without a registered rule, explicit metadata, or an override, such a leaf stays replicated.

## What the layer rules choose

| Layer | Default placement decision |
| --- | --- |
| [`Linear`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Linear) | Prefers output features when the output is wider than two, otherwise input features. Uses an eligible divisible dimension. |
| [`Conv`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Conv) | Applies the affine rule to the final two kernel dimensions, which represent input and output channels. Spatial kernel dimensions remain complete. |
| [`Embed`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Embed) | Partitions the embedding feature dimension when divisible, keeping the category dimension complete. |
| [`LayerNorm`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.LayerNorm) and [`RMSNorm`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.RMSNorm) | Partition divisible feature scale and bias vectors. Reductions retain global feature semantics. |
| [`Sequential`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Sequential) | Alternates output and input partitions across its direct affine children. Small output heads prefer input partitions. |
| [`GraphConv`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.GraphConv) | Its constituent affine modules receive layer rules. A surrounding registered GNN architecture can provide more specific decisions. |

The generic sequential rule only sees its own direct affine children. It does not prove an optimal communication schedule for arbitrary residual connections, reshapes, or nested branches. Architecture-specific rules and explicit overrides let you express those relationships. Inspect the complete plan before relying on a particular weight being divided.

## Specify the contraction pair explicitly

[`TensorParallel`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.TensorParallel) attaches path-based layout rules to a model passed to the variational state. A path is relative to its trainable parameter tree, without an outer `params/` component. Container indices appear as ordinary slash-separated components.

For the model above, these rules make the intended contraction pair explicit:

```python
rules = {
    "body/layers/0/kernel": P(None, "tensor"),
    "body/layers/0/bias": P("tensor"),
    "body/layers/1/scale": P("tensor"),
    "body/layers/1/bias": P("tensor"),
    "body/layers/3/kernel": P("tensor", None),
    "body/layers/3/bias": P(),
}
explicit_model = nqx.nn.init_tensor_parallel(
    lambda: FeatureWavefunction(seed=17), parameter_rules=rules,
)
explicit_state = nqx.vqs.VariationalState(
    sampler,
    nqx.nn.TensorParallel(explicit_model, parameter_rules=rules),
    n_samples=32, sampler_seed=23,
)
np.testing.assert_allclose(
    explicit_state.log_value(probe), state.log_value(probe),
    rtol=1e-11, atol=1e-11,
)
```

Passing the rules to initialization avoids allocating a differently placed model first. Passing them through the adapter makes the execution policy explicit at the variational-state boundary. The initialization helper also retains its supplied overrides on the returned model.

Rules accept exact paths or glob patterns such as `"body/layers/*/kernel"`. When patterns overlap, the last matching entry in the effective rule sequence determines the layout. Each explicitly supplied pattern must match at least one trainable leaf. A misspelled path is an error rather than a silently unused instruction.

Use `P()` to replicate an array. Otherwise, the supported weight plan names the tensor axis in **one array dimension**. It cannot use the sample or operator axes to divide model weights, or place the tensor axis in two dimensions of the same array. The selected dimension must divide the tensor group size. The literal `"tensor"` in an adapter rule is translated to the configured tensor-axis name.

## Put partition metadata on a layer

The layer constructors also support explicit parameter metadata. For example, `Linear` accepts `kernel_axes` and `bias_axes`, `Conv` has the same options, and `Embed` accepts `embedding_axes`. Metadata overrides their default architecture or layer rules.

This complete pair uses the active mesh while constructing the annotated weights:

```python
from neuralqx.jax.tensor_parallel import tensor_axis_name, tensor_mesh

class AnnotatedWavefunction(nnx.Module):
    def __init__(self):
        axis = tensor_axis_name()
        rngs = nnx.Rngs(9)
        self.hidden = nqx.nn.layers.Linear(
            4, 32, rngs=rngs,
            kernel_axes=(None, axis), bias_axes=(axis,),
            dtype=jnp.float64, param_dtype=jnp.float64,
        )
        self.head = nqx.nn.layers.Linear(
            32, 1, rngs=rngs,
            kernel_axes=(axis, None), bias_axes=(None,),
            dtype=jnp.float64, param_dtype=jnp.float64,
        )

    def __call__(self, x):
        return self.head(jnp.tanh(self.hidden(x)))[..., 0]

with jax.set_mesh(tensor_mesh()):
    annotated = nqx.nn.init_tensor_parallel(AnnotatedWavefunction)
for entry in nqx.nn.tensor_parallel_plan(annotated).parameters:
    print(entry.path, entry.spec)
```

Unlike an adapter rule's logical `"tensor"` spelling, explicit Flax metadata describes the actual mesh names used during initialization. Obtaining the name with `tensor_axis_name()` keeps it consistent with renamed meshes. The [Flax multi-device guide](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html) explains eager variable sharding and mesh context.

An annotation describes an array layout. It does not enable tensor parallelism by itself. Set the neuraLQX startup policy as well, so model evaluation, sampling, and gradients use the coordinated execution path. Use adapter rules when you want to keep execution policy separate from a portable model definition and its metadata.

## Linen and pure JAX models

For an arbitrary Linen module, automatic tensor mode chooses a divisible dimension of eligible arrays by shape. It prefers a final dimension larger than two, then other dimensions, with a different preference for small output heads. This is a fallback, not knowledge of the architecture's intended contractions. Supply explicit rules when the choice matters.

The following Linen model uses named submodules so its parameter paths are predictable. It reuses the Hilbert space, sampler, and probe defined earlier:

```python
from flax import linen as linen

class LinenWavefunction(linen.Module):
    @linen.compact
    def __call__(self, x):
        options = dict(dtype=jnp.float64, param_dtype=jnp.float64)
        hidden = jnp.tanh(linen.Dense(32, name="hidden", **options)(x))
        return linen.Dense(1, name="head", **options)(hidden)[..., 0]

linen_state = nqx.vqs.VariationalState(
    sampler,
    nqx.nn.TensorParallel(
        LinenWavefunction(),
        parameter_rules={
            "hidden/kernel": P(None, "tensor"),
            "hidden/bias": P("tensor"),
            "head/kernel": P("tensor", None),
            "head/bias": P(),
        },
    ),
    n_samples=32, seed=11, sampler_seed=23,
)
assert linen_state.log_value(probe).shape == (4,)
```

A pure JAX callable can use the same adapter. Its initializer accepts `(key, input_shape, dtype)`, where `dtype` describes the configuration input. Choose a floating or complex dtype explicitly for trainable weights:

```python
def initialize(key, input_shape, dtype):
    first, second = jax.random.split(key)
    return {
        "w": 0.1 * jax.random.normal(
            first, (input_shape[-1], 32), dtype=jnp.float64,
        ),
        "b": jnp.zeros(32, dtype=jnp.float64),
        "v": 0.1 * jax.random.normal(second, (32,), dtype=jnp.float64),
    }

def log_amplitude(parameters, x):
    return jnp.tanh(x @ parameters["w"] + parameters["b"]) @ parameters["v"]

callable_state = nqx.vqs.VariationalState(
    sampler,
    nqx.nn.TensorParallel(
        log_amplitude,
        parameter_rules={
            "w": P(None, "tensor"), "b": P("tensor"), "v": P("tensor"),
        },
        init_fn=initialize,
    ),
    n_samples=32, seed=11, sampler_seed=23,
)
assert callable_state.log_value(probe).shape == (4,)
```

If the callable expects a complete variable mapping rather than just the parameter tree, set `expects_variables=True` on the adapter. These conventions match the ordinary [custom model interface](../../concepts/networks/custom-models.md). With tensor mode disabled, the adapter unwraps to the ordinary framework and does not independently activate distributed execution.

## Preserve the scientific model

Do not divide a layer's declared width by the number of tensor workers. The model should describe the full wavefunction. Layouts describe which pieces of its arrays live on each device.

Normalization must also retain its full mathematical domain. For real features $h_j$ with width $H$, layer normalization uses

$$
\mu=\frac{1}{H}\sum_{j=0}^{H-1}h_j,
\qquad
\sigma^2=\frac{1}{H}\sum_{j=0}^{H-1}(h_j-\mu)^2.
$$

If hidden features occupy different devices, the global reductions include every feature. Normalizing each local block independently would define a different network. The supplied normalization layers express the global operation, allowing JAX to insert communication. Normalizing across configurations would instead make one amplitude depend on the other batch entries and violate the [wavefunction contract](../../developers/networks/contracts.md).

The same caution applies to softmax, pooling, attention, and reductions you write yourself. Express them on global array dimensions. Do not insert a tensor-axis `psum` into an ordinary global model call to compensate for a sum that JAX already implements.

## Compare values and derivatives

For this small model, we can compare against an ordinary NNX model initialized with the same seed. This is a single-process check, where the complete small arrays are addressable:

```python
from neuralqx.utils.frameworks.flax import FlaxNNXFramework

reference, reference_variables = FlaxNNXFramework.from_model(
    FeatureWavefunction(seed=17)
)
reference_parameters, reference_fixed = reference.split_variables(reference_variables)
fixed = state.model_state

def distributed_loss(parameters):
    variables = state.framework.merge_variables(parameters, fixed)
    values = state.framework.apply_variables(variables, probe)
    return jnp.mean(jnp.abs(values) ** 2)

def reference_loss(parameters):
    variables = reference.merge_variables(parameters, reference_fixed)
    values = reference.apply_variables(variables, probe)
    return jnp.mean(jnp.abs(values) ** 2)

np.testing.assert_allclose(
    state.log_value(probe), reference.apply_variables(reference_variables, probe),
    rtol=1e-11, atol=1e-11,
)
distributed_gradient = jax.jit(jax.grad(distributed_loss))(state.parameters)
reference_gradient = jax.grad(reference_loss)(reference_parameters)
for actual, expected in zip(
    jax.tree.leaves(distributed_gradient),
    jax.tree.leaves(reference_gradient),
    strict=True,
):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
print("Log amplitudes and test-loss gradients agree")
```

This verifies the network's numerical function and its derivative for a fixed probe. The test loss is intentionally simple. For a new scientific model, also exercise [expectations and VMC gradients](../../concepts/vqs/expectations.md) on a small tractable problem, including any projector or connected-state reuse it provides. Comparing the same input configurations avoids confusing layout errors with Monte Carlo fluctuations.

When a model will be reused, continue with [tensor partition rules](../../developers/networks/tensor-rules.md) to put its layout knowledge in a registered architecture rule.
