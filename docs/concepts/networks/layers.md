---
title: Layers and activations
---

# Layers and activations

The built-in wavefunctions are composed from reusable layers in [`neuralqx.nn.layers`](../../api/public/neuralqx/nn/layers/index.md) and functions in [`neuralqx.nn.activations`](../../api/public/neuralqx/nn/activations/index.md). You can use these pieces inside your own model, combine them with native Flax layers, or call the activation functions from pure JAX code.

The wrappers retain the underlying JAX and Flax operations while adding consistent initialisation shortcuts, optional tensor partition annotations, and capabilities used by exact connected-state evaluation. They do not automatically turn an arbitrary composition into a scalar log wavefunction.

## Available layers

| Layer | Action and input convention |
| --- | --- |
| [`Linear`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Linear), also named [`Dense`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Dense) | Applies $xW+b$ to the final feature axis and preserves leading axes. |
| [`Conv`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Conv) | Convolves arrays with spatial axes followed by a channel axis. Circular padding supports periodic boundaries. |
| [`Embed`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Embed) | Looks up a feature vector for each integer category index. |
| [`GraphConv`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.GraphConv) | Combines each node's features with its mean incoming-neighbour features. |
| [`LayerNorm`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.LayerNorm) | Normalises selected feature axes within a configuration, with optional learned scale and bias. |
| [`RMSNorm`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.RMSNorm) | Uses the feature root mean square without subtracting the mean. |
| [`Activation`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Activation) | Wraps a registered activation or callable as an NNX layer, with fixed keyword options. |
| [`Sequential`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Sequential) | Applies modules and functions in order, retaining access to its constituent layers. |

Native options such as initialisers, bias selection, dtypes, and contraction precision are accepted where supported by the underlying layer. `Linear` and `Conv` also accept `kernel_axes` and `bias_axes`, while `Embed` accepts `embedding_axes`. These annotate weight dimensions and override automatic tensor rules when tensor parallelism is enabled. Supported layers also have defaults, so an ordinary composition can participate without explicit annotations. See [custom networks with tensor parallelism](../../distributed/tensor-parallelism/custom-models.md) for a complete model, column and row layouts, initialization, and gradient checks. An annotation alone does not enable the execution policy.

## Compose features and a log-amplitude head

Use one shared RNG stream when constructing a sequence of layers. Each layer then receives its own initialisation keys.

```python
from flax import nnx
import jax.numpy as jnp
import neuralqx as nqx

rngs = nnx.Rngs(0)
layers = nqx.nn.layers.Sequential(
    nqx.nn.layers.Linear(4, 8, rngs=rngs),
    nqx.nn.layers.Activation("tanh"),
    nqx.nn.layers.Linear(8, 2, rngs=rngs),
)
configurations = jnp.zeros((3, 4))
head = layers(configurations)
log_amplitudes = head[..., 0] + 1j * head[..., 1]
print(head.shape)
print(log_amplitudes.shape)
```

```text
(3, 2)
(3,)
```

The sequential object here returns two features. The final combination supplies the scalar log amplitude required by a wavefunction. The [custom model guide](custom-models.md) places that combination inside a complete model class.

The [`layers.linen`](../../api/public/neuralqx/nn/layers/linen/index.md) namespace exposes Linen constructors for individual layers. The general [`to_linen`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.to_linen) bridge can also convert an NNX class. Linen initialisation supplies the random key through `init`, so its layer constructors omit `rngs`.

## Embeddings use category indices

`Embed` expects indices from zero to `num_embeddings - 1`. A physical charge or spin label is not automatically such an index. For a contiguous integer charge range $-J,\ldots,J$, the mapping $q\mapsto q+J$ is sufficient.

```python
import jax.numpy as jnp
import neuralqx as nqx

cutoff = 2
embedding = nqx.nn.layers.Embed(2 * cutoff + 1, 4, rngs=2)
charges = jnp.array([-2, 0, 2], dtype=jnp.int32)
indices = charges + cutoff
print(embedding(indices).shape)
```

```text
(3, 4)
```

For an irregular alphabet, construct an explicit label-to-index map. The [autoregressive model](built-in/autoregressive.md) performs that mapping internally for its supplied `local_states`.

## Select an activation

Model constructors accept a registered name such as `"tanh"` or a callable. [`get_activation`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.get_activation) resolves a name once during construction. [`available_activations`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.available_activations) returns the full registry, and [`activation_spec`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.activation_spec) exposes a function's declared properties.

The supplied functions include real activations such as [`relu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.relu), [`gelu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.gelu), [`selu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.selu), and [`softplus`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.softplus). Native complex implementations include [`tanh`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.tanh), [`sigmoid`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.sigmoid), [`silu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.silu), and the stable [`logcosh`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.logcosh).

Registration records two different properties. `supports_complex` describes the input domain. `elementwise` declares whether each output element depends only on its corresponding input element. The latter matters for local CNN and GNN updates. These properties do not assert that a function is holomorphic.

Real-only activations reject complex inputs. [`reim`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.reim) explicitly extends a real function as

$$
g(x+iy)=f(x)+i f(y).
$$

Registered shortcuts include [`reim_relu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.reim_relu), [`reim_gelu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.reim_gelu), and [`reim_tanh`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.reim_tanh). A model can use `activation="reim_gelu"` or `activation=nqx.nn.reim("gelu", approximate=False)`. This is a deliberate componentwise extension. Its derivatives are generally non-holomorphic, as explained in the [complex guide](complex.md).

Not every registered function preserves feature width. For example, [`glu`](../../api/public/neuralqx/nn/activations/index.md#neuralqx.nn.activations.glu) splits an axis and changes its size. The supplied hidden-layer architectures expect their configured widths to remain consistent. Use a custom architecture when an activation changes those dimensions.

## Register an activation of your own

A custom activation must use JAX-compatible operations and remain pure. The following bounded real function can be selected by its registered name.

```python
import jax.numpy as jnp
import neuralqx as nqx

@nqx.nn.activation(
    name="scaled_tanh_example",
    elementwise=True,
    supports_complex=False,
)
def scaled_tanh(x):
    return 1.5 * jnp.tanh(x / 1.5)

network = nqx.nn.MLP(
    4, hidden_features=(8,),
    activation="scaled_tanh_example", rngs=3
)
print(network(jnp.ones((2, 4))).shape)
print(nqx.nn.activations.activation_spec(scaled_tanh).elementwise)
```

```text
(2,)
True
```

A new registration defaults to real-only and nonlocal unless those properties are declared. An unregistered callable is also accepted, with conservative locality and an unknown complex domain. Declare `elementwise=True` only when it is true, since an incorrect declaration can invalidate reuse of supposedly unchanged features.

Names are registered in the current Python process. Replacing an existing name requires `replace=True`, and models already constructed retain the callable they resolved earlier.

## Normalisation and physical meaning

Normalising features can improve the behaviour of a custom architecture, but it must leave one configuration's amplitude independent of other batch entries. The supplied normalisation layers use no running batch statistics. Keep their reduction axes within the features of each configuration.

Normalising across spatial positions couples those positions and changes locality assumptions. Norms involving a complex modulus also introduce dependence on conjugated parameters and generally break holomorphicity. Neither effect makes the architecture invalid, but both affect which [evaluation and differentiation choices](complex.md) are appropriate.

Finally, layer names do not confer physical symmetries. A `GraphConv` implements a particular message operation, and a periodic `Conv` shares kernels on a chosen lattice. The complete wavefunction, its input interpretation, and any [projector](symmetries.md) determine the resulting variational family.
