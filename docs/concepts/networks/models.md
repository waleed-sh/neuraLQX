---
title: Built-in networks
---

# Built-in networks

The supplied networks provide complete log wavefunctions with consistent input conventions, parameter controls, and access through NNX or Linen. They are useful starting points for a calculation and examples of how to compose the [neuraLQX layers and activations](layers.md).

neuraLQX builds these components on JAX and Flax. Many layers extend native Flax layers, while registered activations use JAX operations. The additional interfaces address the needs of variational simulations, particularly scalar log amplitudes, explicit complex arithmetic, fixed graph data, and repeated evaluation of configurations connected by an operator.

## Why supply networks, layers, and activations together

An operator often connects a sampled configuration to many others that differ in only a few coordinates. Evaluating a complete network independently on every target can repeat work. Several supplied architectures expose exact ways to reuse source features. For example, a dense first layer satisfies

$$
(x+\delta)W+b=(xW+b)+\delta W.
$$

The network can update that layer from the changed coordinates, then evaluate the remaining layers normally. This is an algebraic identity. It does not approximate the target wavefunction or drop operator connections.

Supporting this consistently requires more than a collection of model constructors. The linear layers expose sparse updates. Activations record whether they act independently on each element and whether they accept complex inputs. Convolutional and graph models can then decide whether a local change really leaves other features unchanged. The supplied layers also accept explicit dtypes, contraction precision, initialisers, and optional parameter partition annotations.

These interfaces do not make a custom network compulsory or establish that one architecture is fastest for every problem. You can use ordinary Flax components or [your own JAX model](custom-models.md). The built-ins provide the additional evaluation capabilities where their structure permits them.

## Shortcuts and namespaces

After `import neuralqx as nqx`, all six models are available directly as `nqx.nn.MLP`, `nqx.nn.RBM`, and the other names below. They are the same constructors exported by `nqx.nn.models`.

| Model | Structure and intended use | Detailed guide |
| --- | --- | --- |
| [`MLP`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.MLP) | Dense nonlinear layers over the full label vector, without an imposed spatial symmetry | [Multilayer perceptrons](built-in/mlp.md) |
| [`RBM`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.RBM) | Visible linear terms and a sum of hidden log-cosh contributions | [Restricted Boltzmann machines](built-in/rbm.md) |
| [`Jastrow`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.Jastrow) | Linear and pairwise contributions to the log amplitude | [Jastrow wavefunctions](built-in/jastrow.md) |
| [`CNN`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.CNN) | Periodic convolutions and spatial mean pooling on a specified lattice | [Convolutional networks](built-in/cnn.md) |
| [`GNN`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.GNN) | Message passing on a specified network of input nodes | [Graph networks](built-in/gnn.md) |
| [`AutoregressiveRNN`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.AutoregressiveRNN) | Normalised categorical conditionals in a fixed coordinate order | [Autoregressive networks](built-in/autoregressive.md) |

Layers live in [`nqx.nn.layers`](../../api/public/neuralqx/nn/layers/index.md). [`Dense`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Dense) is an alias for [`Linear`](../../api/public/neuralqx/nn/layers/index.md#neuralqx.nn.layers.Linear). Activation names such as `"tanh"` and `"silu"` are resolved when constructing a model. [`nqx.nn.get_activation`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.get_activation), [`nqx.nn.activation`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.activation), and [`nqx.nn.reim`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.reim) provide short access to lookup, custom registration, and explicit real/imaginary activation.

```python
import jax.numpy as jnp
import neuralqx as nqx

assert nqx.nn.MLP is nqx.nn.models.MLP
assert nqx.nn.layers.Dense is nqx.nn.layers.Linear
network = nqx.nn.MLP(6, hidden_features=(16,), activation="tanh", rngs=0)
print(network(jnp.zeros((3, 6))).shape)
```

```text
(3,)
```

## NNX and Linen use the same implementations

NNX models are initialised objects. Their constructors accept an integer seed, a JAX key, or a shared `flax.nnx.Rngs` stream through `rngs`. Sharing a stream across layers consumes successive keys. Repeatedly passing the same integer seed to independently constructed, identically shaped layers repeats their initialisation.

The [`nqx.nn.models.linen`](../../api/public/neuralqx/nn/models/linen/index.md) namespace supplies Linen constructors for the same six models. Omit `rngs` there and initialise with `model.init(key, states)`.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

network = nqx.nn.models.linen.MLP(
    6, hidden_features=(16,), complex_output=False
)
configurations = jnp.zeros((3, 6))
variables = network.init(jax.random.key(0), configurations)
print(network.apply(variables, configurations).shape)
```

```text
(3,)
```

[`to_linen`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.to_linen) exposes the same bridge for a custom NNX class or individual layer. It takes the class and constructor arguments, not an already initialised instance. The bridge constructs a Linen model with its own initialisation lifecycle. The [custom model guide](custom-models.md) shows how to use these layers within a Linen module.

## Output channels and precision

By default, the supplied models use real `float32` parameters and produce a complex log amplitude. For the dense, RBM, Jastrow, CNN, and GNN families, two output channels are combined as

$$
f_\theta(x)=h_0(x)+i h_1(x).
$$

With real computation, the first channel is the log modulus and the second is the phase. `complex_output=False` selects one channel. It guarantees a real output only when that channel's computation is real.

These five families also accept complex parameters. Set both `param_dtype` and a compatible `dtype`, or use `dtype=None` to follow the parameter dtype. The models reject complex parameter storage combined with a real computation dtype. The autoregressive model requires real parameters and real probability logits, even when it adds a complex phase.

Parameter storage and computation precision are separate settings. The built-in default remains `float32` even when JAX permits 64-bit arrays globally. Choose `dtype=jnp.float64, param_dtype=jnp.float64` explicitly for a real 64-bit calculation. See [complex and holomorphic models](complex.md) for examples and the differentiation implications.

## Exact reuse during operator evaluation

[`ConnectionPolicy`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.ConnectionPolicy) controls whether a supported model attempts exact connected-state reuse. Its default mode is `"auto"`. `"off"` uses ordinary evaluation, while `"on"` requests reuse where the architecture supports it. A request does not create a local update for a model that lacks one.

The MLP reuses its first affine layer. The RBM updates hidden fields and visible terms. Jastrow uses an exact quadratic difference. The CNN can recompute a local receptive field on a sufficiently long one-dimensional lattice. The GNN can update a bounded message dependency region. The autoregressive model can reuse a causal prefix and recompute the remaining suffix.

Automatic decisions include architecture-specific size checks and the relative number of sources and targets. Sparse-update limits are resource choices. If a chunk changes too many coordinates, the complete target evaluation is used. CNN and GNN locality also depends on the activation. A function that mixes all features can disable the local route.

The cache belongs to one evaluation at one set of parameters. Gradients include both cached source features and their updates. No cache is carried across optimisation steps. The ordinary model remains the reference when [profiling](../../developers/profiling/for-scientists.md) whether reuse helps a particular workload.

## Distributed network evaluation

The supplied architectures include [tensor partition rules](../../distributed/tensor-parallelism/models.md) for distributing their weights and contractions across cooperating devices. Use the [tensor overview](../../distributed/tensor-parallelism/index.md) to distinguish this from distributing independent samples. You can inspect each selected weight layout and initialize a large model directly into its shards.
