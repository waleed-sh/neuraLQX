---
title: Custom models
---

# Custom models

A custom model lets you choose features, architecture, and symmetries for the physical state you want to represent. It can use the [supplied layers](layers.md), native Flax components, or a pure JAX function. The essential contract is the same as for the built-ins: accept configuration labels and return one scalar log amplitude per configuration.

neuraLQX accepts initialised Flax NNX modules, Flax Linen modules, and explicit apply functions. These are alternatives for expressing the same variational object. The [Flax NNX guide](https://flax.readthedocs.io/en/stable/nnx_basics.html) explains its module and parameter system, while the examples below show the wavefunction conventions used here.

## Start with the input and output

For inputs of shape `(..., hilbert.size)`, return shape `(...)`. The output is $\log\psi_\theta(\sigma)$. A final feature axis of length two must be combined into one complex scalar if its entries represent log modulus and phase.

Preserve the Hilbert space's meaning when transforming the input. A fixed coordinate permutation, scale, or embedding can be part of the network, but the sampler and operators continue to exchange the original configuration layout. In particular, the [GNN guide](built-in/gnn.md#define-the-node-layout) explains why U(1) component blocks may need a gathering operation before being interpreted as node features.

The forward pass must be deterministic and independent of other batch entries. Construct parameters once, and use JAX operations when evaluating the model. Do not mutate model state or draw new random features on each wavefunction evaluation.

## A custom NNX wavefunction

[`LogWavefunction`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.LogWavefunction) is a convenient base class for scalar log-amplitude models. A subclass can implement only `__call__` and use ordinary connected-state evaluation. Inheriting it does not require implementing an optimised cache.

```python
from flax import nnx
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

class DenseWavefunction(nqx.nn.LogWavefunction):
    def __init__(self, in_features, *, rngs):
        self.hidden = nqx.nn.layers.Linear(in_features, 12, rngs=rngs)
        self.output = nqx.nn.layers.Linear(12, 2, rngs=rngs)

    def __call__(self, states):
        features = jnp.tanh(self.hidden(states.astype(jnp.float32)))
        head = self.output(features)
        return head[..., 0] + 1j * head[..., 1]

network = DenseWavefunction(4, rngs=nnx.Rngs(0))
configurations = (jnp.arange(24).reshape(2, 3, 4) % 3) - 1
evaluate = nnx.jit(lambda model, x: model(x))
values = evaluate(network, configurations)

flat_values = network(configurations.reshape(-1, 4)).reshape(2, 3)
np.testing.assert_allclose(values, flat_values, rtol=1e-5, atol=1e-6)
print(values.shape)
```

```text
(2, 3)
```

The layers share the same RNG stream at construction. Their parameters are tracked as NNX parameters, and the final operation combines two real outputs into a complex log amplitude. The shape check also verifies that flattening the batch does not change the represented values.

An ordinary `flax.nnx.Module` is accepted as well. Use `nnx.Param` for trainable array data and non-parameter variables for fixed array state, such as a topology or feature map. Pass the initialised instance to [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState), following the [overview example](index.md#connect-a-network-to-a-variational-state).

## A custom Linen wavefunction

Linen separates the module definition from its variables. The following example uses the supplied Linen layer bridge inside a conventional `@compact` method.

```python
from flax import linen as nn
import jax
import jax.numpy as jnp
import neuralqx as nqx

class LinenWavefunction(nn.Module):
    width: int = 12

    @nn.compact
    def __call__(self, states):
        x = states.astype(jnp.float32)
        x = nqx.nn.layers.linen.Linear(
            x.shape[-1], self.width, name="hidden"
        )(x)
        x = jnp.tanh(x)
        head = nqx.nn.layers.linen.Linear(
            self.width, 2, name="output"
        )(x)
        return head[..., 0] + 1j * head[..., 1]

network = LinenWavefunction()
configurations = jnp.zeros((3, 4), dtype=jnp.int32)
variables = network.init(jax.random.key(0), configurations)
print(network.apply(variables, configurations).shape)
```

```text
(3,)
```

Pass a Linen module to the variational state and let its `seed` control initialisation, or supply `variables` explicitly to retain an existing initialisation. The complete variables dictionary includes the `"params"` collection and any fixed collections the model requires.

[`to_linen`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.to_linen) can also wrap a custom NNX class directly. It constructs a Linen instance from that class and its constructor arguments. Supply random keys through Linen `init`, rather than passing an NNX `rngs` argument to the bridge.

## An explicit JAX apply function

A model need not be a neural network class. This example represents a complex linear log amplitude with an explicit parameter dictionary.

```python
import jax.numpy as jnp
import neuralqx as nqx

def log_value(parameters, states):
    return jnp.sum(states * parameters["weights"], axis=-1)

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
parameters = {"weights": jnp.array([0.1 + 0.2j])}
state = nqx.vqs.VariationalState(
    hilbert=hilbert,
    apply_fn=log_value,
    parameters=parameters,
    seed=0,
)
configurations = jnp.array([[-1], [0], [1]], dtype=jnp.int32)
print(state.log_value(configurations).shape)
```

```text
(3,)
```

By default, the function receives only the trainable parameter tree as its first argument. With `expects_variables=True`, it receives the complete variables dictionary instead. This is useful when the calculation also needs fixed model state.

If parameters are not supplied, provide `init_fn`. Its call signature is `init_fn(key, input_shape, dtype)`, where `input_shape` is `(1, hilbert.size)` and `dtype` is the sampler's configuration dtype. It can return the parameter tree or a variables dictionary containing `"params"`. Choose floating or complex parameter dtypes explicitly rather than copying an integer configuration dtype into trainable weights.

## Model variables belong to the variational state

During optimisation, update and inspect parameters through the variational state. Assigning `state.parameters` or `state.variables` triggers the required cache invalidation and refreshes sampling probabilities when needed. Mutating a nested dictionary in place bypasses that interface.

Forward evaluations use fixed model collections. The variational state currently accepts `mutable=False` or `None`, rather than arbitrary collection updates during sampling. A layer that needs fixed inference options can receive them through `training_kwargs`, but the resulting wavefunction must still be deterministic and consistent across batches.

## Further extensions

You can combine a custom model with any [projector](symmetries.md), provided its input domain includes the configurations used by that projection. Choose [complex differentiation](complex.md) from the actual operations in the model, not from its class name.

For ordinary scientific use, a correct scalar log-amplitude implementation is sufficient. The [developer guide](../../developers/networks/index.md) covers deeper model contracts and integration work. Exact connected-state reuse is an optional optimisation whose cache and updates must remain differentiable and reproduce the complete target evaluation.
