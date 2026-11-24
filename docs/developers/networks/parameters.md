---
title: Parameters and frameworks
---

# Parameters and frameworks

A model's variables contain everything needed to evaluate its wavefunction. Some leaves are trainable parameters. Others describe fixed state, such as a feature scale, a topology array, or inference statistics. A framework adapter separates those groups and gives the variational state a stable interface for evaluating and differentiating the model.

This separation is mathematical as well as organisational. An optimiser varies the parameter tree $\theta$ while holding fixed model state $s$ constant in $\ell_{\theta,s}(x)$. Accidentally placing fixed geometry in the parameter tree changes the variational problem. Hiding trainable values inside a static Python object can prevent the optimiser from seeing them at all.

## Supplied framework adapters

[`as_framework`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.as_framework) recognises the supported model forms and returns an adapter together with any variables already extracted from the model.

| Model form | Variable representation |
| --- | --- |
| Flax Linen through [`FlaxLinenFramework`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.FlaxLinenFramework) | A mapping with a `params` collection and any fixed non-parameter collections. |
| Flax NNX through [`FlaxNNXFramework`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.FlaxNNXFramework) | A static graph definition plus variables containing parameter state under `params` and remaining state under `state`. |
| An apply function through [`CallableFramework`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.CallableFramework) | A parameter tree, or the complete variables mapping when `expects_variables=True`. |

NNX adaptation splits an already initialised module into its graph definition, `nnx.Param` state, and remaining state. Evaluation reconstructs a module from those pieces. The [Flax NNX guide](https://flax.readthedocs.io/en/stable/nnx_basics.html) explains the underlying split and merge operations. In neuraLQX, this means that the variational state's variables are the values used for subsequent evaluations. Editing the original NNX instance is not the supported way to update an existing variational state.

Linen initialisation calls the model on a zero-filled dummy input with shape `(1, hilbert.size)`. A callable `init_fn` receives `(key, input_shape, dtype)` and may return a parameter tree or a mapping containing `params`. The supplied dtype describes configuration storage. Choose floating or complex trainable dtypes explicitly, especially for Hilbert spaces whose labels are integers.

## Ownership and updates

When constructing [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState), explicit `variables` take precedence. Otherwise, variables extracted from an initialised NNX model take precedence over separately supplied `parameters`. If neither is available, the state merges supplied parameters with `model_state`, or invokes the framework initializer.

After construction, update through `state.parameters`, `state.model_state`, or `state.variables`. These setters rebuild the relevant representation and invalidate dependent state. Mutating a nested mapping in place can bypass that work. Replacing a network architecture or changing parameter shapes requires reconsidering optimiser state, compilation, and [weight transfer](../../concepts/vqs/transfer.md), rather than treating the change as an ordinary numerical update.

The current variational state supports fixed model collections during evaluation. Its `mutable` argument accepts `False` or `None`. An NNX module may support mutation in other applications, but neuraLQX does not use that fact to propagate state changes during sampling. Fixed `training_kwargs` can select deterministic inference behaviour. Treat them as construction-time call settings, rather than a mutable per-batch state channel.

## The framework contract

Implement [`AbstractModelFramework`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.AbstractModelFramework) when the supplied adapters cannot represent a model's variable organisation. It is a [struct](../structs/index.md), with the following required interface.

| Operation | Contract |
| --- | --- |
| `name` | A descriptive framework name. |
| `init(key, input_shape, *, dtype)` | Construct complete model variables. |
| `apply_variables(variables, states, *, mutable=False, training_kwargs=None)` | Evaluate log amplitudes with the model shape contract. |
| `split_variables(variables)` | Return `(parameters, model_state)`. |
| `merge_variables(parameters, model_state=None)` | Reconstruct complete variables from those parts. |

For supported variables $v$, merging the split must preserve the represented function and parameter structure. The inherited `place_variables` normalizes the variable container while retaining supplied array placements. When tensor mode is enabled, a composed tensor adapter places trainable arrays according to a validated partition plan and replicates fixed array collections. Its constraints also apply during ordinary and connected evaluation. See [tensor partition rules](tensor-rules.md) for the planning contract. An adapter needing other specialized placement must implement and test it explicitly.

`apply_connected` has a correct ordinary-evaluation fallback. Override it only when exact connected-state reuse is available. Set `has_projector=True` when the adapter implements amplitude projection that can introduce exact nodes. This metadata is used by support-sensitive consumers, so it must describe the actual model rather than merely its output dtype.

## A complete adapter with fixed feature scales

The following adapter uses a tuple of two dictionaries as its complete variables. This illustrates that the framework boundary can accommodate a representation other than a Linen variables mapping. The model is

$$
\ell_{\theta,s}(x)=\sum_i\theta_i s_i x_i,
\qquad
\frac{\partial\ell_{\theta,s}(x)}{\partial\theta_i}=s_i x_i.
$$

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.frameworks import AbstractModelFramework
from neuralqx.utils.struct import field

class ScaledLinearFramework(AbstractModelFramework):
    features: int = field(static=True)

    def __post_init__(self):
        if type(self.features) is not int or self.features < 1:
            raise ValueError("features must be a positive integer")

    def __hash__(self):
        return hash((type(self), self.features))

    @property
    def name(self):
        return "example.scaled_linear"

    def init(self, key, input_shape, *, dtype):
        if input_shape[-1] != self.features:
            raise ValueError("Input width does not match the framework")
        parameters = {
            "weights": 0.05 * jax.random.normal(
                key, (self.features,), dtype=jnp.float64
            )
        }
        return self.place_variables(self.merge_variables(parameters))

    def split_variables(self, variables):
        parameters, fixed = variables
        return parameters, fixed

    def merge_variables(self, parameters, model_state=None):
        if model_state is None:
            model_state = {"scale": jnp.ones(self.features, dtype=jnp.float64)}
        return parameters, model_state

    def apply_variables(
        self, variables, states, *, mutable=False, training_kwargs=None
    ):
        if mutable is not False and mutable is not None:
            raise ValueError("This adapter has no mutable collections")
        if training_kwargs:
            raise ValueError("This adapter accepts no extra call options")
        parameters, fixed = self.split_variables(variables)
        features = states.astype(jnp.float64) * fixed["scale"]
        return features @ parameters["weights"]

framework = ScaledLinearFramework(features=2)
variables = framework.init(jax.random.key(7), (1, 2), dtype=jnp.int32)
parameters, _ = framework.split_variables(variables)
fixed = {"scale": jnp.array([2.0, 0.5])}
variables = framework.merge_variables(parameters, fixed)
states = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.int32)

round_trip = framework.merge_variables(*framework.split_variables(variables))
np.testing.assert_allclose(
    framework.apply_variables(round_trip, states),
    framework.apply_variables(variables, states),
)

def loss(parameters):
    values = framework.apply_variables(
        framework.merge_variables(parameters, fixed), states
    )
    return jnp.sum(values)

gradient = jax.jit(jax.grad(loss))(parameters)
np.testing.assert_allclose(gradient["weights"], jnp.sum(states * fixed["scale"], axis=0))

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
vstate = nqx.vqs.VariationalState(
    hilbert=hilbert, model=framework, variables=variables, seed=3, n_samples=32
)
np.testing.assert_allclose(vstate.log_value(states), framework.apply_variables(variables, states))
print(vstate.framework.name)
```

```text
example.scaled_linear
```

The scale is held fixed while the loss is differentiated. All changing numerical values remain in the variables supplied to `apply_variables`. The adapter itself contains only the structural feature count.

## Static identity and compilation

[`FrameworkApply`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.FrameworkApply) presents framework evaluation as a stable callable to compiled kernels. Its identity includes the adapter's `jit_cache_key` and fixed call options. The default adapter key uses its type and hash.

Define that identity from structural settings that affect evaluation. Do not hide changing weights, fixed-array updates, or mutable lookup tables behind an unchanged hash. Conversely, numerical parameter updates should normally enter as new array arguments rather than force a new static model identity. The [pytree guide](../structs/pytrees.md) explains this distinction in more detail.

Finally, test the adapter through the variational state, not only by calling its methods directly. Verify parameter updates, fixed-state preservation, JIT evaluation, and gradients. If the model exposes connected-state reuse, also compare the framework's connected path with ordinary target evaluation. Save variables together with the model definition and required construction metadata using the [saving guide](../../concepts/vqs/saving.md).
