---
title: PyTree behaviour
---

# PyTree behaviour

JAX transformations operate on the numerical leaves of a pytree while preserving enough structure to reconstruct their inputs and outputs. For a neuraLQX [`Struct`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct), that structure includes its class, declared child fields, static metadata, and opaque references.

This makes a struct useful as an argument, result, or loop state. It does not remove the ordinary requirements of the transformation. A differentiated argument needs appropriate numerical leaves, a vectorised argument needs compatible batch axes, and a compiled loop must keep its state structure consistent.

## Inspect leaves and paths

An ordinary field can contain a nested pytree. The number of fields is therefore not necessarily the number of leaves. Static and opaque fields contribute no numerical leaves, and `None` follows JAX's usual zero-leaf tree behaviour.

Struct registration includes attribute keys, so leaf paths retain the field names. The following example uses one field containing two parameter arrays and another containing a scalar count.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class ParameterState(Struct):
    parameters: dict
    iteration: jax.Array
    basis: str = field(static=True, default="edge")

state = ParameterState(
    {"weights": jnp.array([1.0, 2.0]), "bias": jnp.array(0.5)},
    jnp.array(0, dtype=jnp.int32),
)
paths, definition = jax.tree_util.tree_flatten_with_path(state)
print([jax.tree_util.keystr(path) for path, value in paths])
print(state.tree_size())

restored = jax.tree.unflatten(definition, [value for path, value in paths])
print(restored.basis)
print(restored.parameters["weights"].tolist())
```

```text
[".parameters['bias']", ".parameters['weights']", '.iteration']
3
edge
[1.0, 2.0]
```

The parameter dictionary is traversed in JAX's key order. The array of two weights is one leaf, rather than two leaves. `tree_size()` counts leaves, not scalar parameters. Ordinary iteration over a struct is different again, since it yields all declared field values, including static, opaque, and derived fields.

## Compilation sees data and metadata differently

For a fixed tree definition, a jitted function traces operations on numerical inputs. Array shape and dtype, static metadata, and the structure of the tree all contribute to whether an existing compiled computation can be reused. Changing only array values with compatible structure, shape, and dtype can reuse a compiled path. Changing an algorithm mode stored as static metadata can require a different one.

A Python branch on static metadata is available while tracing. A Python branch on a changing array value is not. Use JAX control flow for decisions that must happen at execution time. The [JAX compilation guide](https://docs.jax.dev/en/latest/jit-compilation.html) explains the distinction between traced computation and static arguments.

Pass the struct as an ordinary pytree argument. Marking the entire object static would hide its numerical leaves from normal argument tracing, and a plain `Struct` is unhashable by default. Some package metadata classes define their own structural hash, but this does not make arbitrary numerical containers suitable static arguments.

Opaque references also participate in tree identity. They are compared by object identity rather than the value comparison used for static fields. Keep their contents stable while a compiled computation depends on them. `compare=False` affects the struct's Python equality method and has no effect on this JAX metadata rule.

## Differentiate the parameters you intend

For a real parameter vector $\theta$, consider

$$
\mathcal L(\theta)=\frac{1}{2}\sum_i\theta_i^2,
\qquad
\nabla_\theta\mathcal L=\theta.
$$

Here the struct contains only differentiable numerical parameters and a static reduction choice. The gradient has the same container structure. A tree map can then apply one update to matching parameter and gradient leaves.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class Parameters(Struct):
    weights: jax.Array
    reduction: str = field(static=True, default="sum")

def objective(parameters):
    terms = 0.5 * parameters.weights ** 2
    if parameters.reduction == "sum":
        return jnp.sum(terms)
    return jnp.mean(terms)

@jax.jit
def update(parameters, learning_rate):
    value, gradient = jax.value_and_grad(objective)(parameters)
    updated = jax.tree.map(
        lambda parameter, derivative: parameter - learning_rate * derivative,
        parameters,
        gradient,
    )
    return value, gradient, updated

parameters = Parameters(jnp.array([2.0, -4.0]))
value, gradient, updated = update(parameters, 0.25)
print(float(value))
print(gradient.weights.tolist())
print(updated.weights.tolist())
print(gradient.reduction)
```

```text
10.0
[2.0, -4.0]
[1.5, -3.0]
sum
```

A node declaration is not a trainability flag. If you differentiate an entire struct containing floating-point samples, those samples are also part of the differentiated argument. Integer counters and random keys need different treatment and do not become ordinary real-valued parameters. In practice, separate the parameter subtree from sampling and bookkeeping state, as the [VQS interface](../../concepts/vqs/state.md) does.

Static fields are carried in the gradient's tree metadata, not differentiated. Marking a coefficient static merely to exclude it from one gradient also makes it a fixed tracing choice. Prefer selecting the intended argument or subtree for differentiation when the coefficient must remain a changing runtime input.

Ordinary struct equality is a host-side convenience. Array comparisons can transfer or wait for data, and they are not a replacement for a JAX boolean expression inside a compiled function.

## Batch arrays inside the container

`vmap` maps array axes, not an arbitrary Python list of independent objects. A common representation is one struct whose dynamic fields carry a leading batch axis. Static metadata is shared by that batch.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class ConfigurationBatch(Struct):
    values: jax.Array
    label: str = field(static=True, default="two-components")

batch = ConfigurationBatch(jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
squared_norms = jax.jit(jax.vmap(lambda item: jnp.sum(item.values ** 2)))(batch)
print(squared_norms.tolist())
```

```text
[5.0, 25.0, 61.0]
```

When several fields should be mapped together, their mapped axes need compatible lengths. To combine a shared struct with a separately batched argument, an `in_axes` specification such as `(None, 0)` can broadcast the first argument. If every sample needs a different numerical coefficient, represent those coefficients as a batched node field rather than varying static metadata within one batch.

Stacking separate structs with `jax.tree.map` requires compatible tree definitions, including matching static metadata and opaque identities. A different label or a newly created opaque token can therefore prevent an otherwise plausible stack.

## Carry a struct through an iterative calculation

`jax.lax.scan` expresses a sequence of updates with an explicit carried state. Its state must retain the same pytree structure, shapes, and dtypes across iterations. A struct makes that state readable without changing the requirement.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct

class RunningTotal(Struct):
    total: jax.Array
    count: jax.Array

def accumulate(state, value):
    updated = state.replace(total=state.total + value, count=state.count + 1)
    return updated, updated.total / updated.count

@jax.jit
def running_means(values):
    initial = RunningTotal(
        jnp.zeros((), dtype=values.dtype),
        jnp.zeros((), dtype=jnp.int32),
    )
    return jax.lax.scan(accumulate, initial, values)

final, means = running_means(jnp.array([2.0, 4.0, 6.0]))
print(means.tolist())
print(float(final.total), int(final.count))
```

```text
[2.0, 3.0, 4.0]
12.0 3
```

The count is a dynamic integer scalar whose value changes while its shape and dtype remain fixed. Making it static would change the carried tree definition on each update. Replacing a `None` node with an array partway through a scan also changes the structure. Initialise the state in the form the full calculation needs.

## Reconstruction is not an ordinary constructor call

JAX repeatedly flattens and reconstructs containers while tracing and transforming a function. Struct unflattening allocates an instance without calling its usual `__init__` or `__post_init__`, restores non-derived fields through their converters, and recomputes derived fields.

The implementation suppresses ordinary field validation when incoming node values contain recognised JAX tracers. Concrete reconstruction still validates. Trace-related failures in derived calculations can be deferred and retried when the field is accessed. This allows an unused host-oriented derived value to remain unavailable during a transformation, but does not make host conversion of a tracer valid when the value is actually needed.

Transformations can also reconstruct objects with structural placeholders while determining their behaviour. Keep converters and derivations compatible with the transformations you use. In particular, unconditional NumPy conversion, a host reduction over a dynamic array, or shape assumptions about every intermediate object can fail. The [JAX custom-pytree discussion](https://docs.jax.dev/en/latest/101/pytrees.html#custom-pytree-nodes) describes why constructor-like work during unflattening needs care.

`replace` is a different path. It calls the constructor and therefore runs post-init and validation even when invoked inside a transformed function. If your update uses `replace` under `jit`, those hooks must accept the traced values. The examples on this page keep such hooks out of the numerical state.

## Adapting an existing class

An existing Python class can join JAX's tree system without becoming immutable or inheriting from `Struct`. [`register_attrs_type`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.register_attrs_type) registers selected attributes as children or static metadata and supplies an adapter for struct serialisation.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import register_attrs_type

class MeasuredVector:
    def __init__(self, values, unit):
        self.values = values
        self.unit = unit

register_attrs_type(
    MeasuredVector,
    node_fields=("values",),
    static_fields=("unit",),
)

vector = MeasuredVector(jnp.array([3.0, 4.0]), "normalised")
norm = jax.jit(lambda item: jnp.linalg.norm(item.values))(vector)
scaled = jax.tree.map(lambda value: value / norm, vector)
print(scaled.values.tolist())
print(scaled.unit)
```

The numerical output is approximately `[0.6, 0.8]`, with the original unit string retained. This class remains an ordinary mutable Python class. Registration does not add `replace`, the struct field lifecycle, or its static-value validation. The adapter author must maintain the required static metadata contract.

By default, reconstruction first tries `Class(**values)` and falls back to allocating the instance and assigning the registered attributes if that call raises. Supply `constructor=` when reconstruction needs a deliberate rule. Only the registered attributes are available to that rule, so an unregistered cache or handle is not automatically preserved.

For complete control, [`register_pytree_type`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.register_pytree_type) accepts `flatten(obj) -> (children, aux)` and `unflatten(aux, children) -> obj`, plus an optional keyed flatten function. Keep arrays in the children and immutable comparable metadata in `aux`. Optional `serializer` and `deserializer` functions provide a separate persistence representation when the tree representation alone is insufficient.

Define adapters once when their module is imported. Re-registering a Python class with different callbacks is not a reliable way to change its JAX behaviour, since JAX registration and the package's adapter cache have separate lifetimes. Use the [serialisation guide](serialisation.md) to make the same class reconstructible in a later process.
