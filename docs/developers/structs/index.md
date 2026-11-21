---
title: Structs and pytrees
---

# Structs and pytrees

An algorithm usually carries more than one array. A sampler needs configurations, probabilities, random state, and acceptance counts. An estimator returns several related statistics. An operator combines numerical coefficients with information about the space on which it acts. Keeping these values together makes the mathematical object easier to represent and its update rules easier to follow.

[`Struct`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct) is neuraLQX's base class for this kind of container. It supplies declared fields, generated construction, immutable attribute bindings, replacement, JAX pytree registration, and state serialisation. Its field declarations distinguish numerical data from metadata and Python objects that should remain outside ordinary array transformations.

These guides are primarily for people implementing or extending components. If you use the supplied [samplers](../../concepts/samplers/index.md), [operators](../../concepts/operators/index.md), and [variational states](../../concepts/vqs/index.md), you already benefit from this infrastructure without needing to define a struct yourself.

## A struct gives a calculation an explicit state

A useful numerical update can be written as

$$
s_{t+1}=F(s_t,\eta_t),
$$

where $s_t$ contains the evolving state and $\eta_t$ supplies an update parameter. Returning a new state makes the dependency on the old one explicit. The same function can then participate in compilation or a compiled iteration without relying on hidden changes to a Python object.

The following example groups a parameter array and an iteration counter. It applies a simple contraction to the parameters. This is a demonstration of state management, rather than a VMC update or a sampling rule.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class IterationState(Struct):
    parameters: dict
    iteration: jax.Array
    method: str = field(static=True, default="contraction")

@jax.jit
def advance(state, rate):
    parameters = jax.tree.map(lambda value: (1.0 - rate) * value, state.parameters)
    return state.replace(parameters=parameters, iteration=state.iteration + 1)

initial = IterationState(
    parameters={"weights": jnp.array([2.0, -4.0])},
    iteration=jnp.array(0, dtype=jnp.int32),
)
updated = advance(initial, 0.25)

print(initial.parameters["weights"].tolist())
print(updated.parameters["weights"].tolist())
print(int(updated.iteration))
print(updated.method)
```

```text
[2.0, -4.0]
[1.5, -3.0]
1
contraction
```

The class annotations declare its fields. The generated constructor assigns their values, and `replace` constructs the updated instance. The original object retains its parameter bindings and counter. The static `method` accompanies the numerical state without becoming an array argument.

Attribute assignment after construction raises an error. This protects the structure of the object, but it does not recursively freeze a dictionary, list, or NumPy array stored inside it. Treat nested values as immutable too. Build a new dictionary or use JAX array updates when producing a replacement.

## How a struct becomes a pytree

A pytree is a nested arrangement of containers and leaves. Dictionaries, tuples, and lists already participate in JAX's tree operations. A registered custom class can participate as another container. Flattening separates its numerical leaves from the structure needed to reconstruct the object.

A `Struct` subclass is registered automatically. Its ordinary fields are dynamic children, so a field containing a dictionary of arrays contributes the dictionary's array leaves recursively. Static fields become auxiliary metadata in the tree definition. Opaque fields preserve Python references outside the numerical leaves.

This distinction is described in [fields and static data](fields.md). The [JAX behaviour guide](pytrees.md) then shows what it means for compilation, differentiation, batching, and iterative calculations. JAX's own [pytree guide](https://docs.jax.dev/en/latest/101/pytrees.html) provides the broader container model.

Being a pytree does not make every field trainable. A sampler's integer counters and random key are dynamic data, but they are not optimisation parameters. The function being differentiated and its selected arguments determine the derivative. Structs organise that data without choosing a physical model or an optimisation problem for you.

## Where neuraLQX uses this infrastructure

The package uses structs for both evolving numerical state and fixed descriptions of a calculation. The field choices reflect the role of each object.

| Example | Why it is represented as a struct |
| --- | --- |
| [`MetropolisSamplerState`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSamplerState) | Configurations, log probabilities, random state, and counters travel together through sampling updates |
| [`Stats`](../../api/public/neuralqx/utils/stats/index.md#neuralqx.utils.stats.Stats) | Means, variances, errors, and diagnostics can be returned together from numerical calculations |
| [`SiteDof`](../../api/public/neuralqx/hilbert/layout/index.md#neuralqx.hilbert.layout.SiteDof) | Static metadata identifies how a scalar site relates to a graph entity and component |
| [`LocalEstimatorOptions`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.LocalEstimatorOptions) | Static execution choices form one explicit object passed through estimator dispatch |
| [`ScaledOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ScaledOperator) | A wrapped operator and its coefficient form a replaceable numerical object |
| [`QGTOnTheFly`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.QGTOnTheFly) | Parameters and samples remain numerical data, while execution settings and the model framework have different field roles |

For sampling, explicit state supports repeated compiled transitions. For estimator results, it avoids unpacking unrelated positional arrays. For layouts and options, it records the fixed choices that affect interpretation or execution. For operators and the QGT, it lets numerical data coexist with the metadata and runtime interfaces required to evaluate an action.

The [Hilbert space](../../concepts/hilbert/index.md), [operator](../../concepts/operators/index.md), and [sampler](../../concepts/samplers/index.md) base classes already participate in this system. Their extension guides build on the field and reconstruction rules introduced here.

## When to define your own

Use a struct when several values form a meaningful numerical object that must pass through JAX, when a component needs clear replacement semantics, or when the same field description should also determine its saved state. It is particularly useful for a custom transition state, an estimator result, or a small collection of algorithm settings.

A dictionary of arrays is still appropriate when names and nesting are enough. A host-side experiment controller with files, loggers, and mutable bookkeeping often belongs outside the compiled state. A [custom neural network](../../concepts/networks/custom-models.md) should follow its framework's module conventions rather than being converted to a struct merely because it contains parameters.

Existing classes can be adapted through the [pytree registration interfaces](pytrees.md#adapting-an-existing-class) without changing their inheritance. This is useful when integrating a component that already has its own constructor or lifecycle.

Read [fields and static data](fields.md) before choosing a schema, then check the [JAX transformation examples](pytrees.md) against the operations your component needs. Finally, [serialisation](serialisation.md) explains which values are saved, how reconstruction works, and why a generic struct bundle differs from a complete simulation checkpoint.
