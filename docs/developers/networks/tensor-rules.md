---
title: Tensor partition rules
---

# Tensor partition rules

A tensor partition rule describes the weight layout of an NNX module. It runs during host-side planning and returns partition specifications for the module's trainable arrays. It does not implement a distributed forward pass. The model retains global array operations, and JAX compiles their communication and derivatives.

This interface is useful when you maintain a reusable architecture whose contractions deserve a more specific plan than the supplied layer defaults. For an individual calculation, the [custom network guide](../../distributed/tensor-parallelism/custom-models.md) demonstrates the simpler path-based adapter and layer metadata interfaces.

## Register an architecture rule

[`register_tensor_parallel_rule`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.register_tensor_parallel_rule) accepts an NNX module class and registers a function taking `(module, axis_name, partitions)`. The result maps **tuple paths relative to that module** to JAX `PartitionSpec` objects. Integer container indices stay integers in these paths.

The following complete example registers a two-layer model. Run it in a fresh process with `NEURALQX_SHARDING=tensor` and two devices. A local test can use `JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2`.

```python
from flax import nnx
import jax.numpy as jnp
import neuralqx as nqx
from jax.sharding import PartitionSpec as P

class PairedNetwork(nnx.Module):
    def __init__(self, width=32):
        rngs = nnx.Rngs(7)
        self.hidden = nqx.nn.layers.Linear(
            4, width, rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64,
        )
        self.output = nqx.nn.layers.Linear(
            width, 1, rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64,
        )

    def __call__(self, x):
        return self.output(jnp.tanh(self.hidden(x)))[..., 0]

@nqx.nn.register_tensor_parallel_rule(PairedNetwork)
def paired_rule(module, axis_name, partitions):
    if module.hidden.out_features % partitions:
        raise ValueError("Hidden width must be divisible by the tensor group size")
    return {
        ("hidden", "kernel"): P(None, axis_name),
        ("hidden", "bias"): P(axis_name),
        ("output", "kernel"): P(axis_name, None),
        ("output", "bias"): P(),
    }

model = nqx.nn.init_tensor_parallel(lambda: PairedNetwork(width=32))
plan = nqx.nn.tensor_parallel_plan(model)
assert plan.sharded_parameter_count > 0
print(model(jnp.ones((3, 4), dtype=jnp.float64)))
for entry in plan.parameters:
    print(entry.path, entry.spec)
```

Register the rule before initializing a distributed model or constructing its variational state. Rules must work on abstract modules as well as initialized ones. Read shapes and static architecture attributes, rather than parameter values, host copies of arrays, or runtime inputs. A rule may reject an unsuitable architecture explicitly, as above, or return replicated specifications for particular leaves.

Registering a second rule for the same class raises an error unless `replace=True` is supplied. Class lookup follows the module's method-resolution order, so a registered base-layer rule also applies to subclasses unless a more specific rule exists.

## Resolution and validation

The planner visits descendant modules before their containing architectures. A parent architecture can therefore override defaults supplied by its child layers. Explicit NNX parameter metadata is applied after those registered rules. Overrides retained by `init_tensor_parallel` and explicit `TensorParallel.parameter_rules` are applied later.

Registered rules use tuple paths because they operate on an NNX graph. The public adapter uses slash-separated strings and glob patterns because it addresses the flattened parameter tree. The final plan records each normalized string path, global shape, and selected specification. Shared parameters are represented by the unique leaves in that tree, so inspect paths rather than assuming that every reference to a shared layer creates another weight.

An explicit adapter pattern must match a trainable leaf. Registered handlers should likewise return valid paths for the module they describe, but the automatic handler mapping does not require every emitted path to exist. For example, a generic layer rule can mention an optional bias that is absent. Test your architecture's expected plan directly to detect mistakes in its rule.

The supported tensor plan allows either replication or partitioning one array dimension along the tensor axis. The partitioned dimension must divide the tensor group size exactly. It rejects other mesh axes, repeated tensor-axis entries, and specifications longer than the array rank. A multi-worker plan with no partitioned weight raises an error rather than pretending to distribute the model.

## How the plan reaches evaluation

The framework factory adapts NNX, Linen, and callable models, then composes the adapter with tensor placement when `network_sharding_enabled` is true. Initialization uses shape tracing and output shardings. Existing variable trees are placed on the shared sample/operator/tensor mesh. Trainable arrays follow the weight plan, while fixed model collections receive replicated placement.

At model-call boundaries the tensor adapter applies layout constraints to trainable arrays and delegates to the original framework. The same happens for `apply_connected`. The automatic tensor axis retains global dimensions, even when the surrounding sampler or estimator uses manually partitioned sample and operator axes. This separation lets one global contraction serve ordinary evaluation, connected evaluation, and differentiated evaluation.

The immutable [`TensorParallelPlan`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.TensorParallelPlan) contains layout decisions rather than numerical weights. Its `constrain` method uses the active mesh when the configured tensor axis is present. Its `place` method constructs matching global arrays on the configured tensor mesh. These operations should not be confused with slicing a parameter tree into unrelated local models.

## Derivative and normalization contracts

Custom networks must satisfy the existing [model contracts](contracts.md). Their outputs must be independent of unrelated batch entries, deterministic for fixed variables and configurations, and compatible with JAX transformations. Tensor partitioning does not relax those requirements.

A reduction over hidden features must use the global feature count. If a normalization uses $H$ features distributed over $D_T$ workers, its mean is

$$
\mu=\frac{1}{H}\sum_{t=0}^{D_T-1}
\sum_{j\in\mathcal H_t}h_j.
$$

Replacing the denominator and feature set with a local block changes the model. In an ordinary global JAX expression, write the complete reduction and let the compiler partition it. Explicit collectives belong only in a deliberately manual execution region with a defined axis contract.

For derivatives, test a real scalar loss of complex outputs where appropriate, as in the [value-and-gradient example](../../distributed/tensor-parallelism/custom-models.md#compare-values-and-derivatives). Also test the actual variational estimator. A matching forward value alone does not establish that custom derivative rules or manual collective transposes are correct.

## Validate a new rule

Inspect the abstract plan first, checking expected paths, global shapes, and divisibility. Then compare initialized weights, log amplitudes, and gradients with a small unpartitioned reference using the same scientific definition and seeds. Exercise tensor-only, sample-plus-tensor, operator-plus-tensor, and combined layouts when the model is intended for those paths.

For models with [connected-state reuse](connections.md), compare that path with ordinary target evaluation under the same layouts. Check [checkpoint restoration](../../concepts/vqs/saving.md) and [weight transfer](../../concepts/vqs/transfer.md) into a model configured for the destination mesh. Keep layout rules available at reconstruction time, and distinguish scientific model metadata from an execution-only override.

CPU logical devices can establish these numerical and shape properties. They do not establish useful GPU speedup, communication overlap, or multi-node transport behavior. Profile representative calculations on the target system after correctness checks pass.
