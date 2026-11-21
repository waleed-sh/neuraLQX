---
title: Dispatch and contracts
---

# Dispatch and contracts

Registration tells neuraLQX which local calculation belongs to a particular operator type. The public extension point is [`register_local_estimator`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.register_local_estimator). It registers a method of `local_estimator_kernel` in the [estimator dispatch module](../../api/public/neuralqx/vqs/estimators/local/index.md). Its arguments are the variational state, operator, sample array, and resolved evaluation options.

The operator type annotation selects the implementation. Define a distinct operator class for a distinct observable rather than installing a broad rule that changes unrelated operators. Keep registrations in an importable module and import it before evaluating the observable. Registration occurs when Python executes the decorator.

## The input and output contract

The kernel receives configurations with shape `(..., hilbert.size)` and must return a [`LocalEstimators`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.LocalEstimators) whose `data` has shape `(...)`. It returns local values, not their average. The last configuration axis disappears, while every leading axis retains its meaning.

Use the supplied `samples` argument. Reading `vstate.samples` inside the rule would ignore an explicit batch passed by the caller and could cause an unintended sampling operation. The options argument is a [`LocalEstimatorOptions`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.LocalEstimatorOptions), already resolved by the outer interface.

An [`AbstractOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.AbstractOperator) supplies the Hilbert space and operator metadata without requiring connectivity. This is useful for an observable evaluated directly from configurations. It does not inherit the computational operator algebra or dense matrix conversion. If those operations matter, implement a [computational operator](../operators/contracts.md) as well.

## Register an occupation observable

Consider binary degrees of freedom $x_i\in\{0,1\}$ and the occupied fraction

$$
\hat\rho=\frac1N\sum_{i=0}^{N-1}\hat n_i,
\qquad
\rho_{\mathrm{loc}}(x)=\frac1N\sum_i x_i.
$$

Its local estimator needs neither connected states nor the wavefunction. The following complete example constructs three edge degrees of freedom, registers the rule, and checks the uniform state's expectation using every basis configuration.

```python
from typing import Any
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.vqs import LocalEstimatorOptions, LocalEstimators

class OccupiedFraction(nqx.operator.AbstractOperator):
    @property
    def dtype(self):
        return jnp.dtype("float64")

    @property
    def is_hermitian(self):
        return True

    @property
    def is_diagonal(self):
        return True

    @property
    def adjoint(self):
        return self

@nqx.vqs.register_local_estimator
def occupied_fraction_kernel(
    vstate: Any,
    operator: OccupiedFraction,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    values = jnp.mean(jnp.asarray(samples, dtype=operator.dtype), axis=-1)
    return LocalEstimators(values)

def log_value(parameters, configurations):
    return parameters["theta"] * jnp.sum(configurations, axis=-1)

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
state = nqx.vqs.VariationalState(
    hilbert=hilbert,
    apply_fn=log_value,
    parameters={"theta": jnp.array(0.0)},
    seed=0,
)
observable = OccupiedFraction(hilbert)
samples = hilbert.all_states().reshape(2, 4, hilbert.size)
estimates = nqx.vqs.local_estimators(state, observable, samples=samples)
stats = nqx.vqs.expect(state, observable, samples=samples)

np.testing.assert_allclose(estimates.data, jnp.mean(samples, axis=-1))
np.testing.assert_allclose(stats.mean, 0.5)
print(estimates.data.shape)
print(float(stats.mean))
```

```text
(2, 4)
0.5
```

At `theta=0`, all eight configurations have equal Born probability, so their unweighted mean is an exact expectation. The reshaping checks the output contract. These deliberately arranged configurations are not a Markov chain experiment, so the example uses the mean rather than interpreting its estimated sampling error or convergence diagnostics. At another value of `theta`, an unweighted average of this complete basis would no longer be the correct expectation.

In an actual simulation, omit `samples=` to use the state's cached or newly generated samples. The registration works with the usual state expectation interface as well. The [gradient guide](gradients.md) uses the same observable to verify its derivative.

## Dispatch specificity and evaluation options

An annotation for a concrete subclass normally gives a more specific match than the built-in `DiscreteOperator` rule. The optional `precedence` argument can resolve deliberate overlaps, for example `@register_local_estimator(precedence=1)`. Prefer a precise signature over a global precedence change. Broad registrations can silently intercept calls intended for another implementation.

You can pass either an `options` object or individual evaluation keywords to `local_estimators`. Supplying both is rejected. The main controls are explained in [batching and chunking](../../concepts/vqs/batching.md). `chunk_size` controls model evaluation, while `operator_streaming_chunk_size` controls the standard operator connectivity batches. `operator_chunk_size` is an alias for the latter and must not be supplied alongside it.

A custom kernel must implement the controls relevant to its calculation. The occupation rule has no network evaluations or connection tensor to chunk. An off-diagonal rule usually does, as shown in [estimator kernels](kernels.md). Merely accepting the options argument does not make an arbitrary custom computation stream automatically.

## Registration has a defined scope

The public wrapper handles specialised importance and residual estimators before ordinary local dispatch. It may also combine the connections of a sum or sequence of discrete operators before evaluation. Consequently, a registered rule for one factor is not a guarantee that every enclosing expression will call that rule.

For a `DiscreteOperator`, keep a specialised estimator mathematically identical to the estimator implied by its connectivity. This preserves the observable under composition and under [gradient paths that use connectivity directly](gradients.md#what-registration-does-and-does-not-provide). For a new quantity that is not such an operator, use its own type and state its supported measurement and differentiation paths explicitly.
