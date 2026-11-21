---
title: Amplitudes and state vectors
---

# Amplitudes and state vectors

A variational state usually evaluates only the configurations needed for sampling and measurements. You can also inspect selected amplitudes directly or, on a small indexable Hilbert space, construct the complete state vector.

These operations answer different questions. `log_value` evaluates the model on configurations you provide. [`to_array`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.to_array) enumerates the basis and returns its wavefunction amplitudes in a definite order. Neither operation advances the Monte Carlo chains or replaces the cached sample batch.

## Evaluate log amplitudes on selected configurations

[`log_value`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState.log_value) returns

$$
f_\theta(\sigma)=\log\psi_\theta(\sigma).
$$

A single input has shape `(hilbert.size,)` and produces a scalar. A batch `(..., hilbert.size)` produces the corresponding leading shape `(...)`. The state flattens those leading axes for model evaluation and restores them afterwards.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
model = nqx.nn.MLP(1, hidden_features=(8,), rngs=2)
state = nqx.vqs.VariationalState(
    hilbert=hilbert, model=model, chunk_size=2, seed=3
)
configurations = jnp.array([
    [[-1], [0], [1]],
    [[1], [0], [-1]],
], dtype=jnp.int32)

values = state.log_value(configurations)
print(values.shape)
print(state.log_value(configurations[0, 0]).shape)
np.testing.assert_allclose(
    values,
    state.log_value(configurations.reshape(-1, 1)).reshape(2, 3),
)
```

```text
(2, 3)
()
```

The output is not a normalised probability. For two configurations with finite nonzero amplitudes, the ratio needed by a local estimator is

$$
\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}
=\exp\!\left[f_\theta(\sigma')-f_\theta(\sigma)\right].
$$

The state checks the final input width, but `log_value` does not replace the Hilbert space's validity checks. Supply configurations in the correct [label layout](../hilbert/state-layouts.md), including any gauge and recoupling constraints.

`state.apply(...)` calls the model directly with the current variables. Use `log_value` for the shape handling and chunking shown here. `state.apply_variables(variables, configurations)` evaluates an explicitly supplied variables tree without assigning it to the state.

## Construct the normalised state vector

Let the ordered basis from `hilbert.all_states()` be $\sigma_0,\ldots,\sigma_{\mathcal D-1}$. By default, `state.to_array()` returns

$$
v_n
=\frac{\exp f_\theta(\sigma_n)}
{\sqrt{\sum_{m=0}^{\mathcal D-1}
\left|\exp f_\theta(\sigma_m)\right|^2}},
\qquad
\sum_n|v_n|^2=1.
$$

The vector has length `hilbert.dimension`, whereas each basis configuration has length `hilbert.size`. Those quantities are usually very different.

This example uses a simple complex log amplitude on one U(1) edge and checks the returned vector against its explicit expression.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

def log_amplitude(parameters, configurations):
    q = configurations[..., 0]
    return parameters["slope"] * q

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    hilbert=hilbert,
    apply_fn=log_amplitude,
    parameters={"slope": jnp.array(0.2 + 0.3j)},
    seed=0,
)

basis = hilbert.all_states()
vector = state.to_array(max_states=10, chunk_size=2)
raw = state.to_array(normalize=False, max_states=10, chunk_size=2)
reference = jnp.exp((0.2 + 0.3j) * basis[:, 0])

np.testing.assert_allclose(raw, reference)
np.testing.assert_allclose(vector, reference / jnp.linalg.norm(reference))
print(basis[:, 0].tolist())
print(vector.shape)
print(round(float(jnp.sum(jnp.abs(vector)**2)), 6))
```

```text
[-1, 0, 1]
(3,)
1.0
```

The normalised calculation subtracts a common real shift from the log amplitudes before exponentiating. That shift cancels in the final normalisation and reduces overflow risk. `normalize=False` returns `exp(log_value)` without that rescaling, so its magnitude reflects the network's arbitrary overall scale.

A state with zero amplitude everywhere cannot be normalised. In particular, a [projector](../networks/symmetries.md) must retain a nonzero state. Finite amplitudes and a nonzero norm are assumptions of the normalisation calculation.

## Compare with an exact small-space expectation

The vector and [`operator.to_dense()`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.to_dense) use the same Hilbert basis ordering. You can therefore compare a small calculation with

$$
\langle A\rangle=v^\dagger A v
$$

when `v` is normalised.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    hilbert=hilbert,
    model=nqx.nn.Jastrow(1, complex_output=False, rngs=4),
    seed=5,
)
charge = NumberOperator(hilbert, edge=0)
basis = hilbert.all_states()
vector = state.to_array(max_states=10)

from_matrix = jnp.vdot(vector, charge.to_dense(max_states=10) @ vector)
from_probabilities = jnp.sum(jnp.abs(vector)**2 * basis[:, 0])
np.testing.assert_allclose(from_matrix, from_probabilities, atol=1e-7)
print(bool(jnp.isfinite(from_matrix)))
```

```text
True
```

This evaluates the complete basis of the chosen finite model. It removes sampling error, but it does not remove the physical cutoff or any approximation in an operator's own evaluation.

Calling `to_array()` does not turn subsequent `state.expect(...)` calls into exact sums. The state remains a sampled variational state. Likewise, assigning every basis configuration to `state.samples` gives an equal-weight batch, which generally differs from the Born weights of a nonuniform wavefunction.

## Enumeration and memory

`to_array()` requires `hilbert.is_indexable`. Some constrained spaces expose exact enumeration only when explicitly constructed for it. The [Hilbert space guides](../hilbert/index.md) explain those choices.

`max_states` bounds the number of basis configurations that may be enumerated. It defaults to one million. It is a limit on the number of states, not on bytes. A complete basis array requires storage proportional to $\mathcal D\,\texttt{hilbert.size}$, and the returned amplitude vector adds $\mathcal D$ floating or complex entries.

`chunk_size` bounds model evaluation batches. It does not avoid materialising the basis or the final vector. For large spaces, use selected `log_value` evaluations and [sampled expectations](expectations.md).

A vector can be converted to a NumPy array for a small external calculation. Retain its basis ordering alongside it. Saving that array alone does not preserve the network, chain state, or reconstruction information supplied by a [VQS checkpoint](saving.md).
