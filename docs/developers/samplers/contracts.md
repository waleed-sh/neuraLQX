---
title: Sampler contracts
---

# Sampler contracts

[`AbstractSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.AbstractSampler) separates the definition of a sampling algorithm from the evolving state of a particular run. The sampler contains its Hilbert space and algorithm settings. An [`AbstractSamplerState`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.AbstractSamplerState) contains the random key and any other information needed to continue a sequence of draws.

Both are [structs](../structs/index.md). Numerical state participates in JAX transformations, while structural settings such as a chain count belong in [static fields](../structs/fields.md). Use returned state objects to advance a calculation. Hidden mutation of a Python object is not a substitute for this explicit state flow.

## Required operations

The base class implements the public `init_state`, `reset`, and `sample` methods through three implementation hooks.

| Hook | Responsibility |
| --- | --- |
| `_init_state(apply_fn, parameters, key)` | Allocate a state from an explicit key. |
| `_reset(apply_fn, parameters, state)` | Refresh quantities that depend on the target and reset algorithm-specific diagnostics. |
| `_sample_chain(apply_fn, parameters, state, chain_length, *, return_log_probabilities=False)` | Draw the requested chain length and return the resulting state. |

The implementation also exposes `n_batches`, the leading chain dimension, and a chain count appropriate for consumers such as the variational state. The example below uses `n_chains`. Override `is_exact` only when samples are independent draws from the target distribution, without Markov-chain equilibration. This flag does not make a finite-sample expectation exact.

For $B$ chains, $T$ retained configurations per chain, and a state width $N$, samples have shape `(B, T, N)`. The ordinary return is `(samples, new_state)`. With `return_log_probabilities=True`, return `((samples, log_probabilities), new_state)`, where log probabilities have shape `(B, T)`. These values follow `log_probability` and are **unnormalised log target weights**.

Public `sample` accepts either `chain_length` or `n_samples`. The latter is rounded to a positive multiple of the number of chains, using the nearest multiple and rounding ties upward. Set `chain_length` in shape-sensitive tests so the requested output is explicit. When no state is supplied, `sample` initialises through `reset`. With an existing state it continues directly, without resetting it first.

## An independent sampler for a small finite space

The following implementation enumerates the basis, evaluates its log weights, and uses categorical draws. It is useful as a reference for small tests. Its cost grows with the full Hilbert space dimension, so it is not a replacement for Metropolis sampling in large simulations.

This implementation deliberately supports one sampling device and unconstrained indexable spaces. Those restrictions make its enumeration and array placement explicit. A general distributed implementation needs its own placement and key strategy, as discussed under [sampler state](state.md).

```python
from functools import partial

import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import field

class IndependentState(nqx.sampler.AbstractSamplerState):
    rng: jax.Array

class EnumeratedSampler(nqx.sampler.AbstractSampler):
    n_chains: int = field(static=True, default=4)
    max_states: int = field(static=True, default=4096)

    def __post_init__(self):
        super().__post_init__()
        if type(self.n_chains) is not int or self.n_chains < 1:
            raise ValueError("n_chains must be a positive integer")
        if self.hilbert.constrained or not self.hilbert.is_indexable:
            raise ValueError("This example needs an unconstrained indexable space")
        if self.hilbert.n_states > self.max_states:
            raise ValueError("The basis exceeds the enumeration limit")
        if self.sample_device_count != 1:
            raise ValueError("This reference implementation uses one sampling device")

    @property
    def n_batches(self):
        return self.n_chains

    @property
    def is_exact(self):
        return True

    def _init_state(self, apply_fn, parameters, key):
        return IndependentState(rng=key)

    def _reset(self, apply_fn, parameters, state):
        return state

    @partial(
        jax.jit,
        static_argnames=("apply_fn", "chain_length", "return_log_probabilities"),
    )
    def _sample_chain(
        self, apply_fn, parameters, state, chain_length,
        *, return_log_probabilities=False,
    ):
        basis = self.hilbert.numbers_to_states(
            jnp.arange(self.hilbert.n_states, dtype=jnp.int32)
        )
        log_weights = self.log_probability(apply_fn, parameters, basis)
        next_key, draw_key = jax.random.split(state.rng)
        indices = jax.random.categorical(
            draw_key, log_weights, shape=(self.n_batches, chain_length)
        )
        samples = basis[indices].astype(self.dtype)
        new_state = state.replace(rng=next_key)
        if return_log_probabilities:
            return (samples, log_weights[indices]), new_state
        return samples, new_state

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)

def log_value(parameters, states):
    return states @ parameters

parameters = jnp.array([0.2, -0.3])
sampler = EnumeratedSampler(hilbert, n_chains=4)
state = sampler.reset(log_value, parameters, key=jax.random.key(12))
(samples, log_weights), state = sampler.sample(
    log_value, parameters, state=state, chain_length=16,
    return_log_probabilities=True,
)
assert samples.shape == (4, 16, 2)
assert bool(hilbert.is_valid(samples).all())
np.testing.assert_allclose(log_weights, 2 * log_value(parameters, samples))
assert sampler.is_exact
vstate = nqx.vqs.VariationalState(
    sampler=sampler, apply_fn=log_value, parameters=parameters,
    n_samples=16, seed=12,
)
assert vstate.samples.shape == (4, 4, 2)
print(samples.shape, log_weights.shape)
```

```text
(4, 16, 2) (4, 16)
```

The `_reset` hook does nothing here because no target-dependent weights are cached. Every call recomputes them from the supplied parameters. The example assumes a target with finite log amplitudes on this basis. A production enumerating sampler should explicitly handle zero weights and reject an entirely zero or nonfinite target before drawing.

The random key advances even though there is no Markov configuration to retain. Reusing the same initial `IndependentState` would reproduce the same draw. Continuing with the returned state produces the next draw in the sequence.

## Integration and validation

A sampler following this interface can be supplied to [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState). Keep the same callable and variable conventions used by the framework adapter. In particular, do not assume that `parameters` is a flat array merely because it is one in this example.

Check output order with more than one chain and more than one retained step. Test both return forms, continuity across calls, reset after changing parameters, and reproducibility from an identical initial state. For a new Markov algorithm, also compare a small enumerated stationary distribution with the intended target. The [detailed-balance guide](detailed-balance.md) shows an exact finite-state check that is more diagnostic than a noisy histogram alone.

Keep JAX shapes fixed throughout the compiled sampling loop. Use masks instead of constructing variable-length candidate arrays, and pass stochastic choices through explicit keys. JAX's [random-number guide](https://docs.jax.dev/en/latest/101/random.html) explains why key splitting is part of the algorithm's functional interface.
