---
title: Transition rules
---

# Transition rules

A transition rule proposes a candidate for each chain. [`MetropolisSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSampler) then evaluates the target distribution and decides which candidates to retain. Keeping those responsibilities separate allows the same proposal to work with different wavefunctions, including the sampling functions prepared by a variational state.

## The transition interface

Implement `transition(sampler, apply_fn, parameters, sampler_state, key, sigma)` on an [`AbstractTransitionRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.AbstractTransitionRule). Here `sigma` has shape `(B, N)` and the proposed batch must have exactly the same shape and represent labels in the same Hilbert space. Use `sigma.shape[0]` for the batch visible to the rule. In distributed execution this can be a local shard of the full chain ensemble.

Return a [`TransitionOutput`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.TransitionOutput) containing `sigma`, `log_correction`, and optional `rule_state`. A two-tuple `(proposed, log_correction)` or three-tuple with rule state is also accepted. Prefer an explicit result over a bare array when defining a reusable extension.

For a proposal $q(y\mid x)$, the correction is

$$
c(x,y)=\log q(x\mid y)-\log q(y\mid x).
$$

Return one real correction per chain, with shape `(B,)`. Use `None` for a symmetric proposal. Do not include the wavefunction ratio in this correction, since the sampler adds that itself. The next page derives [detailed balance](detailed-balance.md) and explains how state-dependent move choices affect $q$.

The optional `init_state` hook constructs rule state. `reset` refreshes it when the sampler resets. To change initial configuration generation, override `random_state_local`, whose default calls `hilbert.random_state` using `sampler.n_batches` and the sampler dtype. The public `random_state` wrapper handles placement. A valid initial configuration need not already follow the target distribution, but it must lie in a component that the proposal can explore.

## An asymmetric independent proposal

Consider binary configurations $x_i\in\{0,1\}$. Propose every new label independently, with a fixed probability $p_i$ of obtaining one. Then

$$
\begin{aligned}
q(y\mid x)&=q(y)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i},\\
c(x,y)&=\sum_i(x_i-y_i)\log\frac{p_i}{1-p_i}.
\end{aligned}
$$

Although the proposal does not depend on $x$, it is generally asymmetric. The probability of drawing $x$ need not equal the probability of drawing $y$. Its correction is therefore necessary even for this simple independent construction.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import field

class IndependentBinaryRule(nqx.sampler.AbstractTransitionRule):
    probabilities: tuple = field(static=True)

    def __post_init__(self):
        if not isinstance(self.probabilities, tuple) or not self.probabilities:
            raise ValueError("probabilities must be a nonempty tuple")
        if not all(0 < p < 1 for p in self.probabilities):
            raise ValueError("Every probability must lie strictly between zero and one")

    def __hash__(self):
        return hash((type(self), self.probabilities))

    def validate_space(self, hilbert):
        if hilbert.constrained or hilbert.size != len(self.probabilities):
            raise ValueError("Use an unconstrained binary space of the matching width")
        if any(size != 2 for size in hilbert.local_sizes):
            raise ValueError("Every site must have two labels")
        indices = jnp.stack([
            jnp.zeros(hilbert.size, dtype=jnp.int32),
            jnp.ones(hilbert.size, dtype=jnp.int32),
        ])
        if not np.array_equal(hilbert.local_indices_to_states(indices), indices):
            raise ValueError("Each site's ordered labels must be zero and one")

    def log_proposal(self, states):
        p = jnp.asarray(self.probabilities)
        return jnp.sum(states * jnp.log(p) + (1 - states) * jnp.log1p(-p), axis=-1)

    def transition(self, sampler, apply_fn, parameters, sampler_state, key, sigma):
        proposed = jax.random.bernoulli(
            key, jnp.asarray(self.probabilities), shape=sigma.shape
        ).astype(sigma.dtype)
        correction = self.log_proposal(sigma) - self.log_proposal(proposed)
        return nqx.sampler.TransitionOutput(proposed, correction)

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
rule = IndependentBinaryRule(probabilities=(0.8, 0.3))
rule.validate_space(hilbert)
sampler = nqx.sampler.MetropolisSampler(
    hilbert, rule=rule, n_chains=8, sweep_size=2
)

def log_value(parameters, states):
    return states @ parameters

parameters = jnp.array([0.2, -0.3])
state = sampler.reset(log_value, parameters, key=jax.random.key(8))
_, state = sampler.sample(log_value, parameters, state=state, chain_length=32)
(samples, log_weights), state = sampler.sample(
    log_value, parameters, state=state, chain_length=64,
    return_log_probabilities=True,
)
assert bool(hilbert.is_valid(samples).all())
np.testing.assert_allclose(log_weights, 2 * log_value(parameters, samples))

# Compare the rule's returned correction with its explicit proposal probabilities.
basis = hilbert.all_states()
proposal = rule.transition(
    sampler, log_value, parameters, state, jax.random.key(23), basis
)
np.testing.assert_allclose(
    proposal.log_correction,
    rule.log_proposal(basis) - rule.log_proposal(proposal.sigma),
)
print(samples.shape)
```

```text
(8, 64, 2)
```

The validation method is called explicitly before constructing the sampler. The generic sampler does not discover and invoke arbitrary validation methods on custom rules. Likewise, validation belongs outside `transition`, where Python checks can inspect concrete values before JAX tracing begins.

Every configuration has positive proposal probability because $0<p_i<1$. This establishes reachability on the unconstrained binary space. It does not make this proposal efficient for a large correlated target. Proposing many unrelated labels at once can produce very low acceptance. The warmup length above demonstrates the call sequence and is not a convergence recommendation.

## A function-based rule

[`FunctionalRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.FunctionalRule) accepts `transition_fn` and optional `init_state_fn`, `reset_fn`, and `random_state_fn`. The random-state function implements the local generation step. The following complete example flips one uniformly selected binary site. Its proposal is symmetric.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

def flip_one(sampler, apply_fn, parameters, sampler_state, key, sigma):
    rows = jnp.arange(sigma.shape[0])
    sites = jax.random.randint(key, (sigma.shape[0],), 0, sigma.shape[1])
    proposed = sigma.at[rows, sites].set(1 - sigma[rows, sites])
    return proposed, None

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
rule = nqx.sampler.FunctionalRule(transition_fn=flip_one, name="binary_flip")
sampler = nqx.sampler.MetropolisSampler(hilbert, rule=rule, n_chains=4)

def log_value(parameters, states):
    return parameters * jnp.sum(states, axis=-1)

samples, state = sampler.sample(
    log_value, jnp.asarray(0.3), key=jax.random.key(5), chain_length=16
)
assert samples.shape == (4, 16, 1)
assert bool(hilbert.is_valid(samples).all())
```

This function assumes binary labels and an unconstrained space. Applying it to a different domain would require different label handling and a new validity argument. For general local domains, the supplied [`LocalRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.LocalRule) already implements those details.

## Constraints and rule state

For a constrained space, design moves that remain in the allowed set, or replace an invalid candidate with the current configuration. Rejection in this form adds a self transition. Repeatedly drawing until a valid candidate appears changes the normalisation of the proposal and can require an additional correction, as explained under [conditional proposals](detailed-balance.md#conditioning-a-proposal-on-validity).

Keep proposal arrays at fixed shapes and use JAX operations for decisions that depend on sampled values. A rule can split its supplied key for several random choices, but should not introduce a global random generator. If it returns updated rule state, that state is retained even for rejected proposals. Read the [state contract](state.md) before using this mechanism for caches, counters, or adaptation.
