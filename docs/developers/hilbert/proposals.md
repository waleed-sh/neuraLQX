---
title: Proposal moves
---

# Proposal moves

A proposal changes a current configuration into a candidate. It is not yet an accepted Monte Carlo transition. The sampler compares wavefunction probabilities and, where necessary, proposal probabilities before deciding whether to keep that candidate.

For a proposal distribution $q(y|x)$ and target $p_\theta$, Metropolis-Hastings accepts with probability

$$
\alpha(x,y)=\min\left(1,
\frac{p_\theta(y)q(x|y)}{p_\theta(x)q(y|x)}\right).
$$

Under $p_\theta(x)\propto|\psi_\theta(x)|^m$, the log acceptance ratio is

$$
m\operatorname{Re}\bigl(\log\psi_\theta(y)-\log\psi_\theta(x)\bigr)
+\log q(x|y)-\log q(y|x).
$$

The Hilbert space move interface returns candidate states. A sampler transition rule supplies any required log proposal correction. Keeping those responsibilities explicit prevents a valid but asymmetric move from being used with an incorrect acceptance probability.

## Generic moves use local indices

[`UniformSiteUpdate`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.UniformSiteUpdate) selects `n_sites` distinct flat sites per configuration and redraws their local indices. With its default `avoid_current=True`, it selects a different value when the local domain contains more than one value. A one-value domain remains unchanged.

[`AdjacentSiteUpdate`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AdjacentSiteUpdate) takes a positive or negative step in the local basis index. `boundary="wrap"` wraps around that index range, while `boundary="clamp"` retains the endpoint when the selected step leaves the range. This is adjacency in the declared local table, not necessarily equal spacing of physical labels or adjacency on the graph.

Both moves convert back to stored values through the Hilbert space. For a constrained space, invalid candidates are replaced by the original state. The number of selected sites must be positive and no larger than the configuration width.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
graph = nqx.graph.Graph([(a, b), (b, c)], is_planar=True)
hilbert = nqx.hilbert.HeterogeneousDiscreteHilbert(
    graph,
    nqx.hilbert.HeterogeneousLocalSpace(
        edge=(-1, 0, 1), edge_spaces={1: (0, 2, 5, 9)}
    ),
)
key_initial, key_uniform, key_adjacent = jax.random.split(jax.random.key(3), 3)
states = hilbert.random_state(key_initial, size=12)
uniform = hilbert.propose(nqx.hilbert.UniformSiteUpdate(), key_uniform, states)
adjacent = hilbert.propose(
    nqx.hilbert.AdjacentSiteUpdate(boundary="clamp"), key_adjacent, states
)
assert bool(jnp.all(hilbert.is_valid(uniform)))
assert bool(jnp.all(hilbert.is_valid(adjacent)))
assert bool(jnp.all(jnp.sum(uniform != states, axis=-1) == 1))
print(uniform.shape, adjacent.shape)
```

```text
(12, 2) (12, 2)
```

The generic proposals select sites independently of their current values. Their replacement or neighbouring-index probabilities are symmetric, and rejecting invalid candidates adds self-transitions. A custom proposal needs its own symmetry argument. Repeatedly resampling until a move happens to be valid can change the normalisation of $q(y|x)$ in a state-dependent way.

## Validity does not guarantee exploration

For binary occupations with $\sum_i x_i=K$, every single-site flip changes the sum. Rejecting all invalid candidates makes a local sampler remain at its initial configuration. Every output is valid, but no sampling of the sector takes place.

A pair exchange preserves the sum. Choosing a pair independently of the configuration and swapping its entries gives a symmetric proposal, since the same swap reverses the move. Selecting equal sites or equal values produces a self-transition. Over all site pairs, exchanges connect all binary configurations with fixed occupation. Additional spatial or gauge constraints can change that reachability, so the same conclusion does not automatically extend to another sector.

The supplied [`MetropolisExchange`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisExchange) is often sufficient for exchange-based calculations. The example below shows how to implement and connect a custom Hilbert move when the move itself is part of your extension.

## Connect a custom move to a sampler

A move descriptor can subclass [`AbstractProposalMove`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AbstractProposalMove). Override the space's `propose` method for that descriptor and delegate other descriptors to the inherited method.

The module-level [`propose`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.propose) function routes supplied move types through a private registry. It does not expose a public registration decorator. Also, `MetropolisLocal` uses its own local transition implementation. Overriding a Hilbert space method does not silently replace that sampler's rule. Pass an explicit [`AbstractTransitionRule`](../../api/public/neuralqx/sampler/rules/index.md#neuralqx.sampler.rules.AbstractTransitionRule) that calls your method.

This complete example defines a one-particle space, a constructive initialiser, a pair-swap move, and the small rule connecting it to the sampler. The space validates its assumptions at construction.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.sampler.rules import AbstractTransitionRule

class PairSwap(nqx.hilbert.AbstractProposalMove):
    def __hash__(self):
        return hash(type(self))

class OneParticleSpace(nqx.hilbert.HomogeneousDiscreteHilbert):
    def __post_init__(self):
        super().__post_init__()
        expected = nqx.hilbert.LinearConstraint.sum(range(self.size), target=1)
        if self.size < 2 or self.constraint != expected:
            raise ValueError("Require at least two sites and total occupation one")
        for local in self.local_spaces:
            if not np.array_equal(np.asarray(local.all_values()), [0, 1]):
                raise ValueError("Local labels must be 0 and 1")

    def random_state(self, key, size=None, *, dtype=None, max_trials=1024):
        batch = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        positions = jax.random.randint(key, batch, 0, self.size)
        return jax.nn.one_hot(
            positions, self.size, dtype=self.dtype if dtype is None else dtype
        )

    def propose(self, move, key, states):
        if not isinstance(move, PairSwap):
            return super().propose(move, key, states)
        array = jnp.asarray(states)
        flat = array.reshape(-1, self.size)
        keys = jax.random.split(key, flat.shape[0])

        def swap_one(k, configuration):
            i, j = jax.random.randint(k, (2,), 0, self.size)
            return configuration.at[i].set(configuration[j]).at[j].set(configuration[i])

        return jax.vmap(swap_one)(keys, flat).reshape(array.shape)

class PairExchangeRule(AbstractTransitionRule):
    def __hash__(self):
        return hash(type(self))

    def transition(self, sampler, apply_fn, parameters, sampler_state, key, sigma):
        return sampler.hilbert.propose(PairSwap(), key, sigma), None

def log_value(parameters, configurations):
    return parameters["bias"] * configurations[..., 0]

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = OneParticleSpace(
    graph, nqx.hilbert.LocalRange(0, 1, 2),
    constraint=nqx.hilbert.LinearConstraint.sum(range(3), target=1),
)
sampler = nqx.sampler.MetropolisSampler(
    hilbert, rule=PairExchangeRule(), n_chains=4, sweep_size=1
)
state = nqx.vqs.VariationalState(
    sampler=sampler, apply_fn=log_value,
    parameters={"bias": jnp.array(0.3)},
    n_samples=32, n_discard_per_chain=2, seed=0,
)
samples = state.samples
assert bool(jnp.all(hilbert.is_valid(samples)))
print(samples.shape)
print(jnp.unique(jnp.sum(samples, axis=-1)).tolist())
```

```text
(4, 8, 3)
[1]
```

The rule returns `None` for the proposal correction because the chosen pair exchange is symmetric. The sampler still evaluates and applies the wavefunction-dependent acceptance probability. The short run verifies integration and constraint preservation, not statistical convergence.

## Check a move beyond its output shape

For a small sector, enumerate the proposal graph and verify that every intended configuration is reachable. Check forward and reverse probabilities, including boundary cases and self-transitions. A move can preserve the constraint and still split the state space into disconnected sectors.

For an asymmetric proposal, return $\log q(x|y)-\log q(y|x)$ through the transition rule with one correction per chain. A Hilbert method that returns only candidate states is not enough to communicate that information. The [custom sampler guides](../samplers/index.md) are the place for more extensive transition state, proposal mixtures, and sampler implementation work. The [sampling diagnostics](../../concepts/samplers/diagnostics.md) remain essential once the mathematical and implementation checks pass.
