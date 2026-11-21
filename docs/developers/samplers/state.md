---
title: Sampler state
---

# Sampler state

Sampler state records where the chains are, which random numbers come next, and which cached quantities remain valid. Treating this state explicitly makes a sampling sequence reproducible and allows JAX to compile repeated transitions without depending on mutable Python objects.

[`MetropolisSamplerState`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSamplerState) is the state used by the supplied Metropolis algorithms. A different algorithm can define its own [`AbstractSamplerState`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.AbstractSamplerState), as shown in the [sampler-contract example](contracts.md).

## Stored quantities

| Field | Meaning |
| --- | --- |
| `sigma` | Current configurations, with shape `(B, N)`. |
| `log_prob` | Unnormalised log target weights of those configurations, with shape `(B,)`. |
| `rng` | Random state used to generate subsequent transitions. |
| `rule_state` | Optional pytree owned by the transition rule. |
| `n_steps` | Total attempted proposals across the chain ensemble since reset. |
| `n_accepted_per_chain` | Accepted proposals for each chain since reset. |

The derived `acceptance` property divides the total accepted count by `n_steps`. Before any counted proposals it is `NaN`. An accepted self transition contributes to the accepted count, so this statistic does not measure the fraction of proposals that changed the configuration.

For $B$ chains, `sweep_size=s`, and a retained chain length $T$, ordinary sampling attempts $BsT$ proposals. One retained sample follows a complete sweep of $s$ proposal rounds. Each round proposes once for every chain.

## Initialisation and target changes

`init_state` allocates the Metropolis state and initialises rule state. It does not finish evaluating the target. Its `log_prob` values start at negative infinity. With `reset_chains=False`, it also generates initial configurations. With `reset_chains=True`, random generation is deferred to reset.

Use `sampler.reset(apply_fn, parameters, key=key)` for a ready initial state. Reset evaluates log weights, clears the acceptance counters, and calls the rule's reset hook. For an existing state, the default `reset_chains=False` preserves configurations while refreshing their weights. Setting `reset_chains=True` redraws configurations as part of that operation.

This distinction matters when parameters change. Calling `sample` with an existing state does not first recompute its cached weights. In a manual loop, call `reset` with the new parameters before continuing. Otherwise a new candidate's weight may be compared with an old parameter set's cached weight. The [variational-state interface](../../concepts/vqs/state.md) manages its own sampling invalidation when parameters are assigned through its supported setters.

For projected models, the Metropolis implementation also has a finite-support initialisation step. It can attempt transitions from zero-amplitude configurations before recording samples. This is not a replacement for equilibration or for a suitable random initialiser. An initial configuration outside every component of the nonzero support cannot be repaired by a rule that never reaches that support.

## A rule with explicit state

The following rule adds a count of proposal rounds to the supplied [`LocalRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.LocalRule). It uses a scalar array so the pytree structure and dtype remain fixed through the compiled loop.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import Struct

class ProposalCounter(Struct):
    rounds: jax.Array

class CountingLocalRule(nqx.sampler.LocalRule):
    def init_state(self, sampler, apply_fn, parameters, key):
        return ProposalCounter(rounds=jnp.asarray(0, dtype=jnp.int32))

    def reset(self, sampler, apply_fn, parameters, sampler_state):
        return sampler_state.rule_state.replace(rounds=jnp.asarray(0, dtype=jnp.int32))

    def transition(self, sampler, apply_fn, parameters, sampler_state, key, sigma):
        proposed, correction = super().transition(
            sampler, apply_fn, parameters, sampler_state, key, sigma
        )
        counter = sampler_state.rule_state.replace(
            rounds=sampler_state.rule_state.rounds + 1
        )
        return nqx.sampler.TransitionOutput(proposed, correction, counter)

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
sampler = nqx.sampler.MetropolisSampler(
    hilbert, rule=CountingLocalRule(), n_chains=4, sweep_size=3,
    reset_chains=False,
)

def log_value(parameters, states):
    return states @ parameters

parameters = jnp.array([0.2, -0.3])
initial = sampler.reset(log_value, parameters, key=jax.random.key(14))
samples, state = sampler.sample(
    log_value, parameters, state=initial, chain_length=5
)
replayed, replay_state = sampler.sample(
    log_value, parameters, state=initial, chain_length=5
)
np.testing.assert_array_equal(samples, replayed)
np.testing.assert_array_equal(
    jax.random.key_data(state.rng), jax.random.key_data(replay_state.rng)
)
assert int(initial.rule_state.rounds) == 0
assert int(state.rule_state.rounds) == 15
assert int(state.n_steps) == 60

new_parameters = parameters + 0.1
refreshed = sampler.reset(log_value, new_parameters, state=state)
np.testing.assert_array_equal(refreshed.sigma, state.sigma)
np.testing.assert_allclose(
    refreshed.log_prob, 2 * log_value(new_parameters, refreshed.sigma)
)
assert int(refreshed.n_steps) == 0
assert int(refreshed.rule_state.rounds) == 0
print(int(state.rule_state.rounds), int(state.n_steps))
```

```text
15 60
```

Replaying from the same state is intentional in this test. In a simulation, continue from the returned state to advance the random sequence. Supplying another `key` alongside an existing state is not a mechanism for reseeding that state. Start a new state explicitly if a new sequence is wanted.

## What happens to rule state on rejection

The Metropolis step selects between the old and proposed `sigma` and `log_prob` using its acceptance mask. It retains the transition's returned `rule_state` **without applying that mask**. The round counter above is valid because it counts attempted rounds, including rejected moves.

A cache claimed to describe the accepted configuration has a different requirement. Returning a cache computed only for the candidate would leave it inconsistent whenever that candidate is rejected. Recompute such information from the current `sigma`, retain enough information to reconstruct it correctly on the next transition, or implement an algorithm whose state update explicitly includes the acceptance decision. The transition-rule interface does not provide a post-acceptance callback.

Returning `rule_state=None` in a transition means retaining the previous rule state. It is not an instruction to clear it. If a state needs an inactive condition, represent that condition with an explicit field while keeping the pytree structure stable.

## JAX and distributed execution

Inside a compiled sweep, rule state is carried through JAX loops. Its tree structure, leaf shapes, and dtypes must stay fixed. Counters and numerical caches belong in dynamic fields. Descriptions, immutable mode choices, and structural dimensions belong in static fields, following the [struct field guide](../structs/fields.md). Split the supplied random key for independent random choices, as described in the [JAX random-number documentation](https://docs.jax.dev/en/latest/101/random.html).

The current sample-sharded Metropolis runtime partitions configurations and per-chain counters across the sample axis. It passes a local sampler view to transition rules, so `sigma.shape[0]` and `sampler.n_batches` describe the local batch there. Random initialisers should use that local count instead of the global `n_chains`.

Custom `rule_state` is currently passed through this runtime as **replicated state**, not automatically partitioned per chain. A deterministic round counter that agrees on every shard fits that contract. A per-chain cache, a random scalar that differs between shards, or a statistic computed from only local configurations does not become a valid replicated result automatically. Such state needs an explicit distributed design and, where necessary, changes to the sampler runtime. Test on the intended device layout before claiming distributed support for a custom stateful rule.

## Saving and resuming

Struct state dictionaries can preserve nested rule state, arrays, and random keys. The [struct serialisation guide](../structs/serialisation.md) explains both dictionary and file round trips. Keep custom state classes importable when restoring in another process.

A sampler-state checkpoint alone does not define a simulation. Also retain the Hilbert space construction, sampler and rule settings, model variables, and any importance-sampling configuration. For a complete variational calculation, start with the [state saving and loading guide](../../concepts/vqs/saving.md). After changing parameters or reconstructing a different target, refresh cached log weights before continuing. Do not assume that moving a checkpoint to a different number of sampling devices reproduces the same random sequence.
