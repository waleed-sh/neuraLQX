---
title: Transition rules
---

# Transition rules

A transition rule specifies which configurations a chain can propose. Choosing a useful rule requires knowing which labels are independent, which combinations are allowed, and which quantities must remain fixed. The Metropolis acceptance step then weights those proposals using the current wavefunction.

This page covers using the supplied rules. Implementing a new rule, its state, or its proposal correction belongs in the [developer guide to custom samplers](../../developers/samplers/index.md).

## Start with the Hilbert space

Without an explicit `rule`, [`MetropolisSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSampler) chooses a U(1) gauge rule for a constructive U(1) gauge space, an SU(2) rule for a gauge invariant SU(2) space, and a local rule otherwise. The named constructors below let you choose explicitly and expose the settings for each family.

| Constructor | Proposal | Suitable use |
| --- | --- | --- |
| [`MetropolisLocal`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisLocal) | Replace one flat site's value | Unconstrained product spaces, or restrictions that permit single-site motion |
| [`MetropolisNeighbor`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisNeighbor) | Move to an adjacent value in one site's local ordering | A bounded walk through ordered local values |
| [`MetropolisExchange`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisExchange) | Swap values on a selected pair of flat sites | A sector whose allowed configurations are connected by the specified exchanges |
| [`MetropolisU1Gauge`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisU1Gauge) | Update free charges and reconstruct dependent charges | Constructive [U(1) gauge spaces](u1.md) |
| [`MetropolisSU2`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSU2) | Mix loop, edge, intertwiner, and global moves | Gauge invariant [SU(2) spin networks](su2/index.md) |

The named constructors are explicit choices. In particular, `MetropolisLocal(gauge_hilbert)` selects the local rule, rather than asking for the Hilbert-aware default. For a gauge space, start with its specialised constructor or with `MetropolisSampler(hilbert)`.

## Local replacement

[`LocalRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.LocalRule) selects a flat site uniformly and replaces its current value by a uniformly selected different value from that site's local domain. If site $i$ has $d_i>1$ values and there are $D$ sites, the probability of a particular replacement is

$$
q(\sigma'\mid\sigma)=\frac1D\frac1{d_i-1}.
$$

The reverse replacement has the same probability, so no Hastings correction is needed. A binary domain gives a flip. A site with only one value remains unchanged. For heterogeneous spaces, each selected site uses its own domain.

On a constrained space, a proposed replacement that fails `hilbert.is_valid` becomes a stay at the current configuration. This preserves the constraint but can prevent exploration. For example, a single flip in a fixed-total binary occupation space always changes the total. Every nontrivial local proposal is then excluded, even though many valid configurations exist.

For U(1) gauge conditions, coordinated charge updates solve this problem. For SU(2), edge spins and vertex channels must be treated together. The corresponding guides explain these constructions rather than relying on rejection of arbitrary label changes.

## Moves between neighbouring local values

[`NeighborRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.NeighborRule) moves one position up or down in the selected site's local ordering. For an arithmetic range, this changes the label by one range step. For an explicitly listed domain, adjacency refers to positions in that list, which need not have equal numerical spacing.

This rule does not wrap at a boundary. An interior value has two neighbours, while an endpoint has one. If $d(r)$ is the number of available neighbours of local index $r$, the correction is

$$
\log\frac{q(\sigma\mid\sigma')}{q(\sigma'\mid\sigma)}
=\log d(r)-\log d(r').
$$

For example, a proposal from an endpoint to the interior is certain once the site is selected, while its reverse has probability one half. With a flat target, the forward move is accepted with probability one half. The correction prevents the chain from overpopulating interior values merely because they have more neighbours.

Invalid constrained updates again become stays. The modular neighbour option in [U(1) gauge sampling](u1.md#modular-neighbour-moves) is a different rule, with cyclic boundaries and a symmetric proposal.

## Exchange moves and conserved sectors

[`ExchangeRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.ExchangeRule) swaps values on a pair of flat sites. This preserves the multiset of labels in the connected exchange region, including their total. That makes it useful for fixed occupations, but prevents it from exploring a larger space in which those counts should change.

The following example has four binary edge occupations and fixes their total to two. All pairs of edge sites can exchange, so the moves connect the six configurations in this sector. We draw a valid starting batch before entering the Metropolis loop and place it in a [`MetropolisSamplerState`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSamplerState), making the sector of every chain explicit.

```python
from itertools import combinations
import jax
import jax.numpy as jnp
import neuralqx as nqx


def log_amplitude(parameters, configurations):
    return jnp.sum(parameters * configurations, axis=-1)


graph = nqx.graph.Grid2D(2, 2)
constraint = nqx.hilbert.LinearConstraint.sum(
    sites=tuple(range(graph.n_edges)), target=2
)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2), constraint=constraint
)
sampler = nqx.sampler.MetropolisExchange(
    hilbert,
    clusters=tuple(combinations(range(hilbert.size), 2)),
    n_chains=4,
)
parameters = jnp.array([0.1, -0.2, 0.2, -0.1])
initial = hilbert.random_state(jax.random.key(8), size=sampler.n_chains)
chain = nqx.sampler.MetropolisSamplerState(
    sigma=initial,
    log_prob=sampler.log_probability(log_amplitude, parameters, initial),
    rng=jax.random.key(9),
)
_, chain = sampler.sample(log_amplitude, parameters, state=chain, chain_length=32)
samples, chain = sampler.sample(
    log_amplitude, parameters, state=chain, chain_length=64
)
print(samples.shape)
print(bool(jnp.all(jnp.sum(samples, axis=-1) == 2)))
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
(4, 64, 4)
True
True
```

The [`LinearConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.LinearConstraint) sets the sector and ensures that initial configurations belong to it. The sampler state records those configurations, their current log target, and a separate random key for the Markov updates. Passing the returned state onward continues the same chains.

The specified swaps preserve the total occupation. Exchange proposals do not perform a general constraint or local-domain repair. Use pairs with compatible domains, and establish that swapping their values preserves your particular restriction. An exchange of arbitrary U(1) edges or SU(2) channels need not do so.

Only pairs whose current values differ are eligible. If pair $k$ has positive weight $w_k$, define

$$
W(\sigma)=\sum_{k:\,\sigma_{i_k}\ne\sigma_{j_k}}w_k.
$$

The selected pair is drawn with probability $w_k/W(\sigma)$. Its reverse has the same pair weight but can have a different total eligible weight, giving correction $\log W(\sigma)-\log W(\sigma')$. The implementation includes this correction. With no eligible pair, the configuration stays unchanged. The `probabilities` argument supplies the pair weights.

You may provide `graph` and `d_max` instead of explicit `clusters`. With `entity='vertex'`, pairs follow vertex distance. With `entity='edge'`, pairs follow edge adjacency. These indices must match the corresponding flat sites in your layout. For multiple gauge components or mixed edge and vertex labels, explicit clusters make that mapping clear.

## Assess reachability as well as validity

An accepted configuration can satisfy every Hilbert constraint while the chain remains confined to a small subset of the basis. Fixed label counts under exchange, a disconnected exchange graph, or an SU(2) proposal that never changes parity are examples of restrictions introduced by the moves themselves.

The supplied U(1) default connects the constructive space through independent-coordinate replacements. The SU(2) mixture includes a global component for full finite-cutoff support. Actual motion still depends on the wavefunction, especially when it has exact zeros or strongly separated probability peaks. Use [chain diagnostics](diagnostics.md) to assess the resulting simulation.

If a different collective update is needed, follow the [custom transition-rule guide](../../developers/samplers/transition-rules.md). A new rule must account for its forward and reverse proposal probabilities as well as the configurations it produces.
