---
title: SU(2) sampling
---

# SU(2) sampling

A gauge invariant SU(2) configuration carries both edge spins and the intertwiner channels that couple them at vertices. Sampling must explore both. Changing an edge spin can change the allowed channels at its endpoints, while different channels can describe distinct states with exactly the same edge spins.

[`MetropolisSU2`](../../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSU2) provides a coordinated sampler for the [SU(2) spin network basis](../../hilbert/su2.md). It combines four proposal families and includes their forward and reverse probability corrections. It can work without enumerating the physical basis.

## What is being sampled?

Write a complete basis label as $\sigma=(\mathbf n,\mathbf a)$, where $n_e=2j_e$ are doubled edge spins and $\mathbf a$ contains the doubled intermediate spins in the vertex recoupling trees. The target is

$$
p_\theta(\mathbf n,\mathbf a)
=\frac{|\psi_\theta(\mathbf n,\mathbf a)|^2}
{\displaystyle\sum_{(\mathbf m,\mathbf b)\ \mathrm{admissible}}
|\psi_\theta(\mathbf m,\mathbf b)|^2}.
$$

Every admissible choice of channels is a separate basis configuration. If one edge assignment admits several intertwiners, their probabilities are summed when asking for the probability of that edge assignment alone,

$$
p_\theta(\mathbf n)
=\sum_{\mathbf a\ \mathrm{admissible\ for}\ \mathbf n}
p_\theta(\mathbf n,\mathbf a).
$$

Even for a constant wavefunction, edge sectors need not have equal probability. Sectors with more channel assignments contain more basis states. There is no extra multiplicity weight to insert by hand when sampling the complete labels. The sampler already explores the intertwiners explicitly.

For example, on a K5 graph with every edge at $j=\tfrac12$, each of its five four-valent vertices has two channel choices. That one edge assignment represents $2^5=32$ distinct basis states. The [Hilbert space example](../../hilbert/su2.md#exact-bases-and-larger-calculations) shows this counting directly.

## Construct a sampled spin network state

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisSU2(hilbert, n_chains=8)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=17
)

samples, _ = state.sample()
print(samples.shape)
print(hilbert.edge_spins(samples).shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
(8, 32, 15)
(8, 32, 10)
True
```

The cutoff allows $j=0$ and $j=\tfrac12$. Each sample contains ten doubled edge spins followed by five channel labels. A channel can exceed the edge cutoff, so use the Hilbert space's [layout and coupling rules](../../hilbert/su2.md#vertex-coupling-and-intertwiner-labels) when interpreting these entries.

The example uses the default `exact=False`. The sampler does not require an exact dimension or a list of every physical state. The initial network is only an example of a log-amplitude model, and the short sampling settings demonstrate the interface rather than establish convergence.

## The proposal mixture

[`SU2MetropolisRule`](../../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.SU2MetropolisRule) mixes loop, same-parity edge, vertex-channel, and global proposals. Their default relative weights are `(0.45, 0.2, 0.25, 0.1)`, in that order. A separate `stay_probability=0.05` chooses an explicit stay before any physical update.

For normalised weights $w_r$ and stay probability $s$, family $r$ is attempted with probability $(1-s)w_r$. The supplied weights are normalised automatically. They must be finite and nonnegative, with a positive global weight. The stay probability must lie strictly between zero and one.

These settings control attempted proposal families, not the frequencies of accepted physical changes. Some proposals fail admissibility, some draw the current labels again, and others are rejected by the wavefunction acceptance step. The [proposal-family page](moves.md) explains all four modes and their mathematical corrections.

The global component gives every admissible finite-cutoff basis state a positive proposal probability, and the explicit stay makes the kernel aperiodic. At fixed parameters this supports convergence on the positive Born support. It does not guarantee a short mixing time for a sharply concentrated wavefunction. Monitor the actual chains using [samples and diagnostics](../diagnostics.md).

## Initial states and thermalisation

Starting labels come from the Hilbert space's random-state policy. With a cached exact basis, this can draw a basis state uniformly. Otherwise the default uses a finite Metropolis warmup with a uniform target. A vacuum initialisation is also available.

These choices establish valid starting configurations. The production sampler subsequently targets the neural wavefunction, which is generally not uniform. Its `n_discard_per_chain` is therefore a separate choice from the Hilbert initialisation warmup. The [initialisation guide](initialisation.md) explains every option and how to compare starting conditions.

## The unconstrained Peter–Weyl basis

[`SU2PeterWeylHilbert`](../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2PeterWeylHilbert) uses a different representation. One integer per edge encodes a magnetic basis label $(j,m,n)$. It is an unconstrained product space, without the vertex intertwiner coordinates used above.

Use [`MetropolisLocal`](../../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisLocal), or the generic sampler's local default, on this basis. `MetropolisSU2` specifically requires the gauge invariant spin network space.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.SU2PeterWeylHilbert(graph, two_j_max=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=128, n_discard_per_chain=32, seed=19
)

samples, _ = state.sample()
two_j, two_m, two_n = hilbert.decode(samples)
print(samples.shape, two_j.shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
print(bool(jnp.all(jnp.abs(two_m) <= two_j)))
```

```text
(4, 32, 1) (4, 32, 1)
True
True
```

A local replacement selects a different encoded edge state, so it may change the spin and both magnetic labels together. The generic neighbour rule can walk through the encoded ordering, but neighbouring integers need not represent a simple physical spin change. Read the [Peter–Weyl basis conventions](../../hilbert/su2.md#the-link-basis-before-gauss-law) when relating these moves to an observable.

For a new proposal beyond the supplied families, continue with the [custom sampler guide](../../../developers/samplers/index.md). The scientist-facing choices described here use the existing implementations and their tested corrections.
