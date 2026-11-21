---
title: Initialising SU(2) chains
---

# Initialising SU(2) chains

Before the sampler can follow a wavefunction, each chain needs a valid spin network configuration. Drawing every spin and channel independently from its numerical range is usually inefficient because most combinations violate vertex admissibility. The SU(2) Hilbert space therefore provides initialisation methods that construct valid states directly or move between them.

Initialisation is controlled by [`SU2Hilbert`](../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2GaugeInvariantHilbert), also named `SU2GaugeInvariantHilbert`. The production [`MetropolisSU2`](../../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSU2) sampler uses that policy when it needs new starting configurations.

## The default Metropolis warmup

With the default `exact=False` and `random_init='metropolis'`, each requested initial configuration starts from the vacuum and undergoes a finite number of valid Metropolis updates. This preparatory chain targets the uniform law on admissible complete basis states,

$$
p_{\mathrm{init}}(\sigma)\propto1.
$$

The wavefunction contribution to the acceptance ratio is therefore zero. The proposal correction remains necessary,

$$
a_{\mathrm{init}}(\sigma\to\sigma')
=\min\left(1,\frac{q(\sigma\mid\sigma')}{q(\sigma'\mid\sigma)}\right).
$$

The constructor's `init_warmup` controls the number of attempted updates per initial configuration. With no explicit value, it uses `max(32, 4 * n_edges)` for a graph with edges. A call to `hilbert.random_state(..., n_steps=...)` can override that count.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

hilbert = nqx.hilbert.SU2Hilbert(
    nqx.graph.K5Graph(), two_j_max=1, init_warmup=64
)
first_key, longer_key = jax.random.split(jax.random.key(29))
initial = hilbert.random_state(first_key, size=8)
longer = hilbert.random_state(longer_key, size=8, n_steps=128)

print(initial.shape, longer.shape)
print(bool(jnp.all(hilbert.is_valid(initial))))
print(bool(jnp.all(hilbert.is_valid(longer))))
```

```text
(8, 15) (8, 15)
True
True
```

These are valid initial configurations. A finite preparatory run does not guarantee that its endpoints follow the uniform equilibrium distribution. It also does not prepare the Born distribution of a subsequently supplied neural wavefunction.

The initialisation warmup uses the default proposal-family weights and stay probability, with `coherent_loop_probability=0.5`. It does not inherit the weights of a separately constructed production sampler. Thus changing `MetropolisSU2(..., weights=...)` affects production motion, while changing `init_warmup` affects the starting-state preparation.

## Uniform draws from an exact cached basis

For a small space, `exact=True` enumerates and caches all admissible basis configurations. With no per-call method override, `random_state` then draws uniformly from that cache,

$$
p_{\mathrm{init}}(\sigma)=\frac1{\dim\mathcal H}.
$$

This is a direct uniform basis draw. It is still independent of the wavefunction and should not be mistaken for exact Born sampling of an arbitrary model.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=2, exact=True)
uniform_key, vacuum_key, warmup_key = jax.random.split(jax.random.key(31), 3)

uniform = hilbert.random_state(uniform_key, size=8)
vacuum = hilbert.random_state(vacuum_key, size=8, method='vacuum')
warmup = hilbert.random_state(
    warmup_key, size=8, method='metropolis', n_steps=64
)
print(hilbert.dimension)
print(uniform.shape, vacuum.shape, warmup.shape)
print(bool(jnp.all(hilbert.is_valid(uniform))))
print(bool(jnp.all(vacuum == 0)))
```

```text
3
(8, 3) (8, 3) (8, 3)
True
True
```

The explicit `method` takes precedence over the cache. With `method=None`, a cached exact basis takes precedence over the constructor's `random_init` setting. To make production chains start from the vacuum through their default initialisation call, use an uncached space with `random_init='vacuum'`.

The [exact-basis guide](../../hilbert/su2.md#exact-bases-and-larger-calculations) explains the cost and `max_exact_states` guard. Large simulations usually leave enumeration disabled. Initialising chains should not require allocating the full many-body basis.

## Starting deliberately from the vacuum

The vacuum has zero on every edge and every channel. Set `random_init='vacuum'` on an uncached Hilbert space when you deliberately want this starting condition. For a standalone draw, `method='vacuum'` requests it explicitly. A Metropolis initialisation with zero warmup steps also returns its vacuum starting point.

The vacuum is a valid physical configuration. It can appear naturally in random initialisation and in production sampling, and it is the only configuration at cutoff zero. Filtering it out would change the distribution unless the model itself assigns it zero amplitude.

For a nontrivial space, identical vacuum starts are useful for a controlled comparison but provide little evidence of mixing by themselves. Compare with longer warmups or other starting conditions. On long loops, [coherent loop proposals](moves.md#leaving-the-vacuum-on-a-long-loop) can help the production chains leave the vacuum efficiently.

## Thermalise against the actual wavefunction

After initialisation, the production target is $|\psi_\theta|^2$. The state's `n_discard_per_chain` controls discarded stored points under that target. Its units differ from `init_warmup`, which counts attempted preparatory updates. A production discard of $B$ points uses $B\times\text{sweep size}$ attempted moves per chain.

With `reset_chains=False`, the default, later resets retain chain positions and refresh their probabilities. The Hilbert initialiser is used when starting the chains, rather than before every batch. With `reset_chains=True`, resetting redraws positions using the same Hilbert policy.

Check observables after increasing the discard and after changing the starting policy. Agreement across such runs is more informative than validity alone. The [diagnostics guide](../diagnostics.md) explains how to inspect both spin and intertwiner motion while retaining the distinction between independent chains and correlated samples along each chain.
