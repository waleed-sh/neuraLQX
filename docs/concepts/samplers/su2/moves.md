---
title: SU(2) proposal families
---

# SU(2) proposal families

The SU(2) sampler uses several kinds of motion because no single local label change captures all the freedoms of a spin network. Loop moves change spin parity around cycles. Edge moves explore other spins within a parity sector. Vertex moves change intertwiners while holding the geometry's edge labels fixed. Global moves make the complete admissible basis reachable.

All four are part of [`MetropolisSU2`](../../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSU2). The [overview](index.md) gives a complete sampled-state example. This page explains what each mode proposes, why the probability corrections matter, and how to tune the mixture.

## Labels and vertex conditions

Use doubled labels $n_e=2j_e\in\{0,\ldots,J\}$, with $J=\texttt{two\_j\_max}$. A coupling of doubled spins $a,b,c$ is admissible when

$$
|a-b|\le c\le a+b,
\qquad a+b+c\equiv0\pmod2.
$$

At a higher-valence vertex, these conditions apply along its recoupling tree, ending in a singlet. The intermediate channel labels form part of the configuration. The [Hilbert guide](../../hilbert/su2.md#vertex-coupling-and-intertwiner-labels) explains their physical meaning and storage order.

One consequence is that the number of odd doubled edge labels incident at each vertex must be even. Thus the parity pattern $n_e\bmod2$ forms a binary cycle-space element. This motivates both the loop and global proposals. Triangle inequalities impose further conditions beyond parity.

## Loop moves

A loop proposal selects a loop and shifts the doubled spin on each of its edges by one,

$$
n'_e=
\begin{cases}
n_e+\varepsilon_e,&e\in\gamma,\\
n_e,&e\notin\gamma,
\end{cases}
\qquad \varepsilon_e\in\{-1,+1\}.
$$

In physical spins this is a change of $\pm\tfrac12$. Each selected edge changes parity. A closed loop changes the incident parity an even number of times at each vertex, so the parity condition survives. The remaining admissibility conditions are checked when the affected vertex channels are drawn.

The loop collection includes the space's graph loops together with fundamental cycles of a spanning forest, with duplicates removed. Labels that would fall below zero or exceed the cutoff are rejected. There is no cyclic wrapping of spin at $0$ or $J$.

By default, the signs are chosen independently on different edges. Setting `coherent_loop_probability` to a positive value mixes in proposals that use one common sign on the whole loop. Both sign distributions have symmetric reverse shifts. The channel reconstruction can still be asymmetric, and its correction is retained.

### Leaving the vacuum on a long loop

Suppose a loop has $\ell$ edges and the current configuration is the vacuum. For $J\ge1$, only the all-positive sign choice can raise that loop from zero without proposing negative spins. Independent signs choose it with probability $2^{-\ell}$. A common sign chooses it with probability one half.

If the coherent probability is $\eta$, the probability of proposing the all-positive spin shift, conditional on selecting that loop, is

$$
(1-\eta)2^{-\ell}+\frac{\eta}{2}.
$$

This explains why coherent moves can help long cycles leave the vacuum. The expression is a proposal probability before the wavefunction acceptance step, not a promised acceptance rate. The default is `coherent_loop_probability=0.0`, and any value from zero to one is accepted.

## Same-parity edge moves

An edge proposal chooses one edge and redraws its doubled spin from values with the same parity. If $p=n_e\bmod2$, it draws

$$
\begin{gathered}
n'_e\in\{p,p+2,\ldots,p+2(N_p-1)\},\\
N_p=\left\lfloor\frac{J-p}{2}\right\rfloor+1.
\end{gathered}
$$

The draw is uniform and includes the current value. In physical spin units, it can change the selected spin by an integer amount. Preserving parity avoids changing the vertex parity constraints through a single edge, while redrawing channels at the incident vertices checks the remaining coupling conditions.

The spin redraw has equal forward and reverse probabilities within that parity class. Its Hastings correction comes from the affected channel draws. Channels at untouched vertices are retained.

At `two_j_max=1`, each parity class contains only one edge label. This move cannot change the selected edge spin, although it can redraw channels at its endpoints. At larger cutoffs it also explores different spins in the same parity class.

## Vertex-channel moves

A vertex proposal selects one vertex, keeps all edge spins fixed, and redraws its stored recoupling channels. This is needed to explore distinct invariant tensors associated with the same edge assignment.

For four incident spins $j=\tfrac12$, the intermediate spin can be $k=0$ or $k=1$, stored as doubled labels `0` and `2`. A vertex proposal can move between these states without changing an edge. More generally, a vertex of valence $d$ has $\max(d-3,0)$ stored channels.

At an admissible vertex of valence at most three, no independent channel is stored, so a vertex-only proposal leaves the configuration unchanged. On a graph consisting only of such vertices, you can set the vertex weight to zero and assign that effort to other families. The global weight must remain positive.

## How intertwiner channels are drawn

After a spin change, the sampler needs a valid channel assignment at every affected vertex. It builds the channels sequentially along the existing recoupling tree. At each step it chooses uniformly among intermediate spins that satisfy the current triangle condition and still allow the remaining legs to finish in a singlet.

For the remaining doubled spins $b_1,\ldots,b_m$, let $S=\sum_i b_i$. Their reachable total doubled spins run in steps of two from

$$
b_{\min}=\max\left(S\bmod2,\ 2\max_i b_i-S\right)
$$

to $S$. Intersecting this range with the current triangle interval determines which next channels are feasible. This avoids enumerating every complete intertwiner path.

Let $N_{v,r}$ be the number of feasible choices at channel step $r$, conditional on the preceding choices and the incident edge labels. The probability of a complete channel assignment is

$$
r_v(\mathbf a_v\mid\mathbf n_v)
=\prod_r\frac1{N_{v,r}}.
$$

These conditional counts can depend on the earlier channel choices. Consequently, sequential uniform draws need not be uniform over complete intertwiner assignments at higher valence. Simply returning a valid channel would not give the correct sampling distribution.

For the affected vertices, the sampler includes

$$
\Delta_{\mathrm{channels}}
=\sum_{v\ \mathrm{affected}}
\log\frac{r_v(\mathbf a_v\mid\mathbf n_v)}
{r_v(\mathbf a'_v\mid\mathbf n'_v)}
$$

in the [Metropolis log ratio](../metropolis.md#propose-compare-accept). At a four-valent vertex with $N_v$ admissible channels, the draw is uniform over those channels. Its contribution reduces to $\log N'_v-\log N_v$. This accounts for changes in intertwiner multiplicity when edge spins change.

If no admissible completion exists, the whole proposal becomes a stay. The implementation does not retry until it finds a valid configuration. Such retries would condition the proposal on success and change the probability ratio.

## Global proposals

The global family proposes an entirely new admissible spin network through its parity, edge spins, and channels. Let $B$ be a binary cycle basis and let $F$ be its dimension. Draw independent uniform bits $z\in\{0,1\}^F$ and set

$$
\mathbf p=Bz\pmod2.
$$

This generates a uniform cycle-space parity pattern, including the zero pattern. Each edge spin is then drawn uniformly from its parity class, and the channels are drawn by the feasible sequential construction above.

For a valid configuration and $J>0$, the proposal mass is

$$
q_{\mathrm{global}}(\mathbf n,\mathbf a)
=2^{-F}
\prod_e\frac1{N_{p_e}}
\prod_v r_v(\mathbf a_v\mid\mathbf n_v).
$$

Every admissible configuration has positive mass. The graph construction includes bridges, disconnected components, parallel edges, and self-loops in its parity accounting. Parity patterns that fail the additional spin-coupling conditions produce rejected attempts rather than being removed by renormalising this expression.

Because this is an independence proposal, its correction is

$$
\Delta_{\mathrm{global}}
=\log q_{\mathrm{global}}(\sigma)
-\log q_{\mathrm{global}}(\sigma').
$$

It is generally not a uniform draw over physical configurations. A small triangle makes that difference visible. Its bivalent vertices require all three edge spins to agree.

```python
import numpy as np
import neuralqx as nqx

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=2, exact=True)
sampler = nqx.sampler.MetropolisSU2(
    hilbert, weights=(0.0, 0.0, 0.0, 1.0), n_chains=4
)
basis = hilbert.all_states()
proposal_mass = np.exp(np.asarray(sampler.rule.global_log_proposal(hilbert, basis)))

print(basis.tolist())
print(np.round(proposal_mass, 6).tolist())
print(round(float(proposal_mass.sum()), 6))
```

```text
[[0, 0, 0], [1, 1, 1], [2, 2, 2]]
[0.0625, 0.5, 0.0625]
0.625
```

The odd parity has only one available label per edge. The even parity has two, and independent edge draws can disagree at a vertex. The missing mass $1-0.625$ corresponds to invalid attempted draws that stay at the source. It is not a normalisation factor to divide out. `global_log_proposal` reports the conditional global-family mass before the separate family selection and explicit stay.

## Configure the mixture

The weights are ordered as `(loop, edge, vertex, global)`. The following gives more weight to loop motion and mixes common-sign shifts into those proposals.

```python
import jax.numpy as jnp
import neuralqx as nqx

hilbert = nqx.hilbert.SU2Hilbert(nqx.graph.K5Graph(), two_j_max=2)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisSU2(
    hilbert,
    weights=(0.6, 0.15, 0.15, 0.1),
    coherent_loop_probability=0.5,
    stay_probability=0.05,
    sweep_size=8,
    n_chains=4,
)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=128, n_discard_per_chain=32, seed=23
)
samples, _ = state.sample()
print(samples.shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
(4, 32, 15)
True
```

This is an example of configuring the mixture, not an optimum for every graph. Compare movement of spins, parity sectors, channels, and physical observables before choosing weights. Increasing a family's weight only allocates it more attempts. It need not increase useful motion if those attempts are mostly rejected or leave the labels unchanged.

Keep a positive global weight and a nonzero stay probability, as required by the constructor. At cutoff zero the vacuum is the only state and all proposals stay there. For nontrivial spaces, the [diagnostics guide](../diagnostics.md) explains how to distinguish formal reachability from adequate exploration in a finite run.
