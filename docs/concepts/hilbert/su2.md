---
title: SU(2) spaces
---

# SU(2) spaces

SU(2) introduces more than a different set of edge labels. Its representations have magnetic indices, and gauge invariance couples the representations meeting at a vertex through an intertwiner. A useful basis must account for both the edge representations and this vertex structure.

neuraLQX provides two SU(2) spaces. [`SU2Hilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2GaugeInvariantHilbert) is the gauge invariant spin network space. [`SU2PeterWeylHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2PeterWeylHilbert) is the link basis before Gauss law, with magnetic indices retained. Choose the representation appropriate to your operators and variational calculation. Their configuration arrays have different meanings.

## The graph Hilbert space and its cutoff

Before imposing gauge invariance, the continuum link space on a fixed graph $\Gamma$ is

$$
\mathcal H_\Gamma
= L^2\!\left(\mathrm{SU}(2)^E,\mathrm d\mu_H^{\otimes E}\right),
$$

where $E$ is the number of edges and $\mu_H$ is normalised Haar measure. The Peter–Weyl decomposition resolves each link into irreducible representations:

$$
L^2\!\left(\mathrm{SU}(2),\mathrm d\mu_H\right)
\cong \bigoplus_{j=0,\frac12,1,\ldots} V_j\otimes V_j^*.
$$

Each $V_j$ has dimension $2j+1$. Gauge invariant states arise by contracting the representation indices at each vertex with invariant tensors. The [review by Ashtekar and Lewandowski, Section IV](https://arxiv.org/html/gr-qc/0404018#S4) develops this relation between graph connections and spin network states.

The implementation uses a finite edge cutoff and integer **doubled spin labels**:

$$
n_e=2j_e\in\{0,1,\ldots,J\},
\qquad J=\texttt{two\_j\_max}.
$$

Thus `two_j_max=1` allows $j=0$ and $j=\tfrac12$, while `two_j_max=2` also allows $j=1$. Pass integer doubled labels in configurations. A stored value of `1` means spin one half. The gauge invariant constructor also accepts `cutoff` as an alias for `two_j_max`, with the same doubled spin meaning.

Both supplied representations include the trivial $j=0$ representation on each edge. A zero spin label keeps the edge in the computational graph. It does not delete the edge or change the layout. The cutoff is part of the model approximation, so study its effect when interpreting observables.

## Construct a gauge invariant spin network space

[`SU2GaugeInvariantHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2GaugeInvariantHilbert) is the full name of the class also exported as `SU2Hilbert`. It accepts only configurations whose incident spins and vertex channels satisfy the SU(2) coupling rules.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
samples = hilbert.random_state(jax.random.key(12), size=8)

print(hilbert.n_edges, hilbert.n_channel_sites, hilbert.size)
print(samples.shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
print(hilbert.dimension, hilbert.is_indexable)
```

```text
10 5 15
(8, 15)
True
None False
```

[`K5Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.K5Graph) has ten edges and five vertices of valence four. The edge spins occupy ten entries, and each vertex contributes one independent recoupling channel. A complete configuration therefore has fifteen scalar entries. Its basis dimension is a separate quantity, which is left unknown by default to avoid enumerating all states.

Mathematically, the gauge invariant basis can be written as $|\{j_e\},\{\iota_v\}\rangle$, where $\iota_v$ labels a basis of the invariant tensor space at vertex $v$. For each fixed assignment of edge spins, the vertex multiplicities determine how many spin network states exist:

$$
\dim\mathcal H_{\Gamma,J}^{\mathrm{GI}}
= \sum_{\{j_e\leq J/2\}}
  \prod_v
  \dim\operatorname{Inv}\!\left(
    \bigotimes_{e\ni v} V_{j_e}^{\epsilon(v,e)}
  \right).
$$

Here $V_j^{\epsilon(v,e)}$ denotes the representation or its dual according to the edge orientation. For SU(2), the dual is equivalent to the same spin representation, so the admissible spin labels can be determined from the incident spins. Orientation still matters to the conventions used by operators. A self loop contributes two incident legs at its vertex.

The built-in space imposes gauge invariance at every graph vertex, including endpoints. A vertex of valence one can therefore carry only a trivial incident spin. For example, a star whose outer endpoints have valence one does not support arbitrary nonzero edge spins in this space. Boundary charges or open representation indices require a representation and constraints that describe that boundary physics explicitly, such as a calculation starting in the link basis below.

## Vertex coupling and intertwiner labels

Three spins can couple to an invariant tensor when

$$
|j_1-j_2|\leq j_3\leq j_1+j_2,
\qquad j_1+j_2+j_3\in\mathbb Z.
$$

In doubled labels, the parity condition becomes $n_1+n_2+n_3$ even. The triple $(1,1,0)$ is admissible, while $(1,1,1)$ is not. When a trivalent vertex is admissible, its invariant tensor space has dimension one, so there is no extra channel entry to store. A bivalent vertex requires equal incident spins, also without an additional channel.

At higher valence, more than one invariant tensor may be available for the same edge spins. neuraLQX uses a **left comb recoupling basis**, currently the only supported `recoupling` option. At a four-valent vertex, the first pair of incident spins couples to an intermediate spin $k$. The other pair must couple to that same $k$ so that the complete vertex can form a singlet:

$$
\begin{aligned}
|j_1-j_2|&\leq k\leq j_1+j_2,\qquad j_1+j_2+k\in\mathbb Z,\\
|j_3-j_4|&\leq k\leq j_3+j_4,\qquad j_3+j_4+k\in\mathbb Z.
\end{aligned}
$$

For four incident spins $\tfrac12$, the allowed channels are $k=0$ and $k=1$. The stored channel labels are therefore `0` and `2`. They identify two different intertwiners with the same edge spins. A channel label is the doubled intermediate spin, not an ordinal such as “first intertwiner” or “second intertwiner”.

More generally, a vertex of valence $d>3$ has $d-3$ channel entries. The first couples $j_1$ and $j_2$, each subsequent channel couples the previous channel with the next incident spin, and the final channel must couple admissibly with the last two edge spins. The graph edge order determines the order of incident legs in this construction. You can inspect it through `graph_metadata.incident_edges_by_vertex`, or retrieve the ordered labels with `local_vertex_state`.

Intermediate spins can exceed the edge cutoff. For a channel with zero-based position $r$, the layout allocates the doubled upper bound

$$
2k_r\leq\min(r+2,d-r-2)J.
$$

The coupling conditions then select the allowed values inside those bounds. This is why the K5 space with `two_j_max=1` has local dimension two on each edge and local dimension three on each channel. The additional channel label `2` is necessary even though edge labels stop at `1`.

## Build and inspect a configuration

The flat order is all edge doubled spins followed by the channels of each vertex in graph vertex order. The specialised view exposes `edge_spins` and `vertex_channels`. See the [state layout guide](state-layouts.md#the-su2-spin-network-layout) for batch shapes and round trips.

For a fixed assignment of edge spins, `complete_channels_for_edges` finds one admissible completion. It returns a complete NumPy configuration, or `None` when no completion exists.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
state = hilbert.complete_channels_for_edges([1] * graph.n_edges)
assert state is not None

edge_labels, channel_labels = hilbert.local_vertex_state(state, graph.vertices[0])
first_channel = hilbert.layout.channel_site(graph.vertices[0], 0)
alternative = jnp.asarray(state).at[first_channel].set(2)
invalid = jnp.asarray(state).at[first_channel].set(1)

print(hilbert.view(state).vertex_channels.tolist())
print(edge_labels.tolist(), channel_labels.tolist())
print(bool(hilbert.is_valid(alternative)))
print(bool(hilbert.is_valid(invalid)))
print(hilbert.explain_invalid_state(invalid)["reason"])
```

```text
[0, 0, 0, 0, 0]
[1, 1, 1, 1] [0]
True
False
vertex_admissibility
```

Completion chooses the first admissible channel assignment at each vertex. It is useful for preparing a particular configuration, but it does not sample all intertwiners for those edge spins. In this example, replacing one channel by `2` gives another valid basis state. Replacing it by `1` violates the coupling parity rule even though `1` is inside that site's allocated numerical range.

Use `is_valid` for a single configuration or batch. `explain_invalid_state` is a diagnostic for one integer configuration and reports the first failed condition, including bounds or vertex admissibility. `site_info(site)` identifies whether an entry is an edge or channel and gives its allowed bound. These helpers are useful when comparing externally prepared spin network data with neuraLQX's recoupling convention.

Changing the graph order or recoupling convention can change the interpretation of channel coordinates. Keep that information with saved configurations. Matching array lengths alone does not establish that two arrays use the same basis.

## Exact bases and larger calculations

For small graphs and cutoffs, `exact=True` enumerates and caches the valid basis during construction. This supplies the exact dimension, basis numbering, and direct draws from the enumerated states.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(
    graph,
    two_j_max=1,
    exact=True,
    max_exact_states=1000,
)
basis = hilbert.all_states(max_states=1000)
all_half_spins = jnp.all(hilbert.edge_spins(basis) == 1, axis=-1)
numbers = jnp.array([0, 17, hilbert.dimension - 1])
selected = hilbert.numbers_to_states(numbers)

print(hilbert.dimension, basis.shape)
print(int(jnp.sum(all_half_spins)))
print(hilbert.states_to_numbers(selected, validate=True).tolist())
```

```text
140 (140, 15)
32
[0, 17, 139]
```

There are 140 gauge invariant basis states in this example. When every edge has spin $\tfrac12$, each of the five vertices has two intertwiner choices, giving $2^5=32$ states in that edge sector alone. Enumerating only edge spin assignments would miss that multiplicity.

Leave `exact=False`, the default, for calculations that do not need the complete basis. The space then reports `dimension=None` and cannot perform exact basis ranking, while validity checks and state generation remain available. `all_states` can still request enumeration on demand, but it does not turn the object into an exact cached space. Use `exact=True` from construction when you need the numbering methods.

Exact enumeration examines edge sectors and admissible channel combinations. `max_exact_states` limits the number of states retained, but is not a bound on the time spent examining candidate sectors. Increase graph size or cutoff with care when deliberately enumerating a basis. A variational Monte Carlo calculation generally uses the implicit space and avoids this allocation.

### Initial configurations and wavefunction sampling

With a cached exact basis and no method override, `random_state` draws uniformly from that basis. Without the cache, its default method runs a finite Metropolis warmup from the vacuum. This generates valid initial configurations with a uniform stationary target, but a finite run is not a guarantee of independent or uniformly distributed samples.

The constructor's `init_warmup` controls the default number of attempted updates per output configuration. You can override it per call with `n_steps`, or request the all-zero vacuum explicitly:

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

hilbert = nqx.hilbert.SU2Hilbert(nqx.graph.K5Graph(), two_j_max=1)
vacuum_key, warmup_key = jax.random.split(jax.random.key(13))
vacuum = hilbert.random_state(vacuum_key, size=4, method="vacuum")
initial = hilbert.random_state(warmup_key, size=4, n_steps=64)

print(bool(jnp.all(vacuum == 0)))
print(initial.shape)
print(bool(jnp.all(hilbert.is_valid(initial))))
```

```text
True
(4, 15)
True
```

For the variational distribution, use [`MetropolisSU2`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSU2) with your [variational state](../vqs/index.md). Its spin and intertwiner proposals include the required Hastings corrections. Initialisation warmup is separate from equilibration under the wavefunction and from production sampling. Check [sampling diagnostics](../samplers/diagnostics.md) for the actual calculation rather than treating a valid initial batch as an equilibrium result.

## The link basis before Gauss law

Use `SU2PeterWeylHilbert` when you need magnetic indices on unconstrained links. For one edge, the normalised basis functions are

$$
\langle g|j,m,n\rangle
=\sqrt{2j+1}\,D^j_{mn}(g),
\qquad m,n=-j,-j+1,\ldots,j.
$$

At doubled cutoff $J$, the local link dimension is

$$
d_{\mathrm{link}}
=\sum_{r=0}^{J}(r+1)^2
=\frac{(J+1)(J+2)(2J+3)}{6}.
$$

The graph product has dimension $d_{\mathrm{link}}^E$. Each edge stores **one encoded integer**, which identifies the triple $(2j,2m,2n)$. Increasing spin blocks are ordered first by `two_m`, then by `two_n`, with `two_n` changing fastest. Use `encode` and `decode` to translate between quantum numbers and stored values.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.SU2PeterWeylHilbert(graph, two_j_max=1)
state = hilbert.encode(
    two_j=jnp.array([0, 1, 1, 1]),
    two_m=jnp.array([0, -1, -1, 1]),
    two_n=jnp.array([0, -1, 1, 1]),
)
two_j, two_m, two_n = hilbert.decode(state)

print(hilbert.local_size, hilbert.size, hilbert.dimension)
print(state.tolist())
print(two_j.tolist())
print(two_m.tolist())
print(two_n.tolist())
print(bool(hilbert.is_valid(state)))
```

```text
5 4 625
[0, 1, 2, 4]
[0, 1, 1, 1]
[0, -1, -1, 1]
[0, -1, 1, 1]
True
```

The spin-zero block has one magnetic state, and the spin-one-half block has four. The square therefore has $5^4=625$ link basis states. `is_valid` checks membership in that link product. It does not impose a vertex Gauss condition. `encode` returns `-1` for invalid or out-of-cutoff magnetic labels, so validate configurations after assembling them.

This space uses the generic scalar edge layout. Its `random_state` draws product basis configurations, and its dimension is available without enumerating the graph product. Indexability still depends on the total dimension fitting the supported [basis numbering](state-layouts.md#values-local-indices-and-basis-numbers) range.

A gauge invariant spin network is generally a linear combination of magnetic link basis states with coefficients determined by the intertwiners. Converting between these descriptions is a change of basis involving those contractions. It is not a reshape or a call to `decode`. Likewise, setting `is_gauge_invariant=False` on `SU2Hilbert` is not a representation switch and raises an error. Construct `SU2PeterWeylHilbert` explicitly and use operators and samplers that support it.

For gauge invariant LQG calculations, continue with [SU(2) LQX models](../lqx/su2.md) and their [operators](../operators/index.md). For custom boundary sectors or representations, see the [custom Hilbert space guide](../../developers/hilbert/index.md) and the [constraint guide](constraints.md).
