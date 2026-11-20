---
title: Hilbert spaces
---

# Hilbert spaces

A graph tells you which edges and vertices belong to your system. A Hilbert space specifies the quantum labels they carry, the values those labels can take, and which combinations are allowed. It is where the connectivity introduced in the [graph guide](../graphs/index.md) becomes a basis for a quantum calculation.

For a U(1) model, a configuration might assign an integer charge to every edge. For a gauge invariant SU(2) model, it assigns spins to edges and, where needed, intertwiner labels to vertices. The Hilbert object defines how these configurations are represented in arrays so that [operators](../operators/index.md), [samplers](../samplers/index.md), and [variational states](../vqs/index.md) agree on what each entry means.

## A basis for your calculation

Write the set of allowed configurations as $\mathcal C$. A variational wavefunction on the chosen graph has the expansion

$$
|\psi_\theta\rangle
= \sum_{\sigma\in\mathcal C}
  \psi_\theta(\sigma)\,|\sigma\rangle.
$$

The Hilbert space defines $\mathcal C$ and the basis states $|\sigma\rangle$. Your [network](../networks/index.md) supplies the amplitudes through its parametrisation, and the variational state combines that model with a sampling or evaluation strategy. Constructing a Hilbert object does not initialise a wavefunction or select its variational parameters.

In canonical LQG, this choice also fixes which restrictions have already been built into the basis. A gauge invariant spin network basis satisfies the SU(2) Gauss constraint at each graph vertex. That does not by itself solve the Hamiltonian or spatial diffeomorphism constraints. Those questions belong to the [LQX model](../lqx/index.md), its operators, and the calculation you perform with them.

The spaces described here are finite, discrete representations. A charge or spin cutoff makes the basis finite, but its dimension can still be far too large to enumerate. You can construct and sample a space without storing every basis configuration. The scientific effect of the cutoff remains something to check when comparing simulations.

## Choose a supplied space or define the local domains

For the supplied gravity models, begin with the corresponding U(1) or SU(2) space. They already know the representation, its constraints, and how to generate valid initial configurations.

| Space | Basis and intended use |
| --- | --- |
| [`U1Hilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.U1Hilbert) | Finite edge charges before imposing Gauss relations |
| [`U1GaugeInvariantHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.U1GaugeInvariantHilbert) | Edge charges with constructive, cyclic gauge relations |
| [`SU2Hilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2GaugeInvariantHilbert) | Gauge invariant spins and vertex intertwiners |
| [`SU2PeterWeylHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2PeterWeylHilbert) | SU(2) link states with magnetic indices, before Gauss law |

The [U(1) guide](u1.md) explains the cutoff and the cyclic arithmetic used by its gauge relations. The [SU(2) guide](su2.md) explains why its two bases have different coordinates and cannot be interchanged.

For a different finite model, [`HomogeneousHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HomogeneousDiscreteHilbert) assigns the same local domain to every enabled graph entity. [`HeterogeneousHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousDiscreteHilbert) allows different domains on edges and vertices, including individual overrides. Their construction is covered in [Local spaces](local-spaces.md). A generic domain of integer labels does not acquire U(1) or SU(2) physics automatically. The constraints and operators must implement the model you intend.

## Inspect a space before using it

Here is a small U(1) example on a square. [`Grid2D(2, 2)`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.Grid2D) has four edges, and `cutoff=1` gives each edge the three labels $-1,0,1$.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
samples = hilbert.random_state(jax.random.key(0), size=6)

print(hilbert.size, hilbert.dimension)
print(samples.shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
4 81
(6, 4)
True
```

Here `size` is **the number of scalar entries in one configuration**. `dimension` counts **the basis states**. Four entries with three choices each give $3^4=81$ basis states. These are different quantities even though both describe the same space.

The final array axis always holds one complete configuration. The leading axes contain samples or batches. `random_state` generates configurations suitable for inspection or initialisation. It has no wavefunction argument and does not draw from your variational probability distribution proportional to $|\psi_\theta|^2$. In particular, SU(2) initialisation without an enumerated basis uses a finite Metropolis warmup, whose output is not guaranteed to be uniform.

Use `is_valid` when checking configurations assembled by hand. It tests the local label domains together with the space's constraints. `dimension` can be `None` when an exact count is unavailable, and `is_indexable` reports whether the supported integer basis indexing is available. An unknown dimension or unavailable indexing does not prevent Monte Carlo calculations.

## Coming from v1.X

Version 2.0 moves the Hilbert machinery into neuraLQX's own discrete space infrastructure. The main benefit for scientific code is a more explicit contract between local domains, graph coordinates, constraints, and configuration arrays. You can describe mixed edge and vertex degrees of freedom directly, inspect their layout, and reuse constraints across generic spaces.

The old U(1) entry point was `nqx.hilbert.u1.HilbertU1`, with `is_gauge_invariant` selecting its constrained behaviour. In the current API, choose the class that represents the space you need:

```python
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)

unconstrained = nqx.hilbert.U1Hilbert(graph, cutoff=2)
gauge_invariant = nqx.hilbert.U1GaugeInvariantHilbert(
    graph,
    cutoff=2,
    auto_constraint=True,
)

print(unconstrained.dimension, gauge_invariant.dimension)
print(unconstrained.size, gauge_invariant.size)
```

```text
625 5
4 4
```

The constraint reduces the allowed basis, while configurations still contain a value for every edge. The dependent values are reconstructed by the space. This preserves the information needed by edge operators.

You also use the Hilbert object directly. Calls such as `hilbert.view(samples)` and `hilbert.flatten(view)` replace the old pattern of reaching through an interface to `hilbert.hilbert`. New code should not depend on the former NetKet backing object exposed as `hilbert_netket`. Read the [state layout guide](state-layouts.md) when adapting code that manually reshapes arrays or accesses edge values.

SU(2) is now part of the public Hilbert API. Its gauge invariant space includes vertex recoupling channels, and its separate Peter–Weyl space retains the magnetic indices of unconstrained links. In SU(2), `two_j_max` is a doubled spin cutoff, so a value of `2` means $j\leq1$. It has a different meaning from a U(1) charge cutoff.

## Continue through the guide

[Local spaces](local-spaces.md) develops homogeneous and heterogeneous construction. [State layouts](state-layouts.md) shows how to translate between flat configurations and graph quantities. [Constraints](constraints.md) explains validity checks, linear restrictions, and custom predicates, including their consequences for sampling.

If you already know which gauge group your model uses, continue directly to [U(1) spaces](u1.md) or [SU(2) spaces](su2.md). Both pages introduce the mathematical basis alongside executable examples.
