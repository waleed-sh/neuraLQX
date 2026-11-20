---
title: neuraLQX
summary: Neural quantum states for canonical loop quantum gravity.
---

# neuraLQX

neuraLQX is an open-source Python library for studying canonical loop quantum gravity with neural quantum states. It provides the ingredients for describing quantum geometry on a graph, representing a variational wavefunction, and using Monte Carlo methods to estimate observables and optimise the state. Graphs, Hilbert spaces, geometric operators, and constraints are part of the library's vocabulary, so the structure of a calculation follows the physics you want to investigate.

The numerical core is built on [JAX](https://docs.jax.dev/en/latest/), which provides compilation, automatic differentiation, and computation across CPUs and GPUs. You can use the supplied models and algorithms to build a simulation, or introduce your own components as your research requires. To begin, follow the [installation guide](start/installation.md) and the [quick start](start/quickstart.md).

!!! note "What's new in v2.0.0"

    Version 2.0.0 introduces neuraLQX's own simulation infrastructure, a reworked numerical core designed for faster GPU execution, [SU(2) spin networks](concepts/hilbert/su2.md), and [new sampling methods](concepts/samplers/index.md). See the [changelog](api/changelog.md) for the release details.

## Describe the quantum geometry

A calculation starts with a [graph](concepts/graphs/index.md) and a [Hilbert space](concepts/hilbert/index.md). The graph specifies the vertices and edges, while the Hilbert space determines the quantum labels, their allowed values, and the constraints that admissible configurations must satisfy. These choices define the space in which your variational state lives.

[LQX models](concepts/lqx/index.md) connect this description to the operators of a particular theory. The library includes [U(1) models](concepts/lqx/u1.md), [SU(2) models](concepts/lqx/su2.md), and [symmetry-reduced models](concepts/lqx/reduced-models.md). Each model exposes the geometric quantities and constraint operators appropriate to its formulation.

For example, you can construct a gauge-invariant SU(2) Hilbert space on the complete graph with five vertices, then define an area operator:

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
geometry = nqx.lqx.LqxSU2(hilbert)

area = geometry.get_area(surface=[0])
```

Here, `two_j_max=1` restricts edge spins to $j = 0$ and $j = \tfrac{1}{2}$. The Hilbert space also accounts for the intertwiner labels needed to couple the incident spins at each vertex. The final line constructs the area operator for a surface that crosses edge `0` once, away from a vertex. Its default scale expresses area in units of $8\pi |\gamma|\ell_{\mathrm{P}}^2$.

This defines an observable. To obtain its expectation value, you also need a quantum state. The [SU(2) Hilbert space guide](concepts/hilbert/su2.md), [edge labelling guide](concepts/graphs/labels.md), and [operator guide](concepts/operators/index.md) develop the ingredients used here.

## Build a variational calculation

A neural quantum state represents wavefunction amplitudes with a parametrised [neural network](concepts/networks/index.md). Changing its parameters changes the state. A [variational state](concepts/vqs/index.md) brings together the model, its parameters, and a [sampler](concepts/samplers/index.md), allowing you to draw configurations and estimate [expectation values and gradients](concepts/vqs/expectations.md).

To optimise the state, you choose an operator that defines your objective, such as a Hamiltonian or a suitable quadratic constraint. A [variational Monte Carlo driver](concepts/drivers/index.md) coordinates sampling, gradient estimation, and parameter updates. You can use [Optax optimisers](concepts/optimisers/optax.md) and [stochastic reconfiguration](concepts/optimisers/sr.md), then follow the calculation with [callbacks](concepts/callbacks/index.md) and save its progress through [checkpointing](concepts/drivers/checkpointing.md).

These components can be changed independently. You might keep the same physical model while comparing network architectures, change the sampling rule for a constrained space, or add an observable to an existing calculation. The [core concepts](concepts/index.md) explain how the components fit together, and the [tutorials](tutorials/index.md) are the place to follow complete calculations in notebooks.

## Run on the hardware you have

You can develop a calculation on a [CPU](distributed/cpu.md), run it on a [single GPU](distributed/single-gpu.md), or distribute the work across [multiple GPUs](distributed/multi-gpu.md) and [multiple nodes](distributed/multi-node.md). The physical model remains the starting point as you choose how to place arrays and divide the numerical work across devices.

Start with the [JAX parallelism overview](distributed/jax-parallelism.md) to understand how device placement and sharding affect a simulation. When you need to understand where a run spends its time, [profiling for scientists](developers/profiling/for-scientists.md) introduces the tools for examining your calculation.

## Extend the library for your research

Research often calls for an operator, proposal rule, or ansatz that is specific to the problem. The [developer guide](developers/index.md) covers writing [custom operators](developers/operators/index.md), [estimators](developers/estimators/index.md), [samplers](developers/samplers/index.md), and other components that work with the simulation machinery. It also explains the conventions that matter when these components are compiled and differentiated with JAX.

For constructor arguments, method signatures, and individual objects, consult the [public API reference](api/public/index.md). The [experimental API](api/experimental/index.md) documents interfaces that are still evolving.

If you use neuraLQX in your research, please follow the [citation guide](start/citing.md). Bug reports and suggestions are welcome on [GitHub](https://github.com/waleed-sh/neuraLQX/issues), and the [contributing guide](developers/contributing/index.md) is the starting point for changes to the library itself.
