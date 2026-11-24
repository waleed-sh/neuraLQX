---
title: Developer guide
---

# Developer guide

These guides are for researchers and developers who need to understand how neuraLQX performs a calculation, extend one of its components, or contribute to the package. You might be implementing an operator from a new regularisation, introducing a different configuration space, investigating a slow simulation, or adapting a sampling rule to a particular physical problem.

The discussion becomes technical where the subject requires it. We will work with mathematical contracts, array layouts, JAX transformations, differentiation rules, and the relationship between Python code and compiled execution. The purpose is to explain why an implementation must behave in a particular way, as well as how to write it.

You do not need to work on the package itself to use this section. In particular, [profiling for scientists](profiling/for-scientists.md) explains how to investigate an existing calculation through public interfaces, without changing neuraLQX's implementation.

## Start from the scientific object

An extension should begin with the object it is meant to represent. For an operator, that means its action on configurations and any properties assumed by its estimator. For a Hilbert space, it means which labels are allowed and how configurations are enumerated or sampled. For a network, it means the map from those configurations to log amplitudes.

The [core concepts](../concepts/index.md) introduce these objects from the perspective of running a simulation. Read the relevant scientific guide first if its role is unfamiliar. The developer guides build on that picture and explain the interfaces that let the components work together.

This separation matters when debugging. A calculation can execute successfully while representing a different constraint, distribution, or derivative from the one intended. Conversely, a mathematically correct expression may need a different array organisation to behave well under batching, compilation, or distributed execution.

## Follow the component you want to extend

| Task | Relevant guides |
| --- | --- |
| Understand where a calculation spends time | [Profiling](profiling/index.md) |
| Represent parameters, numerical state, and fixed metadata | [Structs and pytrees](structs/index.md) |
| Implement an observable estimator or its derivative | [Custom estimators](estimators/index.md) |
| Define a new matrix free operator action | [Custom operators](operators/index.md) |
| Inspect or extend symbolic operator compilation | [Symbolic compiler](symbolic/index.md) |
| Define allowed configurations and constraints | [Custom Hilbert spaces](hilbert/index.md) |
| Change how configurations are proposed and sampled | [Custom samplers](samplers/index.md) |
| Implement a different wavefunction ansatz | [Custom networks](networks/index.md) |
| Supply graph topology and derived geometric structures | [Custom graphs](graphs/index.md) |

These subjects meet at a few shared boundaries. An operator supplies connected configurations and matrix elements. An estimator combines them with wavefunction ratios. A sampler produces configurations from a distribution, and the gradient calculation must account for that distribution. The [VQS guides](../concepts/vqs/index.md) show how those pieces are used together before discussing their implementation separately.

## Expect both mathematics and execution details

JAX traces Python functions to construct compiled computations. Array shapes, parameter structure, and the distinction between fixed and varying data therefore affect more than notation. They influence which work can be compiled, reused, differentiated, and distributed.

The guides will make those assumptions explicit. When a formula has several possible conventions, the convention used by the implementation matters. When a computation is asynchronous, the location of a timing boundary matters. When a state is constrained, a proposal or an operator connection must be interpreted in the corresponding space.

Small examples are useful for checking these contracts before scaling up. For an indexable Hilbert space, an explicitly evaluated [state vector](../concepts/vqs/arrays.md) can provide a reference for an estimator. For a sampling rule, [diagnostics](../concepts/samplers/diagnostics.md) help distinguish execution from successful exploration. For performance work, a [profile](profiling/index.md) identifies which part of the actual calculation should be investigated.

## From an extension to a contribution

An extension can live in your own research code while you test its scientific behaviour. If it belongs in the package, the [contribution overview](../api/contributing.md) and [development environment](contributing/environment.md) describe the next steps. The [testing guide](contributing/testing.md) belongs alongside the component contract, since correctness includes the behaviour of the object under the transformations used by the rest of the library.

Use the [public API reference](../api/public/index.md) for signatures and available arguments. These guides provide the reasoning and worked examples that make those interfaces easier to apply.
