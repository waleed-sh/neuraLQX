---
title: LQX models
---

# LQX models

Once you have chosen a [graph](../graphs/index.md) and a [Hilbert space](../hilbert/index.md), you still need to specify what acts on your states. Which Hamiltonian constraint are you studying? Which volume or area will you measure? If the constraint has several local contributions, how should they enter a variational objective?

The LQX layer gives these choices a common home. A model holds your Hilbert space and graph, provides methods for constructing the relevant operators, and remembers the operators you select for a calculation. This makes a simulation easier to set up and revisit, particularly when several observables share the same conventions and parameters.

**Using LQX is optional.** The rest of neuraLQX works with operator objects. You can construct them directly, combine them in a script, and pass them to a variational state or driver without creating an LQX model. The model layer is there to organise that work and remove repeated setup. It does not introduce another physical degree of freedom or change the Hilbert space you supply.

## Where a model fits

The [core concepts overview](../index.md) separates the ingredients of a simulation. The graph describes connectivity and geometric information. The Hilbert space defines the configurations you can represent. Operators specify their dynamics and observables. A variational quantum state supplies amplitudes, while a sampler and driver handle estimation and optimisation.

An LQX model sits alongside the operators in that picture. It knows how to build the operators associated with a particular physical model. It does not choose a neural network, start a Markov chain, or perform an optimisation step. Those remain independent decisions, so you can use the same physical model with different [variational states](../vqs/index.md), [samplers](../samplers/index.md), and [drivers](../drivers/index.md).

The common interface is [`LQXModel`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LQXModel). Supplied models specialise it with operator factories such as `get_euclidean_constraint()`, `get_volume(vertex)`, and `get_area(surface)`. The arguments express the physical choice being made. A volume needs a vertex, and an area needs a set of puncturing edges.

## A first model

Here is a small example using [`LqxWCL1D`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxWCL1D), the single component U(1) model. Its default constraint measures curvature around the graph's minimal loops. The example also constructs the same operator directly to show how the two routes relate.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt import wcl

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
model = nqx.lqx.LqxWCL1D(hilbert)

constraint = model.constraint

# The same construction using the underlying operators.
direct = wcl.IdentityOperator(hilbert, const=0)
for loop in graph.dressed_minimal_loops():
    holonomy = wcl.MinimalLoopHolonomyOperator(hilbert, loop=loop)
    direct = direct + 2 * wcl.IdentityOperator(hilbert) - holonomy - holonomy.adjoint

np.testing.assert_allclose(constraint.to_dense(), direct.to_dense())
print(model.graph is graph)
print(constraint.hilbert == hilbert)
```

```text
True
True
```

The matrix comparison is practical here because the space contains only $3^3=27$ configurations. In a simulation, the operators normally provide their action through connected configurations rather than a full matrix. The model does not require matrix materialisation.

There are two useful forms of access. A call such as `model.get_euclidean_constraint(...)` constructs an operator with the arguments you provide. A property such as `model.euclidean_constraint` returns the operator currently selected for the model, constructing its default on first access. Calling a factory does not replace that selection. The [model structure guide](structure.md) explains how to select operators and how the total and quadratic constraints follow those choices.

## Choosing a supplied model

The classes below share an interface, but they describe different state spaces and regularisations. Choose the physical model first, then construct a compatible Hilbert space.

| Model | States and setting | Guide |
| --- | --- | --- |
| [`LqxWCL1D`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxWCL1D), also available as `LqxWCL` | One U(1) charge per edge, with a BF curvature constraint | [U(1) models](u1.md) |
| [`LqxWCL3D`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxWCL3D) | Three U(1) charges per edge on a planar graph, for the $2+1$ weak coupling construction | [2+1 weak coupling limit](wcl-planar.md) |
| [`LqxWCL4D`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxWCL4D) | Three U(1) charges per edge on a nonplanar graph, for the $3+1$ weak coupling construction | [3+1 weak coupling limit](wcl-spatial.md) |
| [`LqxSU2`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2), also available as `LqxEuclideanSU2` | Gauge invariant SU(2) spin networks, with geometric observables and a Euclidean constraint | [SU(2) models](su2.md) |
| [`LqxQR`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxQR) | Three nonnegative reduced spin labels, with Euclidean and Lorentzian terms | [Single vertex quantum reduction](qr.md) |
| [`LqxSphericalBojowaldSwiderski`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSphericalBojowaldSwiderski), also available as `LqxSphericalBoSw` | Radial and transverse labels on a half ladder graph | [Spherical symmetry](spherical.md) |

The `1D` in `LqxWCL1D` refers to one U(1) component. The `3D` and `4D` names refer to the spacetime dimensions of the weak coupling models. In both of the latter models, each edge carries three charge components.

## What to check before a calculation

A shared method name does not imply identical physics. For example, U(1) holonomies use cyclic charge shifts, SU(2) holonomies act through representation coupling and cutoff projection, and the reduced QR shifts stop at their boundaries. Similarly, a supplied Euclidean constraint does not imply that a Lorentzian term is also available. Each model page states what is implemented and gives the corresponding equations.

The quadratic constraint deserves particular attention. For models that preserve a decomposition into vertex contributions $H_v$, the default is

$$
Q = \sum_v H_v H_v^\dagger.
$$

The opposite ordering is available explicitly. These orderings can have different kernels when the constraint is not Hermitian. Read [quadratic constraints](structure.md#quadratic-constraints) before choosing the quantity to minimise.

If the supplied models are close to what you need, you can select different operators or combine existing ones. If your construction should be reusable, [creating a model](custom-models.md) shows how to package it in a subclass without implementing a new operator kernel.
