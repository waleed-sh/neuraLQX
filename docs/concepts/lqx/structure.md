---
title: Model structure
---

# Model structure

An LQX model connects a physical construction to the operators used in a calculation. Its structure is deliberately small. There is a Hilbert space, the graph attached to that space, and a set of named operator selections. Understanding how those selections work lets you change a lapse, replace a constraint, or choose a different observable without accidentally continuing with an earlier default.

## The Hilbert space comes first

Every [`LQXModel`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LQXModel) is constructed from an existing Hilbert space. The model takes its graph from `hilbert.graph`. If you also pass `graph=...`, it must be that same graph object. Constructing a second graph with matching edges is not sufficient.

The space determines the basis on which every selected operator acts. Assignments are checked for Hilbert space compatibility. A model on [gauge invariant SU(2) states](../hilbert/su2.md) therefore cannot accept an open edge operator built on a Peter–Weyl basis, even if both spaces refer to the same graph.

## Factories and selected operators

A method beginning with `get_` constructs an operator. It can take arguments such as a vertex, surface, coupling, or lapse. A property stores a selection. The common pairs are:

| Selected property | Factory | Meaning |
| --- | --- | --- |
| `euclidean_constraint` | `get_euclidean_constraint(...)` | Euclidean component supplied by the model |
| `lorentzian_constraint` | `get_lorentzian_constraint(...)` | Lorentzian component, when implemented |
| `constraint` | `get_constraint()` | Total of the available selected components |
| `quadratic_constraint` | `get_quadratic_constraint(...)` | Positive quadratic objective built from the selected total |
| `area` | `get_area(...)` | Area for a chosen surface |
| `volume` | `get_volume(...)` | Volume for a chosen vertex |
| `holonomy` | `get_holonomy(...)` | Holonomy for a chosen edge, site, or loop |
| `mass` | `get_mass(...)` | Mass observable, if a subclass supplies one |

On first access, a property calls its factory without arguments and caches the result. Subsequent accesses return that selected object. The supplied models can generally construct default constraints this way. Geometric observables usually require an explicit region, so select them before reading the corresponding property.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
model = nqx.lqx.LqxWCL4D(hilbert)

vertex = graph.vertices[0]
surface = graph.edges[:2]

model.volume = model.get_volume(vertex)
model.area = model.get_area(surface)

print(model.volume is model.volume)
print(model.area.hilbert == hilbert)
```

```text
True
True
```

Calling `model.get_volume(vertex)` by itself would return an operator without selecting it. Assigning the result to `model.volume` gives that observable a name within this calculation. It does not replace the volume used internally by a built-in Hamiltonian factory. To change that Hamiltonian's definition, use its supported construction arguments or [supply your own model](custom-models.md).

Assigning `None` clears a selection. The next access attempts default construction again. For an observable that needs a vertex or surface, that attempt will ask you to make an explicit choice. An unavailable factory raises `NotImplementedError` rather than silently supplying an unrelated observable. None of the currently supplied models defines a default mass operator.

## Selecting a constraint

The automatic total is

$$
H = H_E + H_L,
$$

with only the components that the model implements. The weak coupling and SU(2) models supply a Euclidean component. The QR model supplies both components. If an implemented factory fails, the error is propagated so that a construction problem cannot silently remove part of the constraint.

Use a factory followed by an assignment when you want its parameters to determine the model total. In this QR example, both components use the same Barbero–Immirzi parameter, and a constant lapse multiplies both terms explicitly.

```python
import neuralqx as nqx

graph = nqx.graph.SingleVertexGraph(3)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, positive_qn=True)
model = nqx.lqx.LqxQR(hilbert)

gamma = 0.8
lapse = 1.5
model.euclidean_constraint = model.get_euclidean_constraint(
    lapse=lapse, immirzi=gamma
)
model.lorentzian_constraint = lapse * model.get_lorentzian_constraint(
    immirzi=gamma
)

constraint = model.constraint
objective = model.quadratic_constraint
print(constraint.hilbert == objective.hilbert)
```

```text
True
```

You can also assign a complete linear operator to `model.constraint`. This is useful when a study uses a different combination of terms. `get_constraint()` still constructs the automatic component sum when called explicitly. It neither returns nor overwrites your total override.

The linear component and total properties accept [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) instances. A quadratic objective is a separate kind of object and belongs in `quadratic_constraint`, not in a linear component.

## What changes when you make a new selection

Selections have dependencies. Assigning either `euclidean_constraint` or `lorentzian_constraint` clears the cached total and quadratic objective. This also clears an explicitly assigned total. Assigning `constraint` clears the quadratic objective. Assigning an observable such as `volume` does not clear the constraints.

Consequently, set the components first, then any total override, and finally any quadratic override. That order follows the construction itself.

The invalidation happens through these property assignments. Editing an ordinary attribute, such as a lapse dictionary or a coupling stored on a subclass, does not rebuild operators that already exist. For a parameter study, either construct a new model for each parameter set or build and assign the affected components explicitly.

Construct models and make these selections in ordinary Python before entering transformed numerical code. They are mutable setup objects. JAX transformations operate on numerical functions, and Python side effects during tracing do not behave like updates performed on every execution. The [JAX explanation of tracing](https://docs.jax.dev/en/latest/jit-compilation.html#how-jax-transformations-work) gives the relevant background.

## Quadratic constraints

Many constraint calculations minimise a positive quantity rather than the expectation value of a possibly non-Hermitian linear constraint. The default model construction is

$$
Q = \sum_v H_v H_v^\dagger,
\qquad
\langle Q\rangle_\psi
= \sum_v \lVert H_v^\dagger\psi\rVert^2
$$

for a normalised state. Calling `get_quadratic_constraint(alternative=True)` instead constructs

$$
Q_{\mathrm{alt}} = \sum_v H_v^\dagger H_v,
\qquad
\langle Q_{\mathrm{alt}}\rangle_\psi
= \sum_v \lVert H_v\psi\rVert^2.
$$

Thus the alternative ordering directly measures the residual of $H_v\psi=0$. Both expressions are positive, but their null spaces need not agree for non-Hermitian $H_v$. Select the ordering that matches your constraint equation rather than assuming the two are interchangeable.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
model = nqx.lqx.LqxWCL4D(hilbert)

model.quadratic_constraint = model.get_quadratic_constraint(alternative=True)
print(len(model.get_constraint_terms()) == len(model.admissible_vertices))
```

```text
True
```

These objectives use [`Squared`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared) and [`SumOfSquares`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.SumOfSquares). They preserve the factorisation for expectation estimation instead of expanding every two-step transition. In particular, `Squared(A)` means $A^\dagger A$. The default ordering is represented by applying that construction to $H_v^\dagger$.

### Why vertex labels matter

If a model supplies both components at a vertex, the local block is

$$
H_v = H_{E,v} + H_{L,v}.
$$

The model combines these contributions before forming a local quadratic product. Cross terms between Euclidean and Lorentzian contributions at the same vertex are retained. Cross terms between different vertices are not introduced.

This differs from squaring the global sum,

$$
\left(\sum_v H_v\right)
\left(\sum_w H_w\right)^\dagger,
$$

which also couples distinct vertices and can allow cancellations between their residuals. [`sum_constraint_terms`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LQXModel.sum_constraint_terms) preserves the local labels using a [`VertexSum`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.VertexSum). An ordinary operator sum has no such labels and is treated as one global block. Mixing labelled blocks with an unlabelled contribution is rejected because its local interpretation would be ambiguous.

You can inspect the decomposition with `model.get_constraint_terms()`. This returns a dictionary of local operators. A single unlabelled total appears under the key `None`. The [custom model example](custom-models.md#preserving-local-constraints) shows how to retain this structure in your own construction.

Several models also provide a method returning a list of local quadratic terms, named `thiemann_quadratic_constraint()` or `hamiltonian_list()`. These lists contain quadratic objectives, not linear Hamiltonians. They construct their terms from their own factory arguments. To use components you have explicitly selected on the model, use the inherited `get_quadratic_constraint()` method.
