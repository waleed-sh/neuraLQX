---
title: Constructing operators
---

# Constructing operators

Constructing an operator means choosing its Hilbert space and the physical arguments that determine its action. A charge needs an edge and component. A volume needs a vertex. A Wilson loop needs an oriented loop. A Hamiltonian may also depend on a lapse, coupling, and regularisation choices.

For most calculations, you can start with a supplied operator and use the [operator algebra](algebra.md) to build the expression you need. An [LQX model](../lqx/index.md) offers another route to the same operators when several choices should be organised together.

## Start from the Hilbert space

Every operator belongs to a specific Hilbert space. Sharing a graph or a configuration width is not enough to make two operators compatible. The charge range, gauge restrictions, basis labels, and layout determine the space on which the action is defined.

For a U(1) observable, construct a [U(1) space](../hilbert/u1.md) first and pass that object to the operator. In the example below, `copy=1` selects the second gauge component of an edge. The layout's corresponding argument is called `component`.

```python
import neuralqx as nqx
import jax.numpy as jnp
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2, gauge_dimensions=3)
edge = graph.edges[0]
site = hilbert.layout.edge_site(edge, component=1)
charge = NumberOperator(hilbert, edge=edge, copy=1)

configuration = jnp.zeros(hilbert.size, dtype=jnp.int32).at[site].set(2)
_, elements = charge.get_conn_padded(configuration)
print(float(elements[0]))
print(charge.hilbert == hilbert)
```

```text
2.0
True
```

[`NumberOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) accepts an edge object or a base-edge index. Other constructors can use different selectors. For example, [`ChargeColoringOperator`](../../api/public/neuralqx/operator/prebuilt/index.md#neuralqx.operator.prebuilt.ChargeColoringOperator) takes a flat component index, while [`ColoringOperator`](../../api/public/neuralqx/operator/prebuilt/index.md#neuralqx.operator.prebuilt.ColoringOperator) takes a base-edge index and a complete charge vector. The [projector examples](prebuilt.md#charge-projectors) show both conventions.

## Construct directly or through an LQX model

The prebuilt classes can be used without a model. If you already have an LQX model, its factories choose the corresponding class and pass the shared settings. Both routes produce operator objects that can be measured or composed in the same way.

Here we construct the positive loop-curvature observable

$$
K_\alpha=I-\frac12(R_\alpha+R_\alpha^\dagger)
$$

through both interfaces and compare their matrices on a small space.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import OneMinusHalfHolonomyOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
loop = graph.dressed_minimal_loops()[0]

direct = OneMinusHalfHolonomyOperator(hilbert, loop=loop)
model = nqx.lqx.LqxWCL1D(hilbert)
from_model = model.one_minus_half_holonomy(loop)

np.testing.assert_allclose(direct.to_dense(), from_model.to_dense())
print(direct.is_hermitian)
```

```text
True
```

[`OneMinusHalfHolonomyOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.OneMinusHalfHolonomyOperator) uses the cyclic U(1) holonomy. The [U(1) model guide](../lqx/u1.md) explains its cutoff and orientation conventions. An LQX factory call constructs an operator but does not automatically change a selected model property. See [model structure](../lqx/structure.md#factories-and-selected-operators) when the construction should also become the model's selected observable or constraint.

## Build an expression from existing pieces

Suppose you want a charge penalty together with loop curvature,

$$
H=g\sum_e\hat q_e^2+\kappa K_\alpha.
$$

You can express this directly with scalar multiplication, addition, and operator products. The following is an illustrative finite model, rather than a new LQG Hamiltonian prescription.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt import wcl

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)

charges = [wcl.NumberOperator(hilbert, edge=edge) for edge in graph.edges]
charge_penalty = sum(charge @ charge for charge in charges)
curvature = wcl.OneMinusHalfHolonomyOperator(
    hilbert, loop=graph.dressed_minimal_loops()[0]
)
hamiltonian = 0.5 * charge_penalty + 0.2 * curvature

matrix = np.asarray(hamiltonian.to_dense())
np.testing.assert_allclose(matrix, matrix.conj().T)
print(matrix.shape)
```

```text
(27, 27)
```

This construction stores a composition of operators. A dense matrix is created only when `to_dense()` is called at the end for the small verification. In a variational calculation, pass `hamiltonian` directly to the [state or driver](application.md#measure-an-operator-with-a-variational-state).

Use an explicit identity when adding a constant. [`IdentityOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.IdentityOperator) works on a discrete Hilbert space and represents `const` times the identity. Despite its namespace, its action does not require U(1) charges. An energy shift is therefore written as `A + energy_offset * IdentityOperator(hilbert)`, rather than adding a bare scalar to an operator.

## Keep the basis restriction explicit

An operator must act within the space on which you evaluate it, or you must specify how to restrict its action. A closed gauge invariant observable and an open edge holonomy can have very different compatibility with a gauge invariant basis. Supplying a constrained Hilbert space does not establish that an arbitrary physical action preserves it.

Similarly, a change of cutoff can change the operator's boundary action. U(1) cyclic wrapping, bounded QR shifts, and projected SU(2) recouplings are different constructions. The [prebuilt guide](prebuilt.md) links to their definitions.

If your wavefunction lives in a subspace of a larger operator space, use a [`BasisEmbedding`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.BasisEmbedding) and the [restriction and residual interfaces](restricted-spaces.md). They distinguish measuring the restricted action from retaining all contributions to a larger constraint norm.

## Reusing and extending a construction

Operators are immutable objects. Build a new operator or expression when you change a coupling or a selected region. An expression such as `scaled = 2.0 * volume` leaves `volume` available for separate measurement. Construct graph-dependent choices and operator structure before running your numerical calculation.

If the same collection of factories and physical parameters is useful across studies, [create an LQX subclass](../lqx/custom-models.md). If the action itself is new, implement a [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) with its own connectivity kernel. The [custom operator section](../../developers/operators/index.md) is the place for that implementation work. The scientist-facing API described here remains the way to use the resulting object.
