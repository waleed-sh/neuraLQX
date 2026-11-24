---
title: Hilbert spaces and admissibility
---

# Hilbert spaces and admissibility

Symbolic rules act on the flat configuration representation of a native discrete Hilbert space. They can be used with homogeneous and heterogeneous spaces, vector labels, U(1) spaces, gauge invariant SU(2) spin networks, and SU(2) Peter–Weyl spaces. The representation determines what a coordinate means and which targets are admissible.

An operator builder does not create a second Hilbert space or reinterpret its labels. The definition and compiled operator retain the supplied space. Check its [state layout](../../hilbert/state-layouts.md) before choosing iterator domains or update indices.

## Local domains and physical meaning

In a homogeneous space, coordinates share a scalar local domain. A heterogeneous declaration can give them different allowed values. A vector degree of freedom contributes several scalar coordinates, which may be interleaved by component. `for_each_site` visits all of those flat coordinates.

Consequently, a sum of squared scalar labels is not automatically the geometric observable associated with each edge. For a U(1) charge vector it can contribute to a Euclidean norm. For SU(2), an edge spin is stored as $2j$, and area involves $\sqrt{j(j+1)}$. Intertwiner and magnetic labels have different roles again.

Use `hilbert.local_spaces[i]` for a coordinate's local domain and the layout methods for its position. [Modular shifts](updates/modular.md) follow each declared local basis. Other updates use their numerical formulas and need appropriate guards.

## Select components through the layout

The following observable measures the squared three-component charge on the first edge of a U(1)$^3$ space. It uses the layout to locate the components rather than assuming adjacent entries.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
sites = [hilbert.layout.edge_site(0, component=g) for g in range(3)]
components = [s.source_index(i) for i in sites]
ctx = s.ExpressionContext()

charge_norm_squared = (
    s.SymbolicOperatorBuilder(hilbert, "first_edge_norm", hermitian=True)
    .globally()
    .emit(matrix_element=ctx.sq_norm(*components))
    .compile()
)
source = jnp.zeros(hilbert.size, dtype=hilbert.dtype)
source = source.at[jnp.array(sites)].set(jnp.array([1, -1, 1]))
_, values = charge_norm_squared.get_conn_padded(source)
np.testing.assert_allclose(values, [3])
```

Here the observable is $q_{0,0}^2+q_{0,1}^2+q_{0,2}^2$. [`ExpressionContext.sq_norm`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.ExpressionContext.sq_norm) is appropriate because these components are real. An iterator over all flat sites would instead sum the squared charges of both edges.

## SU(2) spins and intertwiner labels

On a gauge invariant spin network, restrict an edge-spin observable to the edge sites. Do not include the trailing intertwiner channel coordinates simply because they are also integers.

```python
from neuralqx.operator.prebuilt.su2 import AreaOperator

spin_space = nqx.hilbert.SU2Hilbert(nqx.graph.K5Graph(), two_j_max=1)
edge_rows = [(spin_space.layout.edge_site(e),) for e in range(spin_space.n_edges)]
j = s.site("edge").value / 2
area = (
    s.SymbolicOperatorBuilder(spin_space, "edge_area", hermitian=True)
    .for_each(("edge",), over=edge_rows)
    .emit(matrix_element=(j * (j + 1)).sqrt())
    .compile()
)
labels = spin_space.complete_channels_for_edges([1] * spin_space.n_edges)
assert labels is not None
source = jnp.asarray(labels, dtype=spin_space.dtype)
_, values = area.get_conn_padded(source)
np.testing.assert_allclose(values.sum(), 5 * np.sqrt(3))

reference = AreaOperator(spin_space, tuple(range(spin_space.n_edges)))
_, reference_values = reference.get_conn_padded(source)
np.testing.assert_allclose(values.sum(), reference_values.sum())
```

Every edge in this configuration has $j=1/2$, and the sum over ten edges gives $10\sqrt{3}/2$. The example uses unit flux scale. [`AreaOperator`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.AreaOperator) supplies the corresponding native reference, and the [SU(2) model guide](../../lqx/su2.md#area) explains its physical units.

An off-diagonal SU(2) action is more demanding. Changing an edge spin may require changes to several vertex channels, and its matrix element must implement the relevant recoupling coefficients. A generic shift does not derive these relations. The supplied [SU(2) operators](../../lqx/su2.md) remain appropriate building blocks when they already implement the required action. Peter–Weyl labels likewise require their own spin and magnetic-index admissibility conditions.

## Preserve the complete constraint

Local membership and global admissibility are separate checks. As a simple illustration, impose $q_0+q_1=0$ on two scalar coordinates. Exchanging them preserves this constraint, while shifting just one generally does not.

```python
constraint = nqx.hilbert.CallableDiscreteConstraint(
    lambda x: jnp.sum(x, axis=-1) == 0,
    hash_key=("symbolic_docs_zero_total",),
)
constrained = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(-1, 1, 3), constraint=constraint
)
exchange = (
    s.SymbolicOperatorBuilder(constrained, "constraint_preserving_exchange", hermitian=True)
    .globally()
    .emit(s.swap(0, 1), matrix_element=1)
    .compile()
)
sources = jnp.array([[-1, 1], [0, 0], [1, -1]], dtype=constrained.dtype)
targets, values = exchange.get_conn_padded(sources)
assert bool(jnp.all(constrained.is_valid(sources)))
assert bool(jnp.all(constrained.is_valid(targets)))
np.testing.assert_allclose(values, 1)
```

[`CallableDiscreteConstraint`](../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.CallableDiscreteConstraint) defines this illustrative total-charge constraint. It is not a substitute for the graph's vertex Gauss laws. For a new constrained operator, establish the relevant conservation or recoupling argument and test active targets against `hilbert.is_valid`.

The compiler does not silently project invalid active connections back into the Hilbert space. Zero padding is replaced by the valid input configuration, but a nonzero branch still carries the target you specified. Diagnostics sample possible violations and help locate mistakes. Their silence does not prove preservation on every configuration.

## Enumeration is optional for execution

Connectivity evaluation and local generic adjoints do not require ranking the global basis. `to_dense` and `to_sparse` do require suitable indexing support, and can be impractical even when indexing exists. Use them for small independent checks, as in [defining an operator](first-operator.md).

If the variational state occupies a restricted space but the operator's action belongs to a larger one, express that relationship through the [restricted-space interfaces](../restricted-spaces.md). Removing transitions that leave the smaller space can change a quadratic residual. That is a physical change, not a harmless way to make a connectivity warning disappear.
