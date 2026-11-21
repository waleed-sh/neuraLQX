---
title: 2+1 weak coupling limit
---

# 2+1 weak coupling limit

[`LqxWCL3D`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxWCL3D) provides the U(1)$^3$ weak coupling construction in $2+1$ spacetime dimensions. Each edge carries a three-component charge vector, while the graph is planar. The model adds a planar volume and a regularised Euclidean constraint to the [shared U(1) operators](u1.md).

Construct it from a U(1) Hilbert space with `gauge_dimensions=3` and a graph with `is_planar=True`. This model does not supply a Lorentzian term, area operator, or mass operator. Its default total is the sum of its Euclidean vertex constraints.

## Constructing a model

The planar Hamiltonian uses pairs of distinct minimal loops meeting at a vertex. A graph with only one triangle therefore does not provide a nontrivial loop pair. Here we use two triangles meeting at their central vertex.

```python
import neuralqx as nqx

center = (0, 0, 0)
right, up = (1, 0, 0), (0, 1, 0)
left, down = (-1, 0, 0), (0, -1, 0)
graph = nqx.graph.Graph(
    [
        (center, right), (right, up), (up, center),
        (center, left), (left, down), (down, center),
    ],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
model = nqx.lqx.LqxWCL3D(hilbert)

constraint = model.constraint
print(hilbert.size)
print(len(model.get_constraint_terms()))
```

```text
18
5
```

There are six edges and three charges on each edge. The model retains one constraint block per graph vertex, including blocks that vanish because no suitable loop pair meets there. Keeping the labels makes the [quadratic construction](structure.md#quadratic-constraints) unambiguous.

## Planar volume

Let $\mathbf q_e=(q_e^{(0)},q_e^{(1)},q_e^{(2)})$. The [`PlanarVolumeOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.PlanarVolumeOperator) is diagonal in this charge basis. In the dimensionless convention implemented here,

$$
\mathbf X_v(q)
=\sum_{(e,f)\text{ incident at }v}
\epsilon_v(e,f)\,\mathbf q_e\times\mathbf q_f,
\qquad
V_v(q)=\lVert\mathbf X_v(q)\rVert_2.
$$

The sum is over ordered pairs of distinct incident edges. Both $(e,f)$ and $(f,e)$ are included. The sign $\epsilon_v(e,f)$ comes from the graph's planar orientation data. Consequently, replacing this sum with an unordered one without adjusting the normalisation changes the operator.

`get_volume(vertex)` returns $V_v$. Setting `squared=True` returns $V_v^2$, and setting `sqrt=True` returns $\sqrt{V_v}$. The helpers `squared_volume(vertex)` and `sqrt_volume(vertex)` express the latter choices directly. If both flags are true, the operations are applied in that order and the result is $V_v$ again.

This example builds a state through the Hilbert layout and checks the relationship between the three observables.

```python
import neuralqx as nqx
import jax.numpy as jnp
import numpy as np

center = (0, 0, 0)
right, up = (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph(
    [(center, right), (right, up), (up, center)], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
model = nqx.lqx.LqxWCL3D(hilbert)
vertex = graph.vertices[0]

state = jnp.zeros(hilbert.size, dtype=jnp.int32)
state = state.at[hilbert.layout.edge_site(graph.edges[0], component=0)].set(1)
state = state.at[hilbert.layout.edge_site(graph.edges[2], component=1)].set(1)

_, value = model.get_volume(vertex).get_conn_padded(state)
_, squared = model.squared_volume(vertex).get_conn_padded(state)
_, root = model.sqrt_volume(vertex).get_conn_padded(state)
np.testing.assert_allclose(squared, value**2)
np.testing.assert_allclose(root**2, value)
print(float(value[0]))
```

```text
2.0
```

The graph is sufficient for a nonzero volume even though it lacks the second loop needed by the planar Hamiltonian. Geometry and available curvature loops are separate ingredients.

## The Euclidean vertex constraint

The [`Thiemann2p1VertexConstraintOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.Thiemann2p1VertexConstraintOperator) combines loop holonomies with differences of $\sqrt{V_v}$. To state its action, let $d_s=+1$ for a stored segment leaving $v$ and $d_s=-1$ for one entering it. For component $c$, define the diagonal factor

$$
D_{s,c}(q)
=\sqrt{V_v(q)}
-\sqrt{V_v\bigl(W(q-d_s h\mathbf e_{s,c})\bigr)}.
$$

Here $W$ is the cyclic wrapping from the [shared U(1) convention](u1.md#edge-and-loop-holonomies). Write $U_{\alpha,c}$ for the ket shift

$$
U_{\alpha,c}|q\rangle
=|W(q+h\ell_{\alpha,c})\rangle.
$$

This $U_{\alpha,c}$ is the adjoint of the positive *row* shift denoted by $R_{\alpha,c}$ on that page.

For each ordered pair of distinct minimal loops through $v$, the first loop supplies $\alpha_t$ and the second supplies two incident segments $k_t,l_t$. Both loops must meet the vertex in exactly two edges. The local constraint is

$$
H_v
=N_v\sum_t\sum_{c=0}^{2}
w_t\,U_{\alpha_t,c}\,D_{k_t,c}D_{l_t,c}.
$$

The rightmost diagonal factors act first. Equivalently, a matrix row connects to $q'=W(q-h\ell_{\alpha_t,c})$ with coefficient $N_vw_tD_{k_t,c}(q')D_{l_t,c}(q')$. This makes the placement of the volume factors explicit.

The weights $w_t$ are products of the permutation signs of the two incident loop pairs. They use the ordering of the dressed loops, with opposite signs for a pair and its reversal. They are distinct from the geometric orientation signs in the volume. Both assignments of the two loops are retained, and the implementation does not divide by the number of loop pairs. The [operator reference](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.Thiemann2p1VertexConstraintOperator) records the construction arguments.

Use `model.thiemann_regularized_constraint(vertex)` for a local operator, or `model.get_euclidean_constraint()` for the labelled sum $H_E=\sum_v H_v$. Passing a vertex to `get_euclidean_constraint(vertex)` selects one local block. The operator is generally not Hermitian. Use `adjoint=True` or the returned operator's `.adjoint` property when you need its adjoint.

## Lapses and quadratic objectives

A lapse dictionary can be supplied as `lapses={vertex: value, ...}`. It is used only when you pass `apply_lapse=True` to a constraint factory. A missing entry contributes a lapse of one. In this model, dictionary lookup uses the vertex argument itself, so use canonical objects from `graph.vertices` consistently rather than mixing them with integer positions.

```python
import neuralqx as nqx

center = (0, 0, 0)
right, up = (1, 0, 0), (0, 1, 0)
left, down = (-1, 0, 0), (0, -1, 0)
graph = nqx.graph.Graph(
    [
        (center, right), (right, up), (up, center),
        (center, left), (left, down), (down, center),
    ],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
lapses = {vertex: 0.5 for vertex in graph.vertices}
model = nqx.lqx.LqxWCL3D(hilbert, lapses=lapses)

model.euclidean_constraint = model.get_euclidean_constraint(apply_lapse=True)
model.quadratic_constraint = model.get_quadratic_constraint(alternative=True)
print(len(model.get_constraint_terms()))
```

```text
5
```

This selects $\sum_v H_v^\dagger H_v$, including the chosen lapse inside each $H_v$. The default property would instead use $\sum_v H_vH_v^\dagger$. For separate local objectives, `thiemann_quadratic_constraint(apply_lapse=True, alternative=True)` returns a list. It supports the factorised form through `lazy=True`, which is the default.

The [3+1 weak coupling limit](wcl-spatial.md) uses the same charge and holonomy conventions but a different volume and a loop-and-segment Hamiltonian construction.
