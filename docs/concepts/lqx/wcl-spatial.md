---
title: Spatial weak coupling model
---

# Spatial weak coupling model

[`LqxWCL4D`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxWCL4D) describes the U(1)$^3$ weak coupling construction in $3+1$ spacetime dimensions. It requires three charge components per edge and a graph marked `is_planar=False`. It provides area, volume, and a regularised Euclidean vertex constraint, together with the [shared U(1) operators](u1.md).

No Lorentzian component or mass observable is supplied. The default total is the Euclidean sum over vertices with suitable loop-and-segment contributions. The model calls these `admissible_vertices`. This is a condition on the Hamiltonian construction, not the gauge admissibility of a state.

## Constructing a model

[`K5Graph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.K5Graph) is a convenient starting point because it supplies nonplanar geometry and closed loops through its vertices.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
model = nqx.lqx.LqxWCL4D(hilbert)

model.area = model.get_area(graph.edges[:2])
model.volume = model.get_volume(graph.vertices[0])
constraint = model.constraint
print(hilbert.size)
print(len(model.admissible_vertices))
```

```text
30
5
```

The graph has ten edges, each carrying three charges. A larger cutoff enlarges the available charges but does not change the length of a configuration. See [U(1) state construction](../hilbert/u1.md) for the distinction between these quantities.

## Areas and area differences

For an explicitly selected collection $S$ of puncturing edges, the dimensionless [`AreaOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.AreaOperator) is diagonal,

$$
A_S(q)=\sum_{e\in S}a_e(q),
\qquad
a_e(q)=\sqrt{\sum_{c=0}^{2}\bigl(q_e^{(c)}\bigr)^2}.
$$

Pass the edges as `model.get_area(surface)`. The model does not infer intersections from a drawn surface. Repeated edges are counted repeatedly, and an empty collection gives zero.

The `squared=True` option squares each edge contribution before summing,

$$
A_{S,\mathrm{edge}^2}(q)
=\sum_{e\in S}a_e(q)^2.
$$

This generally differs from $A_S^2$. To square the total area, construct `area = model.get_area(surface)` and use `area @ area`. The distinction matters for surfaces with several punctures, and differs from the corresponding [SU(2) option](su2.md#area).

The model also supplies [`AreaDifferenceSquaredOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.AreaDifferenceSquaredOperator) and [`AreaDifferenceSquaredSurfacesOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.AreaDifferenceSquaredSurfacesOperator),

$$
\Delta A_{e,f}^2=(a_e-a_f)^2,
\qquad
\Delta A_{S_1,S_2}^2=(A_{S_1}-A_{S_2})^2.
$$

Use `area_difference_squared([edge_a, edge_b])` for exactly two edges and `area_difference_squared_surfaces((surface_a, surface_b))` for two edge collections. The surfaces may overlap.

```python
import neuralqx as nqx
import jax.numpy as jnp

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2, gauge_dimensions=3)
model = nqx.lqx.LqxWCL4D(hilbert)
first, second = graph.edges[:2]

state = jnp.zeros(hilbert.size, dtype=jnp.int32)
state = state.at[hilbert.layout.edge_site(first, component=0)].set(1)
state = state.at[hilbert.layout.edge_site(second, component=1)].set(2)

area = model.get_area([first, second])
edge_squares = model.get_area([first, second], squared=True)
difference = model.area_difference_squared([first, second])
for operator in (area, edge_squares, area @ area, difference):
    _, elements = operator.get_conn_padded(state)
    print(float(jnp.sum(elements)))
```

```text
3.0
5.0
9.0
1.0
```

All operators in this example are diagonal, so summing the row's matrix elements gives the eigenvalue on the chosen configuration.

## Spatial volume

The [`VolumeOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.VolumeOperator) uses the scalar triple products of the charge vectors,

$$
\mathcal V_v(q)
=\sum_{(e,f,g)\text{ incident at }v}
\epsilon_v(e,f,g)\,
\mathbf q_e\cdot(\mathbf q_f\times\mathbf q_g),
$$

$$
V_v(q)=\sqrt{|\mathcal V_v(q)|}.
$$

The sum contains ordered triples of distinct incident edges. The geometric sign $\epsilon_v(e,f,g)$ is the orientation of the corresponding tangents, supplied by the graph. Coplanar triples contribute zero. This formula uses the weak coupling normalisation as implemented, without the separate $1/8$ factor in the SU(2) volume convention.

`model.get_volume(vertex)` returns $V_v$. With `squared=True`, it returns $|\mathcal V_v|=V_v^2$. The result is diagonal in the charge basis. Graph [embedding and orientation data](../graphs/embeddings.md) therefore affect its eigenvalues even though the state itself contains only charges.

## The Euclidean vertex constraint

The [`Thiemann3p1VertexConstraintOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.Thiemann3p1VertexConstraintOperator) combines an oriented loop with a volume difference along a third segment. Let $d_s$ be $+1$ for a stored edge leaving $v$ and $-1$ for one entering it. Define

$$
D_{s,c}(q)
=V_v(q)-V_v\bigl(W(q-d_s h\mathbf e_{s,c})\bigr).
$$

As on the [planar page](wcl-planar.md#the-euclidean-vertex-constraint), $U_{\alpha,c}$ denotes the positive ket shift around a dressed loop. It is the adjoint of the positive row shift returned by the shared holonomy factory.

For a retained incident triple $(i,j,k)$, choose a minimal loop $\alpha$ leaving $v$ along $i$ and returning along $j$. The edge $k$ supplies the commutator segment. The local operator is

$$
H_v=N_v\sum_{(i,j,k)}\sum_{\alpha\in\mathcal L_{ij}}
\sum_{c=0}^{2}
\frac{\varepsilon_{ijk}r_{ijk}}{T_v|\mathcal L_{ij}|}
\left(U_{\alpha,c}-U_{\alpha,c}^\dagger\right)D_{k,c}.
$$

Here $\mathcal L_{ij}$ contains the graph's minimal loops through the selected pair. A pair with no such loop contributes nothing. The difference operator acts before the loop shift, so the volume factor is evaluated on the incoming ket. The corresponding row kernel evaluates it at the connected configuration.

The signs and normalisation follow the implementation's combinatorial convention. Order the incident edges as they occur in `graph.edges`. For every three-edge subset $a<b<c$, retain $(a,b,c)$, $(a,c,b)$, and $(b,c,a)$, with permutation signs $+1,-1,+1$. Reorientation contributes

$$
r_{ijk}=d_i(-d_j)d_k,
\qquad
T_v=2\binom{n_v}{3}
$$

when the valence $n_v$ is at least three. The count $T_v$ includes retained positive-parity triples even when their pair has no loop. The signs $\varepsilon_{ijk}$ in this expression are permutation signs. They must not be substituted for the geometric tangent signs used inside the volume.

Use `thiemann_regularized_constraint(vertex)` to construct one local operator. `get_euclidean_constraint()` constructs the labelled sum over `admissible_vertices`. A vertex with no loop-and-segment contribution has zero local action. The constraint is generally non-Hermitian, and `.adjoint` reverses the full operator action, including the ordering of its volume factors.

## Lapses and the variational objective

As in the planar model, lapse values are used only when `apply_lapse=True`. Use canonical vertex objects as dictionary keys, because lookup uses the vertex argument passed to the factory. Missing entries have lapse one.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
model = nqx.lqx.LqxWCL4D(
    hilbert, lapses={vertex: 2.0 for vertex in graph.vertices}
)

model.euclidean_constraint = model.get_euclidean_constraint(apply_lapse=True)
model.quadratic_constraint = model.get_quadratic_constraint(alternative=True)
local_objectives = model.thiemann_quadratic_constraint(
    apply_lapse=True, alternative=True
)
print(len(local_objectives))
```

```text
5
```

Both constructions above use $H_v^\dagger H_v$ for each vertex. The default ordering is $H_vH_v^\dagger$. The separate list factory builds its own local terms, while `get_quadratic_constraint()` follows the components selected on the model. See [model structure](structure.md#quadratic-constraints) for why the distinction matters when you override a component.

The list factory uses factorised quadratic operators. `lazy=False` is not supported. This avoids expanding the full two-step connectivity merely to estimate a norm squared.
