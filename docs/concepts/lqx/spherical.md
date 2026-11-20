---
title: Spherical symmetry
---

# Spherical symmetry

[`LqxSphericalBojowaldSwiderski`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSphericalBojowaldSwiderski), also available as `LqxSphericalBoSw`, provides the spherical Bojowald–Swiderski construction on an ordered radial graph. The physical variables are radial edge labels $k$ and transverse labels $\mu$ at the radial vertices. They replace the unrestricted graph degrees of freedom of a general spin network.

The model supplies label operators, cyclic holonomies, a local volume, the radial triad, and a vertex Hamiltonian. [Bojowald and Swiderski's Hamiltonian constraint paper](https://arxiv.org/abs/gr-qc/0511108) introduces the spherical variables and their quantisation. This page states the finite lattice action and normalisation used by the implementation, including its spin connection differences and cutoff wrapping.

## The radial graph and state

Construct a [`HalfLadderGraph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.HalfLadderGraph). Its radial chain carries the $k$ labels, and one transverse edge at each radial vertex carries the corresponding $\mu$ label. The transverse edges serve to store these local reduced variables.

```python
import neuralqx as nqx

graph = nqx.graph.HalfLadderGraph(6, connected=False)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=4)
model = nqx.lqx.LqxSphericalBoSw(hilbert, immirzi=1.0, delta=2)

print(len(graph.edges_k), len(graph.edges_mu))
print(hilbert.size)
print(len(model.inner_vertices))
```

```text
6 7
13
3
```

A chain of length six has six radial edges and seven radial vertices. Each radial vertex has its own transverse edge, giving thirteen labels in total. The graph exposes `edges_k`, `edges_mu`, and `vertices_k` in their radial order. Use those collections and the [Hilbert layout](../hilbert/state-layouts.md) when preparing a configuration.

The Hamiltonian requires one gauge component and integer labels with unit spacing. The regularisation parameter `delta` must be a positive even integer, so both $\delta$ and $\delta/2$ shifts stay on that lattice. The Barbero–Immirzi parameter `immirzi` must be nonzero.

The model selects `vertices_k[2:-2]` for its total constraint, exposed as `inner_vertices`. It therefore excludes two radial vertices at each end. This is a stencil choice, not a statement that every other vertex is geometrically absent. An integer vertex argument is interpreted as a position in `vertices_k`. Using the canonical vertex objects avoids ambiguity with the full graph's vertex ordering.

## Label and holonomy operators

`model.number(edge)` uses [`NumberOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) to read either a radial or transverse label,

$$
\widehat n_e|\sigma\rangle=\sigma_e|\sigma\rangle,
\qquad \sigma=(k,\mu).
$$

`model.get_holonomy(edge)` constructs [`U1HolonomyOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.U1HolonomyOperator). In the matrix-row convention,

$$
\langle\sigma|R_e|\sigma'\rangle
=\delta_{\sigma',\,W(\sigma+\mathbf e_e)}.
$$

$W$ wraps each shifted label into the finite local range. `adjoint=True` reverses the row shift. This elementary holonomy moves by one lattice unit. The larger shifts in the Hamiltonian are controlled separately by `delta`.

## Volume and radial triad

At a radial vertex $v$, let $k_-$ and $k_+$ be the adjacent radial labels and define

$$
K_v=k_-+k_+.
$$

The dimensionless [`SphericalVolumeOperator`](../../api/public/neuralqx/operator/prebuilt/bojowald_swiderski/index.md#neuralqx.operator.prebuilt.bojowald_swiderski.SphericalVolumeOperator) is diagonal,

$$
V_v^{(a)}(\sigma)=|\mu_v|\sqrt{|K_v+a|}.
$$

Use `get_volume(vertex, shift=a)` or its model-specific equivalent `spherical_volume(vertex, shift=a)`. The model factory defaults to `shift=1`. Constructing `SphericalVolumeOperator` directly defaults to `shift=0`. State this choice explicitly when comparing the two routes or reproducing a calculation.

The radial triad is supplied by [`SphericalExOperator`](../../api/public/neuralqx/operator/prebuilt/bojowald_swiderski/index.md#neuralqx.operator.prebuilt.bojowald_swiderski.SphericalExOperator),

$$
E_v^x(\sigma)=\frac{\gamma}{2}K_v.
$$

Construct it with `model.spherical_Ex(vertex)`. Unlike the volume, it retains the sign of the radial label sum. Both observables require a radial vertex with the adjacent radial edges available.

```python
import neuralqx as nqx
import jax.numpy as jnp
import numpy as np

graph = nqx.graph.HalfLadderGraph(6)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=4)
model = nqx.lqx.LqxSphericalBoSw(hilbert, immirzi=0.8, delta=2)
vertex = model.inner_vertices[0]
state = jnp.ones(hilbert.size, dtype=jnp.int32)

_, volume = model.get_volume(vertex).get_conn_padded(state)
_, unshifted = model.get_volume(vertex, shift=0).get_conn_padded(state)
_, triad = model.spherical_Ex(vertex).get_conn_padded(state)
np.testing.assert_allclose(volume[0], np.sqrt(3.0))
np.testing.assert_allclose(unshifted[0], np.sqrt(2.0))
print(round(float(triad[0]), 6))
```

```text
0.8
```

Here $k_-=k_+=\mu_v=1$. Changing the selected volume property does not change the volume differences built into the Hamiltonian below. Those differences are part of its specified regularisation.

## The vertex Hamiltonian

[`BojowaldSwiderskiVertexConstraintOperator`](../../api/public/neuralqx/operator/prebuilt/bojowald_swiderski/index.md#neuralqx.operator.prebuilt.bojowald_swiderski.BojowaldSwiderskiVertexConstraintOperator) has nineteen padded branches. Its action consists of a diagonal term, two transverse shifts, and eight shifts on each adjacent radial edge.

### Volume difference coefficients

The two coefficients entering these branches are

$$
X_v(\sigma)
=|\mu_v|\left(\sqrt{|K_v+1|}-\sqrt{|K_v-1|}\right),
$$

$$
P_v(\sigma)
=\operatorname{clip}\left(\frac{2\mu_v}{\delta},-1,1\right)
\sqrt{|K_v|}.
$$

The clipped factor regularises the sign near $\mu_v=0$. These expressions use ordinary real square roots and absolute values of the indicated label combinations. Cyclic wrapping applies to the shifted state labels, not to the scalar expression under each square root.

### Off-diagonal action

There are two transverse ket transitions,

$$
|\sigma\rangle\longmapsto
|W(\sigma\pm2\delta\mathbf e_{\mu_v})\rangle,
$$

each with coefficient $X_v(\sigma)$. For an adjacent radial edge $e$ with left and right endpoints $u,w$, define

$$
\Delta_{e,s}
=s_k\delta\mathbf e_{k_e}
+s_1\frac{\delta}{2}\mathbf e_{\mu_u}
+s_2\frac{\delta}{2}\mathbf e_{\mu_w},
\qquad s_k,s_1,s_2\in\{-1,+1\}.
$$

Each of its eight sign choices gives the ket transition $|\sigma\rangle\mapsto|W(\sigma+\Delta_{e,s})\rangle$ with coefficient $s_ks_2P_v(\sigma)$. For the left radial edge, $(u,w)=(v-1,v)$. For the right one, $(u,w)=(v,v+1)$. Keeping this endpoint order fixes the sign convention.

### Spin connection and diagonal action

For the implemented finite differences, define the rescaled radial triad $\widetilde E_v=K_v/2$ and link averages

$$
I_{v,\pm}=\frac{|\mu_v|+|\mu_{v\pm1}|}{2}.
$$

The spin connection and its forward difference are

$$
\Gamma_v
=\frac14\left[
-\frac{\widetilde E_{v+1}-\widetilde E_v}{I_{v,+}}
+\frac{\widetilde E_v-\widetilde E_{v-1}}{I_{v,-}}
\right],
\qquad
\Gamma'_v=\Gamma_{v+1}-\Gamma_v.
$$

A fraction with zero denominator contributes zero. The triad used inside this difference omits the factor $\gamma$ in the observable $E_v^x$. The Hamiltonian includes its Immirzi factors explicitly. The prime denotes this unit-step lattice difference, not an automatically rescaled derivative with a user-supplied radial spacing.

The diagonal coefficient is

$$
D_v(\sigma)
=-2X_v
-4\gamma^2\delta^2(1-\Gamma_v^2)X_v
-4\gamma^2\delta^2\Gamma'_v P_v.
$$

Combining the branches gives the complete local action,

$$
\begin{aligned}
H_v|\sigma\rangle
={}&D_v(\sigma)|\sigma\rangle\\
&+X_v(\sigma)\sum_{\eta=\pm1}
|W(\sigma+2\eta\delta\mathbf e_{\mu_v})\rangle\\
&+P_v(\sigma)\sum_{e\in\{e_-,e_+\}}
\sum_{s_k,s_1,s_2=\pm1}s_ks_2
|W(\sigma+\Delta_{e,s})\rangle.
\end{aligned}
$$

Setting `include_gamma_terms=False` sets $\Gamma_v^2$ and $\Gamma'_v$ to zero. It does **not** remove the entire term proportional to $\gamma^2$, since the constant part of $1-\Gamma_v^2$ remains.

### Boundaries and matrix rows

The stencil reaches neighbouring radial and transverse labels. Missing outer radial labels use `outer_km_constant` or `outer_kp_constant`, both zero by default. A missing outer transverse label needed by the forward difference is treated as zero. These are specified boundary substitutions, not an inferred boundary state. Select them deliberately when the stencil reaches the edge of your chain.

The formulas above describe ket action. In the forward matrix-row kernel, each off-diagonal transition uses the negative of the listed shift and evaluates its coefficient at the connected configuration. The adjoint uses the positive shift and evaluates the coefficient at the input configuration. This distinction is necessary because the Hamiltonian is generally not Hermitian.

```python
import neuralqx as nqx
import jax.numpy as jnp

graph = nqx.graph.HalfLadderGraph(6)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=4)
model = nqx.lqx.LqxSphericalBoSw(hilbert, delta=2)
vertex = model.inner_vertices[0]
state = jnp.ones(hilbert.size, dtype=jnp.int32)

operator = model.spherical_hamiltonian_on_vertex(vertex)
connected, elements = operator.get_conn_padded(state)
adjoint_connected, adjoint_elements = operator.adjoint.get_conn_padded(state)
print(connected.shape, elements.shape)
print(bool(jnp.all(jnp.isfinite(adjoint_elements))))
```

```text
(19, 13) (19,)
True
```

Padding and coincident shifted configurations can make the number of distinct nonzero targets smaller than nineteen. In particular, small cutoffs can identify different shifts through wrapping. Increasing the cutoff changes this finite operator, so check the cutoff dependence of a physical result.

## The total constraint and objective

`get_euclidean_constraint()` constructs the labelled sum over `inner_vertices`. This is the common model property used for the implemented spherical Hamiltonian. There is no separately supplied Lorentzian component. A selected local vertex can also be passed to this factory, or constructed directly through `spherical_hamiltonian_on_vertex(vertex)`.

```python
import neuralqx as nqx

graph = nqx.graph.HalfLadderGraph(6)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=4)
model = nqx.lqx.LqxSphericalBoSw(hilbert, immirzi=0.8, delta=2)

model.euclidean_constraint = model.get_euclidean_constraint(
    include_gamma_terms=True,
    outer_km_constant=1,
    outer_kp_constant=1,
)
model.quadratic_constraint = model.get_quadratic_constraint(alternative=True)
print(len(model.get_constraint_terms()))
```

```text
3
```

This objective is $\sum_vH_v^\dagger H_v$ with the selected boundary substitutions. The default quadratic ordering is $\sum_vH_vH_v^\dagger$. The method `hamiltonian_list(alternative=...)` returns local quadratic terms constructed with its default vertex settings, not a list of linear Hamiltonians and not a reconstruction of your selected overrides. Use the inherited quadratic factory when the objective should follow those selections.

The model does not provide an area or mass operator. Additional observables should specify the reduced quantisation and units appropriate to your calculation. The [custom model guide](custom-models.md) explains how to package them alongside the existing factories.
