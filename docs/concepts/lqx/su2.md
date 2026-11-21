---
title: SU(2) models
---

# SU(2) models

[`LqxSU2`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2), also available as `LqxEuclideanSU2`, organises the geometric observables and Euclidean Hamiltonian constraint of the fixed graph SU(2) construction. It acts on [gauge invariant spin networks](../hilbert/su2.md), with edge spins and vertex intertwiners both included in each configuration.

The model requires [`SU2GaugeInvariantHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2GaugeInvariantHilbert), commonly constructed through its alias `SU2Hilbert`. It provides area, Ashtekar–Lewandowski volume, fundamental Wilson loops, and a regularised Euclidean vertex constraint. A Lorentzian component and mass observable are not supplied. Its default total therefore contains the Euclidean component only. Choose the full constraint appropriate to your physical setting rather than assuming that a Lorentzian correction has been included.

## Constructing the model and choosing units

The shared geometric scale is

$$
a_0=8\pi|\gamma|\ell_P^2,
$$

where $\gamma$ is the Barbero–Immirzi parameter. Pass its numerical value in your chosen units as `flux_scale`. Areas scale as $a_0$, and volumes as $a_0^{3/2}$. The default `flux_scale=1.0` expresses the operators in dimensionless flux units.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
model = nqx.lqx.LqxSU2(
    hilbert,
    flux_scale=1.0,
    hamiltonian_prefactor=1j,
)

model.area = model.get_area(graph.edges[:2])
model.volume = model.get_volume(graph.vertices[0])
print(hilbert.n_edges, hilbert.n_channel_sites)
```

```text
10 5
```

The cutoff is specified in doubled spins, so `two_j_max=1` allows edge spins $0$ and $\tfrac12$. The model's `gauge_dimensions` property is three because the Lie algebra $\mathfrak{su}(2)$ has dimension three. It does not mean that each edge stores three independent U(1) charges. The [SU(2) layout](../hilbert/state-layouts.md#the-su2-spin-network-layout) determines the stored labels.

## Area

For a surface intersecting the selected edges transversely away from vertices, [`AreaOperator`](../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.AreaOperator) acts diagonally,

$$
\widehat A(S)|j,\iota\rangle
=a_0\sum_{e\in S}\sqrt{j_e(j_e+1)}\,|j,\iota\rangle.
$$

The input `surface` is the collection of puncturing edges. Intersections are not inferred from a geometric surface object. Repeating an edge repeats its contribution. This puncture formula does not implement the more general intersection prescription for a surface passing through a vertex.

`get_area(surface, squared=True)` returns the square of the **total** selected area,

$$
\widehat A(S)^2
=a_0^2\left(\sum_{e\in S}\sqrt{j_e(j_e+1)}\right)^2.
$$

This is different from the edgewise square used by the [U(1)$^3$ area option](wcl-spatial.md#areas-and-area-differences).

```python
import neuralqx as nqx
import jax.numpy as jnp
import numpy as np

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
model = nqx.lqx.LqxSU2(hilbert)
state = hilbert.complete_channels_for_edges([1] * graph.n_edges)
assert state is not None
state = jnp.asarray(state)

surface = graph.edges[:2]
_, area = model.get_area(surface).get_conn_padded(state)
_, squared = model.get_area(surface, squared=True).get_conn_padded(state)
np.testing.assert_allclose(area[0], np.sqrt(3.0))
np.testing.assert_allclose(squared[0], 3.0)
print(round(float(area[0]), 6))
```

```text
1.732051
```

Each of the two spin-$\tfrac12$ punctures contributes $\sqrt{3}/2$ in these units. The intertwiner choice does not change this area eigenvalue.

## Volume and intertwiner mixing

At a vertex $v$, the [`VolumeOperator`](../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VolumeOperator) uses the Ashtekar–Lewandowski convention

$$
\widehat Q_v
=\frac18\sum_{i<j<k}
\epsilon_v(i,j,k)\,
\mathbf J_i\cdot(\mathbf J_j\times\mathbf J_k),
\qquad
\widehat V_v=a_0^{3/2}\sqrt{|\widehat Q_v|}.
$$

The generators act on the incident representations with the orientation conventions of the graph. The sign $\epsilon_v(i,j,k)$ records the orientation of the three tangents. The sum is over unordered triples, with their order fixed by the incident leg ordering.

At fixed edge spins, $Q_v$ is a matrix in the vertex intertwiner space. The volume can therefore mix intertwiner labels while leaving edge spins unchanged. It is not generally a diagonal function of the stored channel labels. If $Q_v=W\operatorname{diag}(\lambda_r)W^\dagger$, its spectral volume is

$$
V_v=a_0^{3/2}
W\operatorname{diag}\bigl(\sqrt{|\lambda_r|}\bigr)W^\dagger.
$$

Use `model.get_volume(vertex)` to construct it. The `volume_method` model option controls numerical evaluation. With `"auto"`, local blocks up to `spectral_max_dimension`, which defaults to 64, use spectral evaluation. Larger blocks use shifted resolvent quadrature, selected explicitly by `"srq"`. The `"spectral"` option provides a reference based on local diagonalisation and can require precomputation across spin sectors. The dedicated [volume guide](su2-volume/index.md) explains the choice and how to inspect a local action without enumerating the full Hilbert space.

[Shifted resolvent quadrature](su2-volume/srq.md) replaces the matrix function by weighted shifted linear solves. Its accuracy depends on `srq_order`, `srq_step`, `srq_solve_rtol`, and `srq_maxiter`. The [accuracy and convergence guide](su2-volume/accuracy.md) develops the separate quadrature and solve errors, shows how to inspect diagnostics, and compares SRQ with a dense reference. Construction limits can be controlled with [`CompilationBudget`](../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.CompilationBudget).

## Closed holonomies

`get_holonomy(loop)` returns [`HolonomyOperator`](../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.HolonomyOperator), the fundamental Wilson loop on the invariant basis,

$$
\widehat W_\alpha
=P_J\,\operatorname{tr}_{1/2}(h_\alpha)\,P_J.
$$

Here $P_J$ projects the external states to the chosen edge spin cutoff. The trace is not divided by two. For an empty loop, the fundamental trace gives $2I$. For a nontrivial loop, coupling with the fundamental representation changes traversed edge spins by $\pm\tfrac12$ and recouples the affected intertwiners.

Contributions outside the external cutoff are projected out. There is no wrap from the largest spin to zero. This Wilson loop is also not a unitary charge translation. The fundamental SU(2) character is real, so the closed-loop operator is Hermitian. `adjoint=True` returns the same mathematical observable.

The aliases `minimal_loop_holonomy(loop)` and `loop_holonomy(loop)` construct the same operator. For example, use one of the graph's dressed loops and inspect the nonzero connected configurations:

```python
import neuralqx as nqx
import jax.numpy as jnp

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
model = nqx.lqx.LqxSU2(hilbert)
loop = graph.dressed_minimal_loops()[0]
holonomy = model.get_holonomy(loop)
vacuum = jnp.zeros(hilbert.size, dtype=jnp.int32)

connected, elements = holonomy.get_conn_padded(vacuum)
active = elements != 0
print(bool(jnp.any(active)))
print(bool(jnp.all(hilbert.is_valid(connected[active]))))
```

```text
True
True
```

The returned entries are matrix rows $\langle\sigma|O|\sigma'\rangle$. Padding has zero matrix elements. If several contributions lead to the same configuration, their amplitudes are added when evaluating the operator action.

### Open edge matrix components

An open holonomy carries magnetic indices and is not an observable within the invariant basis. `model.edge_holonomy(edge, two_a=1, two_b=1)` constructs [`EdgeHolonomyOperator`](../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.EdgeHolonomyOperator) on a companion [`SU2PeterWeylHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2PeterWeylHilbert), with the same graph and edge cutoff,

$$
\widehat h_{e,ab}=P_J D^{1/2}_{ab}(h_e)P_J,
\qquad a,b\in\{-\tfrac12,\tfrac12\}.
$$

`two_a` and `two_b` are twice these magnetic labels. The returned operator acts on a different Hilbert space from the LQX model, so it cannot be assigned directly to `model.holonomy`. Its `gauge_projection(model.hilbert)` maps it into the invariant setting. For an edge with distinct endpoints that projection vanishes by Gauss law.

An operator adjoint conjugates the chosen matrix component. A matrix adjoint also exchanges the two magnetic indices. The underlying operator exposes `matrix_adjoint_component` for the latter distinction. Use a closed Wilson loop when you need a gauge invariant loop observable.

## The Euclidean Hamiltonian constraint

The model's local Hamiltonian is [`EuclideanVertexConstraint`](../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.EuclideanVertexConstraint). It uses fundamental holonomies and a volume commutator in a fixed graph version of the Thiemann construction. For the continuum construction and its motivation, see [Thiemann's Quantum Spin Dynamics](https://arxiv.org/abs/gr-qc/9606089). The precise regulator and normalisation used here are

$$
\widehat H_v^E
=\frac{\kappa_H N_v}{T_v}
\sum_{i<j}\sum_{k\ne i,j}
\epsilon_v(i,j,k)
\frac{1}{|\mathcal L_{v,ij}|}
\sum_{\alpha\in\mathcal L_{v,ij}}
P_J\,\operatorname{tr}_{1/2}\!\left[
(h_\alpha-h_\alpha^{-1})
h_k[h_k^{-1},\widehat V_v]
\right]P_J.
$$

The loop set $\mathcal L_{v,ij}$ contains all shortest paths connecting the other endpoints of legs $i$ and $j$ while avoiding the base vertex, closed by those two legs. Parallel-edge bigons are included. This is a shortest-loop construction for each incident pair, rather than a cycle basis or just the globally shortest cycles. A pair with no connecting path contributes nothing.

$T_v$ counts the noncoplanar unordered triples at the vertex. A vertex without such triples has zero action. The factor $\kappa_H$ is `hamiltonian_prefactor`, whose default is `1j`. It sets dimensionless constraint units including the imaginary commutator prefactor. Supply physical coupling constants or an alternative overall normalisation through this parameter. Changing `flux_scale` changes the volume inside the Hamiltonian as well.

Only the external states are projected by $P_J$. Intermediate states can include spins $j_{\max}+\tfrac12$ and gauge covariant total-spin-$\tfrac12$ sectors. These intermediate contributions are necessary to the operator definition. Multiplying separately cutoff-projected Wilson loops and a volume restricted to invariant states does not reproduce this Hamiltonian.

Use `thiemann_regularized_constraint(vertex)`, also called `vertex_constraint(vertex)`, for one local term. `get_euclidean_constraint()` assembles the labelled sum over all graph vertices. The full operator need not be Hermitian. Its `.adjoint` transposes and conjugates complete amplitudes, including the ordering with the volume.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
model = nqx.lqx.LqxSU2(
    hilbert,
    lapses={vertex: 0.5 for vertex in graph.vertices},
    hamiltonian_backend="sector",
)

model.euclidean_constraint = model.get_euclidean_constraint(apply_lapse=True)
model.quadratic_constraint = model.get_quadratic_constraint(alternative=True)
print(len(model.get_constraint_terms()))
```

```text
5
```

Lapses are applied only when `apply_lapse=True`. The lookup first checks the canonical vertex key, then its integer position, then falls back to one. The example selects $\sum_v(H_v^E)^\dagger H_v^E$. The default [quadratic ordering](structure.md#quadratic-constraints) is $\sum_vH_v^E(H_v^E)^\dagger$. `thiemann_quadratic_constraint()` provides the local quadratic terms as a separate list.

### Evaluation choices

`hamiltonian_backend="auto"` uses precomputed tables at doubled edge cutoffs up to two and evaluation within encountered spin sectors above that. `"sector"` explicitly selects the latter route, while `"table"` selects the precomputed reference. Operator construction and the first JAX compilation are separate costs.

The Hamiltonian shares the model's volume evaluation settings, including any SRQ approximation. Check that matrix elements remain finite and that observables converge as the quadrature and solve settings are refined. A failed runtime shifted solve produces nonfinite amplitudes, while a failed table solve raises during construction. The [volume accuracy guide](su2-volume/accuracy.md#the-volume-inside-a-hamiltonian) explains how to check the constraint's intermediate sectors and carry a refinement through to the objective used in a simulation.
