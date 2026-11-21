---
title: SU(2) volume
---

# SU(2) volume

The volume operator measures three-dimensional quantum geometry at a graph vertex. It also enters the volume commutator in the [SU(2) Euclidean constraint](../su2.md#the-euclidean-hamiltonian-constraint). Its numerical evaluation therefore matters both when you measure an observable and when you optimise a constraint built from it.

neuraLQX provides dense spectral evaluation and **shifted resolvent quadrature**, abbreviated SRQ. Both evaluate the same volume definition. Dense evaluation diagonalises a local matrix. SRQ approximates its action using sparse matrix-vector products and shifted linear solves. The practical choice depends on the size of the local intertwiner space and the accuracy you need.

This page introduces the operator and its interfaces. Continue with [how SRQ works](srq.md) for the quadrature and solver mathematics, then [accuracy and convergence](accuracy.md) for diagnostics and reproducible comparisons with dense evaluation.

## Volume acts on intertwiners

At a vertex with incident spins $j_1,\ldots,j_d$, the gauge invariant local states belong to

$$
\mathcal I_v
=\operatorname{Inv}_{SU(2)}
\left(\bigotimes_{i=1}^{d}\mathcal H_{j_i}\right).
$$

The [intertwiner channels](../../hilbert/su2.md#vertex-coupling-and-intertwiner-labels) specify a basis for this space. Once the incident edge spins are fixed, its dimension $D_v$ determines the size of the local volume problem. The volume keeps those edge spins fixed and can mix the channels at that vertex. It leaves the labels at other vertices unchanged.

The convention implemented by [`VolumeOperator`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VolumeOperator) is

$$
\begin{aligned}
Q_v&=\frac18\sum_{i<j<k}
\epsilon_v(i,j,k)\,
\mathbf J_i\cdot(\mathbf J_j\times\mathbf J_k),\\
V_v&=c_V\sqrt{|Q_v|},
\qquad c_V=a_0^{3/2}.
\end{aligned}
$$

Here $a_0=8\pi|\gamma|\ell_P^2$ is the model's `flux_scale`. Using `flux_scale=1.0` gives dimensionless volume units. The generators act on the incident representations in the graph's orientation convention, and $\epsilon_v(i,j,k)$ is the sign of the three outgoing tangents. The factor $1/8$ accompanies the sum over unordered triples used here. [Ashtekar and Lewandowski's volume construction](https://arxiv.org/abs/gr-qc/9711031) gives the underlying geometric context.

The absolute value is an operator function. For Hermitian $Q_v$,

$$
|Q_v|=(Q_v^2)^{1/2},
\qquad
\sqrt{|Q_v|}=(Q_v^2)^{1/4}.
$$

Taking absolute values or square roots of individual matrix entries would give a different object. Likewise, a stored channel label is an intermediate spin, not a volume eigenvalue. The matrix must be evaluated in the intertwiner basis before it is combined with wavefunction amplitudes.

The resulting volume is positive semidefinite, so its exact expectation is nonnegative. Individual off-diagonal matrix elements can nevertheless be negative. At a coplanar vertex all tangent signs vanish, and a gauge invariant trivalent vertex also has zero volume. Other cancellations can occur for particular embeddings or spin sectors. A zero result should therefore be interpreted together with the [graph's orientation data](../../graphs/embeddings.md) and the chosen spins.

## Construct and inspect a volume

[`LqxSU2.get_volume`](../../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2.get_volume) constructs a vertex volume in the model's units. Here every edge of a [`K5Graph`](../../../api/public/neuralqx/graph/index.md#neuralqx.graph.K5Graph) is assigned spin two, stored as the doubled label `4`.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(
    graph, two_j_max=4, random_init="vacuum"
)
model = nqx.lqx.LqxSU2(hilbert, volume_method="auto")
volume = model.get_volume(graph.vertices[0])

configuration = hilbert.complete_channels_for_edges([4] * graph.n_edges)
assert configuration is not None
configuration = jnp.asarray(configuration)
connected, elements = volume.get_conn_padded(configuration)

print(connected.shape)
print(bool(jnp.all(hilbert.is_valid(connected))))
print(bool(jnp.all(connected[:, :hilbert.n_edges] == configuration[:hilbert.n_edges])))
```

```text
(5, 15)
True
True
```

Each row has capacity for five connected configurations, each containing ten edge labels and five vertex channel labels. The first ten entries remain unchanged. `elements` contains the matrix row needed by the [local estimator](../../operators/application.md#from-the-expectation-value-to-the-local-estimator), including zero entries where applicable. A variational measurement is made with `state.expect(volume)` after constructing your variational state.

This calculation does not enumerate the full Hilbert space. The local intertwiner space can be small even when the number of complete spin network configurations is very large.

## Dense spectral evaluation

If

$$
Q_v=W\operatorname{diag}(\lambda_r)W^\dagger,
$$

then the volume is

$$
V_v=c_V W\operatorname{diag}\bigl(\sqrt{|\lambda_r|}\bigr)W^\dagger.
$$

The API calls this method `"spectral"`. It is the dense evaluation path. For a local block of dimension $D_v$, it stores dense arrays of order $D_v^2$ and performs an eigendecomposition whose usual cost grows as $D_v^3$.

Spectral evaluation provides a useful numerical reference. It still uses floating-point arithmetic. The implementation treats eigenvalues within a scale-dependent roundoff threshold as zero before taking the square root. This avoids turning a numerical remnant of a zero eigenvalue into a spurious volume. In the dense problem being diagonalised, the threshold has the form

$$
\tau_0=32\,\varepsilon_{\mathrm{mach}}\,
D_{\mathrm{eigh}}\,\max_r|\lambda_r|.
$$

$D_{\mathrm{eigh}}$ is the dimension of that dense eigenproblem, including any runtime padding. This is numerical zero detection, not a user-selected physical spin cutoff.

## Select the evaluation policy

The model option is `volume_method`, while a directly constructed volume uses `method`.

| Method | Behaviour |
| --- | --- |
| `"auto"` | Uses spectral evaluation for local blocks with dimension at most `spectral_max_dimension`, and SRQ above it. The default threshold is 64. |
| `"spectral"` | Requests dense diagonalisation. A full `VolumeOperator` precomputes tables across local spin assignments allowed by the cutoff. |
| `"srq"` | Uses sparse quadrature for encountered sectors without precomputing a dense volume table across all spin assignments. |

Automatic selection uses the actual admissible local dimension, not the dimension of the complete Hilbert space. Setting `spectral_max_dimension=0` with `"auto"` selects the SRQ branch throughout. The threshold controls the computational route and does not discard intertwiner channels.

The distinction between a local block and a full spectral operator is important. With doubled-spin cutoff $J$ and vertex valence $d$, the full spectral constructor considers $(J+1)^d$ local spin assignments. Even when each block is manageable, constructing all of them can be expensive. The `"auto"` and `"srq"` paths avoid this cutoff-wide table, although they still allocate padded local data and connected rows.

[`CompilationBudget`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.CompilationBudget) bounds construction resources. Exceeding a budget raises an error. It does not silently remove sectors or channels. Increasing a budget permits more allocation and does not improve quadrature accuracy by itself.

## Work within one spin sector

For a reference calculation or convergence study, use [`fixed_sector_action`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VolumeOperator.fixed_sector_action). Given one valid full configuration, it extracts the incident spins and graph orientation and constructs a [`VertexVolumeAction`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VertexVolumeAction) for that sector alone.

The resulting `action.channels` records the ordering of the local coefficient vector. `action.apply(vector)` returns the volume action and an array of shifted-solve residuals. Spectral evaluation returns an empty residual array. `action.apply_q(vector)` applies the underlying dimensionless $Q_v$.

The default for `fixed_sector_action` is `method="srq"`, even when the parent volume uses another policy. Specify `method="spectral"` explicitly when you want a dense reference. This constructs only the selected local block and requires neither full Hilbert space enumeration nor `exact=True` on the Hilbert space. The [accuracy guide](accuracy.md#compare-with-a-fixed-sector-dense-reference) demonstrates the complete comparison.

For calculations that already specify the incident spins and orientation triples directly, `VertexVolumeAction` can also be constructed without a full graph state. Its `spins` use doubled integers, and its triples refer to positions in that supplied incident-leg order. Using `fixed_sector_action` is convenient when you want those choices to agree with an existing simulation.

## Shared settings and local overrides

Model settings are inherited by both volume factories and the volume evaluations inside the Euclidean constraint. A call such as `model.get_volume(vertex, method="srq", solve_rtol=1e-13)` overrides the returned observable only. It does not change the model's shared configuration or an already constructed Hamiltonian.

The [SRQ guide](srq.md#settings-and-their-api-names) maps the option names across the model, volume operator, and fixed-sector action. The [accuracy guide](accuracy.md#the-volume-inside-a-hamiltonian) explains how to carry a convergence study through to the constraint rather than checking only the standalone volume.
