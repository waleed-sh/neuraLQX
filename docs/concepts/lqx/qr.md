---
title: Single vertex quantum reduction
---

# Single vertex quantum reduction

[`LqxQR`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxQR) provides the three-label single vertex quantum-reduced model. Its state is specified by nonnegative reduced spin labels $q_0,q_1,q_2$, associated with three spatial directions. The model supplies flux powers, bounded shifts, and both Euclidean and Lorentzian constraints.

For the physical setting of single vertex dynamics in quantum-reduced loop gravity, see [Mäkinen's study](https://arxiv.org/abs/2412.01375). The equations below specify the finite, symmetrically ordered operators implemented in neuraLQX, including their boundary behaviour.

## Constructing the reduced space

Use a U(1) Hilbert space with one gauge component, exactly three base edges, and nonnegative local labels. The [`SingleVertexGraph`](../../api/public/neuralqx/graph/index.md#neuralqx.graph.SingleVertexGraph) constructor is a convenient way to provide those three directions.

```python
import neuralqx as nqx

graph = nqx.graph.SingleVertexGraph(3)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2, positive_qn=True)
model = nqx.lqx.LqxQR(hilbert)

print(hilbert.size, hilbert.dimension)
print(model.constraint.hilbert == hilbert)
```

```text
3 125
True
```

With `cutoff=2, positive_qn=True`, each label takes the values $0,1,2,3,4$. The shift to a nonnegative range preserves the number of local states, as explained in the [U(1) space guide](../hilbert/u1.md). These labels encode the reduced variables. This use of a U(1) container does not introduce the full edge and intertwiner basis of an [SU(2) spin network](../hilbert/su2.md).

## Fluxes and inverse powers

[`QRFluxOperator`](../../api/public/neuralqx/operator/prebuilt/qr/index.md#neuralqx.operator.prebuilt.qr.QRFluxOperator) is diagonal. The factory `model.flux(site, power=p)` gives

$$
F_s^{(p)}|q\rangle=q_s^p|q\rangle.
$$

With `inverse=True`, it gives the generalised inverse power

$$
F_s^{(-p)}|q\rangle=
\begin{cases}
q_s^{-p}|q\rangle & q_s\ne0,\\
0 & q_s=0.
\end{cases}
$$

Mapping the zero eigenvalue to zero avoids a divergent inverse at that boundary. The same convention is used by the inverse factors in the Hamiltonians. `site` is the flat label index, which is one of `0`, `1`, and `2` in this model.

## Bounded shifts

The factories `creation(site, n=1)` and `annihilation(site, n=1)` construct [`QRCreationOperator`](../../api/public/neuralqx/operator/prebuilt/qr/index.md#neuralqx.operator.prebuilt.qr.QRCreationOperator) and [`QRAnnihilationOperator`](../../api/public/neuralqx/operator/prebuilt/qr/index.md#neuralqx.operator.prebuilt.qr.QRAnnihilationOperator). In the matrix-row convention,

$$
\langle q|C_s^{(n)}|q'\rangle
=\begin{cases}
1 & q'=q+nh\mathbf e_s\text{ lies inside the cutoff},\\
0 & \text{otherwise}.
\end{cases}
$$

The annihilation operator is its adjoint and uses the negative row shift. There are no bosonic factors such as $\sqrt{q_s+1}$. The names identify these bounded shift operators. As with the [U(1) row convention](u1.md#edge-and-loop-holonomies), a positive connected-state shift corresponds to the opposite shift when the operator acts on a ket.

`get_holonomy(site, n=n)` returns the positive row shift, while `adjoint=True` returns its adjoint. The integer `n` must be positive. The local spacing is $h=1$ in the examples here.

```python
import neuralqx as nqx
import jax.numpy as jnp

graph = nqx.graph.SingleVertexGraph(3)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2, positive_qn=True)
model = nqx.lqx.LqxQR(hilbert)
state = jnp.array([4, 1, 0], dtype=jnp.int32)

_, at_boundary = model.creation(0).get_conn_padded(state)
connected, backward = model.annihilation(0).get_conn_padded(state)
_, inverse_zero = model.flux(2, inverse=True).get_conn_padded(state)
print(float(at_boundary[0]))
print(int(connected[0, 0]), float(backward[0]))
print(float(inverse_zero[0]))
```

```text
0.0
3 1.0
0.0
```

The attempted transition past the upper boundary has zero amplitude. It does not wrap to the smallest label. This is the main operational difference from the shared cyclic U(1) holonomies.

## Euclidean constraint

Let $C_a=C_a^{(1)}$ and define the diagonal sandwich factor

$$
F_{ab|c}(q)=q_a^p q_b^p q_c^{-p},
$$

with the generalised inverse convention above. [`QREuclideanConstraintOperator`](../../api/public/neuralqx/operator/prebuilt/qr/index.md#neuralqx.operator.prebuilt.qr.QREuclideanConstraintOperator) implements

$$
H_E=\frac{N}{4\gamma^2}
\sum_{(a,b,c)\in\mathcal P}
F_{ab|c}(C_a-C_a^\dagger)(C_b-C_b^\dagger)F_{ab|c},
$$

$$
\mathcal P=\{(0,1,2),(0,2,1),(1,2,0)\}.
$$

The default physical flux exponent is $p=\tfrac14$. The lapse $N$ and nonzero Barbero–Immirzi parameter $\gamma$ default to one. Construct another choice with `get_euclidean_constraint(lapse=N, power=p, immirzi=gamma)`.

The factors on both sides are part of the ordering. They evaluate the flux weights at both ends of each transition and make the operator Hermitian, including at the bounded cutoff. Each of the three direction pairs has four shift choices, giving twelve padded connectivity branches.

## Lorentzian constraint

For each direction $a$, let $b,c$ denote the other two directions and define

$$
G_a(q)=q_a^{3/8}q_b^{-1/8}q_c^{-1/8},
\qquad
D_a=G_a\left[\frac12 I-\frac14(C_a+C_a^\dagger)\right]G_a.
$$

The [`QRLorentzianConstraintOperator`](../../api/public/neuralqx/operator/prebuilt/qr/index.md#neuralqx.operator.prebuilt.qr.QRLorentzianConstraintOperator) is

$$
H_L=-16\frac{1+\gamma^2}{\gamma^2}
\sum_{a=0}^{2}D_a^2.
$$

Each $D_a$ is Hermitian, so $H_L$ is Hermitian and negative semidefinite. The finite action includes the intermediate cutoff checks in each two-step product. A path that exits the permitted range contributes zero even if a second shift would return its final label to the range.

Use `get_lorentzian_constraint(immirzi=gamma)`. This factory uses unit lapse. It does not inherit a lapse passed to the Euclidean factory.

## Selecting a total constraint

The default total is $H=H_E+H_L$, with $N=1$, $p=\tfrac14$, and $\gamma=1$. To change the model's selections, assign the operators returned by the factories. Here a constant lapse is applied to both components, and the small matrices are checked against the symmetry properties just described.

```python
import neuralqx as nqx
import numpy as np

graph = nqx.graph.SingleVertexGraph(3)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, positive_qn=True)
model = nqx.lqx.LqxQR(hilbert)
gamma, lapse = 0.8, 1.5

model.euclidean_constraint = model.get_euclidean_constraint(
    immirzi=gamma, lapse=lapse
)
model.lorentzian_constraint = lapse * model.get_lorentzian_constraint(
    immirzi=gamma
)

euclidean = np.asarray(model.euclidean_constraint.to_dense())
lorentzian = np.asarray(model.lorentzian_constraint.to_dense())
np.testing.assert_allclose(euclidean, euclidean.conj().T, atol=1e-12)
np.testing.assert_allclose(lorentzian, lorentzian.conj().T, atol=1e-12)
print(bool(np.linalg.eigvalsh(lorentzian).max() < 1e-12))
```

```text
True
```

The convenience method `total_constraint(lapse=..., power=..., immirzi=...)` constructs $H_E+H_L$ without changing the selected properties. Its lapse argument multiplies the Euclidean contribution only. If your lapse should multiply the entire single vertex constraint, multiply a unit-lapse total by that scalar or select both components as above.

The model treats this total as one block. Since both components are Hermitian for the real parameters used here, the default quadratic objective is $HH^\dagger=H^2$, and the alternative ordering agrees. See [model structure](structure.md) for how selections and total overrides interact.

No default area, volume, or mass is provided by this model. If your reduced construction needs one, define the observable with the intended reduced quantisation and attach it explicitly, or package it in a [subclass](custom-models.md).
