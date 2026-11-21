---
title: Prebuilt operators
---

# Prebuilt operators

The prebuilt operators provide the physical actions used by the supplied models, along with projectors for measuring charge probabilities and correlations. They are available directly under [`neuralqx.operator.prebuilt`](../../api/public/neuralqx/operator/prebuilt/index.md). You do not need to construct an LQX model to use them.

The [LQX guides](../lqx/index.md) already develop most of their equations and conventions. This page maps the operator families to those guides and explains the charge projectors, which are useful measurements across U(1) calculations.

## Physical operator families

| Namespace | Available constructions | Physical guide |
| --- | --- | --- |
| [`prebuilt.wcl`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md) | Charge and identity operators, cyclic edge and loop holonomies, Gauss violation, areas, volumes, and weak coupling vertex constraints | [Shared U(1) operators](../lqx/u1.md), [2+1 weak coupling limit](../lqx/wcl-planar.md), and [3+1 weak coupling limit](../lqx/wcl-spatial.md) |
| [`prebuilt.su2`](../../api/public/neuralqx/operator/prebuilt/su2/index.md) | Area, Ashtekar–Lewandowski volume, fundamental Wilson loops, open holonomy components, and Euclidean vertex constraints | [SU(2) models](../lqx/su2.md) and [volume evaluation](../lqx/su2-volume/index.md) |
| [`prebuilt.qr`](../../api/public/neuralqx/operator/prebuilt/qr/index.md) | Reduced flux powers, bounded creation and annihilation shifts, and Euclidean and Lorentzian constraints | [Single vertex quantum reduction](../lqx/qr.md) |
| [`prebuilt.bojowald_swiderski`](../../api/public/neuralqx/operator/prebuilt/bojowald_swiderski/index.md) | Spherical volume, radial triad, and the spherical vertex Hamiltonian | [Spherical symmetry](../lqx/spherical.md) |

Choose a family that matches the basis and physical construction. U(1) cyclic holonomies wrap at their charge cutoff. QR shifts have zero amplitude beyond the boundary. SU(2) holonomies use representation coupling and projection, with separate invariant and Peter–Weyl bases for closed and open actions. These differences remain present when the operators are used directly.

Direct constructors and LQX factories can also have different defaults. For example, the [spherical volume](../lqx/spherical.md#volume-and-radial-triad) factory uses a shifted radial-label sum by default, while the underlying class defaults to no shift. Consult the linked equations and constructor arguments when reproducing a calculation.

## Charge projectors

A charge projector answers a probability question rather than changing the configuration. [`ChargeColoringOperator`](../../api/public/neuralqx/operator/prebuilt/index.md#neuralqx.operator.prebuilt.ChargeColoringOperator) selects one charge $q$ at one flat U(1) component index $s$,

$$
P_{s,q}|\sigma\rangle
=\delta_{\sigma_s,q}|\sigma\rangle,
\qquad
\langle P_{s,q}\rangle=\Pr_\psi(\sigma_s=q).
$$

The other labels remain unrestricted. Use `hilbert.layout.edge_site(edge, component=c)` to locate the component. This avoids confusing a flat component index with a base-edge index in a space with several gauge copies.

[`ColoringOperator`](../../api/public/neuralqx/operator/prebuilt/index.md#neuralqx.operator.prebuilt.ColoringOperator) instead selects a complete charge vector on one base edge,

$$
P_{e,\mathbf q}|\sigma\rangle
=\left[\prod_{c=0}^{N-1}\delta_{\sigma_{e,c},q_c}\right]|\sigma\rangle.
$$

Its expectation is the joint probability of that edge's component values, including their correlations. Its `site` argument is a base-edge index, and `charge_vector` must contain one allowed label per gauge component.

```python
import neuralqx as nqx
import jax.numpy as jnp
from neuralqx.operator.prebuilt import ChargeColoringOperator, ColoringOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1, gauge_dimensions=3)
edge = graph.edges[0]
first = hilbert.layout.edge_site(edge, component=0)
third = hilbert.layout.edge_site(edge, component=2)

single_component = ChargeColoringOperator(hilbert, site=third, charge=-1)
whole_edge = ColoringOperator(hilbert, site=0, charge_vector=(1, 0, -1))
matching = jnp.zeros(hilbert.size, dtype=jnp.int32).at[first].set(1).at[third].set(-1)
configurations = jnp.stack((matching, jnp.zeros_like(matching)))

_, single_values = single_component.get_conn_padded(configurations)
_, edge_values = whole_edge.get_conn_padded(configurations)
print(single_values[:, 0].tolist())
print(edge_values[:, 0].tolist())
```

```text
[1.0, 0.0]
[1.0, 0.0]
```

Both operators are Hermitian diagonal projectors with $P^2=P$. They preserve the existing gauge conditions and can be used on unconstrained or gauge invariant U(1) spaces. A projector measures the probability within that chosen space. It does not enlarge the space to include a forbidden configuration. Fractional charge labels are supported when they belong to the Hilbert space's local lattice.

Use `state.expect(projector)` to estimate the probability from a wavefunction. The [application example](application.md#measure-an-operator-with-a-variational-state) gives a complete sampled calculation.

## Joint probabilities and correlations

Products of charge projectors select simultaneous events,

$$
\left\langle\prod_{i=1}^{n}P_{e_i,\mathbf q_i}\right\rangle
=\Pr_\psi\left(\sigma_{e_i}=\mathbf q_i\text{ for all }i\right).
$$

Construct such products with `@` or use [`n_point_function`](../../api/public/neuralqx/operator/prebuilt/index.md#neuralqx.operator.prebuilt.n_point_function). This helper returns an operator, not an already evaluated correlation. It takes an operator factory, a sequence of selected indices, and the corresponding charge selections.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt import ColoringOperator, n_point_function

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
first = ColoringOperator(hilbert, site=0, charge_vector=(1,))
second = ColoringOperator(hilbert, site=1, charge_vector=(0,))
joint = n_point_function(
    hilbert, ColoringOperator, edges=(0, 1), charge_vectors=((1,), (0,))
)

# Equal amplitudes give a uniform probability on this small product basis.
basis = hilbert.all_states()
p_first = np.mean(np.asarray(first.get_conn_padded(basis)[1][:, 0]))
p_second = np.mean(np.asarray(second.get_conn_padded(basis)[1][:, 0]))
p_joint = np.mean(np.asarray(joint.get_conn_padded(basis)[1][:, 0]))
print(round(float(p_joint), 6))
print(round(float(p_joint - p_first * p_second), 6))
```

```text
0.111111
0.0
```

Each selected charge has probability $1/3$ in this uniform product state, so the joint probability is $1/9$. The connected correlation

$$
C_{12}=\langle P_1P_2\rangle-\langle P_1\rangle\langle P_2\rangle
$$

vanishes. In a correlated variational state, or after restricting the basis by gauge conditions, the joint probability need not factorise.

Repeated identical projectors have no additional effect. Incompatible charge selections on the same component give zero. An empty selection passed to `n_point_function` gives the identity. With `ChargeColoringOperator`, pass flat component indices and scalar charges instead of base-edge indices and charge vectors.

For a general noncommuting factory, the helper retains the order $O_{n-1}\cdots O_1O_0$. Charge projectors commute, so this does not change their joint event. Use explicit `@` products when the ordering of a different operator family matters.

The same projectors can be parents of [expectation-level penalties](penalties.md), but measuring a probability and penalising it are separate choices. Keep the unmodified projector available as an observable when interpreting the resulting simulation.
