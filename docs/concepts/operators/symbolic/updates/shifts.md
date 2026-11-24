---
title: Shifts and transport
---

# Shifts and transport

[`shift`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.shift) adds a numerical displacement to one coordinate of the working target,

$$
q_i^{\mathrm{new}}=q_i^{\mathrm{current}}+\Delta.
$$

It does not wrap, clip, or search for the next admissible value. If the local basis is $\{-2,0,2\}$, a shift of one generally leaves the basis. A shift of two changes the physical label by two. For a step through positions in a finite basis, use [modular shifts](modular.md) instead.

## A bounded row shift

The following rule connects each U(1) charge below the cutoff to the next charge, with coefficient one. At the upper boundary it contributes no connection.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
bounded = (
    s.SymbolicOperatorBuilder(hilbert, "bounded_shift")
    .for_each_site("e")
    .where(s.site("e") < 1)
    .emit(s.shift("e", 1), matrix_element=1)
    .compile()
)
sources = jnp.array([[-1], [0], [1]], dtype=hilbert.dtype)
targets, coefficients = bounded.get_conn_padded(sources)
np.testing.assert_allclose(coefficients[:, 0], [1, 1, 0])
np.testing.assert_array_equal(targets[:, 0, 0], [0, 1, 1])
```

The final target is the source value one because that row contains only zero padding. It is not a reflected or stationary transition with nonzero coefficient. To define a truncated creation or annihilation operator, also attach the correct occupation-dependent matrix element. The [emission guide](../emissions.md#row-convention-and-ladder-operators) derives the convention.

`delta` can be an expression. It is evaluated using the original input, even if the update program has already changed that coordinate. The arithmetic shift itself adds the result to the current working target. This distinction matters in [update chains](chaining.md).

## Transport between two coordinates

[`hop(i, j, amount=delta)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.hop) abbreviates a subtraction at $i$ followed by an addition at $j$,

$$
q_i'=q_i-\Delta,\qquad q_j'=q_j+\Delta.
$$

For distinct coordinates it preserves their numerical sum, but not necessarily every constraint of the Hilbert space. It also does not supply a bosonic square root, a fermionic sign, or an SU(2) recoupling coefficient. Those belong in the matrix element.

For occupations, the row update $n'=n-e_i+e_j$ corresponds to $a_i^\dagger a_j$ with coefficient $\sqrt{n_i(n_j+1)}$. The operator transfers a particle from $j$ to $i$ when acting on kets. The row coordinates move in the opposite direction.

```python
pair_graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
occupations = nqx.hilbert.HomogeneousDiscreteHilbert(
    pair_graph, nqx.hilbert.LocalRange(0, 1, 3)
)
ni, nj = s.site("i").value, s.site("j").value
t = 0.25
hopping = (
    s.SymbolicOperatorBuilder(occupations, "hopping", hermitian=True)
    .for_each_distinct_pair("i", "j")
    .where((ni > 0) & (nj < 2))
    .emit(s.hop("i", "j"), matrix_element=-t * (ni * (nj + 1)).sqrt())
    .compile()
)

a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]])
reference = -t * (np.kron(a.T, a) + np.kron(a, a.T))
np.testing.assert_allclose(hopping.to_dense(), reference)
```

Both ordered pairs occur, giving the Hermitian operator $-t(a_0^\dagger a_1+a_1^\dagger a_0)$. The guards keep both target occupations within $0,1,2$. The local matrix check uses only nine basis states and verifies the coefficient as well as the direction.

For a larger geometry, replace the all-pairs iterator with the relevant directed coordinate pairs. An unrestricted pair iterator includes self pairs, while `for_each_distinct_pair` excludes them. A self hop has cancelling label shifts but does not automatically become the intended number operator, so do not use it to stand in for a separate diagonal term.

## Physical constraints beyond a conserved sum

A paired shift may preserve total charge while changing the Gauss constraint at a vertex. A gauge invariant loop move must account for edge orientations and the constraint convention. On an SU(2) spin network, a spin change can require corresponding intertwiner changes. These are physical relations to encode explicitly, not properties inferred from the name `hop`.

Use [admissibility checks](../spaces.md) on complete targets before using a new transport rule in a simulation. For more complicated updates, [writing and rearranging labels](writes.md) provides direct assignments and permutations.
