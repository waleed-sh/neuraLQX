---
title: Modular shifts
---

# Modular shifts

[`shift_mod(i, delta)`](../../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.shift_mod) advances cyclically through the local basis at coordinate $i$. Its displacement is an integer number of basis positions, rather than an arbitrary numerical increment of the stored label.

If the declared local values are $v_0,\ldots,v_{d_i-1}$ and the current value is $v_r$, then

$$
T_i(v_r)=v_{(r+\Delta)\bmod d_i}.
$$

For a unit-spaced integer range starting at $m$, this reduces to

$$
T_i(q)=m+\bigl(q+\Delta-m\bmod d_i\bigr).
$$

The basis order therefore matters. Check `hilbert.local_spaces[i].all_values()` when constructing a cyclic action on an explicit or descending basis.

## A cyclic U(1) charge

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
cycle = (
    s.SymbolicOperatorBuilder(hilbert, "charge_cycle")
    .for_each_site("e")
    .emit(s.shift_mod("e", 1), matrix_element=1)
    .compile()
)
sources = jnp.array([[-1], [0], [1]], dtype=hilbert.dtype)
targets, coefficients = cycle.get_conn_padded(sources)
np.testing.assert_array_equal(targets[:, 0, 0], [0, 1, -1])
np.testing.assert_allclose(coefficients, 1)
```

Here the largest charge wraps to the smallest. This is a cyclic operator on a finite charge set. A truncated ladder operator with a vanishing action at the boundary has different physics and should use a [guarded ordinary shift](shifts.md).

## Irregular and heterogeneous bases

The integrated neuraLQX compiler uses the local domain of each site. The basis can have non-unit spacing or explicit irregular values, and different sites can have different domains.

```python
mixed_graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
mixed = nqx.hilbert.HeterogeneousDiscreteHilbert(
    mixed_graph,
    nqx.hilbert.HeterogeneousLocalSpace(edge_spaces={
        0: nqx.hilbert.LocalRange(-1, 2, 2),
        1: nqx.hilbert.ExplicitLocalSpace((-4, 0, 7)),
    }),
)
both_cycles = (
    s.SymbolicOperatorBuilder(mixed, "different_local_cycles")
    .globally()
    .emit(s.shift_mod(0, 1).shift_mod(1, 1), matrix_element=1)
    .compile()
)
source = jnp.array([1, 7], dtype=mixed.dtype)
targets, coefficients = both_cycles.get_conn_padded(source)
np.testing.assert_array_equal(targets, [[-1, -4]])
np.testing.assert_allclose(coefficients, [1])
assert bool(jnp.all(mixed.is_valid(targets)))
```

The first basis is $(-1,1)$ and the second is $(-4,0,7)$. Advancing one position from the last value wraps each coordinate to its own first value. Ordinary `shift(..., 1)` would instead produce the illegal values two and eight.

[`HeterogeneousLocalSpace`](../../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousLocalSpace) associates domains with edge coordinates, and [`ExplicitLocalSpace`](../../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.ExplicitLocalSpace) defines the irregular basis. See [local spaces](../../../hilbert/local-spaces.md) for vector and heterogeneous construction details.

## What wrapping guarantees

Given a valid current local label and an integer displacement, the result remains in that site's local basis. Wrapping does not enforce relations between sites. Independent cyclic shifts can violate a fixed total charge or a gauge constraint, and cycling an SU(2) edge spin does not determine admissible intertwiner channels.

Use the [Hilbert space validity check](../spaces.md) on the complete target. In an update chain, the modular shift reads the current working target. If an earlier operation has already produced a value outside the local basis, do not rely on a later modular shift to give that invalid intermediate value a physical interpretation.

Per-site modular domains currently use the reference JAX lowering when the structured kernel planner cannot represent them. This preserves the declared basis semantics. It is not a requirement to change the operator definition or enumerate the global Hilbert space. The [kernel planning guide](../../../../developers/symbolic/kernels.md) explains how to inspect this fallback.

The expression helper `AmplitudeExpr.wrap_mod()` has no site argument and only supports a common unit-spaced integer range. It should not be used as a substitute for the per-site update shown here.
