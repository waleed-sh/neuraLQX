---
title: Constraints
---

# Constraints

A constraint selects combinations of locally allowed labels. It can express a fixed occupation, a charge relation, a parity sector, or another condition on the complete configuration. It is distinct from the local domains, which determine which values can appear at each scalar site.

For a predicate $C$, the constrained basis is

$$
\mathcal X_C=\left\{x\in\prod_iD_i:C(x)\right\}.
$$

The Hilbert space combines local membership with this predicate in `is_valid`. A constraint must therefore have a clear interpretation for the stored [state representation](states.md). It does not, on its own, supply an efficient way to draw or move between valid states.

## Start with a supplied predicate wrapper

[`CallableDiscreteConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.CallableDiscreteConstraint) adapts a JAX-compatible callable. The callable accepts `(..., hilbert.size)` and returns a boolean array with shape `(...)`. Reduce over the configuration axis, not over the whole array.

[`LinearConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.LinearConstraint) describes sparse equalities

$$
\sum_{i\in I_r}w_{ri}x_i=t_r.
$$

Its `sum` constructor covers a single unit-weight equality. The general constructor takes tuples of sites, weights, and targets. This uses the stored values, so a constraint on local basis indices needs an explicit conversion instead.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
base = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)

def even_occupation(states):
    return jnp.sum(states, axis=-1) % 2 == 0

parity = nqx.hilbert.CallableDiscreteConstraint(
    even_occupation, name="even occupation", hash_key=("even occupation", 1)
)
exactly_two = nqx.hilbert.LinearConstraint.sum(range(base.size), target=2)
sector = base.with_constraint(parity & exactly_two)
basis = base.all_states()
mask = jax.jit(sector.is_valid)(basis)
selected = basis[mask]
np.testing.assert_array_equal(jnp.sum(selected, axis=-1), jnp.full((3,), 2))
print(selected.shape)
print(sector.dimension)
```

```text
(3, 3)
None
```

The predicate combination is valid, but the combined generic constraint does not infer a closed-form dimension. `None` means that the exact count is not supplied cheaply. It does not mean that the space is infinite or empty. This example enumerates the small unconstrained parent only to verify the selected configurations.

Constraints support conjunction with `&`, disjunction with `|`, and negation with `~`. Local-domain validity still applies to the result. Negating a constraint does not make labels outside the local domains admissible.

## Define a reusable constraint class

Subclass [`AbstractDiscreteConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AbstractDiscreteConstraint) when the condition deserves a named, validated object. Implement `__call__` for the predicate, `validate_hilbert` for compatibility checks, and a structural hash for use as static metadata. Implement `dimension` only when you can return the exact number of accepted configurations.

For binary occupations and fixed particle number $K$,

$$
C_K(x)\iff\sum_i x_i=K,
\qquad
\dim\mathcal H_K=\binom NK.
$$

The dimension formula depends on the binary domain. It would be wrong to reuse it unchanged for higher occupations or heterogeneous local ranges. The constructor validation below establishes that assumption before the predicate is used.

```python
from math import comb
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import field

class FixedOccupation(nqx.hilbert.AbstractDiscreteConstraint):
    particles: int = field(static=True)

    def __post_init__(self):
        if type(self.particles) is not int or self.particles < 0:
            raise ValueError("particles must be a nonnegative Python integer")

    def __hash__(self):
        return hash((type(self), self.particles))

    def validate_hilbert(self, hilbert):
        if self.particles > hilbert.size:
            raise ValueError("particles cannot exceed the number of binary sites")
        for local in hilbert.local_spaces:
            if not np.array_equal(np.asarray(local.all_values()), [0, 1]):
                raise ValueError("FixedOccupation requires binary labels 0 and 1")

    def __call__(self, states):
        return jnp.sum(states, axis=-1) == self.particles

    def dimension(self, hilbert):
        return comb(hilbert.size, self.particles)

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
base = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
constraint = FixedOccupation(1)
hilbert = base.with_constraint(constraint)
batch = base.all_states().reshape(2, 4, base.size)
mask = jax.jit(constraint)(batch)
print(mask.shape)
print(int(jnp.sum(mask)), hilbert.dimension)
print(hilbert.is_indexable)
```

```text
(2, 4)
3 3
False
```

The exact dimension does not enable constrained indexing automatically. The [indexing guide](indexing.md) explains which additional operations would be needed. The [random generation guide](random.md) provides a constructive way to draw fixed-occupation configurations without enumerating the parent space.

## Static data, equality, and numerical tolerances

Keep parameters such as an occupation count, a tuple of sites, or fixed coefficients in immutable static fields. The inherited struct equality compares the declared fields, and a custom hash must agree with that equality. Do not base a hash on mutable data or changing array contents.

For a callable wrapper, the default hash uses the callable identity. Supply a `hash_key` only when it is a stable description of the callable's behaviour. Two closures with different physical parameters must not share a key that claims they are equivalent. The [struct guide](../structs/fields.md) explains why static metadata participates in compilation reuse.

Prefer exact integer relations for quantised labels when that represents the physics. `LinearConstraint` uses exact comparison with `atol=0`. A nonzero `atol` selects its approximate comparison path. For a specifically prescribed tolerance rule, write the predicate explicitly, including the intended relative tolerance. Dtype conversion can change a boundary decision, so test values on both sides of the tolerance.

The predicate must be valid JAX code. Avoid converting a traced sum into a Python boolean, or returning a variable-length list of accepted configurations. Return a fixed-shape mask and let the caller decide how to use it. Although the conceptual validity test combines local and global conditions, both array expressions can be evaluated, so the predicate should not rely on Python-style short-circuiting to avoid invalid arithmetic.

## A valid sector still needs a sampling strategy

Generic rejection can produce a valid initial batch when the accepted fraction is reasonable. It becomes expensive for a small sector inside a large product space. A local proposal can be even more restrictive. Flipping a single binary occupation always leaves a fixed-particle sector, so rejecting invalid moves would freeze every chain.

Use [constructive random generation](random.md) for such initial states and [constraint-preserving proposals](proposals.md) for motion within the sector. Check reachability as well as validity. For gauge constraints, the relevant moves can involve a loop or a set of dependent labels, rather than one coordinate. The supplied [U(1) sampling](../../concepts/samplers/u1.md) and [SU(2) sampling](../../concepts/samplers/su2/index.md) guides explain their physical constructions.
