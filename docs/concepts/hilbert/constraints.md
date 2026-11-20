---
title: Constraints
---

# Constraints

A local domain says which values one label may take. A constraint says which combinations of labels belong to the space. Fixed total occupation, charge conservation, and vertex coupling rules all restrict combinations, so local domains alone cannot express them.

In neuraLQX, a discrete constraint is a predicate on configurations. For an input with shape `(..., hilbert.size)`, it returns a boolean array with shape `(...)`. Each boolean answers whether one complete configuration satisfies the condition. The Hilbert object combines that answer with its local domain checks.

This kind of constraint selects a basis. It is distinct from a quantum constraint operator whose kernel you try to find variationally. In particular, selecting gauge invariant basis states does not solve the Hamiltonian constraint of a gravity model.

## Apply a linear restriction

[`LinearConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.LinearConstraint) describes equalities between weighted sums of scalar sites. For a fixed total, its `sum` constructor supplies unit weights. Here four edge occupation labels are restricted to exactly two occupied edges:

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
two_occupied = nqx.hilbert.LinearConstraint.sum(
    sites=tuple(range(graph.n_edges)),
    target=2,
)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(0, 1, 2),
    constraint=two_occupied,
)
samples = hilbert.random_state(jax.random.key(8), size=6)
candidates = jnp.array([[1, 1, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]])

print(jnp.sum(samples, axis=-1).tolist())
print(hilbert.local_states_valid(candidates).tolist())
print(hilbert.constraint(candidates).tolist())
print(hilbert.is_valid(candidates).tolist())
```

```text
[2, 2, 2, 2, 2, 2]
[True, True, False]
[True, False, True]
[True, False, False]
```

The final candidate has the right total, but `2` is outside the local occupation domain $\{0,1\}$. Calling the constraint alone therefore accepts it, while `is_valid` correctly rejects it. Use `is_valid` for membership checks on supplied configurations. Use the separate tests when diagnosing why a configuration was rejected.

Generic spaces default to [`IdentityConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.IdentityConstraint), which adds no restriction beyond the local domains. If you already have a homogeneous or heterogeneous space, `hilbert.with_constraint(condition)` creates a new space with the supplied constraint. It replaces the previous constraint. To preserve a condition while adding another, combine them explicitly as described below.

## Weighted relations and charge conservation

The general linear form is

$$
\sum_{i\in S_r} w_{ri}\sigma_i = b_r
\qquad\text{for every row }r.
$$

`sites`, `weights`, and `targets` specify these rows. Site indices refer to the flat configuration, including component and vertex offsets. Obtain them through the [layout](state-layouts.md) when a space has more than one scalar label per edge.

For example, take three parallel edges, with the first directed from $a$ to $b$ and the other two in the opposite direction. The zero charge condition at either vertex reduces to $q_0-q_1-q_2=0$.

```python
import jax.numpy as jnp
import neuralqx as nqx

a, b = (0, 0, 0), (1, 0, 0)
graph = nqx.graph.Graph(
    [(a, b, 0), (b, a, 1), (b, a, 2)],
    is_planar=True,
)
gauss = nqx.hilbert.LinearConstraint(
    sites=((0, 1, 2),),
    weights=((1, -1, -1),),
    targets=(0,),
)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(-2, 1, 5),
    constraint=gauss,
)
candidates = jnp.array([[2, 1, 1], [-1, 2, 2]])

print(hilbert.is_valid(candidates).tolist())
```

```text
[True, False]
```

These are ordinary integer equalities. They do not wrap at the cutoff. This distinction matters when comparing a generic linear Gauss condition with the [cyclic constructive U(1) space](u1.md#what-the-finite-cutoff-means). The second candidate illustrates the difference: $-1$ and $2+2$ agree modulo five, but they are not equal as integers.

For integer charge constraints, leave `atol=0.0` so equality is exact. With floating point domains, a positive `atol` switches the comparison to `jax.numpy.isclose`, which also uses its default relative tolerance. Use a custom predicate if you need a different numerical definition of equality.

## Write a custom constraint

[`CallableDiscreteConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.CallableDiscreteConstraint) wraps a function, so you can express a new restriction without introducing a new Hilbert class. For example, this function selects configurations with an even total occupation:

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx


def even_occupation(states):
    return jnp.sum(states, axis=-1) % 2 == 0


graph = nqx.graph.Grid2D(2, 2)
condition = nqx.hilbert.CallableDiscreteConstraint(
    even_occupation,
    name="even total occupation",
)
hilbert = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(0, 1, 2),
    constraint=condition,
)
candidates = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
samples = hilbert.random_state(jax.random.key(9), size=(2, 3))

print(jax.jit(condition)(candidates).tolist())
print(condition(candidates[0]).shape)
print(condition(samples).shape)
print(bool(jnp.all(hilbert.is_valid(samples))))
```

```text
[True, False, True]
()
(2, 3)
True
```

The reduction uses `axis=-1`, so it operates on each configuration separately. Reducing every axis would collapse a batch into one answer. Indexing a particular site should similarly use `states[..., site]`, which preserves arbitrary leading batch dimensions.

Write the predicate using `jax.numpy` operations and return boolean values. Keep it deterministic and free of side effects. A Python `if` on a state value, conversion of a traced array to NumPy, or mutation of external state can prevent the condition from working inside compiled sampling. JAX's [compilation guide](https://docs.jax.dev/en/latest/jit-compilation.html) explains this execution model. The explicit `jax.jit` call above checks the predicate itself with the same kind of array operations it will use in sampling.

The function should remain well defined on candidate labels that fail the local checks, since `is_valid` evaluates both the local test and the predicate. Do not assume that a Python short circuit will protect an unsafe operation inside your condition. Keep the function and any captured configuration fixed after constructing the space. Create a new constraint when its meaning changes.

For a parameterised constraint with its own compatibility checks, you can subclass [`AbstractDiscreteConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AbstractDiscreteConstraint). It supports `validate_hilbert` and an optional exact `dimension` implementation in addition to `__call__`. The [developer constraint guide](../../developers/hilbert/constraints.md) is the place for those extensions and specialised generation rules.

## Combine conditions

Constraints support `&`, `|`, and `~` for conjunction, disjunction, and negation. They produce [`AndConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AndConstraint), [`OrConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.OrConstraint), and [`NotConstraint`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.NotConstraint) objects. Use these operators instead of Python's `and`, `or`, and `not`.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
base = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(0, 1, 2),
)
total_two = nqx.hilbert.LinearConstraint.sum((0, 1, 2, 3), target=2)
first_empty = nqx.hilbert.CallableDiscreteConstraint(
    lambda states: states[..., 0] == 0,
    name="first edge empty",
)
hilbert = base.with_constraint(total_two & first_empty)
candidates = jnp.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]])

print(hilbert.is_valid(candidates).tolist())
print(base.constrained, hilbert.constrained)
```

```text
[True, False, False]
False True
```

Negation reverses a constraint's answer. It does not reverse the local domains. The Hilbert object's `is_valid` still rejects values outside those domains after any logical combination.

## Sampling a constrained space

Generic constrained spaces generate random configurations by drawing from the product domain and rejecting candidates that fail the predicate. `max_trials` bounds that search. If the requested batch cannot be filled, the call raises an error rather than returning invalid configurations.

This is convenient for a modest restriction such as the examples above. It can become expensive when the allowed set occupies a tiny fraction of the product domain. Increasing `max_trials` may help with a small acceptance probability, but it cannot resolve an impossible condition or make a severely restricted space efficient. The supplied U(1) and SU(2) spaces use their own state generation methods for this reason.

A constraint also affects movement between valid states. Generic local proposals reject invalid updates by keeping the previous configuration. Validity alone does not imply that those moves explore the space. For example, a move changing exactly one binary occupation cannot connect two different configurations at fixed total occupation. A useful sampler needs moves that preserve the total while reaching the relevant allowed configurations. See [transition rules](../samplers/transition-rules.md) and [sampling diagnostics](../samplers/diagnostics.md).

Splitting random keys is equally important when generating several batches. Reusing the same JAX key repeats the same random operation. The [JAX random number guide](https://docs.jax.dev/en/latest/random-numbers.html) explains how to use `jax.random.split` to obtain fresh keys.

## Dimension and enumeration

A predicate can decide whether a configuration is allowed without knowing how many allowed configurations exist. Consequently, a generic space with a linear or callable constraint normally reports `dimension=None`. Its `n_states` property raises when that count is unavailable. Generic constrained basis numbering and `all_states` are also unavailable, even when you happen to know the dimension analytically.

For a deliberately small example, enumerate the unconstrained product and filter it yourself:

```python
import neuralqx as nqx

graph = nqx.graph.Grid2D(2, 2)
product_space = nqx.hilbert.HomogeneousHilbert(
    graph,
    nqx.hilbert.LocalRange(0, 1, 2),
)
hilbert = product_space.with_constraint(
    nqx.hilbert.LinearConstraint.sum((0, 1, 2, 3), target=2),
)
product_basis = product_space.all_states(max_states=16)
allowed_basis = product_basis[hilbert.is_valid(product_basis)]

print(hilbert.dimension, hilbert.is_indexable)
print(allowed_basis.shape)
```

```text
None False
(6, 4)
```

The six rows agree with $\binom{4}{2}=6$. This inspection does not register a new indexing scheme with `hilbert`, and enumerating the parent product scales exponentially. For realistic constrained models, use their specialised generation methods or a suitable sampler. The [U(1)](u1.md) and [SU(2)](su2.md) pages explain which counts and exact basis operations their implementations provide.
