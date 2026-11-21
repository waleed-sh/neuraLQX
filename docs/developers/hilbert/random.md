---
title: Random state generation
---

# Random state generation

Random state generation supplies valid configurations for initialisation and other basis-level operations. It does not sample the variational Born distribution by itself. The initialiser knows the Hilbert space, while the Metropolis sampler additionally evaluates the wavefunction and applies an acceptance rule.

For an unconstrained product space, uniform independent draws from each local domain give a uniform distribution over the complete product basis. For a constrained space, a generator must ensure that every returned configuration satisfies the constraint. Uniformity within the sector is useful and should be stated when established, but sampler initialisation and equilibrium sampling remain different operations.

## Keys, shapes, and dtypes

The method `hilbert.random_state(key, size=...)` accepts a JAX random key. With `size=None`, the result has shape `(hilbert.size,)`. An integer `B` gives `(B, hilbert.size)`, and a tuple supplies multiple leading axes. `dtype` chooses the stored label dtype, which must retain the meaning of the allowed values.

Split keys for independent draws. Reusing a key with the same generator and arguments reproduces the same output. Do not introduce a hidden NumPy random generator into a compiled initialiser.

```python
import jax
import neuralqx as nqx
import numpy as np

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.ExplicitLocalSpace((-2, 0, 3))
)
key_single, key_batch = jax.random.split(jax.random.key(12))
single = hilbert.random_state(key_single)
batch = hilbert.random_state(key_batch, size=(2, 4))
repeat = hilbert.random_state(key_batch, size=(2, 4))
np.testing.assert_array_equal(batch, repeat)
assert bool(hilbert.is_valid(batch).all())
print(single.shape)
print(batch.shape)
```

```text
(1,)
(2, 4, 1)
```

For the generic heterogeneous implementation, validate arrays with multiple leading axes through the flatten-and-restore pattern in [state representation](states.md#preserve-batch-axes-in-custom-methods).

## Generic constrained rejection

The supplied homogeneous and heterogeneous spaces have a fallback that draws from the unconstrained product and retains candidates satisfying the constraint. Already accepted entries are retained while the remaining entries receive further attempts. If the bounded procedure cannot fill the requested batch, it raises an error rather than returning invalid states. `max_trials` controls its retry budget.

If the accepted fraction of the product basis is $\alpha$, one independent valid draw requires $1/\alpha$ candidates on average. After $T$ independent candidates for one entry, its failure probability is $(1-\alpha)^T$. For a large batch, waiting for the last accepted entry can require more trials than a typical individual draw. A tiny accepted fraction is a reason to change the construction, not simply to increase the budget indefinitely.

Conditioning a uniform product draw on validity gives a uniform draw within the constrained basis. This is a useful reference for a small sector. It is not efficient for every constraint.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2),
    constraint=nqx.hilbert.LinearConstraint.sum(range(3), target=1),
)
samples = hilbert.random_state(jax.random.key(5), size=16, max_trials=128)
assert bool(jnp.all(hilbert.is_valid(samples)))
print(samples.shape)
print(jnp.unique(jnp.sum(samples, axis=-1)).tolist())
```

```text
(16, 3)
[1]
```

The fallback's internal rejection loop is compiled, but its public return path checks success on the host. Do not wrap that generic constrained call in an outer `jax.jit`. Sampler initialisation can itself occur inside compiled code, so a custom constrained space intended for that path should supply a compatible constructive generator, as below.

## Construct fixed-occupation configurations directly

For binary labels with $\sum_i x_i=K$, start from a vector with exactly $K$ ones and randomly permute its entries. Every binary configuration with that occupation has $K!(N-K)!$ permutations of the template, so uniform permutations induce a uniform distribution over the $\binom NK$ configurations.

The subclass below validates the required domain and linear constraint, then overrides only `random_state`. The remaining representation and conversion methods come from [`HomogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HomogeneousDiscreteHilbert).

```python
from math import prod
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

class FixedOccupationSpace(nqx.hilbert.HomogeneousDiscreteHilbert):
    def __post_init__(self):
        super().__post_init__()
        constraint = self.constraint
        if not isinstance(constraint, nqx.hilbert.LinearConstraint):
            raise TypeError("Supply a LinearConstraint.sum for the total occupation")
        if len(constraint.targets) != 1:
            raise ValueError("Supply one total-occupation equality")
        particles = constraint.targets[0]
        if type(particles) is not int or not 0 <= particles <= self.size:
            raise ValueError("The occupation must be an integer between 0 and size")
        expected = nqx.hilbert.LinearConstraint.sum(range(self.size), particles)
        if constraint != expected:
            raise ValueError("The constraint must sum every flat site with unit weight")
        for local in self.local_spaces:
            if not np.array_equal(np.asarray(local.all_values()), [0, 1]):
                raise ValueError("The local labels must be 0 and 1")

    def random_state(self, key, size=None, *, dtype=None, max_trials=1024):
        batch = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        count = prod(batch)
        keys = jax.random.split(key, count)
        template = jnp.arange(self.size) < self.constraint.targets[0]
        states = jax.vmap(lambda k: jax.random.permutation(k, template))(keys)
        out_dtype = self.dtype if dtype is None else dtype
        return states.astype(out_dtype).reshape((*batch, self.size))

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = FixedOccupationSpace(
    graph, nqx.hilbert.LocalRange(0, 1, 2),
    constraint=nqx.hilbert.LinearConstraint.sum(range(3), target=2),
)
draw = jax.jit(lambda key: hilbert.random_state(key, size=(2, 4)))
samples = draw(jax.random.key(8))
assert bool(jnp.all(hilbert.is_valid(samples)))
np.testing.assert_array_equal(samples, draw(jax.random.key(8)))
print(samples.shape)
print(jnp.unique(jnp.sum(samples, axis=-1)).tolist())
```

```text
(2, 4, 3)
[2]
```

`max_trials` is accepted for interface compatibility but is unnecessary for this constructive algorithm. The requested batch shape is fixed during compilation. Pass it as static configuration or close over it as the example does, rather than deriving an allocation size from a traced array value.

This initialiser also handles $K=0$ and $K=N$, where the sector contains one configuration. A useful test compares the empirical frequencies of all configurations in a small nontrivial sector with their expected uniform probabilities. Validity alone would not reveal a generator that returns only part of the sector.

## The method and the module-level dispatcher

[`neuralqx.hilbert.random_state`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.random_state) is the public dispatcher for the supplied space and constraint combinations. Its internal dispatch registry is private. It is not a public `random_state.dispatch` decorator for extensions.

For an application-defined subclass, override and call the **space method** as above. The base transition rule obtains initial states through `sampler.hilbert.random_state`, so this method is the relevant integration point. Calling the module-level dispatcher on an arbitrary subclass does not necessarily select its overridden method and can instead reach the generic fallback.

Initialisation is only one part of using the new space. A fixed-occupation state still needs a move that preserves occupation and explores the sector. The [proposal guide](proposals.md) supplies a complete example connected to a Metropolis sampler. For specialised gauge spaces, follow the supplied [U(1)](../../concepts/samplers/u1.md) and [SU(2) initialisation](../../concepts/samplers/su2/initialisation.md) rules rather than assuming their initial states are independent uniform draws from the full physical basis.
