---
title: Connectivity
---

# Connectivity

Connectivity is the representation of the nonzero entries in an operator row. For input $x$, a pair $(y_k,m_k)$ means a contribution $m_k$ to $\langle x|A|y_k\rangle$. Its meaning does not change when batching, streaming, or operator composition changes how the row is evaluated.

The [operator contract](contracts.md) introduces the single-configuration kernel. This page develops the shape and padding rules, then uses a truncated creation operator to check the row convention explicitly.

## Shapes, padding, and repeated targets

For input shape `(..., N)` and connection capacity $K$, [`get_conn_padded`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator.get_conn_padded) returns shapes `(..., K, N)` and `(..., K)`. Every row reserves $K$ slots even when fewer connections are active. Unused slots carry zero matrix element and should contain a valid configuration, usually the source.

Repeated targets are allowed. If two contributions reach the same $y$, their elements add,

$$
A_{xy}=\sum_{k:y_k=y}m_k.
$$

[`get_conn_coalesced`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.get_conn_coalesced) merges equal targets and handles exact cancellation while retaining fixed output capacity. [`n_conn`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.n_conn) counts distinct nonzero connections after this coalescing. A raw count of nonzero padded entries can therefore be larger.

For inspecting one configuration outside compiled code, [`get_conn`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.get_conn) returns its active connections. [`get_conn_flattened`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.get_conn_flattened) provides a flattened representation and can fill cumulative row sections. Filtering to a data-dependent number of nonzero entries is an inspection or host-side operation. Use the padded interface inside JAX transformations.

## A creation operator at a finite cutoff

Take occupation labels $n=0,\ldots,n_{\max}$. The truncated ket action is

$$
a^\dagger|n\rangle=
\begin{cases}
\sqrt{n+1}|n+1\rangle & n<n_{\max},\\
0 & n=n_{\max}.
\end{cases}
$$

Consequently its matrix row is

$$
\langle n|a^\dagger|m\rangle
=\sqrt n\,\delta_{m,n-1}.
$$

The row kernel lowers the target label for the creation operator. Its adjoint raises the target label with element $\sqrt{n+1}$, subject to the upper bound. The following example includes both actions through a static mode and compares the generated matrices with an independent construction.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import field

class TruncatedLadder(nqx.operator.ComputationalOperator):
    creation: bool = field(static=True, default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.hilbert.size != 1 or self.hilbert.constrained:
            raise ValueError("This example requires one unconstrained occupation")
        values = np.asarray(self.hilbert.local_spaces[0].all_values())
        if not np.array_equal(values, np.arange(self.hilbert.local_sizes[0])):
            raise ValueError("Occupation labels must be 0, 1, ..., n_max")

    @property
    def dtype(self):
        return jnp.dtype("float64")

    @property
    def is_hermitian(self):
        return False

    @property
    def max_conn_size(self):
        return 1

    @property
    def adjoint(self):
        return self.replace(creation=not self.creation)

    def _get_conn_padded_kernel(self, configuration):
        n = configuration[0]
        if self.creation:
            candidate = n - 1
            allowed = n > 0
            element = jnp.sqrt(n.astype(self.dtype))
        else:
            candidate = n + 1
            allowed = candidate < self.hilbert.local_sizes[0]
            element = jnp.sqrt((n + 1).astype(self.dtype))
        target = configuration.at[0].set(jnp.where(allowed, candidate, n))
        return target[None, :], jnp.where(allowed, element, 0)[None]

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 4)
)
creation = TruncatedLadder(hilbert)
basis = hilbert.all_states()
reference = np.diag(np.sqrt(np.arange(1, 4)), k=-1)
np.testing.assert_allclose(creation.to_dense(), reference)
np.testing.assert_allclose(creation.adjoint.to_dense(), reference.T)

direct = jax.jit(lambda x: creation.get_conn_padded(x, use_streaming=False))(basis)
streamed = creation.get_conn_padded(basis, use_streaming=True, chunk_size=2)
for actual, expected in zip(streamed, direct):
    np.testing.assert_allclose(actual, expected)
assert bool(jnp.all(hilbert.is_valid(direct[0].reshape(-1, hilbert.size))))
print(creation.n_conn(basis).tolist())
print(np.round(np.asarray(direct[1][:, 0]), 6).tolist())
```

```text
[0, 1, 1, 1]
[0.0, 1.0, 1.414214, 1.732051]
```

The empty row at $n=0$ contains the source as padding with zero element. The example has no cyclic boundary. At the upper cutoff the truncated commutator differs from its infinite-dimensional counterpart, so a test of $[a,a^\dagger]=I$ would be inappropriate on the full truncated basis. Boundary behaviour belongs in the physical definition and in the tests.

## Streaming changes the batch, not the operator

`use_streaming=True` evaluates flattened source configurations in chunks. `chunk_size` here controls connectivity generation. It is distinct from the variational state's model-evaluation chunk size, which limits network batches after targets have been generated. The older `use_sharding` keyword is a compatibility alias for this operator streaming switch, and must not be supplied alongside `use_streaming`.

Streaming should preserve targets and matrix elements. It does not reduce the mathematical connection capacity of a row. A product with many possible paths can still have a large $K$, which is why the [composition guide](composition.md) discusses when to use a fused kernel or a squared estimator.

## Verify the representation before measuring it

[`to_sparse`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.to_sparse) and [`to_dense`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.to_dense) are useful references on small indexable spaces. They enumerate the basis, so they are not the normal execution path for a large simulation. The default enumeration guard limits the number of states, and a dense matrix still requires storage quadratic in that number.

Check the zero rows, cutoff rows, repeated-target cancellations, and the adjoint separately. For a complex operator, compare with a conjugate transpose rather than an ordinary transpose. On a constrained space, also verify that every active target satisfies the constraint and that projection implements the intended physics. These checks catch errors that a plausible-looking expectation value can hide.
