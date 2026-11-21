---
title: Operator contracts
---

# Operator contracts

The usual base for a custom matrix free operator is [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator). It already owns the Hilbert space, batched connectivity interface, streaming controls, and operator algebra. Your implementation supplies the metadata and a JAX function describing one matrix row.

This page builds a local matrix operator. It stores a small matrix for one scalar degree of freedom, while the global Hilbert space action remains matrix free. The same design works when local labels are charges, occupations, or another finite set of values.

## Metadata is part of the contract

`dtype` describes the matrix elements, which can have a different dtype from the configuration labels. Integer configurations can carry complex amplitudes. `max_conn_size` is a fixed upper bound on the number of slots returned for each input. It must follow from the operator structure, not from the values of a traced configuration.

`is_hermitian` states whether Hermiticity is known. Set it to `True` only when the construction establishes $A=A^\dagger$, since expectation gradients can use that assertion to select a covariance formula. Returning `False` is conservative when a general construction does not establish the property, even if one particular numerical instance happens to be Hermitian.

Implement `adjoint` to return an operator representing the conjugate transpose. Returning `self` is appropriate for an operator known to be Hermitian. The optional `is_diagonal` property defaults to `False`, and should be overridden when diagonality is guaranteed by construction.

## Implement one row with fixed output shapes

The required `_get_conn_padded_kernel(configuration)` takes shape `(hilbert.size,)`. It returns target configurations of shape `(max_conn_size, hilbert.size)` and matrix elements of shape `(max_conn_size,)`.

For a local matrix $M$ acting on flat site $s$, write $i_s(x)$ for the local basis index of the stored label $x_s$. The action is

$$
A_{xy}=M_{i_s(x),i_s(y)}\prod_{r\ne s}\delta_{i_r(x),i_r(y)}.
$$

The matrix is indexed by local basis indices, not by the physical values stored in the configuration. For labels $\{-1,0,1\}$, the indices are $\{0,1,2\}$. The Hilbert space's conversion methods establish that correspondence.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.utils.struct import field

class SiteMatrix(nqx.operator.ComputationalOperator):
    matrix: jax.Array
    site: int = field(static=True, default=0)

    def __post_init__(self):
        super().__post_init__()
        if type(self.site) is not int or not 0 <= self.site < self.hilbert.size:
            raise ValueError("site must be a valid flat site index")
        dimension = self.hilbert.local_sizes[self.site]
        if self.matrix.shape != (dimension, dimension):
            raise ValueError("matrix shape must match the selected local dimension")

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def max_conn_size(self):
        return self.hilbert.local_sizes[self.site]

    @property
    def is_hermitian(self):
        return False

    @property
    def adjoint(self):
        return self.replace(matrix=jnp.conj(self.matrix.T))

    def _get_conn_padded_kernel(self, configuration):
        indices = self.hilbert.states_to_local_indices(configuration)
        target_indices = jnp.broadcast_to(
            indices, (self.max_conn_size, self.hilbert.size)
        )
        target_indices = target_indices.at[:, self.site].set(
            jnp.arange(self.max_conn_size, dtype=jnp.int32)
        )
        targets = self.hilbert.local_indices_to_states(
            target_indices, dtype=configuration.dtype
        )
        elements = self.matrix[indices[self.site], :]
        allowed = self.hilbert.is_valid(targets)
        elements = jnp.where(allowed, elements, 0)
        targets = jnp.where(allowed[:, None], targets, configuration[None, :])
        return targets, elements

a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
graph = nqx.graph.Graph([(a, b), (b, c)], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.ExplicitLocalSpace((-1, 1))
)
matrix = jnp.array([[0, -1j], [1j, 0]])
operator = SiteMatrix(hilbert, matrix, site=0)
basis = hilbert.all_states()

np.testing.assert_allclose(operator.to_dense(), np.kron(matrix, np.eye(2)))
np.testing.assert_allclose(operator.adjoint.to_dense(), operator.to_dense().conj().T)

evaluate = jax.jit(lambda op, x: op.get_conn_padded(x))
targets, elements = evaluate(operator, basis.reshape(2, 2, hilbert.size))
print(targets.shape)
print(elements.shape)
```

```text
(2, 2, 2, 2)
(2, 2, 2)
```

The numerical matrix is a dynamic field. The selected site is static because it determines indexing and connection capacity. The class deliberately makes no general Hermiticity promise, even though the displayed matrix is Hermitian. A more specialised class could establish that property by its mathematical construction.

The example checks a global matrix of dimension four. That allocation is a verification step. Applying the operator to sampled configurations uses the small local matrix and generated targets directly.

## The meaning of constraint masking

The kernel tests each target with `hilbert.is_valid`. On a constrained space, this defines the restricted action $PAP$, where $P$ projects onto valid configurations. Replacing a rejected target with its source keeps padding safe, while its zero matrix element removes the contribution.

This is an explicit design choice in this example. The base class does not infer whether your physical operator should project, reconstruct dependent coordinates, or preserve the constraint through a different action. In particular,

$$
PA^\dagger P A P
\quad\text{and}\quad
PA^\dagger A P
$$

can differ. The first expression removes excursions outside the subspace before the second action. If those excursions belong in a constraint norm, use the [embedding and residual machinery](../../concepts/operators/restricted-spaces.md).

## Construction and compilation

Build static graph selections and local dimensions in construction code. Inside the row kernel, use JAX array operations and shape-stable control flow. Python conditionals are suitable for fixed modes known during tracing, but not for questions such as whether a sampled charge is positive. Use array masks or JAX control flow for the latter.

The base implementation batches the row kernel and restores all leading axes. Override `_get_conn_padded_batch_kernel(states_2d)` only when you have a measured reason to provide a direct batched implementation. Its output still needs the same row semantics and fixed connection capacity. The [connectivity guide](connectivity.md) describes the resulting public interfaces.

When adding fields, follow the [struct field rules](../structs/fields.md). Do not put changing coefficients into static or opaque metadata to avoid passing them as numerical inputs. The operator does not automatically make its coefficients trainable VQS parameters. Differentiating the wavefunction parameters and differentiating a physical coupling are separate calculations.
