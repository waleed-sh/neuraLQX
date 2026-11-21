---
title: Composition
---

# Composition

A custom [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) participates in the same algebra as the supplied computational operators. Addition, scalar multiplication, products, and adjoints build objects that retain the constituent actions. They do not allocate the global operator matrix.

Composition is often the most direct way to express a physical model. It is also a useful reference implementation when you later write a fused kernel. The [scientist guide to operator algebra](../../concepts/operators/algebra.md) introduces the public notation. Here the focus is on what a custom implementation must preserve.

## Sums, products, and adjoints

`A + B` adds matrix row contributions. Equal targets can appear in both terms and are combined by summing their elements. Scalar multiplication `c * A` scales the elements without changing the targets. To add a constant energy shift, use an explicit [`IdentityOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.IdentityOperator), since a bare scalar is not an operator on a specified Hilbert space.

`A @ B` represents the matrix product

$$
(AB)_{xz}=\sum_y A_{xy}B_{yz}.
$$

In row connectivity, this means following a row of $A$ from $x$ to $y$, then a row of $B$ from $y$ to $z$. This is consistent with $B$ acting first on a ket. Mixing those two viewpoints is a common source of an accidental reversed product.

The adjoint reverses factor order and conjugates coefficients,

$$
(cAB)^\dagger=c^*B^\dagger A^\dagger.
$$

Every factor needed by this construction must provide a correct `adjoint`. The composed operators check Hilbert space compatibility. Sharing an array width is not enough when local domains or constraints differ.

Hermiticity metadata is conservative. A general product of Hermitian factors is not necessarily Hermitian, since $AB=BA$ is an additional requirement. The product wrapper does not attempt to prove commutation or discover that an arbitrary multi-factor expression is Hermitian. A fused class whose definition establishes that property can expose it explicitly, after verification.

## An end-to-end custom diagonal operator

For real coefficients $w_i$, define $N_w(x)=\sum_iw_ix_i$. Its connectivity has one target, the source, with element $N_w(x)$. This example defines the class, composes it with an identity, and compares the positive squared estimator with an explicit product and an exact gradient.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt.wcl import IdentityOperator

class WeightedOccupation(nqx.operator.ComputationalOperator):
    weights: jax.Array

    def __post_init__(self):
        super().__post_init__()
        if self.weights.shape != (self.hilbert.size,):
            raise ValueError("weights must contain one coefficient per flat site")
        if not np.issubdtype(self.weights.dtype, np.floating):
            raise TypeError("Use real floating-point weights")

    @property
    def dtype(self):
        return self.weights.dtype

    @property
    def is_hermitian(self):
        return True

    @property
    def is_diagonal(self):
        return True

    @property
    def max_conn_size(self):
        return 1

    @property
    def adjoint(self):
        return self

    def _get_conn_padded_kernel(self, configuration):
        element = jnp.sum(self.weights * configuration)
        return configuration[None, :], element[None]

def log_value(parameters, configurations):
    return parameters["theta"] * jnp.sum(configurations, axis=-1)

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 3)
)
number = WeightedOccupation(hilbert, jnp.array([1.0]))
shifted = 2.0 * number + 0.25 * IdentityOperator(hilbert)
np.testing.assert_allclose(shifted.to_dense(), np.diag([0.25, 2.25, 4.25]))

state = nqx.vqs.VariationalState(
    hilbert=hilbert, apply_fn=log_value,
    parameters={"theta": jnp.array(0.0)}, seed=0,
)
basis = hilbert.all_states()
positive = number.squared()
explicit_product = number.adjoint @ number
stats, gradient = nqx.vqs.expect_and_grad(state, positive, samples=basis)
reference_stats = nqx.vqs.expect(state, explicit_product, samples=basis)

def exact_square(theta):
    probabilities = jax.nn.softmax(2 * theta * basis[:, 0])
    return jnp.sum(probabilities * basis[:, 0] ** 2)

np.testing.assert_allclose(stats.mean, reference_stats.mean)
np.testing.assert_allclose(gradient["theta"], jax.grad(exact_square)(0.0))
print(f"{float(stats.mean):.6f}")
print(f"{float(gradient['theta']):.6f}")
```

```text
1.666667
2.666667
```

The complete basis is an exact equally weighted sample only because the model is uniform at `theta=0`. This gives a deterministic check of the mean and derivative without making a claim about Monte Carlo error bars. In a simulation, use the variational state's sampled configurations and [sampling diagnostics](../../concepts/samplers/diagnostics.md).

## An operator square and a squared estimator

`A @ A` and `A ** 2` represent $A^2$. `A.adjoint @ A` represents $A^\dagger A$ with explicit product connectivity. `A.squared()` returns a [`Squared`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared) estimator marker for the positive quadratic form. These agree only in the appropriate mathematical cases, and they need not use the same local estimator.

With full sampled support, the marker uses

$$
\langle A^\dagger A\rangle
=\mathbb E_{p_\theta}\left[|A_{\mathrm{loc},\theta}(x)|^2\right].
$$

It can avoid the product connection expansion. It does not provide its own `get_conn_padded` or `to_dense` interface. Use the explicit product when you need those rows or a small reference matrix. The [squared estimator guide](../../concepts/operators/squared-estimators.md) explains the mathematics, and the [gradient guide](../estimators/gradients.md) explains why its derivative needs special treatment. Wavefunction nodes and restricted supports require the additional handling described in [restrictions and residuals](../../concepts/operators/restricted-spaces.md).

## Connection growth and fused implementations

If $A$ and $B$ reserve $K_A$ and $K_B$ slots per row, a straightforward product reserves up to $K_AK_B$ paths. Many paths may lead to the same target, but the padded capacity can still grow rapidly with the number of factors. Streaming source batches reduces batch memory without removing this per-row expansion.

When a repeated expression has a simpler direct action, implement that action as a new computational operator and compare it with the composed expression on a tiny basis. A fused implementation must preserve interference between paths, including signs and complex phases. It must also preserve the chosen cutoff and subspace projection order.

If only the expectation admits a cheaper calculation, consider a [specialised local estimator](../estimators/index.md). Keep the operator connectivity as a consistent reference, since some composed and gradient paths use it directly. For a complete simulation, the finished operator can be passed to the [VMC driver](../../concepts/drivers/index.md) or exposed through an [LQX model factory](../../concepts/lqx/custom-models.md).
