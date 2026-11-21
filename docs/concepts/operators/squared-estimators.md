---
title: Squared estimators
---

# Squared estimators

Many variational constraint calculations minimise the norm of a linear operator acting on a state. If the condition is $A|\psi\rangle=0$, the associated positive objective is

$$
\mathcal L(\theta)
=\frac{\lVert A\psi_\theta\rVert^2}{\lVert\psi_\theta\rVert^2}
=\langle A^\dagger A\rangle_\theta.
$$

You could construct the product $A^\dagger A$ and evaluate its ordinary local estimator. The squared estimator trick gives another route: evaluate the local action of $A$, take its absolute square, and average. This can avoid the much larger connectivity of the product while preserving the same objective.

In neuraLQX, [`Squared(A)`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared), or equivalently `A.squared()`, records this factorisation for the expectation engine. The derivation below uses Born sampling and initially assumes that the wavefunction is nonzero wherever $A\psi$ is nonzero. [Wavefunction zeros](#wavefunction-zeros-and-support) are addressed explicitly later.

## Deriving the estimator

The [local estimator guide](application.md#from-the-expectation-value-to-the-local-estimator) defines

$$
p_\theta(\sigma)=\frac{|\psi_\theta(\sigma)|^2}{Z_\theta},
\qquad
A_{\mathrm{loc}}(\sigma)
=\frac{(A\psi_\theta)(\sigma)}{\psi_\theta(\sigma)}.
$$

Insert a resolution of the identity between $A^\dagger$ and $A$:

$$
\begin{aligned}
\langle A^\dagger A\rangle_\theta
&=\frac{1}{Z_\theta}
\sum_\sigma
\langle\psi_\theta|A^\dagger|\sigma\rangle
\langle\sigma|A|\psi_\theta\rangle\\
&=\frac{1}{Z_\theta}
\sum_\sigma |(A\psi_\theta)(\sigma)|^2\\
&=\sum_\sigma p_\theta(\sigma)
\left|\frac{(A\psi_\theta)(\sigma)}{\psi_\theta(\sigma)}\right|^2.
\end{aligned}
$$

Consequently, the factorised local estimator is

$$
Q_{\mathrm{sq}}(\sigma)
=|A_{\mathrm{loc}}(\sigma)|^2
=\left|
\sum_{\sigma'}A_{\sigma,\sigma'}
\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}
\right|^2,
$$

and its Monte Carlo estimate is

$$
\mathcal L(\theta)
\approx\frac1M\sum_{r=1}^{M}
|A_{\mathrm{loc}}(\sigma_r)|^2.
$$

The complex conjugation is essential. `Squared(A)` uses an absolute square, not $A_{\mathrm{loc}}^2$. Every sampled contribution is real and nonnegative. Averaging these contributions estimates an action norm, rather than the absolute square of the mean $|\langle A\rangle|^2$.

### Keep the interference

The sum over connected configurations is completed before taking the absolute square. If two branches give amplitudes $a_1$ and $a_2$, the contribution is

$$
|a_1+a_2|^2
=|a_1|^2+|a_2|^2+2\operatorname{Re}(a_1^*a_2).
$$

Squaring the branches individually would discard the interference term and change the objective. This also applies when several padded branches reach the same target. Their amplitudes add, including any cancellation, before the squared local value is formed.

## What computation is avoided?

If one row of $A$ has up to $K$ connections, the factorised estimator needs its row elements and wavefunction ratios at those connected configurations. A generic explicit product $A^\dagger A$ follows two sets of connections and can have up to $K^2$ paths before duplicate targets are combined. Its actual cost depends on both factors' capacities and any specialised structure.

`Squared` keeps the one-action representation. It provides no `get_conn_padded`, `to_sparse`, or `to_dense` method of its own. When an explicit quadratic matrix or its connectivity is needed, construct `A.adjoint @ A`. When the desired quantity is its expectation or gradient, pass `A.squared()` to the variational state.

This saves product construction, but it is not a general guarantee of a smaller statistical error. The two estimators have different fluctuations, as the next example demonstrates.

## Same expectation, different local values

The ordinary local estimator of the product is

$$
Q_{\mathrm{matrix}}(\sigma)
=\frac{(A^\dagger A\psi_\theta)(\sigma)}{\psi_\theta(\sigma)}.
$$

It generally differs point by point from $|A_{\mathrm{loc}}(\sigma)|^2$. Their Born averages agree under the support condition above. Their sample variances and finite-batch means need not agree.

Here $A=\hat qR$, with the charge operator on the left of the cyclic shift. We use the small wavefunction $(1,2,1)/\sqrt6$ and evaluate both estimators exactly.

```python
import neuralqx as nqx
import jax.numpy as jnp
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=0)
shift = U1HolonomyOperator(hilbert, edge=0)
action = charge @ shift
psi = jnp.array([1.0, 2.0, 1.0]) / jnp.sqrt(6.0)
probabilities = jnp.abs(psi)**2

connected, elements = action.get_conn_padded(hilbert.all_states())
columns = hilbert.states_to_numbers(connected)
action_local = jnp.sum(elements * psi[columns] / psi[:, None], axis=-1)
squared_local = jnp.abs(action_local)**2
product_local = ((action.adjoint @ action).to_dense() @ psi) / psi

squared_mean = jnp.sum(probabilities * squared_local)
product_mean = jnp.sum(probabilities * product_local)
np.testing.assert_allclose(squared_mean, product_mean)
print(squared_local.tolist())
print(product_local.tolist())
print(round(float(squared_mean), 6))
```

```text
[4.0, 0.0, 1.0]
[1.0, 1.0, 0.0]
0.833333
```

The factorised values fluctuate more in this particular example. Another operator and state can behave differently. `Stats.variance` returned for `Squared(A)` describes the chosen local estimator's variance. It should not be interpreted as the quantum variance $\langle(A^\dagger A)^2\rangle-\langle A^\dagger A\rangle^2$.

## Choosing the ordering

The action recorded inside `Squared` determines which positive operator is evaluated:

| Construction | Objective | Residual |
| --- | --- | --- |
| `A.squared()` | $A^\dagger A$ | $A\psi$ |
| `A.adjoint.squared()` | $AA^\dagger$ | $A^\dagger\psi$ |
| `A @ A` or `A ** 2` | $A^2$ | An ordinary matrix product |

The first two have nonnegative expectations, but their kernels can differ when $A$ is not Hermitian. The third is not generally positive. Select the construction from the equation you intend to solve.

For [LQX models](../lqx/structure.md#quadratic-constraints), the default local quadratic ordering is $H_vH_v^\dagger$. It is represented by `Squared(H_v.adjoint)`. The alternative returned by `get_quadratic_constraint(alternative=True)` uses $H_v^\dagger H_v$ and measures the residual of $H_v\psi=0$.

## Several local constraints

To impose separate conditions $A_v\psi=0$, use [`SumOfSquares`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.SumOfSquares):

$$
Q=\sum_v A_v^\dagger A_v,
\qquad
Q_{\mathrm{sq}}(\sigma)=\sum_v|(A_v)_{\mathrm{loc}}(\sigma)|^2.
$$

Each local action is summed internally before its absolute square is taken. Distinct vertex residuals are then added without cross terms. By contrast, squaring $\sum_v A_v$ allows interference and cancellation between different vertices. The [algebra example](algebra.md#keep-separate-constraints-separate) shows the difference explicitly.

If a single local constraint is $A_v=H_{E,v}+H_{L,v}$, its Euclidean and Lorentzian contributions belong inside the same square. Their cross terms are part of that local equation. [Vertex labels in LQX](../lqx/structure.md#why-vertex-labels-matter) preserve this grouping.

## Expectations and gradients in code

Construct the linear action, keep its factorisation, and pass the objective directly to `expect` or `expect_and_grad`.

```python
import neuralqx as nqx
import jax
import numpy as np
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=128, n_discard_per_chain=16, seed=1
)

action = NumberOperator(hilbert, edge=0) @ U1HolonomyOperator(hilbert, edge=0)
objective = action.squared()
stats, gradient = state.expect_and_grad(objective)
print(bool(stats.mean >= 0))
print(all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(gradient)))
```

```text
True
True
```

The derivative must account for the parameter dependence of the local action itself. To see why, let $a_\theta(\sigma)=A_{\mathrm{loc}}(\sigma)$ and $O_k(\sigma)=\partial_{\theta_k}\log\psi_\theta(\sigma)$ for a real parameter coordinate. Then

$$
\begin{aligned}
\partial_{\theta_k}\mathcal L
={}&2\operatorname{Re}\mathbb E_p
\left[a_\theta^*\,\partial_{\theta_k}a_\theta\right]\\
&+2\operatorname{Re}\left(
\mathbb E_p[|a_\theta|^2O_k]
-\mathbb E_p[|a_\theta|^2]\mathbb E_p[O_k]
\right).
\end{aligned}
$$

The first term is lost if the squared local values are treated as fixed observations when differentiating the sampling average. The usual Hermitian covariance formula for $A^\dagger A$ uses its matrix local estimator $Q_{\mathrm{matrix}}$. Substituting $|A_{\mathrm{loc}}|^2$ into that formula does not preserve the derivative, even though the two estimators have the same expectation.

`expect_and_grad` selects the appropriate derivative for the factorised objective. Do not force `use_covariance=True` or use `expect_and_forces` for this marker. Those unsupported covariance requests are rejected.

## Wavefunction zeros and support

There is a specific limit to the elementary Born-average proof. If $\psi(\sigma)=0$ but $(A\psi)(\sigma)\ne0$, that output contributes to the action norm but can never be drawn from $p_\theta$. In that case,

$$
\langle A^\dagger A\rangle
=\mathbb E_p[|A_{\mathrm{loc}}|^2]
+\frac1Z\sum_{\sigma:\,\psi(\sigma)=0}|(A\psi)(\sigma)|^2.
$$

The missing term is not repaired by replacing a zero denominator with a small number. For example, take $|\psi\rangle=|0\rangle$ and an action that exchanges $|0\rangle$ and $|1\rangle$. Its norm squared is one, but the only Born-sampled configuration is $0$, where $(A\psi)(0)=0$. The naive squared local average would report zero.

A finite log-amplitude network has nonzero amplitudes mathematically. Exact projectors or a state embedded in a smaller basis can introduce genuine zeros. neuraLQX recognises its supported [projected wavefunctions](../../api/public/neuralqx/nn/projectors/index.md) and routes non-diagonal squared actions through residual estimation to cover those outputs. A custom wavefunction with undeclared exact zeros should not be assumed to receive that treatment automatically.

[`FullResidual`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.FullResidual) provides an explicit action-norm objective with output coverage. It can also be used with an identity [`BasisEmbedding`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.BasisEmbedding) when source and target are the same Hilbert space. If the state occupies a smaller physical space, the [restricted-space guide](restricted-spaces.md) explains how to measure the full action and the part that leaves that space.

The relevant question is therefore both which operator is being squared and where its action can have support. Keeping the factorisation and the basis identification explicit lets the expectation engine evaluate the intended residual.
