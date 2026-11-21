---
title: Quantum geometric tensor
---

# Quantum geometric tensor

The quantum geometric tensor, or QGT, measures how changes in parameters change a normalised wavefunction. [SR](sr.md) uses it to transform the objective gradient. The QGT depends on the variational state and the sampling distribution, rather than on which observable is being minimised. Two objectives evaluated at the same state can therefore share a QGT while supplying different gradients. A Hessian measures the curvature of a particular objective, which is a different quantity.

This page explains its sampled form, the treatment of complex wavefunctions, and the two representations supplied by neuraLQX. The [linear solver guide](solvers.md) explains how those representations are used to solve the SR system.

## A covariance of log derivatives

For a real parameter coordinate $\theta_a$, write

$$
O_a(\sigma)=\partial_{\theta_a}\log\psi_\theta(\sigma),
\qquad
\overline O_a=\sum_{r=1}^{M}w_r O_a(\sigma_r).
$$

For ordinary [Born samples](../samplers/index.md#why-we-sample-the-wavefunction), $w_r=1/M$. The sampled complex covariance and real metric are

$$
\begin{aligned}
\widehat S_{ab}
&=\sum_{r=1}^{M}w_r
\bigl(O_a(\sigma_r)-\overline O_a\bigr)^*
\bigl(O_b(\sigma_r)-\overline O_b\bigr),\\
\widehat G&=\operatorname{Re}\widehat S.
\end{aligned}
$$

The weights sum to one. The built-in constructors use this population covariance convention, with $1/M$ for equal weights, rather than a sample-covariance correction of $1/(M-1)$.

The sample layout remains `(n_chains, chain_length, hilbert.size)` at the user interface. The constructors combine the chain and time axes internally to form the $M$ configurations in the estimate. Correlation reduces the statistical information in those configurations even though the algebraic matrix uses all of them. See [sampling diagnostics](../samplers/diagnostics.md).

With [importance sampling](../samplers/importance.md), the QGT must describe the physical Born distribution rather than the proposal distribution. The built-in constructors obtain the state's associated `sample_weights` automatically. Explicit weights must be finite, nonnegative, have nonzero total, and match `samples.shape[:-1]`. The constructors normalise them before forming the covariance.

## Real and complex wavefunctions

Write the log amplitude as

$$
\log\psi_\theta(\sigma)=a_\theta(\sigma)+i\phi_\theta(\sigma).
$$

In real parameter coordinates, the metric is

$$
\begin{aligned}
G_{ab}
&=\operatorname{Cov}_p(\partial_a a,\partial_b a)\\
&\quad+\operatorname{Cov}_p(\partial_a\phi,\partial_b\phi).
\end{aligned}
$$

The phase term matters even though the Born probabilities depend only on the amplitude. Changing the relative phase changes the quantum state and the expectation of off-diagonal observables. A metric built only from the probability distribution would miss those directions.

For a strictly real log amplitude, the classical Fisher information of $p_\theta\propto|\psi_\theta|^2$ is $4G$. With a varying phase, the probability Fisher matrix still sees only the amplitude contribution. This is why the wavefunction geometry is the relevant quantity for SR on general quantum states.

The QGT constructors expose three differentiation modes.

| Mode | Coordinates and output used |
| --- | --- |
| `mode="real"` | Real parameter coordinates and the real part of the log amplitude. Suitable when the log amplitude is real. |
| `mode="complex"` | Real parameter coordinates and both real and imaginary log-amplitude channels. Suitable for general complex wavefunctions. |
| `mode="holomorphic"` | Complex parameter coordinates and holomorphic derivatives of the complex log amplitude. Requires a holomorphic model. |

When the mode is omitted, the output dtype determines whether the real or complex mode is used. Complex parameters are split into their real and imaginary coordinates in these two modes. Complex output alone does not select holomorphic differentiation. If both the output and parameters are complex and no declaration is supplied, the implementation uses the general real-coordinate treatment and issues a warning about the undeclared choice.

For a parameter $z=x+iy$, holomorphicity means that the model depends on $z$ without an independent dependence on $z^*$. Operations such as taking a modulus, real part, or conjugate can break that property. `holomorphic=True` is a declaration by the user. The library checks relevant dtypes but does not prove the analytic property of the network. Pass either `mode` or `holomorphic`, not both.

In holomorphic coordinates, SR uses the complex Hermitian covariance $S$ directly and solves against the compatible complex gradient. The full real-coordinate representation uses $G$ and can be larger. When comparing learning rates across conventions, account for the gradient and metric normalisations, not just the number of stored parameters.

The supplied [`RBM`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.RBM) uses real parameters by default, including when `complex_output=True` creates amplitude and phase channels. That configuration belongs to `mode="complex"`, not holomorphic mode.

## Dense sample Jacobians

[`qgt_jacobian_dense`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.qgt_jacobian_dense) creates a [`QGTJacobianDense`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.QGTJacobianDense). It stores a centred, weighted Jacobian and uses it to apply or construct the regularised metric.

For real output, define

$$
J_{ra}=\sqrt{w_r}\bigl(O_a(\sigma_r)-\overline O_a\bigr).
$$

The matrix is $J^{\mathsf T}J+\lambda I$. In general complex mode, separate real and imaginary derivative rows are stacked, giving up to $2M$ rows. In holomorphic mode, the complex Jacobian gives $J^\dagger J+\lambda I$.

Storing the Jacobian costs an amount proportional to $MP$, where $P$ is the number of solve coordinates. Forming the dense metric adds $P^2$ storage. The representation is useful when these arrays fit comfortably and you want a [direct solver](solvers.md#dense-direct-solvers). Model evaluation and differentiation also need memory, so the Jacobian and metric sizes are not the entire memory budget.

The sample count also limits the rank before regularisation. With real log amplitudes, centring leaves at most $M-1$ independent sample rows. With both amplitude and phase derivatives in real coordinates, the bound becomes $2(M-1)$. In holomorphic mode, the complex rank is at most $M-1$. The actual rank can be lower because of redundant parameters or sampled configurations. If there are more solve coordinates than these bounds allow, a singular unshifted matrix is expected. The [diagonal shift or a spectral cutoff](solvers.md#pseudoinverses-and-small-eigenvalues) controls how the solve handles such directions.

## Matrix-free products

[`qgt_onthefly`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.qgt_onthefly) creates a [`QGTOnTheFly`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.QGTOnTheFly). It applies the metric to a direction using automatic differentiation, without storing a full sample Jacobian or parameter matrix.

For a real log amplitude, let $B$ denote the uncentred derivative map, $w$ the weight vector, and $W=\operatorname{diag}(w)$. A product has the form

$$
Av=B^{\mathsf T}(W-ww^{\mathsf T})Bv+\lambda v.
$$

The implementation computes the forward derivative along $v$, centres it over the samples, and applies the transpose derivative. Complex output uses both real channels, while holomorphic mode uses the corresponding complex adjoint. The [JAX autodiff guide](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions) explains the forward and transpose operations underlying these products.

The SR solve reuses a linearisation within that solve and rebuilds it for the next state and sample batch. This representation avoids the explicit $P^2$ matrix and pairs naturally with [conjugate gradients](solvers.md#conjugate-gradients). It still retains samples and differentiation intermediates. A smaller `chunk_size` can reduce the memory required for model evaluation, but does not make the calculation independent of sample count or network size.

## Compare the representations on a known state

Consider one U(1) edge with charges $q\in\{-1,0,1\}$ and log amplitude

$$
\log\psi_{a,b}(q)=a q+bq^2.
$$

At $a=b=0$, the Born distribution is uniform, so using every basis configuration once gives the exact covariance. The log derivatives are $q$ and $q^2$, and

$$
G=\begin{pmatrix}2/3&0\\0&2/9\end{pmatrix}.
$$

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np


def log_amplitude(parameters, configurations):
    q = configurations[..., 0]
    return parameters["a"] * q + parameters["b"] * q**2


def initialise(key, shape, dtype):
    return {"a": jnp.array(0.0), "b": jnp.array(0.0)}


graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    hilbert=hilbert,
    apply_fn=log_amplitude,
    init_fn=initialise,
    n_samples=3,
    seed=11,
)
state.samples = hilbert.all_states()[None, ...]

dense = nqx.optimizer.qgt_jacobian_dense(state, diag_shift=0.01)
matrix_free = nqx.optimizer.qgt_onthefly(state, diag_shift=0.01)
expected = jnp.diag(jnp.array([2.0 / 3.0, 2.0 / 9.0])) + 0.01 * jnp.eye(2)

np.testing.assert_allclose(dense.to_dense(), expected, atol=1e-7)
np.testing.assert_allclose(matrix_free.to_dense(), expected, atol=1e-7)
print(dense.mode, dense.n_parameters)
```

```text
real 2
```

The complete basis is a valid equal-weight batch here because the chosen wavefunction is uniform. For a nonuniform wavefunction, enumerating every configuration once and giving all of them equal weight would compute the wrong physical covariance. Use Born samples or the corresponding probability weights.

`to_dense()` in this example is a diagnostic for a two-coordinate model. Calling it on a large matrix-free QGT materialises the matrix and removes the memory advantage you selected that representation for.

## How SR chooses a representation

With its default settings, [`SR`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR) uses [`qgt_auto`](../../api/public/neuralqx/optimizer/qgt_auto/index.md#neuralqx.optimizer.qgt_auto.qgt_auto) together with [`solver.auto`](../../api/public/neuralqx/optimizer/solver/auto/index.md#neuralqx.optimizer.solver.auto.auto). Up to 2048 solve coordinates, the default policy uses the dense Jacobian and a dense solve. Above that threshold it uses the matrix-free QGT and an iterative solve.

The count is the number of coordinates used by the QGT. A complex scalar parameter contributes two real coordinates in the general treatment and one complex coordinate in holomorphic mode. It need not equal the number of scalar entries displayed for the original parameter tree.

Passing a recognised direct solver selects the dense Jacobian, while passing CG selects the matrix-free representation. Put solver options in `solver_kwargs`, as shown in the [solver examples](solvers.md#the-default-solver-policy), so the policy can identify the solver function. You can also pass `qgt=nqx.optimizer.qgt_jacobian_dense` or `qgt=nqx.optimizer.qgt_onthefly` explicitly when comparing representations.

By contrast, [`state.quantum_geometric_tensor()`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.AbstractVariationalState.quantum_geometric_tensor), also available as `state.qgt()`, directly constructs the matrix-free QGT. It does not apply SR's automatic representation policy.

Every constructed QGT describes the parameters and samples at construction time. Rebuild it after changing either if you need the metric of the updated state. The driver handles this as part of each SR step.
