---
title: Linear solvers
---

# Linear solvers

At each [SR](sr.md) step, the objective gradient is transformed by solving

$$
Ad=g,
\qquad A=G+\lambda I.
$$

The gradient $g$ and metric $G$ come from the current variational state and samples. The linear solver computes $d$. The [Optax optimiser](optax.md) then turns $d$ into a parameter increment. A solver tolerance therefore controls the accuracy of this linear system, not the convergence of the physical simulation.

The [QGT representation](qgt.md) determines how the solver accesses $A$. A direct method uses a dense matrix. An iterative method can work through products $Av$ without storing the matrix explicitly.

## The default solver policy

[`solver.auto`](../../api/public/neuralqx/optimizer/solver/auto/index.md#neuralqx.optimizer.solver.auto.auto) is the default for [`SR`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR). It selects a dense solve for at most 2048 solve coordinates and conjugate gradients above that count. SR's automatic QGT selection uses the same threshold to choose a dense Jacobian or matrix-free representation.

The threshold is a policy choice, not a measured performance boundary for every machine and network. Dense work can be attractive for a moderate system size, while large Jacobians or parameter matrices can make an iterative method preferable. The sample count, differentiation mode, model evaluation cost, hardware, and available memory all affect that comparison.

You can change the threshold and the settings passed to the selected solver.

```python
import neuralqx as nqx

sr = nqx.optimizer.SR(
    diag_shift=0.01,
    solver_kwargs={
        "dense_threshold": 1024,
        "iterative_kwargs": {"tol": 1e-5, "maxiter": 300},
    },
)
```

Here the iterative tolerance and iteration limit apply only when the iterative branch is selected. The default dense method is [`solver.solve`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.solve), and the default iterative method is [`solver.cg`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.cg). `solver.auto` also accepts `dense_solver`, `iterative_solver`, and a separate `dense_kwargs` dictionary.

If you already know the intended method, pass its function directly through `solver` and put its numerical options in `solver_kwargs`. This also lets SR recognise the solver when choosing a QGT representation.

## Dense direct solvers

[`solver.solve`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.solve) delegates to JAX's dense linear solve and defaults to `assume_a="pos"`, meaning a positive definite system. [`solver.cholesky`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.cholesky) explicitly uses a Cholesky factorisation with the same positivity requirement.

These are natural choices for a regularised QGT that fits comfortably in memory. They require a dense $P\times P$ matrix, where $P$ counts solve coordinates. Factoring that matrix has cubic cost in $P$, in addition to the work of forming it from the sample Jacobian. A positive diagonal shift helps with singular directions, although a shift too small relative to numerical precision may not resolve them reliably.

Direct solvers accept parameter pytrees and return a solution with the same structure. The [`DenseLinearOperator`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.DenseLinearOperator) below wraps a small known matrix so the solver interfaces can be compared without sampling noise.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

operator = nqx.optimizer.DenseLinearOperator(
    matrix=jnp.array([[4.0, 1.0], [1.0, 3.0]])
)
rhs = {"weights": jnp.array([1.0, 2.0])}
expected = jnp.linalg.solve(operator.to_dense(), rhs["weights"])

methods = (
    nqx.optimizer.solver.solve,
    nqx.optimizer.solver.cholesky,
    nqx.optimizer.solver.cg,
    nqx.optimizer.solver.gmres,
)
for method in methods:
    solution, info = method(operator, rhs)
    np.testing.assert_allclose(solution["weights"], expected, atol=1e-6)
    print(info["method"])
```

```text
solve
cholesky
cg
gmres
```

The agreement concerns this small positive definite system. It does not imply identical costs or convergence behaviour on a sampled QGT. Direct methods ignore the optional initial guess `x0`, while iterative methods can use it.

## Conjugate gradients

[`solver.cg`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.cg) solves a Hermitian positive definite system through matrix-vector products. With a positive shift, the sampled QGT has the intended algebraic structure. Choosing CG directly makes the automatic QGT policy use the matrix-free representation.

```python
import neuralqx as nqx

sr = nqx.optimizer.SR(
    solver=nqx.optimizer.solver.cg,
    diag_shift=0.01,
    solver_kwargs={"tol": 1e-5, "atol": 0.0, "maxiter": 300},
)
```

For residual $r=g-Ad$, the JAX stopping criterion is

$$
\|r\|_2\leq
\max\bigl(\mathtt{tol}\,\|g\|_2,\mathtt{atol}\bigr).
$$

`maxiter` limits the solve even if this tolerance has not been reached. `tol` is relative to the right-hand side, while `atol` supplies an absolute scale that can matter when the gradient becomes small. The [JAX CG reference](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html) describes this criterion and the required operator properties.

For the unregularised metric eigenvalues $s_{\min}$ and $s_{\max}$, a positive shift changes the condition number to

$$
\kappa(A)=\frac{s_{\max}+\lambda}{s_{\min}+\lambda}.
$$

Increasing the shift can make the solve easier, but also changes the SR direction. Increasing `maxiter` gives the solver more work without changing the regularised system. These address different causes of an inaccurate update.

The optional `M` argument is a numerical preconditioner approximating $A^{-1}$ inside CG. This is a second use of the word preconditioner. SR preconditions the optimisation gradient, while `M` helps solve SR's linear system. A suitable `M` changes the convergence of the numerical solve without changing its exact solution.

Choose tolerances in relation to both sampling uncertainty and dtype. Solving a noisy estimated system to many more digits does not remove that noise. Conversely, a loose residual can change the direction even when sampling is adequate. The supplied neural networks commonly use float32 parameters unless configured otherwise, so a requested tolerance far below that arithmetic's useful accuracy may be ineffective. Check the actual residual instead of relying on the requested tolerance alone.

## GMRES

[`solver.gmres`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.gmres) is available for more general linear operators. The built-in regularised QGT already has the symmetry and positivity that CG uses, making CG the more direct starting point for ordinary SR.

GMRES retains a Krylov basis and exposes `restart` to limit its size. In the underlying JAX solver, `maxiter` counts restart cycles, so its numerical value should not be interpreted as the same work limit as CG's `maxiter`. `solve_method="batched"` and `"incremental"` select different ways of solving the small inner problem. See the [JAX GMRES reference](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html) when tuning those options.

The neuraLQX wrapper expects a callable operator, such as a QGT or `DenseLinearOperator`. Wrap a raw matrix as in the example above. Choosing a more general solver does not correct an incorrectly constructed metric or an inappropriate gradient.

## Pseudoinverses and small eigenvalues

[`solver.pinv`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.pinv) uses a Hermitian pseudoinverse. If $A=U\operatorname{diag}(a_k)U^\dagger$, its spectral action is

$$
\begin{aligned}
d&=\sum_k f(a_k)u_k(u_k^\dagger g),\\
f(a_k)&=\begin{cases}
1/a_k,& |a_k|>r_{\mathrm{tol}}\max_j|a_j|,\\
0,&\text{otherwise}.
\end{cases}
\end{aligned}
$$

The cutoff removes unresolved directions instead of dividing by arbitrarily small eigenvalues. This is a different regularisation from adding a shift. You can use both, but then the spectral cutoff is applied to the already shifted matrix.

[`solver.pinv_smooth`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.pinv_smooth) adds a smooth suppression above the hard cutoff. For $r_k=|a_k|/\max_j|a_j|$, its retained inverse factor is

$$
f(a_k)=\frac{1}{a_k}
\frac{1}{1+(r_{\mathrm{smooth}}/r_k)^6},
\qquad r_k>r_{\mathrm{tol}}.
$$

It sets discarded or zero modes to zero. `rtol` controls the hard cutoff, while `rtol_smooth` controls the gradual suppression. Both default to $10^{-12}$. Those defaults are numerical settings, not estimates of the noise floor of your sampled metric.

```python
import jax.numpy as jnp
import neuralqx as nqx

matrix = jnp.diag(jnp.array([1.0, 1e-8, 0.0]))
rhs = jnp.array([1.0, 1.0, 0.0])

hard, _ = nqx.optimizer.solver.pinv(matrix, rhs, rtol=1e-6)
smooth, info = nqx.optimizer.solver.pinv_smooth(
    matrix,
    rhs,
    rtol=1e-12,
    rtol_smooth=1e-6,
    return_eigvals=True,
)

print(hard.tolist())
print(int(info["rank"]))
print(bool(0.0 < smooth[1] < 0.001))
```

```text
[1.0, 0.0, 0.0]
2
True
```

The second mode passes the smooth solver's hard cutoff, so `rank` reports two, but it is strongly attenuated. `rank` counts hard-retained modes and does not measure the effective strength of the smoothed inverse. The information dictionary also includes the smallest and largest eigenvalues, a condition-number estimate from nonzero eigenvalue magnitudes, and optionally the eigenvalues themselves. It does not turn those quantities into an automatic convergence decision.

Pseudoinverse methods require a dense eigendecomposition. They are useful for examining small systems and controlling weak directions, but can be expensive at large parameter counts. Strongly negative eigenvalues in a metric intended to be positive semidefinite warrant investigation. A pseudoinverse does not turn an indefinite matrix into the intended SR metric.

## Initial guesses and solver information

An SR instance retains its latest solution as `sr.x0`. With the default `solver_restart=False`, the next solve uses that direction as an initial guess. This can help when consecutive systems are similar. `solver_restart=True` asks each solve to start without that retained guess. It does not reset the sampler, model parameters, or Optax momentum.

After a solve, `sr.lhs` is the regularised operator that was solved, and `sr.info` is the solver information. The automatic solver records `selected="dense"` or `"iterative"` and places the selected method's information under `inner`. The direct methods report their method name. CG and GMRES include the underlying JAX `info` value.

In the current JAX CG interface, `info` is `None`, which is not a certificate that the tolerance was reached. Compute $\|Ad-g\|$ when diagnosing a solve. The [SR example](sr.md#inspect-one-sr-direction) demonstrates this using parameter pytrees and `sr.lhs`, on the same gradient and samples used by SR.

Also distinguish the residual of a regularised system from a physical optimisation error. An excellent solve of $Ad=g$ can still use an inaccurate sampled metric or gradient. A deliberately truncated pseudoinverse may leave a residual in discarded directions even when it has correctly computed its intended filtered solution.

## Recovering from nonfinite solutions

[`solver.nan_fallback`](../../api/public/neuralqx/optimizer/solver/index.md#neuralqx.optimizer.solver.nan_fallback) tries a second solver if the primary solution contains NaN or infinity. For example, a pseudoinverse can provide a finite result when a direct solve encounters a singular matrix.

```python
import jax.numpy as jnp
import neuralqx as nqx

robust_solve = nqx.optimizer.solver.nan_fallback(
    nqx.optimizer.solver.solve,
    nqx.optimizer.solver.pinv,
)
matrix = jnp.diag(jnp.array([1.0, 0.0]))
solution, info = robust_solve(matrix, jnp.array([1.0, 0.0]))

print(info["solver_fallback"])
print(bool(jnp.all(jnp.isfinite(solution))))
```

```text
True
True
```

The fallback checks finiteness only. It does not catch arbitrary exceptions or reject a finite but inaccurate direction. For an SR calculation with real parameters, this pair can be used with `qgt=nqx.optimizer.qgt_jacobian_dense` specified explicitly. The automatic policy cannot generally infer the requirements hidden inside a custom wrapper. The fallback also makes its decision in Python, so it is intended for the ordinary host-driven solve rather than being enclosed in another `jax.jit`.

Once the solve behaves consistently, use the [schedule guide](schedules.md) to vary regularisation over the run. Reassess the residual and update scale as the shift changes, since a setting that was easy to solve initially can become more demanding later.
