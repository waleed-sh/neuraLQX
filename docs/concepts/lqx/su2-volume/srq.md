---
title: Shifted resolvent quadrature
---

# Shifted resolvent quadrature

SRQ evaluates the [volume action](index.md) without diagonalising the local volume density $Q$ or constructing a dense volume matrix. The name means **shifted resolvent quadrature**. The quadrature is deterministic. Monte Carlo sampling enters later when volume rows are used to estimate wavefunction expectations.

This also gives SRQ a different role from [stochastic reconfiguration](../../optimisers/sr.md), usually abbreviated SR. SR changes the direction of a parameter update. SRQ evaluates the matrix function defining a geometric operator.

The useful starting point is that $Q$ has a sparse action in the recoupling basis, even though $\sqrt{|Q|}$ can mix many intertwiners. SRQ keeps the sparse action and replaces the matrix function by a weighted collection of shifted linear solves.

## Turn the square root into a quarter power

For a Hermitian density $Q$, the dimensionless volume is

$$
\sqrt{|Q|}=(Q^2)^{1/4}.
$$

The implementation computes a positive scale $B$ that bounds $\|Q\|_2$. For the explicitly compiled sparse matrix, it uses the maximum absolute row sum. The runtime sparse representation likewise supplies a row-sum bound. Define

$$
A=\frac{Q^2}{B^2},
\qquad
\sqrt{|Q|}=\sqrt B\,A^{1/4}.
$$

$A$ is positive semidefinite with spectrum contained in $[0,1]$ when the bound is satisfied. It is applied by two successive products with $Q$, divided by $B^2$. Neither $Q^2$ nor $A$ needs to be stored as a matrix. If $B=0$, the density and volume action are zero, and the implementation returns a zero action.

The factor `volume_scale`, denoted $c_V$ in the [overview](index.md#volume-acts-on-intertwiners), is applied after this dimensionless action.

## An integral of shifted systems

For a scalar $x\geq0$,

$$
x^{1/4}
=\frac{1}{\sqrt2\,\pi}
\int_0^\infty t^{-3/4}\frac{x}{x+t}\,dt.
$$

With $t=e^s$, this becomes

$$
x^{1/4}
=\frac{1}{\sqrt2\,\pi}
\int_{-\infty}^{\infty}
e^{s/4}\frac{x}{x+e^s}\,ds.
$$

Applying the same identity to the eigenvalues of $A$ gives a matrix-function formula. For a coefficient vector $b$,

$$
A^{1/4}b
=\frac{1}{\sqrt2\,\pi}
\int_{-\infty}^{\infty}
e^{s/4}(A+e^s I)^{-1}Ab\,ds.
$$

Every integrand requires a positive shifted system. This remains well defined when $A$ has a kernel. The right-hand side is $Ab$, so a vector in the exact kernel has zero right-hand side and zero volume action. The construction does not add a physical offset to the volume or lift its zero modes.

Here, a **resolvent** is the shifted inverse $(A+tI)^{-1}$. SRQ approximates the integral by a weighted sum of these resolvent actions, which gives the method its name. Each action is evaluated by solving a linear system.

## The finite quadrature

Choose spacing $h>0$ and integer order $K$. Sampling the logarithmic integration coordinate at $s_\ell=\ell h$ gives

$$
\begin{aligned}
t_\ell&=e^{\ell h},\\
\omega_\ell&=\frac{h}{\sqrt2\,\pi}e^{\ell h/4},\\
(A+t_\ell I)y_\ell&=Ab,
\qquad -K\leq\ell\leq K.
\end{aligned}
$$

The approximate volume action is

$$
V_{K,h}b
=c_V\sqrt B\sum_{\ell=-K}^{K}\omega_\ell y_\ell.
$$

There are **$2K+1$ shifted systems**. `srq_order=96` therefore uses 193 shifts. The two quadrature controls have separate effects. Increasing $K$ at fixed $h$ extends the interval $[-Kh,Kh]$. Decreasing $h$ makes the grid finer, but also shortens that interval unless $K$ increases.

The integral decays differently at its two ends. For fixed $x>0$, its scalar integrand behaves as $e^{s/4}$ for large negative $s$ and as $x e^{-3s/4}$ for large positive $s$. The negative tail decays more slowly. Small nonzero eigenvalues also move the transition region towards more negative $s$, which is one reason to check sectors representative of your calculation.

Neither increasing the spin cutoff nor adding more Monte Carlo samples refines this quadrature. The [accuracy guide](accuracy.md#refine-the-interval-and-grid-separately) shows how to vary $K$ and $h$ independently.

### See the approximation on individual eigenvalues

This small NumPy calculation evaluates the same rational function on known density eigenvalues. There are no iterative solves or sampling errors here, so the remaining difference comes from the finite quadrature and floating-point arithmetic.

```python
import numpy as np

eigenvalues = np.array([-3.0, -0.2, 0.0, 0.2, 3.0])
bound = np.max(np.abs(eigenvalues))
x = (eigenvalues / bound)**2

order, step = 128, 0.75
log_shifts = step * np.arange(-order, order + 1)
shifts = np.exp(log_shifts)
weights = step / (np.sqrt(2.0) * np.pi) * np.exp(log_shifts / 4.0)
approximation = np.sqrt(bound) * np.sum(
    weights[:, None] * x[None, :] / (x[None, :] + shifts[:, None]),
    axis=0,
)

np.testing.assert_allclose(
    approximation, np.sqrt(np.abs(eigenvalues)), rtol=1e-9, atol=1e-10
)
print(bool(approximation[2] == 0.0))
print(bool(np.all(approximation >= 0.0)))
```

```text
True
True
```

Opposite eigenvalues of $Q$ give the same volume eigenvalue, and the exact zero stays zero. With exact shifted solves, the finite rational function is nonnegative on $A\geq0$. Finite solver error and roundoff still need to be assessed in an actual action.

## Solve the shifts together

The shifted systems share the same right-hand side and differ only by multiples of the identity. Their Krylov spaces agree,

$$
\mathcal K_m(A,r)=\mathcal K_m(A+tI,r),
$$

where the space is spanned by $r,Ar,\ldots,A^{m-1}r$. This lets the implementation use a shared conjugate-gradient recurrence, based on the smallest shift, to advance the other systems. The method follows the shifted-system construction described by [Jegerlehner](https://arxiv.org/abs/hep-lat/9612014).

The shared recurrence avoids repeating the main $Q^2$ product independently for every shift at every iteration. Each shift still needs vector storage and arithmetic. The final diagnostic also recomputes its true residual. Increasing the quadrature order therefore has a real memory and runtime cost, even though the costly recurrence is shared.

For local dimension $D_v$, the shifted solution and direction arrays scale with $(2K+1)D_v$. The sparse density and the padded local representation also consume memory. SRQ removes the dense eigendecomposition requirement, but the most economical choice for a small block can still be spectral evaluation. The [automatic policy](index.md#select-the-evaluation-policy) makes that choice according to a dimension threshold.

These shifted solves are internal to volume evaluation. Their tolerances are the SRQ settings below, not the solver options of an SR optimiser.

## Settings and their API names

The same controls appear at three levels. [`LqxSU2`](../../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2) stores a shared model policy. [`VolumeOperator`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VolumeOperator), including `model.get_volume(...)` overrides, uses the middle column. [`VertexVolumeAction`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VertexVolumeAction), including `fixed_sector_action(...)` overrides, uses the last column.

| Meaning and default | Model | Volume operator | Fixed-sector action |
| --- | --- | --- | --- |
| Quadrature order $K=96$ | `srq_order` | `srq_order` | `order` |
| Logarithmic spacing $h=0.75$ | `srq_step` | `srq_step` | `step` |
| Solve tolerance $10^{-12}$ | `srq_solve_rtol` | `solve_rtol` | `rtol` |
| Maximum CG iterations 256 | `srq_maxiter` | `solve_maxiter` | `maxiter` |
| Evaluation policy | `volume_method` | `method` | `method` |

The model and volume operator default to `"auto"`, while a fixed-sector action defaults to `"srq"`. `spectral_max_dimension` has the same name at all three levels and defaults to 64. All scales and tolerances must be positive, and orders and iteration limits must be positive integers. The product $Kh$ must not exceed 650 because the shifts are computed exponentially. That limit prevents overflow and is not an accuracy recommendation.

The public fixed-sector action runs a fixed iteration loop with inactive updates once convergence is reached. Runtime volume rows can terminate the shared loop early. Both paths recompute shifted residuals afterwards. The next page explains [what those residuals establish](accuracy.md#what-the-shifted-residuals-measure) and how to check the quadrature error separately.
