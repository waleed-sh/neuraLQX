---
title: Stochastic reconfiguration
---

# Stochastic reconfiguration

Stochastic reconfiguration, or SR, uses the geometry of the variational wavefunction to choose a parameter direction. Its purpose is to account for how strongly different combinations of parameters change the physical state. Two parameter variations with the same ordinary Euclidean norm can have very different effects on a wavefunction.

In neuraLQX, [`SR`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR) is a preconditioner. It receives an objective gradient, solves a linear system, and returns a direction. [Optax](optax.md) then decides how that direction becomes a parameter update. This page develops the geometry and the usual SR interpretation. The [optimiser comparison](sr-optimisers.md) explains what changes when momentum or an adaptive optimiser is added.

## Measure change in the wavefunction

Let $|\Psi_\theta\rangle$ be the normalised variational state and let $\theta_a$ initially denote real parameter coordinates. For complex parameter arrays, the real and imaginary parts can be treated as separate coordinates. Define the log derivatives

$$
O_a(\sigma)=\partial_{\theta_a}\log\psi_\theta(\sigma).
$$

A parameter displacement changes the log amplitude through

$$
\delta\log\psi(\sigma)
\approx\sum_a O_a(\sigma)\,\delta\theta_a.
$$

An overall normalisation or phase change does not change the physical ray represented by the state. Removing that unphysical component gives the quantum geometric tensor,

$$
S_{ab}
=\langle O_a^*O_b\rangle_p
-\langle O_a^*\rangle_p\langle O_b\rangle_p,
$$

where $p(\sigma)=|\psi_\theta(\sigma)|^2/Z_\theta$. Its real part $G=\operatorname{Re}S$ is the metric for real parameter displacements. To second order, it measures the loss of overlap between nearby normalised states,

$$
1-|\langle\Psi_\theta|\Psi_{\theta+\delta\theta}\rangle|^2
=\delta\theta^{\mathsf T}G\delta\theta
+O(\|\delta\theta\|^3).
$$

This gives a physical meaning to the size of a step. A direction with a large metric eigenvalue changes the state strongly, while a direction with a small eigenvalue changes it only weakly. The [QGT guide](qgt.md) explains the Monte Carlo estimate and how both amplitude and phase enter it.

## From steepest descent to SR

Write $g=\nabla_\theta\mathcal L$. Ordinary gradient descent minimises a local linear approximation to the objective while penalising the Euclidean size of the displacement. Replacing that penalty with the wavefunction metric gives

$$
\begin{aligned}
\delta\theta
&=\mathop{\operatorname{argmin}}_{\delta}
\left[g^{\mathsf T}\delta
+\frac{1}{2\eta}\delta^{\mathsf T}G\delta\right],\\
G\delta\theta&=-\eta g.
\end{aligned}
$$

When the metric is invertible, this is a natural-gradient step. In practice neuraLQX solves the regularised system

$$
\begin{aligned}
A&=G+\lambda I,\\
Ad&=g,\\
\delta\theta&=-\eta d
\quad\text{for plain SGD}.
\end{aligned}
$$

Here `diag_shift` is $\lambda$, and the Optax learning rate is $\eta$. The implementation solves for $d$ rather than explicitly forming $A^{-1}$. The [linear solver](solvers.md) determines how that solve is performed.

For a positive definite $A$ and an exact nonzero gradient,

$$
g^{\mathsf T}\delta\theta
=-\eta\,g^{\mathsf T}A^{-1}g<0.
$$

This is a local descent statement. A finite learning rate, an inaccurate solve, and Monte Carlo noise can all prevent the measured objective from decreasing at a particular iteration. SR also cannot ensure that a stationary point is the global minimum.

The unregularised metric accounts for smooth changes of parameter coordinates, which is one reason SR can behave differently from Euclidean gradient descent. The identity shift introduces a scale in the chosen coordinates, so exact coordinate invariance should not be claimed for a regularised implementation.

## The connection to imaginary time

For a fixed Hermitian operator $H$, let $E=\langle H\rangle$ and let $E_{\mathrm{loc}}$ be its [local estimator](../operators/application.md#from-the-expectation-value-to-the-local-estimator). The energy derivative in real coordinates is

$$
\begin{aligned}
F_a&=\left\langle
O_a^*(E_{\mathrm{loc}}-E)
\right\rangle_p,\\
g_a&=2\operatorname{Re}F_a.
\end{aligned}
$$

Normalised imaginary-time evolution has tangent

$$
\partial_\tau|\Psi\rangle=-(H-E)|\Psi\rangle.
$$

Projecting this tangent onto the variational manifold gives, in real coordinates,

$$
G\,\dot\theta=-\operatorname{Re}F=-\frac12g.
$$

Thus unregularised SR followed by plain SGD follows the same local direction as projected imaginary-time evolution. With the gradient convention above, the infinitesimal time increment is $\Delta\tau=2\eta$. The factor matters when comparing formulae that solve against $F$ with code that solves against the full objective gradient. It can be absorbed into a learning rate when the aim is simply minimisation.

Regularisation and finite steps modify this correspondence. Momentum and coordinate-dependent rescaling modify it further. SR is still useful for a more general real objective, including [penalties](../operators/penalties.md), but such an objective need not describe imaginary-time evolution under one fixed operator.

For a squared residual, use `expect_and_grad` on the [squared estimator](../operators/squared-estimators.md). Its local expression depends on the parameters, and the implementation includes that derivative. Substituting its local values into a covariance formula without the required derivative terms can produce a different gradient.

## Stabilise the metric

The QGT is positive semidefinite. It can have zero eigenvalues because parameters are redundant, an overall amplitude or phase is irrelevant, or the sample batch does not resolve every direction. Small eigenvalues are particularly sensitive to sampling noise.

If $G u_k=s_k u_k$, adding a positive diagonal shift changes the solved component to

$$
u_k^{\mathsf T}d
=\frac{u_k^{\mathsf T}g}{s_k+\lambda}.
$$

The shift suppresses large responses in poorly resolved directions. A large shift makes the direction approach a uniformly rescaled gradient, $d\approx g/\lambda$. A small shift retains more of the metric's anisotropy but can expose noise and conditioning problems.

`SR` defaults to `diag_shift=0.01`. This is a starting value, not a scale appropriate to every wavefunction or objective. Compare the resulting update sizes, solver residuals, and sampling behaviour. If reducing the shift causes very large updates, reducing the learning rate alone may hide an inadequately estimated metric. More samples or stronger regularisation may be needed.

The shift accepts a scalar or an iteration-dependent function, including an [Optax schedule](schedules.md#schedule-the-sr-shift). `diag_scale` is not implemented by the current built-in SR path. Use `diag_shift` for its supported regularisation rather than assuming that a diagonal-scaling option from another library is available.

## Inspect one SR direction

The following example computes an ordinary gradient and its SR direction on one shared batch. It checks the residual of the solved system before any parameter update occurs.

```python
import jax
import jax.numpy as jnp
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=7
)
charge = NumberOperator(hilbert, edge=0)
objective = charge @ charge

stats, gradient = state.expect_and_grad(objective)
sr = nqx.optimizer.SR(diag_shift=0.01)
direction = sr(state, gradient, step=0, samples=state.samples)
residual = jax.tree.map(
    lambda lhs, rhs: lhs - rhs, sr.lhs @ direction, gradient
)
relative_residual = optax.global_norm(residual) / jnp.maximum(
    optax.global_norm(gradient), 1e-12
)

print(sr.info["selected"])
print(bool(relative_residual < 1e-4))
```

```text
dense
True
```

`sr.lhs` stores the regularised operator used for the latest solve, `sr.x0` stores its solution, and `sr.info` stores solver information. These refer to the state and samples at that solve. They are not automatically recomputed when parameters subsequently change.

In a normal run, pass the SR object to the driver as shown in the [overview](index.md#a-small-optimisation-calculation). Keep that instance across iterations so iterative solvers can reuse the previous direction as an initial guess. The solver guide explains [restarts and residual checks](solvers.md#initial-guesses-and-solver-information).
