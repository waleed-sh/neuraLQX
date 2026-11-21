---
title: Choosing an optimiser for SR
---

# Choosing an optimiser for SR

[SR](sr.md) and an [Optax optimiser](optax.md) act in sequence. SR solves for a direction using the wavefunction metric. Optax receives that direction and applies its own transformation. The combination is therefore determined by both choices.

Plain SGD is the most transparent starting point for SR. It preserves the direction of the solve and lets you study the effects of the learning rate, sampling, and regularisation separately. Other combinations can be useful, but it helps to understand what they change before interpreting their results as natural-gradient or imaginary-time dynamics.

Throughout this page, use real parameter coordinates and write

$$
A_t=G_t+\lambda_t I,
\qquad d_t=A_t^{-1}g_t.
$$

Here $g_t$ is the objective gradient and $d_t$ is the SR direction. Complex wavefunctions can still use this description through their [real-coordinate representation](qgt.md#real-and-complex-wavefunctions).

## Plain SGD preserves the SR direction

With [`optax.sgd`](https://optax.readthedocs.io/en/latest/api/generated/optax.sgd.html) and no momentum,

$$
\Delta\theta_t=-\eta_t d_t.
$$

Only one scalar changes the solved direction. A [learning-rate schedule](schedules.md) changes the distance travelled along that direction at each iteration. It does not introduce an additional coordinate-dependent transformation.

For exact gradients and a positive definite $A_t$, the infinitesimal objective change is negative,

$$
g_t^{\mathsf T}\Delta\theta_t
=-\eta_t g_t^{\mathsf T}A_t^{-1}g_t<0.
$$

This is the clearest baseline when the aim is to retain the [projected imaginary-time interpretation](sr.md#the-connection-to-imaginary-time). Finite steps and noisy estimates still need to be checked. A sequence of Monte Carlo energy estimates need not decrease at every iteration.

## Momentum averages directions from different states

Ordinary SGD with momentum replaces the current direction by a running trace,

$$
\begin{aligned}
m_t&=d_t+\mu m_{t-1},\\
\Delta\theta_t&=-\eta_t m_t.
\end{aligned}
$$

The previous direction was computed at previous parameters, with a different sample batch and potentially a different metric. Momentum combines these directions in the parameter coordinates used by the network. It does not transport them according to the changing wavefunction geometry.

That history can smooth fluctuations and carry progress through regions where individual estimates are noisy. It can also continue a direction after the current metric or gradient has changed substantially. The update is no longer the instantaneous SR direction, and the simple descent expression above no longer applies.

Using momentum with SR is a valid optimisation choice. Compare it with plain SGD at comparable step sizes, rather than assuming that the same learning rate should work in both cases. For a nearly constant direction, the trace approaches $d/(1-\mu)$, so momentum can increase the effective step substantially. Nesterov momentum introduces another extrapolation and likewise changes the basic SR trajectory.

## Adam and other adaptive optimisers

[`optax.adam`](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html) estimates moments of its input. When it follows SR, that input is $d_t$, so its real-coordinate update has the form

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)d_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)d_t^{\odot2},\\
\Delta\theta_t
&=-\eta_t\frac{\widehat m_t}
{\sqrt{\widehat v_t}+\varepsilon}.
\end{aligned}
$$

The hats denote bias correction, and the division is componentwise. Adam is tracking the **SR directions**, not the original gradients. This can damp components that repeatedly receive large SR updates, but it also rescales components whose relative sizes were already chosen by the metric solve.

Even without momentum, a positive diagonal rescaling $D_t$ after SR would give

$$
\Delta\theta_t=-\eta_t D_tA_t^{-1}g_t.
$$

Although $D_t$ and $A_t$ are positive definite, their product need not be symmetric. Consequently $g_t^{\mathsf T}D_tA_t^{-1}g_t$ need not be positive. Coordinatewise rescaling can change a downhill SR direction into an uphill direction for the local objective. Adam's history adds another reason why a step may differ from the current natural gradient.

### A concrete example of the change

Consider this positive definite metric and gradient,

$$
A=\begin{pmatrix}100&9\\9&1\end{pmatrix},
\qquad
g=\begin{pmatrix}2\\1\end{pmatrix}.
$$

The SR direction is $d=(-7,82)^{\mathsf T}/19$, so $g^{\mathsf T}d=68/19>0$. Plain SGD follows $-d$ and is locally downhill. On its first step, Adam approximately normalises each nonzero component of $d$ to its sign. Its increment then points approximately along $(1,-1)$, whose inner product with $g$ is positive.

```python
import jax.numpy as jnp
import optax

metric = jnp.array([[100.0, 9.0], [9.0, 1.0]])
gradient = jnp.array([2.0, 1.0])
direction = jnp.linalg.solve(metric, gradient)
parameters = jnp.zeros_like(gradient)

sgd = optax.sgd(learning_rate=0.01)
adam = optax.adam(learning_rate=0.01)
sgd_update, _ = sgd.update(direction, sgd.init(parameters), parameters)
adam_update, _ = adam.update(direction, adam.init(parameters), parameters)

print(bool(jnp.all(jnp.linalg.eigvalsh(metric) > 0)))
print(bool(jnp.dot(gradient, sgd_update) < 0))
print(bool(jnp.dot(gradient, adam_update) > 0))
```

```text
True
True
True
```

This example establishes a possibility, not a prediction that Adam will fail on a particular simulation. It also does not make ordinary Adam without SR problematic in the same way. The point is that applying an adaptive rescaling after a coupled metric solve changes its geometric properties.

### Is Adam with SR still reasonable?

Yes, if the goal is optimisation and the combined method performs well under appropriate checks. A useful algorithm does not have to preserve an imaginary-time trajectory. The combination can be treated as an empirical update rule whose sampling, regularisation, and learning-rate choices are validated together.

Compare plain SR with SGD, Adam without SR, and the combined method on the same physical objective. Examine the final observables and uncertainties, behaviour across seeds, and time needed to reach a comparable result. A lower noisy training estimate in one run is weaker evidence than consistent improvement on independently collected measurements.

[`optax.rmsprop`](https://optax.readthedocs.io/en/latest/api/generated/optax.rmsprop.html) and [`optax.adagrad`](https://optax.readthedocs.io/en/latest/api/generated/optax.adagrad.html) also rescale coordinates based on past inputs. [`optax.lion`](https://optax.readthedocs.io/en/latest/api/generated/optax.lion.html) and sign-based updates can discard much of the solved direction's relative magnitude. These choices are available as optimisation experiments, but should be described as modifications of SR rather than equivalent ways of taking its usual step.

## Global clipping changes the length of a step

[`optax.clip_by_global_norm`](https://optax.readthedocs.io/en/latest/api/generated/optax.clip_by_global_norm.html) rescales the whole direction by one nonnegative scalar,

$$
\widetilde d_t=c_t d_t,
\qquad
c_t=\min\!\left(1,\frac{C}{\|d_t\|_2}\right),
$$

with the zero direction left unchanged. Followed by plain SGD, this preserves the direction of SR and limits the step length in the chosen parameter coordinates. It can be useful when occasional noisy solves produce large directions.

The threshold is an ordinary parameter norm, not a bound on the wavefunction distance $\Delta\theta^{\mathsf T}G\Delta\theta$. Componentwise clipping, clipping each parameter leaf separately, or applying Adam after clipping changes the direction as well. The [Optax guide](optax.md#compose-transformations-deliberately) gives a runnable global-clipping example and explains where it acts in the driver.

Clipping should not replace checking why updates are large. Persistent clipping can indicate a very small shift, inaccurate sampling, or a learning rate that is inconsistent with the objective scale.

## Weight decay changes the intended update

Weight decay encourages small network parameters. It is distinct from a penalty on a physical observable or constraint. There is no general reason that small parameter norm corresponds to the desired LQG state, particularly when different parameter values can represent the same normalised wavefunction.

To see the distinction, an explicit Euclidean parameter penalty would define

$$
\widetilde{\mathcal L}
=\mathcal L+\frac{\gamma}{2}\|\theta\|_2^2.
$$

Applying SR to its full gradient gives

$$
d_{\mathrm{penalised}}
=A^{-1}(g+\gamma\theta).
$$

Adding a decay term after the SR solve instead uses $A^{-1}g+\gamma\theta$. These differ because the second term has not passed through the metric. [`optax.adamw`](https://optax.readthedocs.io/en/latest/api/generated/optax.adamw.html) additionally uses adaptive moments and decoupled parameter shrinkage. It therefore changes both the SR direction and the update's regularisation.

Weight decay is allowed when it serves a deliberate modelling or optimisation choice. It should not be used as a substitute for [physical penalties](../operators/penalties.md), and a decayed run should not be interpreted as minimising the original expectation by unmodified SR.

## Quasi-Newton methods and line searches

A method such as [`optax.lbfgs`](https://optax.readthedocs.io/en/latest/api/generated/optax.lbfgs.html) builds an approximation to inverse curvature from changes in parameters and gradients. Feeding it SR directions changes the quantities entering those curvature relations. The input need not be the Euclidean gradient of any scalar objective, and it also changes as the metric is re-estimated.

Sampling noise complicates curvature estimation and trial-step comparisons further. In addition, the default Optax L-BFGS line search needs objective evaluations that the [standard VMC update interface](optax.md#which-optax-transformations-fit-the-driver) does not supply. Disabling a line search can change the interface requirements, but does not restore the missing curvature interpretation. Treat such combinations as methods requiring their own implementation and validation.

## A practical starting point

Begin with plain SGD, a positive diagonal shift, and sampling that resolves the important observables. Check the [linear solve residual](solvers.md#initial-guesses-and-solver-information) and the scale of the resulting direction. Then adjust the [learning rate and shift](schedules.md) according to their different roles.

If another optimiser helps, keep the comparison grounded in the scientific result and the computational cost. The important distinction is between an update rule that works for minimisation and an update that retains the particular geometric or dynamical interpretation you intend to study.
