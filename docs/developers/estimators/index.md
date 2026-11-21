---
title: Custom estimators
---

# Custom estimators

A local estimator connects an operator to the configurations drawn from a variational state. The operator describes an action on the Hilbert space. The estimator evaluates the contribution of each sampled configuration to a measurement, and the expectation layer combines those contributions into statistics and, when supported, gradients.

Most new operators do not need a new estimator. A [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) that supplies matrix row connectivity already works with the standard discrete estimator. Write a custom estimator when an observable has a simpler direct expression, when its evaluation can avoid expensive intermediate connections, or when you are developing a different statistical quantity. The [custom operator guides](../operators/index.md) explain the connectivity route.

## From a quantum expectation to a sampled quantity

Let $x$ label a basis configuration, let $\psi_\theta(x)$ be its variational amplitude, and write $A_{xy}=\langle x|A|y\rangle$. With the ordinary Born distribution,

$$
p_\theta(x)=\frac{|\psi_\theta(x)|^2}{Z_\theta},
\qquad
Z_\theta=\sum_x|\psi_\theta(x)|^2,
$$

the normalised expectation is

$$
\langle A\rangle_\theta
=\frac{\sum_{x,y}\psi_\theta(x)^*A_{xy}\psi_\theta(y)}{Z_\theta}
=\sum_xp_\theta(x)A_{\mathrm{loc},\theta}(x),
$$

where, on configurations with nonzero amplitude,

$$
A_{\mathrm{loc},\theta}(x)
=\frac{(A\psi_\theta)(x)}{\psi_\theta(x)}
=\sum_y A_{xy}
  \exp\bigl(\log\psi_\theta(y)-\log\psi_\theta(x)\bigr).
$$

For a diagonal observable $A_{xy}=a(x)\delta_{xy}$, this reduces to $a(x)$. No connected-state evaluation is needed. For an off-diagonal operator, its nonzero matrix row entries determine which additional wavefunction amplitudes must be evaluated. This is why a matrix free representation is useful even when the Hilbert space is too large to enumerate.

Given $M$ retained samples, the basic estimate is

$$
\widehat{\langle A\rangle}=\frac1M\sum_{s=1}^M A_{\mathrm{loc},\theta}(x_s).
$$

The entries need not be real individually, even when $A$ is Hermitian and the exact expectation is real. Keep the complex values until the appropriate reduction. Also retain the chain and sample axes, since correlated Markov chain samples require more information than a flattened mean to assess their uncertainty. The [scientist guide to expectations](../../concepts/vqs/expectations.md) explains the resulting statistics.

## Where a custom implementation enters

[`local_estimators`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.local_estimators) selects samples, checks the operator's Hilbert space, resolves evaluation options, and routes the calculation. For an ordinary individual observable, it reaches a type-dispatched local kernel. Register that kernel with [`register_local_estimator`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.register_local_estimator) and return [`LocalEstimators`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.LocalEstimators) containing one scalar per sample.

[`expect`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.expect) then turns those values into statistics. [`expect_and_grad`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.expect_and_grad) additionally selects a gradient calculation. That selection carries mathematical assumptions. A registered forward estimator does not, by itself, define how an arbitrary parameter-dependent observable should be differentiated.

These guides focus on ordinary sampling with `machine_pow=2`. Importance sampling and [embedded residuals](../../concepts/operators/restricted-spaces.md) use additional numerator, denominator, and support information. Their specialised routes can run before the registered local kernel. Likewise, an operator sum can be evaluated through a combined connectivity plan. An optimisation of one operator's estimator must preserve its meaning when the standard connectivity route is used elsewhere.

## Develop the implementation in stages

Start with [dispatch and contracts](dispatch.md), which builds a diagonal occupation observable and registers its estimator. [Estimator kernels](kernels.md) develops an off-diagonal example, including model evaluation, chunking, and the treatment of padded connections. [Gradients](gradients.md) derives the covariance and score terms, explains which extension paths currently support them, and checks a gradient against an exact calculation.

For every new estimator, establish the mathematical quantity first and compare it with an enumerated small system. Then check batching, compilation, and chunk sizes. Only after those agree should a timing comparison influence the implementation. The [profiling guides](../profiling/index.md) describe how to measure a complete expectation or gradient calculation, including device execution.
