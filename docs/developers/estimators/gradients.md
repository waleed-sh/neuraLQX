---
title: Gradients
---

# Gradients

An expectation changes when the wavefunction changes, even if the local observable is independent of the parameters. The sampling probabilities change too. Differentiating a plain mean over a fixed sample array accounts only for the dependence visible in that array's local values, so it is not generally the gradient of the expectation.

[`expect_and_grad`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.expect_and_grad) handles this distinction for its supported operator paths. A custom estimator must still satisfy the mathematical assumptions of the selected path. This page uses real parameters first, then explains the complex parameter convention and the current registration boundary.

## The covariance identity for a Hermitian operator

For a parameter-independent Hermitian operator $A$ and Born sampling, define

$$
O_k(x)=\partial_{\theta_k}\log\psi_\theta(x),
\qquad E=\langle A\rangle_\theta.
$$

Differentiating the normalised bra and ket gives

$$
\partial_{\theta_k}E
=2\operatorname{Re}\left[
\langle O_k^* A_{\mathrm{loc}}\rangle
-\langle O_k^*\rangle\langle A_{\mathrm{loc}}\rangle
\right].
$$

The quantity inside the real part is the covariance force,

$$
F_k=\left\langle O_k^*(A_{\mathrm{loc}}-E)\right\rangle.
$$

[`expect_and_forces`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.expect_and_forces) computes these forces using a vector-Jacobian product. It does not need to build the entire log-derivative matrix explicitly. [`force_to_grad`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.force_to_grad) performs the corresponding conversion to the parameter gradient convention.

The identity already accounts for the parameter dependence of the usual linear local estimator. It relies on Hermiticity and the Born distribution. A local value being real or positive does not establish those assumptions.

## Check a registered estimator against an exact result

For the occupied fraction from [dispatch and contracts](dispatch.md), take

$$
\log\psi_\theta(x)=\theta\sum_i x_i.
$$

Each binary variable then has probability $q(\theta)=e^{2\theta}/(1+e^{2\theta})$ of being occupied. Consequently $\langle\rho\rangle=q$ and $\partial_\theta\langle\rho\rangle=2q(1-q)$. At $\theta=0$, these are both $1/2$.

This standalone example verifies the registered estimator's covariance gradient against differentiation of the exact normalised sum.

```python
from typing import Any
import jax
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.vqs import LocalEstimatorOptions, LocalEstimators

class OccupiedFraction(nqx.operator.AbstractOperator):
    @property
    def dtype(self):
        return jnp.dtype("float64")

    @property
    def is_hermitian(self):
        return True

    @property
    def is_diagonal(self):
        return True

    @property
    def adjoint(self):
        return self

@nqx.vqs.register_local_estimator
def occupied_fraction_kernel(
    vstate: Any,
    operator: OccupiedFraction,
    samples: Any,
    options: LocalEstimatorOptions,
) -> LocalEstimators:
    return LocalEstimators(jnp.mean(samples.astype(operator.dtype), axis=-1))

def log_value(parameters, configurations):
    return parameters["theta"] * jnp.sum(configurations, axis=-1)

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.HomogeneousDiscreteHilbert(
    graph, nqx.hilbert.LocalRange(0, 1, 2)
)
state = nqx.vqs.VariationalState(
    hilbert=hilbert, apply_fn=log_value,
    parameters={"theta": jnp.array(0.0)}, seed=0,
)
basis = hilbert.all_states()
samples = basis.reshape(2, 4, hilbert.size)
stats, gradient = nqx.vqs.expect_and_grad(
    state, OccupiedFraction(hilbert), samples=samples,
)

def exact_expectation(theta):
    probabilities = jax.nn.softmax(2 * theta * jnp.sum(basis, axis=-1))
    return jnp.sum(probabilities * jnp.mean(basis, axis=-1))

exact_gradient = jax.grad(exact_expectation)(jnp.array(0.0))
np.testing.assert_allclose(gradient["theta"], exact_gradient)
print(float(stats.mean))
print(f"{float(gradient['theta']):.6g}")
```

```text
0.5
0.5
```

Uniform sampling is exact for this particular parameter value. For a general parameter value, compare the exact sum with sufficiently converged Monte Carlo estimates, or explicitly use the exact probabilities in a separate reference calculation. Enumerating configurations does not cause `samples=` to weight them by their Born probabilities automatically.

## When the estimator needs its own derivative

For a general parameter-dependent scalar quantity $a_\theta(x)$ sampled from $p_\theta$, differentiation gives

$$
\partial_k\mathbb E_{p_\theta}[a_\theta]
=\mathbb E_{p_\theta}[\partial_k a_\theta]
+\mathbb E_{p_\theta}\left[
(a_\theta-\mathbb E[a_\theta])\partial_k\log p_\theta
\right].
$$

The first term differentiates the estimator at fixed configurations. The second is the score term from the changing distribution. For $p_\theta\propto|\psi_\theta|^m$,

$$
\partial_k\log p_\theta(x)
=m\left(\operatorname{Re}O_k(x)
-\langle\operatorname{Re}O_k\rangle\right).
$$

A useful fixed-sample surrogate makes both terms visible. Define $L_s=m\operatorname{Re}\log\psi_\theta(x_s)$ and let $\operatorname{sg}$ mean stop-gradient. Then

$$
\widetilde E(\theta)=\frac1M\sum_s
\left[a_\theta(x_s)
+\operatorname{sg}(a_\theta(x_s)-\bar a)
\bigl(L_s-\operatorname{sg}(L_s)\bigr)\right]
$$

has forward value $\bar a$, while its derivative contains both the direct estimator derivative and the centred score contribution. This explains the role of stop-gradient in the general calculation. Applying it to the entire estimator would discard the first term.

The [positive squared estimator](../../concepts/operators/squared-estimators.md) is an important example. Its local quantity is $|A_{\mathrm{loc},\theta}(x)|^2$. Although its expectation represents $\langle A^\dagger A\rangle$ when the support conditions are satisfied, this quantity is not the ordinary local estimator of the explicit product $A^\dagger A$. The squared-estimator path therefore includes the direct derivative. It must not reuse the plain covariance formula with $|A_{\mathrm{loc}}|^2$ substituted as though it were a linear local estimator.

## What registration does and does not provide

For a registered observable with a valid Hermitian linear estimator, the covariance path can consume the registered local values. Setting `is_hermitian=True` is a mathematical assertion by the implementer. It is not an instruction to make an arbitrary loss function differentiable.

The automatic selector uses covariance for supported Hermitian linear operators under ordinary Born sampling. At other `machine_pow` values, its generic covariance selection is restricted to diagonal observables. `use_covariance=True` requests that path explicitly and is rejected when the operator is known to be incompatible. Changing the sampler exponent also changes the measured distribution unless a supported importance correction is used.

`use_covariance=False` requests the general supported gradient path. In the current implementation, that path understands discrete connectivity, supported operator expressions, squared operators, and specialised penalty or residual handling. **It does not automatically differentiate an arbitrary function registered with `register_local_estimator`.** For an otherwise unsupported `AbstractOperator`, a non-covariance gradient raises an error. For a discrete operator with a specialised registration, the general path evaluates its connectivity rather than calling the custom rule with trial parameters.

This is why a specialised discrete estimator must agree with its connectivity. A newly invented parameter-dependent statistical quantity needs its own explicitly derived differentiation implementation, or an expression in one of the supported operator forms. The public `expect` and `expect_and_grad` functions are not generic `.dispatch` registration interfaces. The separate [penalty interfaces](../../concepts/operators/penalties.md) apply to their documented cost constructions.

## Complex parameters and useful verification

For a real objective and a complex parameter $z=a+ib$, neuraLQX represents the update gradient as $\partial E/\partial a+i\partial E/\partial b$. Real parameter leaves receive real derivatives. The [VQS expectation guide](../../concepts/vqs/expectations.md) explains this convention alongside forces and preconditioning. Check both real directions separately when verifying a custom complex calculation.

Use a tiny enumerated space to compare the normalised expectation and its derivative with an independent dense expression. Include a nonzero parameter value, a complex wavefunction when supported, and a case where the local estimator changes with the parameters. Repeating only a constant-wavefunction check can miss a dropped direct derivative or a missing complex conjugation. After that, compare chunk sizes and sampled estimates with the exact reference, allowing for Monte Carlo uncertainty.
