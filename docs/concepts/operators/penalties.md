---
title: Penalties and objectives
---

# Penalties and objectives

A variational calculation often has more than one requirement. You may want to reduce a constraint residual while discouraging a vanishing volume, favouring a particular charge sector, or bringing an observable's mean towards a chosen value. A penalty turns such a requirement into an additional contribution to the objective.

The distinction to keep in mind is **where the penalty is applied**. A matrix expression such as $(A-a_*I)^2$ acts on the wavefunction. An expectation-level penalty such as $(\langle A\rangle-a_*)^2$ acts on a scalar measurement of the state. These choices generally describe different objectives.

## A function of the expectation

Let $m_\theta=\langle A\rangle_\theta$. An expectation-level penalty contributes

$$
\mathcal P(\theta)=f(m_\theta).
$$

The state first estimates the parent expectation, then applies $f$. For a nonlinear function,

$$
f\!\left(\mathbb E_p[A_{\mathrm{loc}}]\right)
\ne\mathbb E_p\!\left[f(A_{\mathrm{loc}})\right]
$$

in general. Transforming each local value and then averaging is therefore not a substitute for an expectation-level penalty.

The supplied wrappers inherit their Hilbert space from their parent. They describe objectives for the expectation engine and do not provide connectivity or a matrix representation. Evaluate them with `state.expect(penalty)` and differentiate them with `state.expect_and_grad(penalty)`.

## Linear penalties

[`PenaltyCost`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.PenaltyCost) supplies the linear case,

$$
f(m)=\lambda m,
\qquad f'(m)=\lambda.
$$

For a computational operator this has the same mean as measuring `factor * operator`. The wrapper is also useful for objects such as [`Squared`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.Squared), which do not participate in scalar operator algebra. For example, a weighted charge residual is

$$
\mathcal P=\lambda\langle\hat q_e^2\rangle.
$$

```python
import neuralqx as nqx
from neuralqx.operator import PenaltyCost
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
charge = NumberOperator(hilbert, edge=0)
weighted_residual = PenaltyCost(charge.squared(), factor=0.2)

print(float(weighted_residual.expectation_value(0.5)))
print(float(weighted_residual.expectation_gradient(0.5)))
```

```text
0.1
0.2
```

The scalar calls illustrate the cost map and its derivative at a given parent mean. In a calculation, passing `weighted_residual` to the variational state supplies that mean from the samples.

For a nonnegative penalty to be minimised, choose a real nonnegative weight and a nonnegative parent objective. A linear penalty on a signed charge is a bias towards one sign, rather than a penalty on its magnitude. Similarly, weighting `A.squared()` by $\lambda$ differs from squaring `lambda * A`, which introduces a factor $|\lambda|^2$.

## Inverse expectation penalties

[`InverseExpectationCost`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.InverseExpectationCost) penalises a small real expectation through

$$
f(m)=\frac{\lambda}{(\alpha+m+\varepsilon)^2},
\qquad
f'(m)=-\frac{2\lambda}{(\alpha+m+\varepsilon)^3},
\qquad m=\operatorname{Re}\langle A\rangle.
$$

Its parent must declare itself Hermitian. `factor` is $\lambda$, `alpha` is an offset, and `eps` is an additional small offset with default $10^{-24}$. The denominator is not clipped or replaced by an absolute-value floor. The full expression remains singular if $\alpha+m+\varepsilon=0$.

For a nonnegative observable such as volume, a positive offset can make the cost finite at zero while still favouring larger means. The scale and units of the offset should match the observable. A raw volume, its square, and a dimensionless rescaling define different penalties.

The following uses a charge projector as a bounded example. The parent mean is a probability in $[0,1]$, and `alpha=1.0` keeps the denominator away from zero.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator import InverseExpectationCost
from neuralqx.operator.prebuilt import ChargeColoringOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=128, n_discard_per_chain=16, seed=1
)
projector = ChargeColoringOperator(hilbert, site=0, charge=0)
penalty = InverseExpectationCost(projector, factor=0.2, alpha=1.0)

parent_stats = state.expect(projector)
penalty_stats = state.expect(penalty)
np.testing.assert_allclose(
    penalty_stats.mean, penalty.expectation_value(parent_stats.mean)
)
print(bool(penalty_stats.mean > 0))
```

```text
True
```

Both measurements reuse the same cached sample batch. This verifies that the cost is evaluated from the estimated probability, rather than averaging inverse costs of the individual zero-or-one projector values.

## Mean targets and fluctuations

For a Hermitian observable, a penalty on its mean is

$$
\mathcal P_{\mathrm{mean}}
=\lambda(\langle A\rangle-a_*)^2.
$$

By contrast, penalising the shifted operator's action gives

$$
\begin{aligned}
\mathcal P_{\mathrm{spread}}
&=\lambda\langle(A-a_*I)^2\rangle\\
&=\lambda\left[(\langle A\rangle-a_*)^2
+\operatorname{Var}_\psi(A)\right].
\end{aligned}
$$

The second objective also suppresses fluctuations. A state can have the desired mean and still pay a nonzero spread penalty. Use a shifted operator and its square when that additional requirement is intended.

A custom expectation-level target can be expressed by subclassing `PenaltyCost`. The inherited affine-estimator method uses the value and derivative supplied below. This example checks that the parent is Hermitian and implements a real target mean.

```python
import neuralqx as nqx
import jax.numpy as jnp
from neuralqx.operator import PenaltyCost
from neuralqx.operator.prebuilt import ChargeColoringOperator
from neuralqx.utils.struct import field


class TargetExpectationCost(PenaltyCost):
    target: float = field(static=True, default=0.0)

    def __post_init__(self):
        super().__post_init__()
        if not self.operator.is_hermitian:
            raise ValueError('The target observable must be Hermitian.')

    def expectation_value(self, parent_expectation):
        difference = jnp.real(parent_expectation) - self.target
        return self.factor * difference**2

    def expectation_gradient(self, parent_expectation):
        difference = jnp.real(parent_expectation) - self.target
        return 2.0 * self.factor * difference


graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
projector = ChargeColoringOperator(hilbert, site=0, charge=0)
penalty = TargetExpectationCost(projector, factor=4.0, target=0.5)
print(float(penalty.expectation_value(0.25)))
print(float(penalty.expectation_gradient(0.25)))
```

```text
0.25
-2.0
```

Pass this object to `state.expect` or `state.expect_and_grad` just like a supplied penalty. The base class recognises subclasses as nonlinear unless they explicitly provide a different linearity rule. More specialised estimator behaviour can be implemented through the [custom estimator interfaces](../../developers/estimators/index.md).

## How gradients and reported errors are formed

For a scalar penalty, the derivative follows the chain rule,

$$
\partial_\theta f(m_\theta)
=f'(m_\theta)\,\partial_\theta m_\theta.
$$

For the ordinary Hermitian parent case, the expectation engine uses an affine transformation of the parent local values,

$$
P_{\mathrm{loc}}(\sigma)
=f(m)+f'(m)\bigl(A_{\mathrm{loc}}(\sigma)-m\bigr).
$$

Its mean is $f(m)$, and its fluctuations carry the local sensitivity $f'(m)$. It is not the pointwise function $f(A_{\mathrm{loc}})$. The supplied wrappers implement this through `local_value_coefficients`, which returns the scale and shift.

The reported variance and error are transformed by $|f'(m)|^2$ and $|f'(m)|$ respectively. For a nonlinear penalty, these describe local error propagation around the estimated mean. Evaluating $f$ on a finite-sample mean can also introduce bias. Check sample-size convergence when the penalty is strongly curved, particularly near a small denominator. These statistics are not the quantum variance of a new penalty matrix.

Let `expect_and_grad` select the derivative route. A penalty whose parent is a [factorised square](squared-estimators.md) still needs the full derivative of that parent estimator. Wrapping it in a penalty does not make the ordinary Hermitian covariance shortcut valid.

## Combine objectives and optimise

To add computational operators, squared objectives, and penalties, pass a nonempty tuple or list to the variational state or VMC driver. This represents a sum of objective contributions. `SumOperator` and the computational `+` operator accept computational terms, so they are not the way to combine these wrappers.

Here a BF curvature residual is combined with an inverse penalty favouring a nonzero probability for charge zero on one component. The short run demonstrates the objective interface. It is not intended as a converged physical calculation.

```python
import neuralqx as nqx
import optax
from neuralqx.operator import InverseExpectationCost, PenaltyCost
from neuralqx.operator.prebuilt import ChargeColoringOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
model = nqx.lqx.LqxWCL1D(hilbert)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=128, n_discard_per_chain=16, seed=1
)

projector = ChargeColoringOperator(hilbert, site=0, charge=0)
constraint_loss = PenaltyCost(model.constraint.squared(), factor=1.0)
probability_cost = InverseExpectationCost(projector, factor=0.05, alpha=0.5)
objective = (constraint_loss, probability_cost)

driver = nqx.driver.VMC(
    objective,
    optax.sgd(learning_rate=0.01),
    variational_state=state,
    loss_name='Objective',
)
driver.run(2, obs={'zero_charge_probability': projector}, show_progress=False)
print(driver.step_count)
```

```text
2
```

The objective is what the driver minimises. The `obs` dictionary requests a separate measurement and does not add another penalty. Keep both the total objective and its physical components available when interpreting a run, since a decrease in their weighted sum can accompany an increase in one component.

The weights set a tradeoff between physical requirements and should reflect their scales and units. A soft penalty does not impose an exact restriction on every configuration. If a condition defines which states are allowed, consider encoding it in the [Hilbert space constraints](../hilbert/constraints.md) instead.
