---
title: Optimisers
---

# Optimisers

A variational simulation changes the parameters of a wavefunction until it describes the state you are looking for. The [Hilbert space](../hilbert/index.md) fixes the allowed configurations, the [neural network](../networks/index.md) assigns their amplitudes, and the [objective](../operators/penalties.md) specifies what should be minimised. The optimiser decides how the parameters move in response to that objective.

In a canonical LQG calculation, the objective might measure a constraint residual, combine several constraints, or include a penalty that selects a physically relevant state. A small objective is meaningful only in relation to that choice. Optimisation cannot compensate for a missing term in the model, and reaching a stationary point does not by itself establish that every physical constraint is satisfied.

## From samples to an update

Write the objective as $\mathcal L(\theta)$, where $\theta$ collects the trainable parameters. At each iteration, the [sampler](../samplers/index.md) draws configurations from the current wavefunction. The variational state uses those configurations to estimate the objective and its gradient,

$$
g_t=\nabla_\theta\mathcal L(\theta_t).
$$

The gradient says how the objective changes under small parameter variations. A **preconditioner** can transform it into a more useful direction $d_t$. The optimiser then converts that direction into an increment $\Delta\theta_t$, possibly using information from previous iterations,

$$
\begin{aligned}
g_t&\longrightarrow d_t,\\
(d_t,\text{optimiser state})&\longrightarrow\Delta\theta_t,\\
\theta_{t+1}&=\theta_t+\Delta\theta_t.
\end{aligned}
$$

neuraLQX uses [Optax](optax.md) for the optimiser and provides [stochastic reconfiguration](sr.md), usually called SR, as a preconditioner. These are separate arguments to the [`VMC`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.VMC) driver. Without a preconditioner, $d_t=g_t$. With SR, the direction is obtained by solving a system involving the geometry of the wavefunction.

The driver performs this sequence and refreshes the sample batch for each iteration. SR uses the same configurations as the gradient. After the update, subsequent sampling follows the changed wavefunction. This connection between sampling and optimisation is why [chain diagnostics](../samplers/diagnostics.md) remain relevant throughout a training run.

## A small optimisation calculation

The following U(1) example minimises

$$
H=\hat q^2-\frac14(R+R^\dagger).
$$

Here [`NumberOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) measures the charge on one edge, and [`U1HolonomyOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.U1HolonomyOperator) couples neighbouring charges with the [cyclic cutoff convention](../lqx/u1.md#edge-and-loop-holonomies). This small model makes it easy to see the optimisation setup. In a larger calculation, the objective can instead come from an [LQX model](../lqx/index.md) or from your own operator expression.

```python
import jax.numpy as jnp
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator, U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=3
)

charge = NumberOperator(hilbert, edge=0)
shift = U1HolonomyOperator(hilbert, edge=graph.edges[0])
objective = charge @ charge - 0.25 * (shift + shift.adjoint)

sr = nqx.optimizer.SR(diag_shift=0.01)
driver = nqx.driver.VMC(
    objective,
    optax.sgd(learning_rate=0.02),
    variational_state=state,
    preconditioner=sr,
)
driver.run(n_iter=5)

print(driver.step_count)
print(bool(jnp.isfinite(state.expect(objective).mean)))
```

```text
5
True
```

The five iterations demonstrate a complete calculation. They are not a convergence criterion. For a scientific result, examine the objective, the observables that matter to your model, and their sampling uncertainty over a substantially longer run.

Plain [`optax.sgd`](https://optax.readthedocs.io/en/latest/api/generated/optax.sgd.html) multiplies the SR direction by a negative learning rate. The [`SR`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR) object supplies the geometry and linear solve. Its `diag_shift` stabilises that solve. Changing the learning rate changes how far the parameters move, while changing the shift changes which directions SR favours. They should be tuned with these different roles in mind.

## Choosing the pieces

For a first SR calculation, plain SGD gives the most direct connection between the solved direction and the parameter update. Adam is also useful as a baseline without SR. Combining Adam with SR is permitted, but introduces another transformation of the direction. The [optimiser comparison](sr-optimisers.md) explains that choice, including momentum, clipping, and weight decay.

The [SR guide](sr.md) develops the natural-gradient and imaginary-time interpretations. The [quantum geometric tensor guide](qgt.md) explains how the metric is estimated, how complex wavefunctions are handled, and when to use a dense or matrix-free representation. The [linear solver guide](solvers.md) then covers the numerical choices used to obtain the direction.

Once a fixed configuration behaves sensibly, [schedules](schedules.md) let you vary the learning rate and diagonal shift over the run. The [VMC driver section](../drivers/index.md) provides the wider context for running, monitoring, and continuing a simulation.
