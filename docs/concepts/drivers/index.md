---
title: VMC drivers
---

# VMC drivers

A variational Monte Carlo driver turns a wavefunction and an objective into an optimisation calculation. The [variational state](../vqs/index.md) knows how to sample, measure operators, and differentiate an expectation. The driver repeats those operations, passes the resulting direction to an optimiser, and updates the state's parameters. It also provides the places where you can record measurements, save a state, or stop a run.

[`VMC`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.VMC) optimises one variational state. [`MultiVMCDriver`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.MultiVMCDriver) optimises several states together and can penalise their overlap. Both use the same run and callback interfaces.

## What is being minimised

For a Hermitian operator $H$, the usual objective is the normalised expectation

$$
\mathcal L(\theta)
=\frac{\langle\psi_\theta|H|\psi_\theta\rangle}
{\langle\psi_\theta|\psi_\theta\rangle}.
$$

In a loop quantum gravity calculation, the objective may instead measure a constraint residual or combine several [penalties](../operators/penalties.md). An [LQX model](../lqx/structure.md#quadratic-constraints) can construct a suitable quadratic objective, but using an LQX model is optional. The driver accepts the objective directly.

The constructor calls this argument `hamiltonian`. That name does not require your calculation to be an ordinary ground-state problem, and it does not automatically square a constraint. Choose the operator or objective whose minimum has the scientific meaning you want. In particular, the expectation of a constraint can vanish by cancellation even when its action on the state is nonzero. The [squared estimator guide](../operators/squared-estimators.md) explains objectives based on the norm of that action.

## One iteration

At parameters $\theta_t$, the state draws configurations and estimates the objective and its gradient $g_t$. A preconditioner can transform this gradient into a direction $d_t$. With plain SGD, the final update is

$$
\begin{aligned}
g_t&\approx\nabla_\theta\mathcal L(\theta_t),\\
d_t&=P_t(g_t),\\
\theta_{t+1}&=\theta_t-\eta_t d_t.
\end{aligned}
$$

Without a preconditioner, $d_t=g_t$. With [stochastic reconfiguration](../optimisers/sr.md), the direction comes from a regularised quantum-geometric linear system. An adaptive optimiser such as Adam introduces its own moment estimates and update rule, so its parameter increment is not simply $-\eta_t d_t$.

Sampling, differentiation, preconditioning, and the optimiser have separate jobs. More Monte Carlo samples can improve the estimates, but they do not change the network's expressiveness. A smaller learning rate changes the parameter step, but it does not repair poorly mixed chains. Keeping these distinctions in mind makes the [monitoring guide](monitoring.md) easier to interpret.

The iteration index labels optimisation updates. It is not physical time. The relation between SR and projected imaginary-time evolution requires additional assumptions, discussed in the [SR derivation](../optimisers/sr.md#the-connection-to-imaginary-time).

## A complete small calculation

This example minimises the expected squared charge on one U(1) edge. Its basis contains $q=-1,0,1$, so the exact minimum is supported on $q=0$. The short run demonstrates the driver interface, rather than establishing convergence.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=0),
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=1,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge,
    optax.sgd(learning_rate=0.02),
    variational_state=state,
    loss_name="ChargeSquared",
)

history = driver.run(5, store_history=True)
print(driver.step_count)
print(len(history))
print(driver.state is state)
```

```text
5
5
True
```

The driver updates the same `state` object that was passed to it. After the run, that state contains the learned parameters and can be used for further [measurements](../vqs/expectations.md), [amplitude evaluation](../vqs/arrays.md), or [weight transfer](../vqs/transfer.md).

Each retained loss was measured before its corresponding update. In particular, `driver.loss_stats` and the alias `driver.energy` describe the last computed objective, not a fresh measurement of the final parameters. Use `driver.estimate({"ChargeSquared": charge @ charge})` when you want a measurement of the state after the last update.

## Where to continue

[Running simulations](running.md) covers construction, SR, repeated runs, and iteration counts. [Monitoring and callbacks](monitoring.md) explains measurements and convergence checks. [Checkpointing](checkpointing.md) shows how to save states during optimisation and distinguishes restoring a wavefunction from restoring the complete optimisation process.

For joint calculations, continue with [multiple states](multiple-states.md). The separate [callbacks section](../callbacks/index.md) develops the event interface, supplied monitoring tools, and ways to control a calculation without rewriting the driver.
