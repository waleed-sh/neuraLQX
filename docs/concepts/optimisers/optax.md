---
title: Optimisation with Optax
---

# Optimisation with Optax

[Optax](https://optax.readthedocs.io/en/latest/) provides the parameter update rules used by neuraLQX. You create an optimiser such as SGD or Adam and pass it to a [VMC driver](../drivers/index.md). The driver owns its evolving state and applies its updates to the [variational state](../vqs/index.md).

An optimiser does not create the samples or define the observable being minimised. Those responsibilities belong to the sampler, variational state, and objective. Keeping them separate lets you compare update rules while leaving the physical problem unchanged.

## The Optax interface

An Optax transformation has two main operations. `init(parameters)` creates its state, which may contain momentum buffers, running second moments, or a schedule counter. `update(direction, optimiser_state, parameters)` returns a parameter increment and the next optimiser state. [`optax.apply_updates`](https://optax.readthedocs.io/en/latest/api/generated/optax.apply_updates.html) adds the increment to the parameters.

For plain [SGD](https://optax.readthedocs.io/en/latest/api/generated/optax.sgd.html),

$$
\Delta\theta_t=-\eta_t d_t.
$$

With no preconditioner, $d_t$ is the estimated objective gradient. With [SR](sr.md), $d_t$ is the result of the SR solve. Optax receives whichever direction the driver supplies. It does not know whether that direction was an ordinary gradient or a natural-gradient direction.

Initialise the optimiser state once and retain each returned state. Reinitialising Adam at every step discards its history. Reinitialising an optimiser with a learning-rate schedule also restarts its counter.

## Adam without SR

[`optax.adam`](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html) uses running first and second moments to adapt the update scale for individual parameters. It is a useful comparison when assessing whether the extra geometric information and solve cost of SR help a particular problem.

This example optimises the expected squared U(1) charge on one edge. [`NumberOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) supplies the charge operator, and `charge @ charge` supplies its matrix square.

```python
import jax.numpy as jnp
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=5
)
charge = NumberOperator(hilbert, edge=0)
objective = charge @ charge

driver = nqx.driver.VMC(
    objective,
    optax.adam(learning_rate=0.003),
    variational_state=state,
)
driver.run(n_iter=5)

print(driver.step_count)
print(bool(jnp.isfinite(state.expect(objective).mean)))
```

```text
5
True
```

Omitting `preconditioner` selects the identity transformation, so Adam receives the objective gradient directly. Adding `preconditioner=nqx.optimizer.SR(...)` makes Adam receive the SR direction instead. The [SR optimiser guide](sr-optimisers.md#adam-and-other-adaptive-optimisers) discusses when that combination is worth trying and why it is a different algorithm.

## Compose transformations deliberately

[`optax.chain`](https://optax.readthedocs.io/en/latest/api/generated/optax.chain.html) applies transformations in the order supplied. For example, global norm clipping can limit the direction before an SGD step.

```python
import jax.numpy as jnp
import optax

parameters = {"weights": jnp.array([1.0, -1.0])}
direction = {"weights": jnp.array([3.0, 4.0])}
optimiser = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.sgd(learning_rate=0.1),
)
optimiser_state = optimiser.init(parameters)
updates, optimiser_state = optimiser.update(
    direction, optimiser_state, parameters
)
new_parameters = optax.apply_updates(parameters, updates)

print([round(float(x), 2) for x in new_parameters["weights"]])
```

```text
[0.94, -1.08]
```

The original direction has norm five. Clipping rescales it to norm one, then SGD multiplies it by $-0.1$. In a driver using SR, this clipping acts **after the SR solve**. It therefore limits the solved direction, not the raw objective gradient. Reversing the chain would instead clip the already scaled increment, which gives the threshold a different meaning.

An ordinary optimiser constructor already includes the sign and learning-rate scaling needed for minimisation. Adding another negative scaling transformation to `optax.sgd(...)` would reverse that sign. When assembling a chain from lower-level transformations, account for the sign once.

## Complex parameters

A complex wavefunction can be represented using real parameters, for example by giving the network separate amplitude and phase outputs. In that case Optax still updates real parameter arrays. Complex parameter arrays require an update rule that supports complex values.

[`expect_and_grad`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.expect_and_grad) returns gradients in the convention expected by neuraLQX optimisers. For a real objective and a complex parameter $z=x+iy$, this packages the real-coordinate derivatives as

$$
g_z=\frac{\partial\mathcal L}{\partial x}
+i\frac{\partial\mathcal L}{\partial y}.
$$

Use this result directly in the driver or an Optax update. Adding a conjugation by analogy with a standalone `jax.grad` example can change the imaginary-coordinate update. The [QGT guide](qgt.md#real-and-complex-wavefunctions) explains the corresponding geometry and the separate meaning of a holomorphic model.

## Which Optax transformations fit the driver

The standard driver calls `optimizer.update(direction, optimizer_state, parameters)`. SGD, Adam, their ordinary schedules, and chains that use this interface fit naturally. An optimiser that requires additional objective evaluations or extra arguments needs additional integration.

For example, the default [`optax.lbfgs`](https://optax.readthedocs.io/en/latest/api/generated/optax.lbfgs.html) uses a line search that needs the objective value, its gradient, and an objective function to evaluate at trial parameters. The standard VMC update does not provide those arguments. A metric-dependent plateau scheduler can similarly need information beyond an iteration counter. Creating an Optax transformation successfully does not establish that these additional inputs are available in the driver.

For ordinary time-dependent learning rates, use the [schedule guide](schedules.md). For decisions that depend on measured simulation quantities, the [callbacks section](../callbacks/control.md) is the place to connect those decisions to the run. The [developer guide to gradients](../../developers/estimators/gradients.md) covers extensions to the differentiation side of the calculation.
