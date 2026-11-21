---
title: Running simulations
---

# Running simulations

Construct the [variational state](../vqs/state.md) before the driver. The state determines the Hilbert space, network, sampler, sample count, and computational chunk sizes. The driver then needs a compatible objective and an Optax optimiser. A preconditioner is optional.

This division lets you change the optimisation method without rebuilding the physical setup. It also means that replacing the driver creates a new optimiser state, even when the same wavefunction is supplied.

## Configure the objective and optimiser

[`VMC`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.VMC) takes `hamiltonian` and `optimizer` as its first two arguments, with `variational_state` passed by name. Use `loss_name` to give the objective a meaningful label in logs and progress displays. Its default is `"Energy"`.

The objective must be compatible with the state's Hilbert space. An operator list or tuple is interpreted as a sum, not as several independent optimisation targets. For extra measurements that should not change the gradient, use the `obs` argument to `run`, as described in [monitoring](monitoring.md).

Use a real objective appropriate to the calculation. The driver delegates its derivative to [`expect_and_grad`](../vqs/expectations.md), including the specialised paths for [squared estimators](../operators/squared-estimators.md) and [expectation penalties](../operators/penalties.md). It does not turn a general complex expectation into a well-defined minimisation problem on your behalf.

## Run with stochastic reconfiguration

Pass [`SR`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR) as `preconditioner`. Plain SGD makes the relation between the SR direction and the parameter step particularly direct,

$$
\begin{aligned}
(G_t+\lambda I)d_t&=g_t,\\
\theta_{t+1}&=\theta_t-\eta_t d_t.
\end{aligned}
$$

Here $G_t$ is the sampled wavefunction metric, $\lambda$ is `diag_shift`, and $\eta_t$ is the learning rate. The [optimiser comparison](../optimisers/sr-optimisers.md) explains what changes when SR is combined with momentum or an adaptive optimiser.

The following calculation uses a scheduled learning rate and a fixed SR shift. The two calls to `run` continue one optimisation process.

```python
import jax.numpy as jnp
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.RBM(1, hidden_features=4, complex_output=False, rngs=2),
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=3,
)
charge = NumberOperator(hilbert, edge=0)
learning_rate = optax.linear_schedule(0.02, 0.005, transition_steps=100)
sr = nqx.optimizer.SR(diag_shift=0.05)
driver = nqx.driver.VMC(
    charge @ charge,
    optax.sgd(learning_rate),
    variational_state=state,
    preconditioner=sr,
)

driver.run(3)
driver.run(2)
estimates = driver.estimate({"ChargeSquared": charge @ charge})
print(driver.step_count)
print(bool(jnp.isfinite(estimates["ChargeSquared"].mean)))
```

```text
5
True
```

`driver.optimizer_state` holds the optimiser's counters and any moments. Reusing the driver preserves these values. Creating another `VMC` initialises them again. An SR shift can also be scheduled, using the driver's current accepted-update count. See [learning rates and schedules](../optimisers/schedules.md) for the relationship between the two counters.

## What `run` counts

[`run`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.AbstractVariationalDriver.run) performs up to `n_iter` attempted iterations during that call. With no stopping or rejected updates, `run(100)` makes one hundred parameter updates. A later `run(50)` continues for fifty more. It does not interpret 50 as an absolute target step.

`driver.step_count` counts accepted parameter updates and starts at zero. If a [callback rejects an update](../callbacks/control.md#skip-an-update), that attempt still consumes one of the requested iterations, but neither `step_count` nor the Optax state advances. The next attempt uses the same step label with a newly collected sample batch. Sampling and preconditioning work from the rejected attempt has already happened.

The loop resets the state's cached samples before computing each gradient. Normally, persistent chain endpoints are retained and sampling continues from them, with the state's configured discard count applied. A sampler configured to reset chains changes that behaviour. The [state lifecycle](../vqs/state.md) and [sampling diagnostics](../samplers/diagnostics.md) explain the distinction.

## Progress, records, and return values

`show_progress=True` enables a progress bar. `progress_every` controls how often its displayed objective is refreshed, and `progress_min_interval` limits refresh frequency in seconds. Neither option changes the number of optimisation updates or the measurements requested through `obs`.

`run` returns `driver.history`, a list of retained log dictionaries. The list accumulates across calls. Set `store_history=True` to retain a record for each completed attempt, or `False` to avoid this retention. If you keep a reference to the returned list, a later run appends to that same list.

With the default `store_history=None`, ordinary `VMC` enables history when observables are requested or a callback declares that it stores history. Merely enabling the progress bar, live monitoring, or a runtime logger does not enable driver history. An empty return list therefore does not mean that the simulation did no work. The [logging guide](../callbacks/logging.md) explains how to keep a separate, optionally thinned record with a logger.

The current `run` signature accepts `step_size`, but the loop does not use it to thin measurements or change its update cadence. Use the explicit logger and callback frequency controls instead.

## Use the iterator when managing the loop yourself

[`iter`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.AbstractVariationalDriver.iter) exposes a smaller interface that applies updates and yields their step labels. It does not run callbacks, measure an `obs` dictionary, write logs, or append history.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=4),
    n_samples=32,
    sampler_seed=5,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)

labels = list(driver.iter(3))
print(labels)
print(driver.step_count)
```

```text
[0, 1, 2]
3
```

Use the default `step_size=1` for one yield per update. At a yield, parameters have already been updated, but the counter increment happens when the iterator resumes. Fully consuming it, as above, completes that bookkeeping. For controlled early stopping, prefer `run` with a callback rather than abandoning an iterator between its yield and counter increment.

For `iter` specifically, a larger `step_size` groups updates and can round the total up to a complete group when it does not divide `n_steps`. This is separate from the unused `run(step_size=...)` argument.

## Prepare a longer calculation

Check the objective and its scale on a small problem before increasing the graph or representation cutoff. Choose sample counts from the quality of the statistics, and choose [chunk sizes](../vqs/batching.md) from memory and execution behaviour. They are different controls.

Then add [monitoring](monitoring.md) and [periodic state snapshots](checkpointing.md). For hardware setup and placement, continue with [distributed simulations](../../distributed/index.md). The driver interface remains the coordination point, while the state and JAX execution configuration determine where arrays and calculations live.
