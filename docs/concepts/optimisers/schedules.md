---
title: Learning rates and schedules
---

# Learning rates and schedules

A schedule is a function of the update count. It lets a run use a different learning rate or SR diagonal shift at different stages, without rebuilding the optimiser on every iteration. [Optax supplies schedules](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html) that can be passed directly to the relevant neuraLQX configuration.

The two most useful quantities to schedule have different roles. The learning rate sets the size of the parameter step. The [SR shift](sr.md#stabilise-the-metric) changes the metric system from which the direction is computed.

## Learning rate and shift are different controls

For plain SGD with SR,

$$
\begin{aligned}
(G_t+\lambda_t I)d_t&=g_t,\\
\theta_{t+1}&=\theta_t-\eta_t d_t.
\end{aligned}
$$

Changing $\eta_t$ scales the whole direction. Changing $\lambda_t$ changes the relative response along metric eigenvectors. In an eigenvector with eigenvalue $s_k$, the update coefficient contains

$$
\frac{\eta_t}{s_k+\lambda_t}.
$$

Reducing the shift can therefore increase some components even while the learning rate decreases. A schedule for one quantity is not automatically a substitute for a schedule for the other.

Early in a run, a conservative step and stronger regularisation can be useful while the state and chains settle. Later, weaker regularisation may allow more detailed use of the geometry. This is a strategy to test, not a requirement that both quantities decrease monotonically. If the sample batch remains noisy, continuing to reduce the shift can make the calculation less reliable.

## Schedule the learning rate

An Optax optimiser accepts a callable in place of a constant learning rate. Its internal counter starts at zero and advances when its update function is called. A warmup followed by cosine decay is one possible choice.

```python
import optax

learning_rate = optax.warmup_cosine_decay_schedule(
    init_value=0.001,
    peak_value=0.02,
    warmup_steps=20,
    decay_steps=200,
    end_value=0.002,
)
optimiser = optax.sgd(learning_rate=learning_rate)

print([round(float(learning_rate(t)), 4) for t in (0, 20, 200)])
```

```text
[0.001, 0.02, 0.002]
```

[`warmup_cosine_decay_schedule`](https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.warmup_cosine_decay_schedule.html) reaches the peak after `warmup_steps` and the final value at `decay_steps`. Here `decay_steps=200` includes the warmup, leaving 180 steps for the cosine part. The three values printed above let you check the intended timeline before running a simulation.

The same schedule can be passed to Adam, but the resulting parameter increments also depend on its moment estimates. A learning-rate curve alone does not specify the actual step sizes of an adaptive optimiser.

## Schedule the SR shift

[`SR`](../../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR) evaluates a callable `diag_shift` at the `step` supplied by the driver. An exponential schedule with a positive floor is convenient when gradually reducing regularisation.

```python
import neuralqx as nqx
import optax

diagonal_shift = optax.exponential_decay(
    init_value=0.1,
    transition_steps=100,
    decay_rate=0.5,
    end_value=0.001,
)
sr = nqx.optimizer.SR(diag_shift=diagonal_shift)

print([round(float(diagonal_shift(t)), 4) for t in (0, 100, 1000)])
```

```text
[0.1, 0.05, 0.001]
```

With these settings, [`exponential_decay`](https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.exponential_decay.html) gives

$$
\lambda_t=\max\!\left(0.001,\,0.1\,2^{-t/100}\right).
$$

`staircase=True` would instead change it in discrete stages. A [linear schedule](https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.linear_schedule.html) can interpolate to a chosen final shift over a fixed number of steps. Whichever form you use, check that it remains positive when your solver requires a positive definite system. The presence of a callable does not itself impose bounds on its values.

When calling SR manually, pass an explicit step, for example `sr(state, gradient, step=t)`. Omitting `step` when `diag_shift` is callable raises an error. The [VMC driver](../../api/public/neuralqx/driver/index.md#neuralqx.driver.VMC) supplies its current update count automatically. The current built-in SR path supports scheduling `diag_shift`, but does not implement `diag_scale`.

## Use both schedules in a calculation

This complete example combines a decaying learning rate with a decaying shift. It uses a small expected charge-square objective, as in the [Optax guide](optax.md#adam-without-sr).

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
    sampler, network, n_samples=256, n_discard_per_chain=32, seed=13
)
charge = NumberOperator(hilbert, edge=0)

learning_rate = optax.linear_schedule(
    init_value=0.02, end_value=0.005, transition_steps=100
)
diagonal_shift = optax.exponential_decay(
    init_value=0.05, transition_steps=50, decay_rate=0.5, end_value=0.005
)
sr = nqx.optimizer.SR(diag_shift=diagonal_shift)
driver = nqx.driver.VMC(
    charge @ charge,
    optax.sgd(learning_rate=learning_rate),
    variational_state=state,
    preconditioner=sr,
)

driver.run(n_iter=5)
driver.run(n_iter=5)

print(driver.step_count)
print(bool(jnp.isfinite(state.expect(charge @ charge).mean)))
```

```text
10
True
```

The second `run` continues with the same driver and optimiser state. It does not restart either schedule. Ten accepted updates have used schedule indices zero through nine, and the next update will use index ten.

The Optax counter and the driver's `step_count` belong to different objects. In the ordinary VMC loop, one optimiser update is made for each accepted driver step, so the counts advance together. If a callback skips an update, the standard `run` loop advances neither. More elaborate Optax transformations, such as accumulation wrappers, can have their own internal counters and should be checked according to their update semantics.

Reconstructing an optimiser or driver starts new state. Restoring only network parameters is therefore insufficient to continue the same learning-rate history or Adam moments. Keep the optimiser state and update count with the simulation state when implementing a [checkpoint workflow](../drivers/checkpointing.md). The method `driver.reset_step()` refreshes cached sampling data for a step and does not reset the optimisation counter.

## Write a custom schedule

A schedule passed through a compiled Optax update should use JAX-compatible array operations. For example, this function keeps a shift constant initially and then decays it to a positive floor.

```python
import jax
import jax.numpy as jnp


def diagonal_shift(step):
    step = jnp.asarray(step)
    elapsed = jnp.maximum(step - 50, 0)
    return jnp.maximum(0.002, 0.05 * jnp.exp(-elapsed / 100.0))


compiled_schedule = jax.jit(diagonal_shift)
print([round(float(compiled_schedule(t)), 3) for t in (0, 50, 1000)])
```

```text
[0.05, 0.05, 0.002]
```

This callable can be supplied as `diag_shift=diagonal_shift`. Using JAX operations also makes it suitable for evaluation with array-valued counters. A Python `if` on a traced counter can fail inside a compiled optimiser update.

A predetermined schedule does not observe the objective or sampling quality. If you want a change triggered by convergence, uncertainty, or another physical quantity, use the [run-control and callback interfaces](../callbacks/control.md) to make that decision with the required information. Keep recording the current learning rate and shift alongside your observables so the resulting trajectory can be interpreted and reproduced.
