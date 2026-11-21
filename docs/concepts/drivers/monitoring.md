---
title: Monitoring and callbacks
---

# Monitoring and callbacks

A useful run record should explain both how the objective evolves and how reliable its estimates are. The loss alone cannot distinguish a well-converged calculation from one with poorly mixed chains or an insufficient sample batch.

The driver can measure additional observables, retain statistics, and pass each iteration through [callbacks](../callbacks/index.md). These mechanisms share a precise timing: ordinary loss and observable measurements happen before the parameter update for that iteration.

## Measure observables during optimisation

Pass a dictionary of named operators to `obs`. These operators contribute measurements to the record, but they do not contribute to the optimised objective or its gradient.

This example minimises the expected squared charge and also records the mean charge. A [`RuntimeLogger`](../../api/public/neuralqx/utils/io/logging/index.md#neuralqx.utils.io.logging.RuntimeLogger) retains the resulting numerical histories.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator
from neuralqx.utils.io.logging import RuntimeLogger

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=6),
    n_samples=128,
    n_discard_per_chain=16,
    sampler_seed=7,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge,
    optax.adam(0.01),
    variational_state=state,
    loss_name="ChargeSquared",
)
logger = RuntimeLogger()
history = driver.run(
    4, obs={"Charge": charge}, logger=logger, store_history=False
)

print(logger["ChargeSquared"]["Mean"].steps)
print(len(logger["Charge"]["Sigma"]))
print(history)
final_estimates = driver.estimate({"ChargeSquared": charge @ charge})
print(final_estimates["ChargeSquared"].n_samples == state.n_samples)
```

```text
[0, 1, 2, 3]
4
[]
True
```

During each ordinary iteration, the loss and these observables reuse the state's current sample batch. Every additional operator still requires its own estimator work. To measure an expensive observable only occasionally, use the [periodic measurement callback](../callbacks/logging.md#measure-an-observable-less-often) rather than expecting a logger's retention frequency to avoid that work.

Choose observable names distinct from `loss_name` and the diagnostic names. The driver merges observables into the same dictionary, so reusing a key can replace another entry.

## Match a record to the correct parameters

If the accepted-update counter is $t$, the loss and `obs` entries are evaluated at $\theta_t$. The optimiser then produces $\theta_{t+1}$. This matters when comparing logs with saved wavefunctions.

| Quantity or event | What it describes |
| --- | --- |
| Loss and `obs` in record `t` | Measurements at the parameters before update `t` |
| `update_computed` callback | Those measurements and the proposed direction, before any parameter update |
| `step_finished` callback | The updated state if accepted, alongside the earlier measurements |
| `driver.loss_stats` after a run | The last computed loss, which is not automatically remeasured after the update |
| `driver.estimate({...})` after a run | Measurements of the current final state |

For an accepted step, `context.step` remains $t$ in `step_finished`, while `context.driver.step_count` is $t+1$. A rejected attempt retains $\theta_t$ and leaves the counter unchanged. The callback's `context.accepted` distinguishes the two cases.

A logger receives the pre-update record together with the current state after the attempt. Saving that state from a logger or end-of-step callback therefore does not save the parameters used for the recorded loss. See [checkpoint timing](checkpointing.md#choose-when-to-save) if you want to associate a best measured value with its exact parameters.

## Read the diagnostics together

The native [`Stats`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.Stats) object contains the mean, local-value variance, standard error, sample count, and split-chain diagnostic. In a runtime log these become `Mean`, `Variance`, `Sigma`, `N`, and `R_hat`. The [sampling diagnostics guide](../samplers/diagnostics.md) explains their interpretation and limitations.

`Sigma` is the estimated uncertainty of the sampled mean. It does not quantify bias from inadequate equilibration, a finite physical cutoff, an unsuitable network, or an approximate operator calculation. Likewise, a small observed variance is not by itself proof that the chains have explored every important region.

For a converged Born estimate of an ordinary Hermitian operator, its local-estimator variance is related to the quantum variance. A zero exact variance identifies an eigenstate, but it does not identify which eigenstate. For a squared residual or an expectation-level penalty, the reported estimator variance must be interpreted according to that estimator. See [expectations and gradients](../vqs/expectations.md) and [squared estimators](../operators/squared-estimators.md).

The driver also records `acceptance` when the single state's sampler provides it. A plausible acceptance rate does not establish mixing. Compare chains, increase the sample count, and repeat measurements at fixed final parameters before interpreting a change smaller than the sampling uncertainty.

## Live feedback and timing

For a terminal display, use `show_progress=True`. For a local browser dashboard, attach [`LiveMonitoringCallback`](../../api/public/neuralqx/callbacks/index.md#neuralqx.callbacks.LiveMonitoringCallback), described in [built-in callbacks](../callbacks/built-in.md#live-monitoring). It displays existing scalar measurements and does not request new operator expectations.

The `wallclock` log entry is a value from Python's `time.perf_counter()`. Its differences can help describe the recorded run, but it is neither a calendar timestamp nor a measurement of one device kernel. Compilation, logging, and [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html) affect what host-side timings include.

Printing device values or converting them to Python or NumPy can wait for outstanding computation. Reduce display and logging frequency when that overhead matters. For detailed timing, use the [profiling guide for scientists](../../developers/profiling/for-scientists.md).

## Decide how a run should stop

A fixed iteration budget is straightforward, but longer calculations may need a target residual, a time budget, or a check for invalid numerical values. [Controlling a run](../callbacks/control.md) shows how to make those decisions at the appropriate event.

Use scientific criteria together with the sampling diagnostics. A brief plateau in a noisy objective is not enough evidence of convergence, and a lower finite-sample mean after one update does not prove that the underlying expectation decreased.
