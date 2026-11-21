---
title: Profiling a simulation
---

# Profiling a simulation

The [VMC drivers](../../../concepts/drivers/index.md) already contain annotations for the main parts of an optimisation step. You can record those annotations without changing the sampler, network, objective, or optimiser. The first useful capture is usually a short run with the same physical setup and array sizes as the calculation you want to investigate.

The package option `profile=True` enables automatic capture around `driver.run(...)`. With the default `profile_level="basic"`, this records application regions such as sampling, gradient evaluation, and parameter updates. Native JAX collection and GPU telemetry are separate choices, introduced in [choosing what to record](capture-options.md).

## Capture a complete run

This example minimises a squared charge on a one-edge U(1) space. The small system keeps the demonstration inexpensive. In a research calculation, keep your existing state and objective and apply the same configuration context around its `run` call.

The [`NumberOperator`](../../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) supplies the charge, and [`VMC`](../../../api/public/neuralqx/driver/index.md#neuralqx.driver.VMC) performs three updates. `profile_detail="detailed"` adds estimator and operator stages to the major driver phases.

```python
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
state = nqx.vqs.VariationalState(
    nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
    nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=1),
    n_samples=64,
    n_discard_per_chain=8,
    sampler_seed=2,
)
charge = NumberOperator(hilbert, edge=0)
driver = nqx.driver.VMC(
    charge @ charge, optax.sgd(0.01), variational_state=state
)

with nqx.config.patch(
    profile=True,
    profile_level="basic",
    profile_detail="detailed",
    profile_output_dir="profiles",
    profile_run_id=None,
):
    driver.run(3)

result = driver.profile_result
print(driver.step_count)
print(result.trace_path.is_file())
print(result.manifest_path.is_file())
```

```text
3
True
True
```

The configuration context restores the previous settings when it exits. Each capture in this single-process example receives a generated run identifier, so repeating it creates another directory instead of replacing the first result.

`result.trace_path` is the combined trace that you can open in [Perfetto](https://ui.perfetto.dev/). `result.manifest_path` points to the local manifest, which records capture settings, collection status, and diagnostics. The [reading guide](reading-results.md) explains both files. Keep the generated directory if you want to compare this capture with a later one.

## What the driver includes

A driver capture follows the calculation through sampling, the objective gradient, optional preconditioning, and the Optax update. Requested observables, callback execution, and logging also receive annotations. A profile can therefore reveal that an apparently expensive optimisation step is spending much of its time on an additional measurement or callback.

At `summary` detail, the trace emphasises these broad phases. At `detailed` detail, it separates stages such as sample collection, local estimator evaluation, shared network evaluation, and the SR solve when those paths are used. The [instrumentation guide](../instrumentation.md#the-supplied-annotations) describes the names in more detail.

The driver also supplies the current result arrays to the profiler. At a capture boundary, the default synchronisation policy waits for those results before closing collection. This includes relevant state, gradient, update, optimiser, and logged measurement values. It does not insert a wait at every annotated operation.

## Separate startup from repeated work

The first use of a JAX computation may compile it. Its cost can dominate a short capture even when subsequent steps are fast. This first example deliberately records the complete run, including any compilation that occurs within it.

If you want to understand startup, retain that first iteration. If you want to understand repeated execution, warm up the same workload and then capture later iterations, or use a [recording schedule](capture-options.md#record-a-short-window). Warming up a different graph size, sample shape, network structure, or dtype does not necessarily prepare the computation you intend to measure.

A warmup run performs real updates. It changes parameters, sampler state, and optimiser state. For a controlled comparison between implementations, start each experiment from the same intended initial condition and apply the same warmup procedure. The [state persistence guide](../../../concepts/vqs/saving.md) explains what a saved variational state contains, and the [driver checkpointing guide](../../../concepts/drivers/checkpointing.md) explains the distinction from saving a complete optimisation process.

## Automatic and explicit sessions

Automatic capture is convenient when one bounded `run` call is the experiment. Each subsequent call creates a separate capture when automatic profiling remains enabled.

An explicit [`Profiler`](../../../api/public/neuralqx/profile/index.md#neuralqx.profile.Profiler) is useful when the experiment includes preparation, several driver calls, or analysis outside the driver. A driver borrows an already active session and adds its instrumentation to that session. In this case, retrieve the result from the explicit profiler. `driver.profile_result` reports a session owned by the driver and should not be used to identify the current borrowed capture.

The lower-level `driver.iter(...)` interface can also participate in an explicit session, but it does not create an automatic session merely because `profile=True`. Its lifetime is controlled by the code consuming the iterator. See [profiling your own code](custom-code.md) for explicit sessions and completion boundaries.

For a first investigation, use a short automatic capture, open the trace, and identify which phase deserves a closer look. Then choose additional detail or native collection to answer that specific question.
