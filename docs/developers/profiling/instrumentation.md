---
title: Instrumentation
---

# Instrumentation

Instrumentation connects execution intervals to the operations a scientist recognises. A native runtime can report a dispatch or kernel, but it does not by itself explain that the work belongs to sampling a constrained space, evaluating connected configurations, or applying an SR preconditioner. neuraLQX adds that context at selected boundaries in the calculation.

There are two distinct mechanisms. Application annotations record elapsed host intervals and small pieces of metadata. Compiled scopes attach names to JAX computations when they are traced. Understanding their different lifetimes is essential when interpreting repeated executions of a compiled function.

## What an application span measures

[`span`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.span) reads a monotonic host clock on entry and exit. For entry time $t_a$ and exit time $t_b$, its duration is

$$
\Delta t=t_b-t_a.
$$

This is elapsed time, including any time spent waiting or suspended inside the region. It is not CPU time consumed exclusively by that function, and it is not a sum of device kernel durations.

A span around an asynchronous JAX call usually measures the host work needed to submit that call. A span around a conversion that reads a device value can include a completion wait. The code inside the boundary determines which experiment is being measured. Ordinary spans never add a device wait of their own.

[`annotate`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.annotate) applies the same model to a function or coroutine. With `sync=True`, it additionally waits for the returned array tree before leaving the span while recording. This deliberately changes the boundary to include result completion. It can also serialise dispatch and include waiting for earlier dependent work.

An exception leaving a span is recorded through its exception type and is propagated normally. If the span crosses the end of its recording window, the retained duration is clipped at that boundary and marked with `clipped_at_boundary=True`. Such a span describes the observed portion of an operation, not necessarily its complete lifetime.

## Python boundaries and compiled scopes

Place a function annotation outside `jax.jit` when it should run on each invocation. A Python context inside a jitted function is normally evaluated while JAX traces the function, so its recorded occurrence describes tracing rather than each later execution.

[`compiled_scope`](../../api/public/neuralqx/profile/index.md#neuralqx.profile.compiled_scope) uses JAX named scopes to attach names during tracing. It does not insert a Python callback, start a host span, or wait for a result. The names remain attached to the compiled computation when it is reused.

These compiled names are supplied independently of the active profiling level or annotation depth. Changing `profile_detail` therefore does not require a new compiled computation merely to add or remove a host region. The compiler still controls fusion and optimisation. A named scope may describe part of a fused operation rather than a separate kernel with a directly measurable duration.

Public application spans and compiled scopes are complementary. neuraLQX's own instrumentation also pairs selected host regions with native JAX annotations while a native collector is active. Driver iterations receive native step annotations. This places the package's scientific structure alongside runtime dispatch information. Public `span` alone records the application region, with optional NVTX emission, and does not create that native JAX annotation.

## The supplied annotations

The broad phases appear at `summary` detail. The exact set depends on the driver, estimator, sampling mode, preconditioner, requested observables, and callbacks actually used.

| Region | Meaning in the calculation |
| --- | --- |
| `vmc.step` | One attempted driver iteration |
| `sampling` | Obtaining the configurations needed for the calculation |
| `gradient` | Evaluating the objective and its parameter derivative |
| `preconditioner` | Transforming the derivative through the configured preconditioner |
| `optimizer.update` | Computing and applying the Optax update |
| `observables` and `observable/<name>` | Additional measurements requested from the driver |
| `callbacks.<event>` | Dispatching callbacks at a particular lifecycle event |
| `logging.write` and `logging.flush` | Writing and flushing simulation records |

For [multiple-state optimisation](../../concepts/drivers/multiple-states.md), names such as `state[i]` and `orthogonality[i,j]` retain the state or pair being processed. They help distinguish the objective for one state from coupling or overlap work involving several states.

At `detailed` depth, sampling can separate `sampling.refresh`, `sampling.discard`, `sampling.collect`, and `sampling.measure`. Estimator regions include `estimator.local_values`, `estimator.statistics`, and preparation specific to residual or importance estimators. Gradient paths can distinguish covariance evaluation, a full derivative, and pullbacks. Preconditioning can expose `preconditioner.construct`, `preconditioner.qgt`, and `preconditioner.solve`.

The `developer` depth adds implementation boundaries such as `vmc.reset`, `sampler.initialize`, `sampler.reset`, `sampler.chain_dispatch`, `operators.prepare_bucket`, `operators.deduplicate`, `optimizer.cast_update`, and `logging.history`. These can explain where setup or dispatch work occurs. Their durations remain host observations.

Compiled scopes provide a second vocabulary inside the numerical computation. Examples include `sampling.proposal`, `sampling.acceptance`, `sampling.model_probability`, `model.reference`, `model.connected`, `gradient.center_cotangent`, and `qgt.matvec`. These names describe mathematical or algorithmic stages that the compiler can transform together.

## Shared work and operator attribution

Detailed operator annotations can use names of the form `operator[index]/Class/operation`. The index refers to a term in the flattened operator list being evaluated. The class and operation identify the work performed at that point. When identical terms share a calculation, `source_indices` records the original term indices associated with that work.

This is not a decomposition of total runtime into independent costs for every formal term in an objective. Several terms may share connected configurations or a network evaluation. Regions such as `operators.shared_network`, `connectivity.coalesce`, and shared compiled evaluations identify work performed collectively.

Suppose two operators require the same wavefunction values. Assigning the complete shared network duration to both would count that cost twice. Conversely, looking only at their individual connectivity spans would omit the common evaluation. Interpret the individual and shared regions together, using the [operator and estimator structure](../../concepts/operators/application.md) to understand why that reuse occurs.

Individual callbacks are also named at detailed depth, using a pattern such as `callback[index]/Class/event`. This separates the total cost of an event from the contributions of its participating callbacks.

## Threads, tasks, and operation lifetimes

The active session is process-wide. Worker threads and asynchronous tasks can contribute annotations without receiving a separate profiler object. Each application thread or task receives a trace lane, subject to the configured lane limit.

An asynchronous span includes time spent awaiting other work. Distinct task lanes prevent overlapping task lifetimes from being mistaken for a single nested call stack. They do not convert elapsed task duration into exclusive CPU usage.

This complete example records two tasks and waits for both before closing the session.

```python
import asyncio
import gzip
import json
from tempfile import TemporaryDirectory

from neuralqx.profile import Profiler, annotate

@annotate(name="analysis/task")
async def analyse(offset):
    await asyncio.sleep(0)
    return sum(range(offset, offset + 100))

async def main(directory):
    with Profiler(directory, level="basic") as profiler:
        values = await asyncio.gather(analyse(0), analyse(100))
    return values, profiler.result

with TemporaryDirectory() as directory:
    values, result = asyncio.run(main(directory))
    with gzip.open(result.trace_path, "rt") as stream:
        records = json.load(stream)["traceEvents"]
    tasks = [item for item in records if item.get("name") == "analysis/task"]
    print(values)
    print(len({item["tid"] for item in tasks}))
```

```text
[4950, 14950]
2
```

Use the same lifetime discipline with threads by joining them before leaving the profiler context. Only the owning thread may control the session or advance its schedule. An annotation decorator rejects generator and asynchronous generator functions because decorating their creation would not measure the later iteration. Put spans inside those generators around the work being measured.

## Metadata and recording overhead

Attributes accept ordinary Python `str`, `int`, `float`, `bool`, and `None` values. Arrays, arbitrary objects, and NumPy scalar objects are not converted implicitly. Counter values accept finite Python integers or floats, excluding booleans. This keeps instrumentation from accidentally transferring a device result or executing an arbitrary object representation.

Names, categories, and attribute keys are nonempty and limited to 256 characters. Attributes allow up to 32 fields, with individual string values limited to 1024 characters and a combined metadata budget of 4096 characters. Integers have a maximum bit length of 64, and floating-point values must be finite. Prefer stable names with small scalar context over a new name for every configuration.

Outside a recording interval, a normal span returns a shared inactive context without reading clocks or copying metadata. The inactive decorator wrapper checks the active-session reference and calls the function directly. Its static name and attributes are still validated when the decorator is created. Active recording necessarily adds timestamp, metadata, and buffer work, so a highly instrumented capture should not be treated as an overhead-free benchmark.

Use these public annotation functions for extensions and research code. The package's private detail-filtering helpers describe its current implementation, rather than a separate extension interface. Continue with [profiler internals](internals.md) for buffer ownership, scheduling, and finalisation.
