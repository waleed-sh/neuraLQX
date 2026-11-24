---
title: Compilation pipeline
---

# Compilation pipeline

[`SymbolicCompiler`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompiler) coordinates validation, analysis, cache lookup, and lowering. A compilation operates on a fresh [`SymbolicCompilationContext`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompilationContext). That context holds the definition, IR, effective options, selected target, intermediate analyses, and pass reports.

The pipeline has an explicit cache boundary. Checks before that boundary run for every invocation. Analysis and lowering after it run only when there is no reusable artifact. This matters when measuring compilation or deciding where a custom pass belongs.

## Resolve the target once

Before running passes, the compiler resolves the requested operator target through its operator-lowering registry. With the default configuration this is the native `"neuralqx"` target. The descriptor identifies the operator class, connection method, and optional factory.

This target is distinct from the numerical backend. The backend is currently JAX, selected by `backend_preference="auto"` or `"jax"`. An operator target determines the wrapper receiving the kernel. Registering another wrapper does not create a new execution backend.

The resolved descriptor is stored in the context and included in the cache identity. An explicit `operator_lowering` choice takes precedence over the configured default for that invocation. A missing or unknown target fails before the pass pipeline begins.

## Passes before cache lookup

The default [`SymbolicPassPipeline`](../../api/public/neuralqx/operator/symbolic/compiler/core/index.md#neuralqx.operator.symbolic.compiler.core.SymbolicPassPipeline) begins with three passes.

| Pass name | Responsibility |
| --- | --- |
| `symbolic_validation` | Check structural invariants, iterator scopes, and supported update parameters |
| `symbolic_diagnostics` | Collect structural findings and bounded connectivity checks, then enforce diagnostic policy |
| `symbolic_normalization` | Record the IR fingerprint, resolve the backend, and collect normalisation metadata |

Normalisation does not promise arbitrary algebraic simplification or canonicalisation of equivalent mathematics. It prepares a consistent compilation context. Two different expression trees that describe the same operator can still have different fingerprints.

Diagnostics run before the cache check. A cache hit can therefore still emit warnings or spend time sampling source configurations. The [diagnostics guide](diagnostics.md) explains how to control this workload without confusing fewer checks with a corrected definition.

## Cache lookup

After the first stage, the compiler constructs a signature from the IR, options, backend, Hilbert identity, and resolved target descriptor. If caching is enabled and the store contains that key, the existing artifact is returned immediately.

A returned cached artifact retains the reports from its original compilation. Its recorded pass durations do not describe the newest cache-hit invocation. To measure that invocation, time the call itself or enable [compiler debug events](configuration.md#symbolic-debug-configuration).

If caching is disabled, the second stage still runs. Disabling the cache is useful when developing passes because the pipeline implementation itself is not automatically hashed into an artifact key.

## Passes after a cache miss

The second stage derives how the operator will produce its connection arrays.

| Pass name | Responsibility |
| --- | --- |
| `symbolic_max_conn_size_analysis` | Derive raw per-term connection capacities |
| `symbolic_deduplication` | Group identical targets, analyse possible collisions, and derive a reduced output bound |
| `symbolic_connectivity_cost` | Estimate connection assembly work and report substantial overhead |
| `symbolic_fusion_planning` | Record compatible term groups for compiler consumers |
| `symbolic_kernel_planning` | Plan instantiated actions, scalar expression batches, and deferred target construction |

The last two passes have different roles. The fusion pass records term-group metadata. The current JAX kernel planner separately uses `enable_fusion` to combine compatible scalar expression templates across mapped actions. Disabling fusion therefore affects kernel planning even though the reference lowering path does not execute the legacy term groups as a separate fused program.

Analyses are stored under named context keys. A pass that consumes one should use `context.require_analysis(...)` when its absence is an error, or explicitly select a conservative fallback when the analysis is optional. The [connectivity](connectivity.md) and [kernel planning](kernels.md) guides explain those dependencies.

## Lower and package the result

The lowerer registry selects the first registered lowerer whose `supports(context)` returns true. The supplied JAX lowerer creates a native connection callable and wraps it in the selected operator target. It then packages the operator, planning summaries, and pass reports into a [`SymbolicCompiledArtifact`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompiledArtifact).

The compiler attaches its cache key and stores the artifact when caching is enabled. `compiler.compile(definition)` returns that artifact. `compiler.compile_operator(definition)` returns its executable operator. A symbolic definition's `.compile(compiler=compiler)` also returns the operator.

## Inspect the recorded stages

This example checks the default pass order and demonstrates reuse within one compiler instance. The operator is small enough that the example tests compiler behaviour without requiring a simulation.

```python
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "pipeline_example", hermitian=True)
    .globally()
    .emit(matrix_element=1.0, tag="identity")
    .build()
)

compiler = s.SymbolicCompiler()
first = compiler.compile(definition)
second = compiler.compile(definition)

expected = (
    "symbolic_validation",
    "symbolic_diagnostics",
    "symbolic_normalization",
    "symbolic_max_conn_size_analysis",
    "symbolic_deduplication",
    "symbolic_connectivity_cost",
    "symbolic_fusion_planning",
    "symbolic_kernel_planning",
)
assert tuple(report.pass_name for report in first.pass_reports) == expected
assert second is first
assert compiler.cache_size == 1

for report in first.pass_reports:
    print(report.pass_name, round(report.duration_ms, 3))
```

Pass durations measure host-side compilation work. They are not GPU kernel timings. Device compilation and execution belong to the [lowering and profiling workflow](lowering.md#measure-the-right-stage).

## Failures retain their stage

Each pass executes through a wrapper that records elapsed time and report metadata. When a pass fails, its exception is re-raised and the compiler wraps it with the surrounding stage, preserving the cause. A failed compile does not return a completed artifact simply because a failed pass report was recorded internally.

Read the original diagnostic or exception before changing pipeline options. An unresolved expression, a missing analysis dependency, and a target-construction failure require different corrections. The [extension guide](extensions.md) shows how to add a pass while retaining the default dependencies and reports.
