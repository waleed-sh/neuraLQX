---
title: Caching and configuration
---

# Caching and configuration

Symbolic compilation has two configuration interfaces. [`SymbolicCompilerOptions`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompilerOptions) controls a compiler or an individual compilation. `neuralqx.operator.symbolic.config` controls the default operator target and symbolic debug instrumentation.

These settings are separate from the main [neuraLQX configuration](../../start/configuration.md). In particular, enabling symbolic debug logging does not enable the package's simulation profiler, and the symbolic artifact cache is not JAX's device compilation cache.

## Reuse a compiler deliberately

A new `SymbolicCompiler()` owns a fresh in-memory artifact store unless one is supplied explicitly. Reuse that compiler when compiling related definitions. Calling a definition's `.compile()` without a compiler creates a compiler for that call, so repeated calls are not an application-level cache strategy.

The convenience function [`compile_symbolic_operator`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.compile_symbolic_operator) reuses a module-level compiler. If multiple explicit compilers need the same store, pass the same store object to them, for example the object returned by [`default_symbolic_artifact_store`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.default_symbolic_artifact_store).

```python
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "cached_identity", hermitian=True)
    .globally()
    .emit(matrix_element=1.0, tag="identity")
    .build()
)

options = s.SymbolicCompilerOptions(
    cache_namespace="my_symbolic_study_v1",
    diagnostics_min_severity="warning",
)
compiler = s.SymbolicCompiler(options=options)
first = compiler.compile(definition)
second = compiler.compile(definition)
assert second is first
assert compiler.cache_size == 1

operator = definition.compile(compiler=compiler)
assert operator is first.compiled_operator
compiler.clear_cache()
assert compiler.cache_size == 0
```

Clearing an artifact store removes its retained entries. It does not invalidate an operator object already held by the application or clear JAX's compiled executables.

## What determines an artifact key

The key includes the IR fingerprint, backend, configuration size, matrix-element dtype, and the static compiler-option signature. It also includes the identity of the Hilbert object and the resolved target descriptor. The descriptor identifies its name, operator class, connection method, and factory, including their in-process identities.

Consequently, two separately constructed but mathematically equal Hilbert spaces need not reuse one artifact. Changing a compiler option can also create a distinct key even when it does not change the resulting operator matrix.

The key does not automatically fingerprint custom pass implementations or every arbitrary item of context metadata. When changing the compiler pipeline, use a fresh store or namespace, or disable caching while developing. Do not use an existing key as proof that two different compiler implementations are equivalent.

Artifacts are Python objects held in process memory. The supplied store is not a persistent executable cache or a portable checkpoint format. JAX compilation remains a later stage, described in [lowering and execution](lowering.md).

## Compiler options

Pass an options object to `SymbolicCompiler(options=...)`, or override the options for one `compiler.compile(definition, options=...)` call. The builder and definition also offer convenient arguments for backend, operator target, deduplication, kernel optimisation, and cache use. Use the options object when configuring diagnostic policy or fusion.

### Execution and planning

| Option | Default | Accepted values and effect |
| --- | --- | --- |
| `backend_preference` | `"auto"` | `"auto"` resolves to JAX. `"jax"` selects it explicitly. No other backend is supplied |
| `operator_lowering` | `None` | `None` follows the compiler or registry default at compilation time. A nonempty registered name selects that operator target |
| `deduplicate_connected_components` | `True` | Boolean. Combine equal targets, sum their matrix elements, and remove inactive output entries when true. False retains raw contributions |
| `optimize_kernels` | `True` | Boolean. Enable structured scalar and coordinate planning. False selects the reference JAX implementation |
| `enable_fusion` | `True` | Boolean. Permit compatible scalar templates to share batches and record legacy fusion groups. False keeps those action batches separate |
| `strict_validation` | `True` | Boolean. Reject validation errors and visible error diagnostics. False relaxes this policy but cannot make malformed expressions executable |

### Artifact caching

| Option | Default | Accepted values and effect |
| --- | --- | --- |
| `cache_enabled` | `True` | Boolean. Read and write the compiler's artifact store when true |
| `cache_namespace` | `"nqx_symbolic_v2"` | Nonempty string included in artifact keys. Use a separate namespace for a different experimental pipeline |

### Diagnostics and instrumentation

| Option | Default | Accepted values and effect |
| --- | --- | --- |
| `diagnostics_enabled` | `True` | Boolean. Run the diagnostic rules when true. Structural validation remains a separate pass |
| `diagnostics_min_severity` | `"warning"` | `"info"` shows all findings, `"warning"` shows warnings and errors, and `"error"` shows errors only |
| `fail_on_warnings` | `False` | Boolean. Reject visible warnings when true, including the connection assembly cost warning |
| `max_diagnostics` | `200` | Positive integer limiting findings retained by the main diagnostic collector |
| `lint_state_sample_size` | `32` | Positive integer limiting sampled source configurations |
| `lint_branch_sample_cap` | `4096` | Positive integer limiting active branch checks in the sampled connectivity rule |
| `lint_max_exact_hilbert_states` | `8192` | Positive integer bounding the full support lookup used to check sampled targets on small indexable spaces |
| `debug_flags` | `()` | Tuple of key-value pairs for extension metadata. `SymbolicCompilerOptions.from_mapping(debug_flags={...})` normalises a mapping. The default pipeline does not interpret named flags here as debug-log switches |

The numerical limits bound particular checks. For example, the branch cap does not prevent a term runner from first constructing its output for a sampled source. See [compiler diagnostics](diagnostics.md) before treating these values as a total compilation-memory budget.

## Symbolic debug configuration

The following eight options are registered in `s.config`. Each can be changed in Python with `s.config.set("OPTION", value)` or temporarily with `s.config.patch(...)`. Environment values are read through the symbolic configuration manager, so set them before importing the package in a new process.

| Option | Default | Values and behaviour | Environment variable |
| --- | --- | --- | --- |
| `LOWERING` | `"neuralqx"` | Nonempty registered target name used when no explicit target is requested | `NEURALQX_SYMBOLIC_LOWERING` |
| `EXPERIMENTAL` | `False` | Boolean flag reserved for experimental behaviour. Currently enabling it emits a warning, but does not select a different supplied compiler algorithm | `NEURALQX_SYMBOLIC_EXPERIMENTAL` |
| `DEBUG` | `False` | Boolean master switch for symbolic debug events | `NEURALQX_SYMBOLIC_DEBUG` |
| `DEBUG_VERBOSITY` | `"info"` | `"debug"`, `"info"`, `"warning"`, `"error"`, or `"critical"`, in increasing severity order | `NEURALQX_SYMBOLIC_DEBUG_VERBOSITY`, with `NEURALQX_SYMBOLIC_LOG_LEVEL` as an alias |
| `DEBUG_SCOPES` | `("all",)` | Tuple or comma-separated string selecting `all`, `dsl`, `compile`, `ir`, `cache`, `passes`, `lowering`, or `runtime`. `pass:<pass_name>` selects a particular pass | `NEURALQX_SYMBOLIC_DEBUG_SCOPES` |
| `DEBUG_PASSES` | `()` | Tuple or comma-separated pass names filtering pass events. Empty leaves pass events unfiltered | `NEURALQX_SYMBOLIC_DEBUG_PASSES` |
| `DEBUG_LOG_TO_FILE` | `True` | Boolean. Write a session log when debugging is enabled. False keeps debug output without a session file | `NEURALQX_SYMBOLIC_DEBUG_LOG_TO_FILE` |
| `DEBUG_LOG_DIR` | `None` | Directory string, or `None` to use the symbolic cache directory's `logs` subdirectory | `NEURALQX_SYMBOLIC_DEBUG_LOG_DIR` |

For Boolean environment values, `true` and `false` are unambiguous choices. Python values use ordinary `True` and `False`. Scope and pass filters use commas in the environment.

This scoped example records compilation and cache events without creating log files. The patch restores the previous configuration when its context exits.

```python
previous = s.config.snapshot()
with s.config.patch(
    DEBUG=True,
    DEBUG_VERBOSITY="info",
    DEBUG_SCOPES="compile,cache",
    DEBUG_LOG_TO_FILE=False,
):
    compiler.compile(definition)
    compiler.compile(definition)
assert s.config.snapshot() == previous
```

An equivalent environment setup can be inspected in a fresh process as follows.

```sh
NEURALQX_SYMBOLIC_DEBUG=true \
NEURALQX_SYMBOLIC_DEBUG_SCOPES=compile,cache \
NEURALQX_SYMBOLIC_DEBUG_LOG_TO_FILE=false \
python - <<'PY'
from neuralqx.operator import symbolic as s
print(s.config.snapshot())
PY
```

The default symbolic cache directory is `.neuralqx/symbolic` relative to the working directory, making the default log location `.neuralqx/symbolic/logs`. Session filenames distinguish timestamp, process, rank, and session. Choose an explicit directory for reproducible collection in batch jobs.

Debug events describe construction and compiler activity. They do not replace device traces or timings. Use the [profiling configuration](../profiling/configuration.md) for simulation instrumentation and the [extension guide](extensions.md) when registering another lowering target.
