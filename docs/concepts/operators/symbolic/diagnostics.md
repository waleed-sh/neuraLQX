---
title: Diagnostics and errors
---

# Diagnostics and errors

The symbolic compiler checks a definition before generating its executable action. Some checks inspect the expression structure, such as whether an iterator label exists. Others evaluate a bounded collection of valid configurations to look for illegal targets. Both help find mistakes, but they answer different questions.

A structural failure can prevent compilation. A sampled warning can show a concrete counterexample to the intended action. Neither successful compilation nor a clean sample proves the physical correctness of an arbitrary operator. Keep the small independent checks introduced in [defining an operator](first-operator.md).

## Read a message from the inside out

A diagnostic contains a stable code, severity, operator name, and sometimes a term name. It can also include a source configuration, an emitted target, and the coordinates with illegal values. Start with that specific finding, then use the surrounding compiler exception to identify the stage that failed.

For example, an unresolved named coefficient is reported as `NQXDSL-E001`. The outer exception says that a pass before cache lookup failed. The useful correction is to supply the coefficient, not to change the cache.

```python
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
unresolved = (
    s.SymbolicOperatorBuilder(hilbert, "missing_coupling")
    .for_each_site("e")
    .named("charge_cost")
    .emit(matrix_element=s.symbol("g") * s.site("e").value.pow(2))
    .build()
)

try:
    unresolved.compile()
except s.SymbolicCompilerError as error:
    assert "NQXDSL-E001" in str(error)
    print("The coefficient g needs a value before compilation.")
else:
    raise AssertionError("Expected an unresolved-symbol diagnostic")
```

The exception is caught deliberately so this example can be run. In a calculation, keep the original message and traceback while correcting the definition. A declared default such as `symbol("g", default=0.5)` resolves this example. See [named constants](expressions.md#named-constants) for the supported binding model.

The `NQXDSL-` prefix identifies diagnostics from the neuraLQX symbolic compiler. The letter distinguishes errors (`E`), warnings (`W`), and informational findings (`I`). Use the complete code to find its explanation in the [diagnostic message reference](messages.md).

## Construction, compilation, and execution failures

Some mistakes are detected immediately by the builder. An empty term, a missing iterator, or an `emit_else` without an open conditional chain is a construction problem. These failures can use native value or type exceptions rather than a numbered diagnostic.

[`SymbolicCompilerError`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompilerError) identifies a failed compilation stage and preserves the underlying exception as its cause. A [`SymbolicDiagnosticsError`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiagnosticsError) indicates that the diagnostic policy rejected the definition. Inspect the code and term rather than treating every compilation failure as an unsupported backend.

[`SymbolicOperatorExecutionError`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicOperatorExecutionError) means that an uncompiled definition was asked to execute. Call `.compile()` and use the returned operator. Later failures inside a JAX calculation can instead reflect invalid input shapes, unsupported values, or a kernel issue. Reproduce them first with a single valid configuration.

## Locate the responsible rule

Use `.named(...)` for physical terms and `tag=...` for branches. Print `definition.to_ir()` to inspect the assembled rules, particularly after conditional chains or reusable update programs. The [IR guide](ir.md) explains the output fields.

For a reported target outside the Hilbert space, inspect the source and active matrix element together. Confirm the coordinate mapping, local domain, and global constraint. Check a boundary configuration explicitly. A modular shift can fix an accidental out-of-range value only if periodic wrapping is the intended physical rule.

For numerical expressions, inspect intermediate values as well as final coefficients. A zero prefactor does not justify an undefined division. The [message reference](messages.md) explains the current conservative zero-division rule and its limitations.

## Inspect reports and control verbosity

Use [`SymbolicCompiler`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompiler) when you want its artifact and recorded diagnostics. The ordinary definition `.compile()` returns only the executable operator.

```python
resolved = (
    s.SymbolicOperatorBuilder(hilbert, "known_coupling", hermitian=True)
    .for_each_site("e")
    .emit(matrix_element=0.5 * s.site("e").value.pow(2))
    .build()
)
options = s.SymbolicCompilerOptions(
    diagnostics_min_severity="info",
    fail_on_warnings=False,
    lint_state_sample_size=32,
)
compiler = s.SymbolicCompiler(options=options)
artifact = compiler.compile(resolved)
for finding in artifact.metadata_map()["dsl_diagnostics"]:
    print(finding["code"], finding["message"])
```

[`SymbolicCompilerOptions`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicCompilerOptions) separates diagnostic policy from the operator's physics. The default display threshold is `"warning"`. `"info"` also shows limitations of sampled checks. `fail_on_warnings=True` rejects visible warnings, which can be useful during validation but also includes the structural cost estimate `NQXDSL-W401`.

`lint_state_sample_size`, `lint_branch_sample_cap`, and `lint_max_exact_hilbert_states` bound the diagnostic workload. Raising them extends checks but does not turn bounded sampling into a proof. Disabling diagnostics or reducing validation strictness does not repair an invalid operator. The [compiler configuration guide](../../../developers/symbolic/configuration.md) documents every option and the separate debug settings.

## Interpret cost warnings separately

[`NQXDSL-W401`](messages.md#lint-code-nqxdsl-w401) compares estimated connection assembly work with and without deduplication. It is not a benchmark or a correctness failure. A reduction in connected configurations can still save substantial neural network work even if assembly itself becomes more expensive.

For performance investigations, compare equivalent operators with `deduplicate_connected_components=True` and `False`, first checking that their aggregated actions agree. Measure the [complete workload](../../../developers/profiling/for-scientists.md), including connected-state network evaluation, rather than deciding from the warning alone.
