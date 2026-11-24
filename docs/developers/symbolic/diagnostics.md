---
title: Compiler diagnostics
---

# Compiler diagnostics

The diagnostic system helps distinguish malformed definitions, suspicious expressions, invalid generated connections, and expensive assembly plans. Its findings are useful evidence about specific checks. They are not a general proof of operator correctness, Hermiticity, constraint preservation, or numerical stability.

The [scientist diagnostics guide](../../concepts/operators/symbolic/diagnostics.md) explains how to respond to a message, and the [message reference](../../concepts/operators/symbolic/messages.md) documents every supplied code. Here we examine how those findings are produced and what their limits imply for compiler development.

## Structural validation and lint rules

Structural validation runs before the diagnostic pass. It checks the IR contract, including expression scopes and supported update structure. The diagnostic collector then runs rules over a context containing the original operator, its IR, and compiler options.

Structural rules inspect declarations without evaluating all source configurations. Examples include missing symbol values, statically invalid coordinate indices, constant-false predicates, repeated emission definitions, and inconsistent capacity hints. Such rules can identify a definite construction mistake, or a suspicious pattern that requires scientific interpretation.

The potential-zero-division rule is deliberately limited. Its current pattern matching flags division by a direct runtime value without proving safety from surrounding predicates or a zero-free local domain. A guard alone therefore does not necessarily remove `NQXDSL-E003`. The [message reference](../../concepts/operators/symbolic/messages.md#lint-code-nqxdsl-e003) gives a finite-basis alternative. Do not describe this rule as a complete symbolic domain solver.

## Obtain bounded source samples

Dynamic connectivity checks first request `hilbert.random_state` with a fixed local PRNG key and the configured sample cap. This permits checks on large non-indexable spaces and uses native constraint-aware sampling where available.

If direct sampling fails, the implementation tries a bounded prefix of `numbers_to_states` on an indexable space. It falls back to `all_states()` only when the known cardinality fits the sample cap. A non-indexable space is not forced through an indexing API merely to run diagnostics.

Failure to obtain samples produces `NQXDSL-I303`. A reproducible sample helps reproduce a finding, but it does not guarantee coverage of rare or boundary configurations. Unit tests for an update should include those configurations explicitly.

## Check generated targets

The connectivity rule uses the reference term-runner builder, which shares expression and update semantics with lowering. For each sampled source it obtains targets and branch-validity flags. It then checks local values, an available global constraint predicate, and exact support membership where a small support lookup can be built.

These checks address different questions. Local membership asks whether each coordinate belongs to its declared basis. A global constraint asks whether the complete configuration is admissible. Exact support membership compares a target with a fully enumerated small Hilbert space. A target can satisfy every local basis and still violate a Gauss constraint.

The rule checks valid emitted branches before final deduplication and cancellation. A coefficient cancellation is therefore not a substitute for defining sensible branch targets. Findings retain example source and target configurations, emission indices, and illegal coordinate positions where relevant.

The exact lookup has an important limit. It makes membership testing exact for each checked target, but the source configurations and branches can still be sampled. It does not establish that every row of the operator preserves the space.

## Interpret workload bounds accurately

`lint_state_sample_size` limits source samples. `lint_branch_sample_cap` limits checked active branches across the rule. The term runner can still construct its full branch output before this cap is applied, so the cap is not a bound on every temporary array allocated during diagnostics.

`lint_max_exact_hilbert_states` limits the support lookup. When enumeration is unavailable or too large, local-domain and constraint checks can still run, and `NQXDSL-I301` records the missing exact lookup. `NQXDSL-I302` records evaluation failures rather than silently treating them as successful checks.

The collector retains at most `max_diagnostics` findings from its ordered rules and sorts retained findings by severity, code, and location. Truncation is not an exhaustive audit. With very large definitions, inspect individual terms and retain targeted tests alongside compilation diagnostics.

## Structured findings and display policy

[`DSLDiagnostic`](../../api/public/neuralqx/operator/symbolic/compiler/diagnostics/models/index.md#neuralqx.operator.symbolic.compiler.diagnostics.models.DSLDiagnostic) records a code, severity, message, operator, optional term, suggested correction, and frozen context. The formatting layer adds a link to the corresponding message explanation.

The options determine which findings are visible and which reject compilation. Visible error diagnostics fail when `strict_validation=True`. Visible warnings fail when `fail_on_warnings=True`. Filtering a warning out of display also removes it from that visible-warning failure policy. The full retained finding payload is still available in a successful artifact's `dsl_diagnostics` metadata.

This example deliberately includes a constant-false term, captures its warning, and inspects its structured finding. The ordinary term keeps the operator itself equal to the identity.

```python
import warnings
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "diagnostic_example", hermitian=True)
    .globally()
    .named("disabled_term")
    .where(False)
    .emit(matrix_element=7.0, tag="disabled")
    .globally()
    .named("identity")
    .emit(matrix_element=1.0, tag="active")
    .build()
)

compiler = s.SymbolicCompiler(
    options=s.SymbolicCompilerOptions(diagnostics_min_severity="info")
)
with warnings.catch_warnings(record=True) as messages:
    warnings.simplefilter("always")
    artifact = compiler.compile(definition)

findings = artifact.metadata_map()["dsl_diagnostics"]
finding = next(item for item in findings if item["code"] == "NQXDSL-W101")
assert finding["term_name"] == "disabled_term"
assert any("NQXDSL-W101" in str(message.message) for message in messages)
print(finding["code"], finding["message"])
```

## Cost diagnostics run after connectivity analysis

`NQXDSL-W401` is produced by the connectivity cost pass after cache lookup, using the assembly plan. It is distinct from the earlier rule collector. Its estimate compares raw assembly with grouping, collision checks, and compaction, while excluding neural network evaluations.

A cache hit returns the existing artifact rather than rerunning this post-cache analysis. Do not expect a cached report's timing or cost metadata to describe a newly modified pipeline. See [caching and configuration](configuration.md) for how to separate experimental compiler variants.

## Tests remain independent evidence

Reusing the reference runner keeps sampled diagnostics aligned with execution, but also means a shared implementation defect can affect both. A compiler test should therefore include an independent mathematical reference when feasible, such as a small dense matrix built directly from a known action.

For larger constrained spaces, check selected admissible sources and boundary transitions without global enumeration. Test the same aggregate action under raw, deduplicated, planned, and reference execution when changing those paths. The [extension guide](extensions.md#testing-a-compiler-change) develops this verification approach.
