---
title: Diagnostic messages
---

# Diagnostic messages

This reference covers every built-in symbolic diagnostic code. The [diagnostics guide](diagnostics.md) explains the message format and compiler options. A code identifies a specific check, while an operator name, term name, and any sampled configuration locate the finding in your definition.

## Unresolved symbols {#lint-code-nqxdsl-e001}

**`NQXDSL-E001`, error.** An expression contains a free name with no supplied default. For example, `symbol("g")` names a coefficient but does not provide its value. The standard connectivity interface has no runtime coefficient dictionary.

Replace the symbol with a numeric expression or declare it with `symbol("g", default=...)`. Rebuild the definition if the coefficient changes. An iterator label is different from a free coefficient, so also check spelling and scope when a name was intended to refer to a site. The [expression guide](expressions.md#named-constants) gives a complete example.

## Static index outside the configuration {#lint-code-nqxdsl-e002}

**`NQXDSL-E002`, error.** A `source_index(k)` or `target_index(k)` read lies outside $0\le k<\texttt{hilbert.size}$. Negative static reads are rejected by this diagnostic even if a low-level array operation could interpret them relative to the end.

Check whether the index belongs to a graph vertex, an edge, or a flat configuration coordinate. Recompute it through the [layout](spaces.md) when the number of gauge components or intertwiner channels changes. A configuration's width is not its basis dimension.

## Potential division by zero {#lint-code-nqxdsl-e003}

**`NQXDSL-E003`, error.** A division uses a direct runtime site value as its denominator. This includes source and emitted selectors and fixed source or target reads.

The current rule is structural. It does not prove that a surrounding `where` condition excludes zero, or that the site's basis contains no zero. Consequently, adding `where(q != 0)` does not by itself silence the diagnostic. Conversely, rewriting a direct read as a more complicated denominator can evade this particular pattern without making the expression safe.

For a small finite domain, an explicitly defined piecewise observable avoids ambiguous division at runtime. Here we define the reciprocal on nonzero charges and choose zero at charge zero. That value at zero is part of the observable's definition.

```python
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
builder = s.SymbolicOperatorBuilder(
    hilbert, "reciprocal_on_nonzero_charges", hermitian=True
).for_each_site("e")

for charge in (-1, 1):
    builder.emit_if(
        s.site("e") == charge,
        matrix_element=1.0 / charge,
        tag=f"charge_{charge}",
    )

reciprocal = builder.compile()
np.testing.assert_allclose(reciprocal.to_dense(), np.diag([-1, 0, 1]))
```

The division occurs in Python between known nonzero constants. For a larger domain, choose an analytically safe expression or a [direct kernel](../../../developers/operators/index.md) with the necessary numerical treatment. Do not change the physical operator merely to remove a message.

## Constant false term {#lint-code-nqxdsl-w101}

**`NQXDSL-W101`, warning.** A term's shared predicate is the constant false value, so the term cannot contribute. Remove an intentionally inactive term or correct the condition.

One subtle cause is using `==` between two `AmplitudeExpr` objects and obtaining a structural comparison instead of a runtime predicate. Use `ExpressionContext.eq` for that case, as explained in [conditions and comparisons](conditions.md#compare-values-rather-than-expression-trees). This check does not discover every logically impossible conjunction.

## Repeated emission definitions {#lint-code-nqxdsl-w103}

**`NQXDSL-W103`, warning.** A term contains structurally repeated emissions. This often results from adding the same branch twice while assembling a model.

Repeated contributions add to the coefficient. Deduplication does not remove their multiplicity from the operator. If the repetition is intentional, write its multiplicity explicitly in the matrix element when that makes the definition clearer. Do not treat changing a diagnostic tag as a correction to the underlying physics.

## Connection hint below the raw bound {#lint-code-nqxdsl-w104}

**`NQXDSL-W104`, warning.** The term's explicit connection hint is below the product of iterator visits and emissions. This can happen when a hint was chosen before more branches were added.

Prefer automatic capacity analysis. If a hint is retained, justify it against the complete rule and test boundary cases. The compiler's deduplication analysis derives its own safe output bound and does not accept an arbitrary small hint as proof that connections can be discarded. See [connectivity analysis](../../../developers/symbolic/connectivity.md).

## Untagged multiple emissions {#lint-code-nqxdsl-i201}

**`NQXDSL-I201`, information.** A term contains multiple emissions without branch tags. The operator can still be valid, but identifying a particular branch in the IR is harder.

Add short physical labels such as `tag="forward"` and `tag="reverse"`. Tags aid inspection and do not affect the numerical action or target equality.

## Target outside enumerated support {#lint-code-nqxdsl-w301}

**`NQXDSL-W301`, warning.** A sampled branch marked as active produces a target absent from the explicitly enumerated Hilbert support. The check happens before final coefficient cancellation. The message can show an input and resulting target that demonstrate the problem.

Check local cutoffs, coordinate mappings, and the complete constraint. If the physical operator should act in a larger space, use the [restricted-space machinery](../restricted-spaces.md) to express that relation. Suppressing an outgoing residual can change the objective.

## Target violates the constraint {#lint-code-nqxdsl-w302}

**`NQXDSL-W302`, warning.** A sampled target fails the Hilbert constraint, for example a fixed-charge relation or a vertex admissibility condition.

Inspect all coordinates affected by the rule. A move preserving a pairwise sum need not preserve the graph's Gauss law, and a locally allowed SU(2) spin change may require accompanying channel changes. Correct the physical update or its guard and verify the complete target with `hilbert.is_valid`.

## Illegal local label {#lint-code-nqxdsl-w303}

**`NQXDSL-W303`, warning.** A sampled target contains a value outside at least one selected site's local domain. Typical causes include an unguarded shift at a cutoff, a write using an incorrect unit, or a swap between different local bases.

Use the declared per-site domain to check the target. An ordinary numerical shift and a [cyclic basis-position shift](updates/modular.md) are different operations. Choose between them from the intended boundary condition.

## Exact support unavailable {#lint-code-nqxdsl-i301}

**`NQXDSL-I301`, information.** The diagnostic could not form an exact support lookup, either because indexing is unavailable or because the configured size cap would be exceeded.

This does not prevent matrix free execution. Other local-domain and constraint checks may still run. Use small reference problems when exact comparisons are useful, and retain checks on valid sampled configurations for larger spaces.

## Sampled rule evaluation failed {#lint-code-nqxdsl-i302}

**`NQXDSL-I302`, information.** The diagnostic evaluator could not finish a sampled evaluation of a term. The finding records the term and a failure count, rather than the original exception. Inspect the term's IR and reproduce its execution on a valid configuration to obtain the underlying error. Possible causes include an unsupported expression or missing data required by the rule.

An informational severity here does not establish that runtime execution will succeed. Evaluate the compiled operator on a valid configuration and resolve the underlying failure before using it in a simulation.

## No source sample available {#lint-code-nqxdsl-i303}

**`NQXDSL-I303`, information.** The bounded diagnostic sampling procedure could not obtain a valid input sample. This can occur when a custom constrained space lacks an appropriate random-state generator or when its generic proposal mechanism is unsuitable.

Provide a native constraint-aware [`random_state`](../../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.random_state) implementation where needed. The [custom Hilbert sampling guide](../../../developers/hilbert/random.md) explains that contract. In the meantime, test rows using valid configurations obtained through a known construction. Compilation may continue, but this sampled check has not been performed.

## Estimated deduplication cost {#lint-code-nqxdsl-w401}

**`NQXDSL-W401`, warning.** Static analysis estimates more connection assembly work with deduplication than with raw branch output. The estimate includes aggregation, collision handling, key encoding, and compaction. It is not elapsed time on a CPU or GPU.

Deduplication remains enabled as requested. The estimate excludes downstream wavefunction evaluation, so fewer targets may still improve a complete VMC step. Compare mathematically equivalent compiled variants and profile the full calculation before selecting an option. The [developer connectivity guide](../../../developers/symbolic/connectivity.md#interpret-assembly-costs-in-context) explains the reported quantities.
