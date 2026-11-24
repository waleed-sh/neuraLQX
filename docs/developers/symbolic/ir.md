---
title: Language and intermediate representation
---

# Language and intermediate representation

The fluent builder is a construction interface. The compiler operates on a structured intermediate representation, or IR, that records the operator independently of the sequence of Python calls used to create it. This separation allows the same definition to be validated, inspected, analysed, and lowered without executing a neural network or enumerating the full Hilbert space.

The [printed IR walkthrough](../../concepts/operators/symbolic/ir.md#read-the-ir) explains an actual operator dump from its header through its iterator, guard, update, and matrix element. This page describes the data model and the invariants needed when writing a pass or language extension.

## From builder state to immutable terms

[`SymbolicOperatorBuilder`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator) holds the current term while its clauses are being assembled. Starting another iterator seals that term. Calling `.build()` seals the final term and returns a [`SymbolicOperator`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicOperator) containing the completed IR terms.

The IR records the result of fluent control flow. An `emit_elseif` does not remain a Python conditional to be replayed. Its predicate includes the negation of earlier conditions in its chain. Likewise, `emit_else` becomes a final emission guarded by the remaining condition. Independent `emit_if` chains remain independent contributions.

The core records are frozen dataclasses. Modify a definition by constructing replacement records, rather than mutating a shared expression or update in place. This is especially important when definitions participate in compilation signatures.

| Record | Information carried |
| --- | --- |
| [`SymbolicOperatorIR`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicOperatorIR) | Operator name, mode, configuration size, matrix-element dtype, Hermiticity declaration, ordered terms, and metadata |
| [`SymbolicIRTerm`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicIRTerm) | Iterator, common predicate, emissions, optional capacity hint, and term metadata |
| [`KBodyIteratorSpec`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.KBodyIteratorSpec) | Label names and ordered tuples of flat configuration indices |
| [`EmissionSpec`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.EmissionSpec) | One update program, coefficient expression, branch predicate, and optional tag |
| [`AmplitudeExpr`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.AmplitudeExpr) and [`PredicateExpr`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.PredicateExpr) | Immutable scalar and Boolean expression trees |
| [`UpdateProgram`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.UpdateProgram) and [`UpdateOp`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.UpdateOp) | An ordered sequence of target mutations and their parameters |

## Iterator bindings are static

All supplied site, pair, edge, and path iterators reduce to `KBodyIteratorSpec`. For labels `("i", "j")`, a row `(2, 5)` binds those names to coordinates 2 and 5. Row order and repeated rows are significant because they define contributions to a sum.

A global term uses empty labels and one empty row. It is visited once. It is not a sum over every basis configuration. Neither the iterator nor its labels infer graph geometry after construction. A graph-aware clause must resolve the desired structures into the appropriate flat indices before creating the IR.

For a term with rows $\mathcal I_t$ and emissions $\mathcal E_t$, the primitive actions are indexed by

$$
r=(t,i,e),\qquad i\in\mathcal I_t,\quad e\in\mathcal E_t.
$$

This ordering is retained in the kernel plan's action origins and in the first-occurrence ordering used for connection assembly.

## Traverse effective emissions

`SymbolicIRTerm` retains single-emission fields for compatibility with direct term construction. Its `effective_emissions` property presents either the explicit emissions tuple or an equivalent single emission. Passes should traverse this property. Reading only `term.amplitude` or `term.update_program` loses additional emissions from a fluent definition.

```python
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "cyclic_charge")
    .for_each_site("e")
    .named("both_directions")
    .emit(s.shift_mod("e", 1), matrix_element=0.5, tag="plus")
    .emit(s.shift_mod("e", -1), matrix_element=0.5, tag="minus")
    .build()
)
ir = definition.to_ir()

assert ir.mode == "symbolic"
assert ir.terms[0].iterator.labels == ("e",)
assert ir.terms[0].iterator.index_sets == ((0,),)
assert len(ir.terms[0].effective_emissions) == 2

for term in ir.terms:
    for emission in term.effective_emissions:
        print(term.name, emission.branch_tag, emission.update_program)
```

## Source reads and ordered target writes

The evaluation environment has two roles. Source selectors and named constants determine scalar expressions. An update program maintains a separate working target, initially equal to the source. Its operations run in order.

For example, `shift("e", 1).write("e", site("e").value)` first changes the target and then restores the original source value. The second expression does not read the partially updated target. By contrast, a second `shift` adds to the current working value. A compiler transformation must preserve this distinction as well as intermediate dtype conversions.

An emitted selector in a matrix element reads the completed target. The kernel planner can often substitute the corresponding coordinate expression directly. A dynamically indexed update may instead require an explicit target read after construction. Emitted selectors are not valid inputs to update expressions or term guards.

## Expression structure and numerical meaning

Nodes carry operation names and arguments. Constants preserve their values and types. Source and target reads preserve their index bindings. An extension must not assume that algebraically equivalent expressions are interchangeable under finite-precision arithmetic. Reassociating a sum, moving a cast, or evaluating a partial function outside its guard can change results or derivatives.

Expression equality also has a construction-time meaning. Comparing two `AmplitudeExpr` objects with `==` tests their structure. To create an equality predicate between two symbolic expressions, use `ExpressionContext.eq`. The [conditions guide](../../concepts/operators/symbolic/conditions.md) demonstrates this distinction.

## Fingerprints and inspection formats

`ir.as_dict()` provides a dictionary view for inspection, while `ir.static_fingerprint()` identifies the encoded definition for compilation caching. A printed IR can abbreviate a long iterator domain, so it should not be parsed to reconstruct a program.

```python
fingerprint = ir.static_fingerprint()
assert fingerprint == definition.to_ir().static_fingerprint()
assert ir.as_dict()["operator_name"] == "cyclic_charge"
print(fingerprint)
```

These inspection forms are not a versioned interchange or checkpoint format. Preserve the Python definition and its physical parameters when recording a simulation. Compiler extensions should use the records directly, include their semantic inputs in the relevant signatures, and treat changes to representation as changes that need [regression tests](extensions.md#testing-a-compiler-change).
