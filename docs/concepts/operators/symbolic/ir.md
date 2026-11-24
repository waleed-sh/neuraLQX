---
title: Reading the intermediate representation
---

# Reading the intermediate representation

The intermediate representation, or IR, is the operator definition after the fluent builder has assembled its terms. It records iterator domains, conditions, updates, and coefficients in a form that the compiler can inspect. Reading it is useful when a definition is built by several functions or contains multiple conditional branches.

The IR is not a matrix, a list of sampled configurations, or JAX's device code. It describes the rule from which those matrix rows will be generated. The [compiler developer guides](../../../developers/symbolic/index.md) explain how analyses and lowering use it.

## Print the IR

The example below defines a row shift with a charge-dependent coefficient. Its upper cutoff is guarded explicitly.

```python
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0))], is_planar=True
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "weighted_shift")
    .for_each_site("e")
    .named("forward")
    .where(s.site("e") < 1)
    .emit(
        s.shift("e", 1),
        matrix_element=2 + s.site("e").value,
        tag="increase_charge",
    )
    .build()
)
ir = definition.to_ir()
print(ir)
```

## Read the IR

This produces the following output. The block is the actual printed representation of the definition above.

```text
symbolic.operator @"weighted_shift" [dtype=float64, hermitian=false, hilbert_size=1] {
  ; 1 term(s)

  term #0 "forward" [kbody, n_iter=1, max_conn_size=1] {
    iterate: for (e,) in [(0,)]
    where:   (x[e] < 1)
    emit #0 [tag='increase_charge']:
      update:    x'[e] = (x[e] + 1)
      amplitude: (2 + x[e])
  }

}
```

You can inspect one term with `print(ir.terms[0])` when an operator has many terms. Printing the complete IR retains the operator header and the numbered term blocks, which makes it easier to connect each rule to the enclosing definition.

Start with the opening `symbolic.operator` line. `@"weighted_shift"` is the operator's name. `dtype=float64` describes its matrix elements, while `hermitian=false` records that this definition has not been declared Hermitian. It is not the result of a compiler proof. In this example, the shift has no matching reverse contribution and is indeed non-Hermitian.

`hilbert_size=1` gives the configuration width. It is not the Hilbert space dimension. There is one stored charge, but three possible basis states, $q=-1,0,1$. The following comment reports the number of terms in the definition.

The `term #0 "forward"` block contains the first term, with the name supplied through `.named("forward")`. `kbody` describes a tuple-based iterator. It does not imply a two-body interaction. Here the tuple has just one entry. `n_iter=1` means that the term has one static iterator visit.

The term's displayed `max_conn_size=1` is its raw capacity estimate or explicit hint, before compiler target analysis. This example has one visit and one emission. A more complicated definition can have a smaller compiled capacity if contributions share targets or cancel. See [repeated targets and output capacity](emissions.md#repeated-targets-and-output-capacity).

Read `iterate: for (e,) in [(0,)]` as “bind `e` to coordinate zero once”. The single-element tuple notation is ordinary Python notation. It does not introduce a new dimension in the configuration. The name `e` is a label for that coordinate, not its charge value.

`x[e]` reads the source charge. The term's `where: (x[e] < 1)` permits only the rows below the upper cutoff. `emit #0` identifies the first emission within this term, and `tag='increase_charge'` is the label supplied for inspection.

The update line gives the connected configuration. `x'[e]` denotes the target coordinate, so `x'[e] = (x[e] + 1)` increases the charge by one relative to the source. This is the [matrix-row convention](emissions.md#row-convention-and-ladder-operators), with the source selecting a row and the target selecting a column.

Finally, `amplitude: (2 + x[e])` gives the matrix element for that connection. The printer uses the internal name `amplitude` for the expression supplied as `matrix_element=...`. It is not the neural network's wavefunction amplitude, and the source read still refers to the original charge.

## Translate the printed rule into matrix rows

The rule contributes

$$
A_{q,q'}=\mathbf 1_{q<1}(2+q)\delta_{q',q+1}.
$$

At $q=-1$ it connects to zero with coefficient one. At $q=0$ it connects to one with coefficient two. At $q=1$ it is inactive. This translation from a printed rule to a few concrete rows is often the quickest way to find a sign or indexing mistake.

In the ordered basis $(-1,0,1)$, the matrix is therefore

$$
A=\begin{pmatrix}
0 & 1 & 0 \\
0 & 0 & 2 \\
0 & 0 & 0
\end{pmatrix}.
$$

The following small check confirms that the compiled operator implements those rows. Dense conversion is useful for this three-state example, but is not needed to print or compile the IR.

```python
import numpy as np

expected = np.array([
    [0, 1, 0],
    [0, 0, 2],
    [0, 0, 0],
])
np.testing.assert_allclose(definition.compile().to_dense(), expected)
```

## Inspect fields without parsing text

[`SymbolicOperatorIR`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicOperatorIR) and its term records expose typed fields. Use those fields when checking a generated definition programmatically.

```python
assert ir.hilbert_size == 1
assert ir.term_count == 1
term = ir.terms[0]
assert term.name == "forward"
assert term.iterator.labels == ("e",)
assert term.iterator.index_sets == ((0,),)

for emission in term.effective_emissions:
    print(emission.branch_tag)
    print(emission.predicate)
    for operation in emission.update_program.ops:
        print(operation.kind, operation.params)
    print(emission.amplitude)
```

`effective_emissions` is the consistent traversal interface. It also accommodates older single-emission term records. The underlying update records retain their order, which matters when several operations touch the same site.

Printed iterator lists can be abbreviated for long domains. Inspect `iterator.index_sets` for every row. `ir.as_dict()` and `ir.static_fingerprint()` support inspection and compiler caching, but the printed form and dictionary are not a versioned interchange format for saving and reconstructing operators across releases.

## Conditional chains become ordinary predicates

An `emit_elseif(p1, ...)` after `emit_if(p0, ...)` is represented with the effective condition $\neg p_0\land p_1$. An `else` contains the remaining condition. The compiler does not retain an instruction to run Python's `if` statement later.

This helps explain unexpected inactive branches. Check whether a condition belongs to the term or just one emission, whether a new `emit_if` started an independent chain, and whether a term modifier closed the previous chain. The [emission guide](emissions.md) develops these cases with runnable examples.

## Relate the IR to compiled output

```python
compiler = s.SymbolicCompiler()
artifact = compiler.compile(definition)
compiled = artifact.compiled_operator
print("Raw estimate:", definition.estimate_max_conn_size())
print("Compiled capacity:", compiled.max_conn_size)
print("Kernel plan:", artifact.metadata_map()["kernel_planning"]["enabled"])
```

The raw IR preserves the declared mathematical contributions. Compiler metadata explains how they are assembled. Equal targets may be grouped, opposite coefficients can cancel, and some target collisions may still require runtime checks. These transformations can change the output capacity without changing the aggregate matrix elements.

Use the IR to understand the definition, [diagnostics](diagnostics.md) to investigate suspicious rules, and [compiler reports](../../../developers/symbolic/pipeline.md) to investigate the chosen implementation. Together they separate a modelling error from a compilation or performance issue.
