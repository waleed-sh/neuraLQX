---
title: Extending the language and compiler
---

# Extending the language and compiler

An extension can introduce a useful scientific notation without changing numerical execution. For example, a clause selecting a family of graph structures can produce ordinary iterator rows, and a named physical transition can expand into existing updates and amplitudes. This is the smallest extension boundary when the current IR can already express the desired action.

A new compiler pass, operator wrapper, or numerical primitive has a wider contract. The implementation must preserve the [IR semantics](ir.md), static capacity, dtype behaviour, and native operator interface. Begin with the mathematical action and choose the narrowest boundary that can represent it correctly.

## Register fluent clauses

[`register`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.register) dispatches a clause class to the appropriate registry. An iterator clause opens a term. A predicate clause adds a common condition to that term. An emission clause returns one normalised emission action.

| Extension base | Method to implement | Result |
| --- | --- | --- |
| [`AbstractIteratorClause`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.AbstractIteratorClause) | `build_iterator(self, hilbert, ...)` | A `KBodyIteratorSpec` or accepted labels-and-rows specification |
| [`AbstractPredicateClause`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.AbstractPredicateClause) | `build_predicate(self, ctx, ...)` | A symbolic predicate or supported Boolean value |
| [`AbstractEmissionClause`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.AbstractEmissionClause) | `build_emission(self, ctx, ...)` | One [`EmissionClauseSpec`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.EmissionClauseSpec) |

The context in the last two methods is an expression-construction context. These methods run in Python while building the definition. They must return symbolic records, not values computed from runtime JAX tracers.

The following complete example defines three clauses. Together they measure positive charges on even flat coordinates. The choice of even coordinates is explicit application notation, not a claim about graph vertex or edge numbering.

```python
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

@s.register(name="research_even_coordinates")
class EvenCoordinates(s.AbstractIteratorClause):
    """Visit flat coordinates with even indices."""

    def build_iterator(self, hilbert, label="e"):
        return s.KBodyIteratorSpec(
            labels=(label,),
            index_sets=tuple((i,) for i in range(0, hilbert.size, 2)),
        )

@s.register(name="research_positive_charge")
class PositiveCharge(s.AbstractPredicateClause):
    """Restrict a term to a positive source charge."""

    def build_predicate(self, ctx, label="e"):
        return ctx.site(label).value > 0

@s.register(name="research_charge_squared")
class ChargeSquared(s.AbstractEmissionClause):
    """Emit one diagonal squared-charge contribution."""

    def build_emission(self, ctx, label="e"):
        return s.EmissionClauseSpec(
            mode="emit",
            update=s.identity(),
            matrix_element=ctx.site(label).value.pow(2),
            tag="positive_charge_squared",
        )

graph = nqx.graph.Graph(
    [
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 0)),
    ],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "selected_charge", hermitian=True)
    .research_even_coordinates("e")
    .research_positive_charge("e")
    .research_charge_squared("e")
    .build()
)
operator = definition.compile()

basis = np.asarray(hilbert.all_states())
selected = basis[:, ::2]
diagonal = np.sum(np.where(selected > 0, selected**2, 0), axis=-1)
expected = np.diag(diagonal)
np.testing.assert_allclose(operator.to_dense(), expected)
```

Use descriptive, application-specific method names to avoid collisions with built-in clauses or other extensions. Duplicate registration warns and retains the existing implementation unless `replace=True` is requested. Replacement is appropriate when deliberately redefining an extension, but should not happen as an incidental import side effect.

An emission clause returns one `EmissionClauseSpec`, with mode `"emit"`, `"emit_if"`, `"emit_elseif"`, or `"emit_else"`. The builder applies the same conditional-chain rules as for built-in emissions. For a reusable construction requiring several terms or emissions, an ordinary Python function that accepts a builder is often clearer than forcing several actions into one clause result.

## Add an analysis pass

[`AbstractSymbolicPass`](../../api/public/neuralqx/operator/symbolic/compiler/passes/base/index.md#neuralqx.operator.symbolic.compiler.passes.base.AbstractSymbolicPass) requires a stable `name` and a `run(context)` method. The method can store analyses in the context and return a small metadata mapping. The framework's `execute` wrapper records timing and failures, so custom passes should implement `run` rather than bypassing that wrapper.

This pass counts the declared raw actions. It is appended after the default analyses, without replacing their dependency order. Caching is disabled while testing this custom pipeline.

```python
from neuralqx.operator.symbolic.compiler.core.pipeline import SymbolicPassPipeline
from neuralqx.operator.symbolic.compiler.passes.base import AbstractSymbolicPass

class CountDeclaredActions(AbstractSymbolicPass):
    """Record the number of iterator-row and emission combinations."""

    @property
    def name(self):
        return "research_count_declared_actions"

    def run(self, context):
        count = sum(
            len(term.iterator.index_sets) * len(term.effective_emissions)
            for term in context.ir.terms
        )
        context.set_analysis("research_declared_actions", count)
        return {"declared_actions": count}

default = s.default_symbolic_pass_pipeline()
pipeline = SymbolicPassPipeline(
    pre_cache_passes=default.pre_cache_passes,
    post_cache_passes=(*default.post_cache_passes, CountDeclaredActions()),
)
compiler = s.SymbolicCompiler(
    pipeline=pipeline,
    options=s.SymbolicCompilerOptions(cache_enabled=False),
)
artifact = compiler.compile(definition)
report = next(
    item for item in artifact.pass_reports
    if item.pass_name == "research_count_declared_actions"
)
assert report.metadata_map()["declared_actions"] == 2
np.testing.assert_allclose(artifact.compiled_operator.to_dense(), expected)
```

A later consumer can retrieve the analysis by name. Arbitrary analyses are not automatically copied into top-level artifact metadata. This example exposes its count through the pass report instead.

An IR-rewriting pass needs additional care. Rewrites affecting compilation identity belong before signature construction, and analyses derived from the old IR must be recomputed. A pass that changes an update after connectivity analysis cannot safely retain the previous capacity or collision proofs. Document required inputs and produced analyses for every nontrivial pass.

## Register another operator target

The operator-target registry selects the class receiving the generated callable. It is separate from the lowerer registry that generates numerical execution. A native subclass is useful when an application needs a recognisable operator type while retaining the compiled kernel implementation.

The example below uses a local registry so it does not change other compilations in the process. Adding its target leaves the local native default unchanged, and the compiler selects the custom target explicitly.

```python
class ResearchCompiledOperator(s.CompiledOperator):
    """Native symbolic operator type used by this application."""

    def __init__(self, hilbert, **kwargs):
        super().__init__(hilbert, **kwargs)

registry = s.SymbolicOperatorLoweringRegistry(
    targets=[s.LOWERING_REGISTRY.resolve("neuralqx")],
    default_target="neuralqx",
)
registry.register(
    name="research",
    operator_type=ResearchCompiledOperator,
    connection_method="_get_conn_padded_kernel",
)
assert registry.default_target == "neuralqx"

custom_compiler = s.SymbolicCompiler(
    operator_lowering_registry=registry,
    options=s.SymbolicCompilerOptions(operator_lowering="research"),
)
custom = custom_compiler.compile_operator(definition)
assert isinstance(custom, ResearchCompiledOperator)
np.testing.assert_allclose(custom.to_dense(), expected)
```

The explicit constructor forwards the compiled operator's kernel-construction arguments. It prevents the native struct machinery from generating a field-based constructor with a different signature for this subclass. See [fields and static data](../structs/fields.md) for the underlying struct contract.

For a custom constructor, a target descriptor can supply a factory accepting `hilbert` and the keyword arguments `name`, `fn`, `is_hermitian`, `dtype`, and `max_conn_size`. The returned object must honour the intended target contract. Registering an arbitrary class does not automatically grant it correct estimator, pytree, or algebra behaviour.

The shared `LOWERING_REGISTRY` is available when a package deliberately registers a process-wide target. Registration alone retains its current default. Select a new default with `set_default=True` or choose the target explicitly at compilation. The [configuration guide](configuration.md) explains target precedence and cache identity.

## Extend numerical lowering only when needed

[`AbstractSymbolicLowerer`](../../api/public/neuralqx/operator/symbolic/compiler/lowering/base/index.md#neuralqx.operator.symbolic.compiler.lowering.base.AbstractSymbolicLowerer) defines `name`, `backend`, `supports(context)`, and `lower(context)`. A lowerer consumes validated IR and analyses and returns a compiled artifact. Lowerers are resolved in registry order, with the first supporting implementation selected.

The supplied compiler options currently recognise `"auto"` and `"jax"`. An alternative JAX lowerer can specialise a supported subset and fall back to the default lowerer for the rest. Supporting a genuinely different backend requires changes beyond registering a name, including backend validation and the resulting operator's execution contract.

A new expression or update primitive also needs all relevant IR consumers. These include construction and validation, symbol traversal, fingerprinting, diagnostic evaluation, target analysis, reference lowering, and planned lowering or a deliberate fallback. Implementing only the builder method can leave a definition printable but unusable, or worse, executable with inconsistent semantics across paths.

## Testing a compiler change

Start with a small independent mathematical reference. For an operator on a small indexable space, build the expected matrix directly from its physical rule, as in the clause example. Test accumulated matrix elements rather than requiring raw connection arrays to have the same shape or order across optimisation modes.

Compare raw and deduplicated output, planned and reference execution, scalar and batched inputs, and JIT execution when those paths are affected. Include boundary labels, repeated targets, cancellations, complex coefficients, conditional branches, and supported dtypes relevant to the change. For heterogeneous spaces, test the actual per-coordinate basis rather than assuming one shared range.

An adjoint change should satisfy the conjugate-transpose matrix relation on a small reference. A cache change should distinguish incompatible Hilbert objects and target descriptors. A transformation of guarded arithmetic should be checked for both values and derivatives where differentiation is supported.

Keep these tests alongside `tests/symbolic/`, and update the scientist explanation when the user-visible language changes. Follow the [contribution guide](../../api/contributing.md) before introducing a large new feature or a compiler-wide redesign.
