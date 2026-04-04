.. _advanced_symbolic_extension_points:

=====================================
Extension Points and Customization
=====================================

This page describes how to extend the symbolic compiler without breaking core
semantic contracts.

Supported extension surfaces:

- pass pipeline composition,
- lowerer registry and backend selection,
- artifact-store policy,
- compiler options and namespace strategy.


Extension map
===============

Default control flow:

.. code-block:: text

   DOperator/SymbolicOperator
      -> SymbolicCompiler
      -> SymbolicPassPipeline (pre-cache / post-cache)
      -> SymbolicLowererRegistry.resolve(...)
      -> selected lowerer.lower(...)
      -> SymbolicCompiledArtifact
      -> artifact store put/get

Each stage is replaceable through ``SymbolicCompiler(...)`` constructor
arguments.


Custom pass design
=====================

Use custom passes for analysis/policy that should run before lowering.

Pass contract summary:

- inherit ``AbstractSymbolicPass``,
- implement ``name`` and ``run(context)``,
- write results via ``context.set_analysis(...)``,
- return a metadata mapping for pass reports.

Example pass:

.. code-block:: python

   from collections.abc import Mapping
   from typing import Any

   from neuralqx.experimental.operators.symbolic.compiler.core.context import (
       SymbolicCompilationContext,
   )
   from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
       AbstractSymbolicPass,
   )


   class TermStatsPass(AbstractSymbolicPass):
       @property
       def name(self) -> str:
           return "term_stats"

       def run(
           self,
           context: SymbolicCompilationContext,
       ) -> Mapping[str, Any] | None:
           payload = {
               "term_count": context.ir.term_count,
               "free_symbol_count": len(context.ir.free_symbols),
           }
           context.set_analysis("term_stats", payload)
           return payload

Pipeline insertion pattern:

.. code-block:: python

   import neuralqx.experimental.operators.symbolic as sym

   from neuralqx.experimental.operators.symbolic.compiler.core.pipeline import (
       SymbolicPassPipeline,
   )

   default = sym.default_symbolic_pass_pipeline()

   pipeline = SymbolicPassPipeline(
       pre_cache_passes=[*default.pre_cache_passes, TermStatsPass()],
       post_cache_passes=default.post_cache_passes,
   )

   compiler = sym.SymbolicCompiler(pipeline=pipeline)
   artifact = compiler.compile(sym_op)
   print([r.pass_name for r in artifact.pass_reports])


Pass placement guidance
=========================

Use **pre-cache** for:

- structural validation,
- deterministic normalization,
- analyses that must influence cache identity indirectly.

Use **post-cache** for:

- heavier analyses used only on misses,
- lowering planning metadata (fanout buckets, clustering, scheduling hints).

Rule of thumb: if pass output affects whether cached artifacts are valid,
it belongs before key derivation.


Custom lowerer registration
=============================

Lowerers are resolved by first-match priority in ``SymbolicLowererRegistry``.
Use ``register_first(...)`` for overrides.

Example: delegate to built-in JAX lowerer, then annotate artifact metadata.

.. code-block:: python

   import neuralqx.experimental.operators.symbolic as sym

   from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
       SymbolicCompiledArtifact,
   )
   from neuralqx.experimental.operators.symbolic.compiler.core.context import (
       SymbolicCompilationContext,
   )
   from neuralqx.experimental.operators.symbolic.compiler.lowering.base import (
       AbstractSymbolicLowerer,
   )
   from neuralqx.experimental.operators.symbolic.compiler.lowering.jax_lowerer import (
       JAXSymbolicLowerer,
   )


   class TaggedJAXLowerer(AbstractSymbolicLowerer):
       def __init__(self) -> None:
           self._delegate = JAXSymbolicLowerer()

       @property
       def name(self) -> str:
           return "jax_symbolic_tagged_v1"

       @property
       def backend(self) -> str:
           return "jax"

       def supports(self, context: SymbolicCompilationContext) -> bool:
           selected = context.selected_backend or context.options.backend_preference
           use_tagged = bool(context.metadata.get("use_tagged_jax_lowerer", False))
           return context.ir.mode == "symbolic" and selected in {"jax", "auto"} and use_tagged

       def lower(self, context: SymbolicCompilationContext) -> SymbolicCompiledArtifact:
           artifact = self._delegate.lower(context)
           meta = artifact.metadata_map()
           meta["custom_mode"] = "tagged"
           context.set_selected_lowerer(self.name)
           return SymbolicCompiledArtifact.create(
               operator_name=artifact.operator_name,
               backend=artifact.backend,
               lowerer_name=self.name,
               compiled_operator=artifact.compiled_operator,
               cache_key=artifact.cache_key,
               pass_reports=artifact.pass_reports,
               metadata=meta,
           )


   registry = sym.default_symbolic_lowerer_registry()
   registry.register_first(TaggedJAXLowerer())

   compiler = sym.SymbolicCompiler(lowerer_registry=registry)
   artifact = compiler.compile(sym_op, metadata={"use_tagged_jax_lowerer": True})
   print(artifact.lowerer_name)


Lowerer safety checklist
==========================

A custom lowerer should preserve:

- branch multiset semantics (no implicit dedup),
- static shape contracts (respect analyzed/predicted fanout bounds),
- predicate/update/amplitude semantics parity with IR definitions,
- deterministic behavior for identical IR + options.

If you intentionally diverge from default semantics, surface that explicitly in
artifact metadata and documentation.


Custom artifact store
=======================

To use non-default persistence policy, implement
``AbstractSymbolicArtifactStore``.

.. code-block:: python

   from neuralqx.experimental.operators.symbolic.compiler.cache.store import (
       AbstractSymbolicArtifactStore,
   )


   class MyArtifactStore(AbstractSymbolicArtifactStore):
       def get(self, key):
           ...

       def put(self, key, artifact):
           ...

       def invalidate(self, key):
           ...

       def clear(self):
           ...

       def __len__(self):
           ...

Store contract requirements:

- key identity is namespace + token,
- return full ``SymbolicCompiledArtifact`` objects,
- ensure thread-safe get/put under concurrent compiles.


Assembling a fully custom compiler
====================================

.. code-block:: python

   import neuralqx.experimental.operators.symbolic as sym

   options = sym.SymbolicCompilerOptions(
       backend_preference="jax",
       cache_enabled=True,
       cache_namespace="my_project_symbolic_v2",
       enable_fusion=True,
       strict_validation=True,
   )

   compiler = sym.SymbolicCompiler(
       pipeline=pipeline,
       lowerer_registry=registry,
       artifact_store=MyArtifactStore(),
       options=options,
   )

   compiled = compiler.compile_operator(sym_op)


Extension testing strategy
============================

Minimum regression matrix for extension safety:

1. cache off/on parity on connectivity outputs,
2. repeated compile cache-hit behavior and stable token,
3. representative ``expect`` / ``expect_and_grad`` integration runs,
4. strict vs non-strict validation behavior where relevant,
5. pass report presence and metadata sanity checks,
6. static-shape assertions on ``get_conn_padded`` outputs.


Operational guardrails
========================

Keep these invariants explicit in your extension review process:

- preserve determinism,
- preserve static-shape guarantees,
- preserve multiset branch semantics,
- avoid in-place IR mutation,
- expose actionable diagnostics.

This keeps custom pipelines compatible with both symbolic semantics and VMC
runtime expectations.


What is pluggable today vs what requires core edits
=====================================================

Pluggable without modifying core modules:

- custom passes via ``SymbolicPassPipeline``,
- custom lowerers via ``SymbolicLowererRegistry``,
- custom artifact stores via ``AbstractSymbolicArtifactStore``,
- custom compiler assembly (`pipeline + registry + store + options`).

Requires core edits (no standalone plugin hook yet):

- introducing a brand-new iterator IR kind,
- introducing new amplitude/predicate op codes,
- introducing new update-op kinds consumed by stock lowerers.

This distinction is important for project planning: some extensions are
configuration-time, others are source-level compiler development.


Pattern A: domain iterators without compiler changes
======================================================

If your goal is ergonomic domain-specific iterators (edges, plaquettes, stars),
you often do **not** need a new IR iterator kind. Use ``for_each(..., over=)``
with helper functions that build static index tuples.

Example: an edge iterator helper layered on top of existing API:

.. code-block:: python

   from collections.abc import Iterable

   def for_each_edge(
       builder,
       *,
       label_src: str = "src",
       label_dst: str = "dst",
       edges: Iterable[tuple[int, int]],
   ):
       edge_rows = tuple((int(a), int(b)) for a, b in edges)
       if not edge_rows:
           raise ValueError("edge iterator requires at least one edge row")
       return builder.for_each((label_src, label_dst), over=edge_rows)

   op = (
       for_each_edge(DOperator(hi, "edge_hop"), edges=edge_list)
       .where(site("src") > 0)
       .emit(shift("src", -1).shift("dst", +1), amplitude=-1.0)
       .build()
   )

Advantages:

- zero compiler-core changes,
- full compatibility with validation, analysis, lowering, and caching,
- minimal maintenance burden.

For many production DSLs, this wrapper pattern is the highest-leverage option.


Pattern B: adding a new iterator IR kind (core extension)
============================================================

When you truly need a distinct iterator representation, treat it as a
cross-cutting change.  A minimal checklist:

1. Add iterator dataclass in IR layer (e.g. new class in ``ir/term.py``).
2. Extend ``SymbolicIRTerm`` typing/formatting paths.
3. Add builder entry points in ``dsl/op.py``.
4. Extend validation label-scope handling in ``ir/validate.py``.
5. Extend fanout analysis in ``passes/analysis.py``.
6. Extend lowerer runner dispatch (e.g. ``jax_lowerer.py``).
7. Update serialization/fingerprint paths in ``ir/program.py``.
8. Add docs + regression tests (IR print, validation, lowering shape).

Skeleton (illustrative):

.. code-block:: python

   # in ir/term.py
   @dataclasses.dataclass(frozen=True, repr=False)
   class EdgeIteratorSpec:
       edges: tuple[tuple[int, int], ...]
       src_label: str = "src"
       dst_label: str = "dst"

       @property
       def kind(self) -> str:
           return "edge"

       @property
       def labels(self) -> tuple[str, str]:
           return (self.src_label, self.dst_label)

Then, in JAX lowering, branch on ``term.iterator.kind`` and map to a dedicated
runner (or normalize to K-body rows early).  Keep semantics identical to the
existing model: source-env predicate, update, emitted-env amplitude, multiset
branch output.


Adding a new update op end-to-end
===================================

Suppose you want ``clamp_site`` semantics (bounded write).

Touch points:

1. Add new kind in ``ir/update.py`` ``_UPDATE_OP_KINDS``.
2. Add builder method in ``dsl/rewrite.py`` (factory + chainable API).
3. Add render support in ``_render_update_op`` for IR readability.
4. Add validation requirements in ``ir/validate.py``.
5. Add runtime semantics in ``jax_lowerer._apply_single_update_op``.
6. Add tests for builder -> IR -> lowered behavior.

Representative lowerer branch:

.. code-block:: python

   if op.kind == "clamp_site":
       idx = jnp.int32(_eval_amplitude(op.get("site"), env))
       lo = _eval_amplitude(op.get("low"), env)
       hi = _eval_amplitude(op.get("high"), env)
       cur = x_prime[idx]
       return x_prime.at[idx].set(jnp.clip(cur, lo, hi))

When adding update ops, ensure they compose with ``cond_branch`` and preserve
deterministic sequential semantics.


Adding a new amplitude op end-to-end
======================================

Example target: ``exp`` amplitude op.

Touch points:

1. Add ``"exp"`` to ``_AMPLITUDE_OPS`` in ``ir/expressions.py``.
2. Add constructor convenience method ``AmplitudeExpr.exp(...)``.
3. Add string rendering support in ``_render_amplitude``.
4. Add lowering semantics in ``jax_lowerer._eval_amplitude``.
5. Ensure symbol collection/serialization still traverses nested args.
6. Add docs and tests (builder arithmetic, IR print, runtime eval).

Representative lowering clause:

.. code-block:: python

   if op == "exp":
       return jnp.exp(_eval_amplitude(expr.args[0], env, ...))

Do not ship new IR ops without corresponding lowering support, validation should
fail early if coverage is incomplete.


Custom lowerer from scratch: recommended structure
====================================================

For a genuinely different backend strategy, start from
``AbstractSymbolicLowerer`` and keep scope narrow:

1. implement ``supports(context)``,
2. implement ``lower(context)`` returning ``SymbolicCompiledArtifact``,
3. set ``context.set_selected_lowerer(self.name)`` before returning,
4. preserve static output shape and branch multiset behavior.

A practical implementation approach is:

- first delegate to ``JAXSymbolicLowerer`` and add metadata/instrumentation,
- then incrementally replace internals once behavioral parity tests pass.

This keeps migration risk low while still enabling aggressive optimization work.


Extension acceptance tests
=============================

Before merging extension work, run a focused matrix:

1. IR determinism:
   same input definition yields same fingerprint and cache key.
2. Cache parity:
   cache-enabled and cache-disabled compiles produce identical connectivity.
3. Shape stability:
   ``get_conn_padded`` output shapes are invariant across representative samples.
4. Semantics parity:
   duplicate-branch multiset behavior preserved; no accidental dedup.
5. Error quality:
   invalid definitions fail with actionable messages at validation/lowering stage.
6. Integration parity:
   ``expect`` / ``expect_and_grad`` behave identically for baseline operators.

For compiler extensions, these tests are part of semantic correctness.


Read next
===========

For full runtime and cache-stage behavior, see :doc:`compiler_pipeline`.

For user-facing DSL construction patterns and operator examples, see
:ref:`symbolic_operators`.
