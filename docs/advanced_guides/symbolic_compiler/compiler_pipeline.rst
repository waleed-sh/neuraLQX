.. _advanced_symbolic_compiler_pipeline:

===============================
Compiler Pipeline and Caching
===============================

This page describes how ``SymbolicCompiler`` turns IR into an executable
operator and how cache reuse is determined.


Entry points
==============

Two common entry patterns:

.. code-block:: python

   import neuralqx.experimental.operators.symbolic as sym

   # convenience path on SymbolicOperator
   compiled = sym_op.compile(cache=True)

   # explicit compiler path
   compiler = sym.SymbolicCompiler()
   compiled = compiler.compile_operator(sym_op)

If you need pass diagnostics and metadata, use artifact-returning ``compile``:

.. code-block:: python

   artifact = compiler.compile(sym_op)
   print(artifact.backend, artifact.lowerer_name)
   print(artifact.cache_token())
   print([r.pass_name for r in artifact.pass_reports])


Compilation context
=====================

Each compile call creates a ``SymbolicCompilationContext`` containing:

- source symbolic operator,
- ``SymbolicOperatorIR``,
- effective ``SymbolicCompilerOptions``,
- mutable analysis map (pass-to-pass communication),
- pass report history,
- selected backend/lowerer fields.

Think of this context as the pipeline's working state object.


Stage-by-stage pipeline
=========================

Stage 1: IR extraction
------------------------

Compiler calls ``operator.to_ir()``. Any failure at this step is surfaced as a
``SymbolicCompilerError`` with extraction context.

Stage 2: pre-cache passes
---------------------------

Default pre-cache sequence:

1. ``symbolic_validation``
2. ``symbolic_normalization``

Key outcomes written to ``context.analyses``:

- ``validation_summary``,
- ``ir_fingerprint``,
- ``resolved_backend``,
- ``term_order``.

These passes run on every compile call, including cache hits.

Stage 3: signature and cache key
----------------------------------

After pre-cache passes, compiler builds ``SymbolicCompilationSignature`` and
then ``SymbolicCacheKey``.

Signature inputs include:

- IR fingerprint,
- selected backend target,
- hilbert size,
- dtype,
- options static signature.

Any change to those inputs yields a different cache key.

Stage 4: cache lookup
-----------------------

If caching is enabled, compiler checks ``artifact_store.get(cache_key)``.

- hit: return cached ``SymbolicCompiledArtifact`` immediately,
- miss: continue to post-cache passes and lowering.

Stage 5: post-cache passes (miss only)
----------------------------------------

Default post-cache sequence:

1. ``symbolic_fanout_analysis``
2. ``symbolic_fusion_planning``

Key outcomes written to analyses:

- ``term_fanouts``,
- ``total_fanout``,
- ``fusion_groups``.

``term_fanouts`` and ``total_fanout`` are consumed directly by the stock JAX
lowerer for padded-shape sizing. ``fusion_groups`` is currently planning
metadata: it is available for custom lowerers and diagnostics, but the default
``jax_symbolic_v1`` lowerer still lowers one runner per term.

Stage 6: lowerer resolution + lowering
----------------------------------------

Compiler asks ``SymbolicLowererRegistry`` for the first lowerer whose
``supports(context)`` returns true. The selected lowerer runs ``lower(context)``
and returns ``SymbolicCompiledArtifact``.

Default lowerer name: ``jax_symbolic_v1``.

Stage 7: artifact finalization + cache store
----------------------------------------------

When caching is enabled, compiler re-packages artifact with cache key attached
and stores it in the configured artifact store.


Cache behavior in practice
============================

Default store is in-memory, process-local, and thread-safe.

Properties:

- O(1) lookups by namespace/token composite key,
- optional soft max-entry bound with oldest-entry eviction,
- no persistence across process restarts.

Manual cache control:

.. code-block:: python

   compiler = sym.SymbolicCompiler()
   print(compiler.cache_size)
   compiler.clear_cache()


Options that materially affect behavior
=========================================

``SymbolicCompilerOptions`` fields with operational impact:

- ``backend_preference``: ``"auto"`` or ``"jax"``,
- ``cache_enabled``: lookup/store on/off,
- ``cache_namespace``: namespace partition for cache keys,
- ``enable_fusion``: controls fusion planning pass behavior,
- ``strict_validation``: fail-fast or continue-with-error-summary.

For reproducibility, treat options as part of experiment configuration and log
them alongside IR fingerprints.


Observability: what to log for reproducibility
================================================

In benchmark/research runs, record at minimum:

1. ``ir.static_fingerprint()``,
2. cache namespace,
3. pass name sequence,
4. selected lowerer name,
5. total padded size from artifact metadata.

This gives enough information to reproduce compiler decisions and explain
performance differences.


Failure triage map
====================

Compiler wraps stage failures as ``SymbolicCompilerError``. Fast triage:

- extraction errors: inspect ``to_ir`` contract and operator construction,
- pre-cache errors: validate label scope and update-op parameterization,
- post-cache errors: inspect fanout hints/fusion assumptions,
- lowering errors: inspect backend assumptions and IR op coverage.

When using artifact-returning ``compile``, inspect pass reports first before
opening deep stack traces.


Operational checklist
=======================

For deterministic symbolic compile behavior in production-like workflows:

1. pin neuraLQX version,
2. pin compiler options,
3. pin cache namespace policy,
4. clear cache between incompatible benchmark scenarios,
5. persist pass/lowerer metadata with run artifacts.

Concrete call graph (reference)
=================================

For contributors, the current ``SymbolicCompiler.compile`` execution path is:

.. code-block:: text

   operator.to_ir()
     -> SymbolicCompilationContext(...)
     -> pipeline.run_pre_cache(context)
     -> SymbolicCompilationSignature.from_context(context)
     -> signature.build_cache_key(namespace=...)
     -> artifact_store.get(cache_key)
        -> hit: return cached artifact
        -> miss:
           -> pipeline.run_post_cache(context)
           -> lowerer_registry.resolve(context)
           -> lowerer.lower(context)
           -> attach cache_key + artifact_store.put(...)
           -> return artifact

This call graph is intentionally linear and easy to instrument.  If you add
stages, preserve that transparency.

Cache-key anatomy and invalidation behavior
=============================================

The cache key token is a SHA-256 hash over a payload that includes:

- namespace,
- IR fingerprint,
- backend target,
- hilbert size,
- dtype string,
- compiler options static signature.

Operational consequences:

- changing any semantically relevant IR field produces a new key,
- changing options (even if runtime behavior is unchanged) also changes key,
- changing namespace provides manual partitioning for experiments/migrations.

The in-memory default store is process-local and non-persistent; restart clears
all entries by design.

Hit vs miss semantics (what actually reruns)
==============================================

On a cache **hit**:

- pre-cache passes run,
- key derivation runs,
- post-cache passes do not run,
- lowering does not run.

On a cache **miss**:

- full pipeline runs through post-cache + lowering,
- resulting artifact is optionally rewrapped with cache key and stored.

This distinction matters when pass side effects are used for diagnostics:
metadata from post-cache passes is only refreshed on misses.

Debugging checklist by stage boundary
=======================================

When compilation behavior differs from expectation, isolate by boundary:

1. ``to_ir`` boundary:
   if IR already looks wrong, fix DSL construction, not compiler passes.
2. pre-cache boundary:
   inspect ``validation_summary`` and ``resolved_backend`` analyses.
3. cache boundary:
   confirm namespace/options/fingerprint; unexpected hit/miss is often key drift.
4. post-cache boundary:
   inspect ``term_fanouts``/``total_fanout`` before blaming lowering.
5. lowering boundary:
   verify selected lowerer and artifact metadata; compare output shapes first.

A disciplined boundary-first workflow usually resolves issues faster than
stack-trace-first debugging.

Production observability template
===================================

In long-running sweeps, persist at least:

- ``ir.static_fingerprint()``,
- ``artifact.cache_token()``,
- pass sequence + per-pass durations,
- lowerer name,
- ``total_padded_size`` from artifact metadata.

This metadata is usually sufficient to explain both reproducibility and
performance anomalies after the fact.

Read next
===========

Continue with :doc:`extension_points` for custom passes/lowerers and custom
pipeline assembly.

For DSL authoring and operator-construction patterns, see
:ref:`symbolic_operators`.
