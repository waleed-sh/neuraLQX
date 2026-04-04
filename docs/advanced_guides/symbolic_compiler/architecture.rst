.. _advanced_symbolic_architecture:

==============================
Symbolic System Architecture
==============================

The symbolic operator stack is a staged system that maps declarative operator
intent into an executable static-shape connectivity kernel.

At a high level:

.. code-block:: text

   DOperator (DSL builder)
      -> SymbolicOperator / SymbolicOperatorSum
      -> SymbolicOperatorIR (immutable typed IR)
      -> SymbolicCompiler
         -> pre-cache passes (validation + normalization)
         -> cache signature + cache lookup
         -> post-cache passes (fanout + fusion planning)
         -> lowering (JAXSymbolicLowerer)
      -> CompiledOperator (runtime get_conn_padded kernel)


Why the architecture is staged
================================

The separation is deliberate and solves three engineering constraints at once:

- **Authoring clarity**: users write physics-level logic with iterator/predicate/
  emission primitives rather than low-level JAX control flow.
- **Compiler observability**: explicit IR + pass stages make validation,
  diagnostics, and extension significantly easier.
- **Runtime correctness/performance**: the lowerer can enforce static-shape
  padding and deterministic branch semantics required by VMC/JIT workflows.


Lifecycle phases
==================

Definition phase (Python, no JAX tracing)
--------------------------------------------

``DOperator`` executes eagerly in Python. Each iterator call opens a term,
``where(...)`` contributes predicate logic, and each ``emit(...)`` contributes
an emission branch.

At this phase there is no numerical execution; the output is a declarative
symbolic object.

Compilation phase (IR + passes + lowering)
--------------------------------------------

Compilation starts from ``to_ir()`` and then runs through ``SymbolicCompiler``:

1. pre-cache pass stage,
2. cache signature/key derivation,
3. cache lookup,
4. post-cache passes on miss,
5. lowerer resolution and lowering,
6. artifact packaging and optional cache store.

The output is a ``SymbolicCompiledArtifact`` (or directly a ``CompiledOperator``
via ``compile_operator``).

Execution phase (JAX runtime)
-------------------------------

The lowered operator exposes ``get_conn_padded`` with static output shapes.
For each input configuration ``x`` the runtime semantics are:

1. iterate according to each term iterator,
2. evaluate predicate in source environment,
3. apply update program to obtain connected state ``x'``,
4. evaluate amplitude,
5. emit valid branches,
6. pad/truncate to static fanout budget.

Connected outputs are treated as a **multiset** (**duplicates are not coalesced**).


One-sample semantic view
==========================

A useful mental model for architecture debugging is this pseudo-code:

.. code-block:: text

   branches = []
   for term in ir.terms:
       for visit in term.iterator.index_sets:
           env = bind_labels(x, visit)
           if eval_predicate(term.predicate, env):
               for emission in term.effective_emissions:
                   x_prime = apply_update(emission.update_program, x, env)
                   mel = eval_amplitude(emission.amplitude, x, x_prime, env)
                   branches.append((x_prime, mel))

   xp, mels = static_pad(branches, total_fanout)

The lowerer materializes a vectorized JAX realization of this behavior.


Core objects and responsibilities
===================================

``DOperator``
---------------

Builder facade that captures user intent.

Responsibilities:

- term segmentation at iterator boundaries,
- predicate accumulation,
- emission accumulation,
- optional fanout hints / metadata,
- construction of ``SymbolicOperator``.

``SymbolicOperator`` / ``SymbolicOperatorSum``
------------------------------------------------

Declarative operator containers.

Responsibilities:

- expose ``to_ir()`` contract,
- support symbolic algebra composition,
- guard execution before compilation,
- provide max-connectivity estimation.

``SymbolicOperatorIR``
------------------------

Immutable typed payload consumed by passes/lowerers.

Responsibilities:

- deterministic serialization/fingerprint,
- free-symbol introspection,
- stable representation for cache signatures.

``SymbolicCompilationContext``
--------------------------------

Mutable envelope shared by passes and lowerer.

Responsibilities:

- carry source operator + IR + effective options,
- carry pass analysis outputs,
- carry pass reports,
- track selected backend/lowerer.

``SymbolicCompiler``
----------------------

Coordinator that wires pipeline, cache, and lowerers.

Responsibilities:

- run staged passes,
- compute signatures/keys,
- route cache hit/miss,
- invoke lowerer,
- package artifact.

``JAXSymbolicLowerer``
------------------------

Backend translator from IR to executable ``CompiledOperator``.

Responsibilities:

- interpret amplitude/predicate/update IR semantics,
- build per-term runners,
- compose a single batched kernel,
- enforce static fanout output shape.


What flows across stages
==========================

Data that remains stable from definition to lowering:

- operator identity: ``operator_name``, ``hilbert_size``, ``dtype_str``,
- structural IR payload: terms, predicates, emissions,
- metadata payloads (operator + term level).

Data derived by compilation stages:

- normalized backend selection (analysis key ``resolved_backend``),
- IR fingerprint (analysis key ``ir_fingerprint``),
- fanout planning (``term_fanouts``, ``total_fanout``),
- fusion grouping (``fusion_groups``),
- pass timing/metadata reports.


Inspection workflow
=====================

.. code-block:: python

   import neuralqx.experimental.operators.symbolic as sym

   sym_op = (
       sym.DOperator(hi, "demo")
       .for_each_site("i")
       .where(sym.site("i") > 0)
       .emit(sym.shift("i", -1), amplitude=1.0)
       .build()
   )

   # Definition output
   ir = sym_op.to_ir()
   print(ir.operator_name, ir.term_count)
   print(ir.static_fingerprint())

   # Compilation output
   compiler = sym.SymbolicCompiler()
   artifact = compiler.compile(sym_op)

   print(artifact.backend, artifact.lowerer_name)
   print(artifact.cache_token())
   print([r.pass_name for r in artifact.pass_reports])

   # Runtime output
   compiled = artifact.compiled_operator
   xp, mels = compiled.get_conn_padded(x_batch)


Debugging by stage
====================

When behavior is unexpected, triage by stage rather than by symptom:

1. **Definition mismatch**: inspect ``print(sym_op.to_ir())`` and verify term
   segmentation / labels / emissions first.
2. **Validation failures**: check symbol scope and update-op parameters.
3. **Static-shape surprises**: inspect ``total_fanout`` and term fanout hints.
4. **Runtime branch differences**: validate multiset semantics and emission
   ordering assumptions.


Design constraints worth preserving
=====================================

- **Determinism**: fingerprints/signatures must be stable.
- **Immutability at IR layer**: transforms produce new objects.
- **Static shape first**: compiler analyses exist to keep JAX outputs static.
- **Backend pluggability**: lowerer registry isolates backend-specific concerns.

Static-shape contract in concrete terms
=========================================

The most important architectural invariant for runtime interoperability is the
static-shape contract of ``get_conn_padded``:

- for fixed compiled artifact + fixed input rank, output rank is fixed,
- connection-axis width is fixed by compile-time fanout analysis/hints,
- inactive branches are represented by zeroed matrix elements (and padded rows),
- duplicate connected states are preserved (branch multiset semantics).

Why this matters:

- NetKet expectation paths assume predictable layout for batching and JIT.
- Shape drift across calls would invalidate compilation caches and break tracing.
- Compiler passes are therefore designed around conservative upper bounds rather
  than exact per-sample branch counts.

Execution invariants by phase
===============================

Definition phase invariants
-----------------------------

- iterator calls partition terms deterministically,
- each term has at least one emission before build,
- term-local label scope is explicit and finite.

Compilation phase invariants
------------------------------

- IR is immutable, passes annotate context, not IR objects in place,
- pre-cache analyses required for key generation are deterministic,
- cache key identity is a function of IR + options + backend target,
- lowering consumes analyzed/static payload and emits one executable operator.

Runtime phase invariants
--------------------------

- one sample is evaluated as term-runner concatenation,
- branch validity gates matrix elements (invalid branch => zero mel),
- output is padded/truncated to the static budget exactly once at composition.

Cost model at architecture level
===================================

A practical architecture-level cost proxy for one operator evaluation is:

.. math::

   C \propto \sum_t \left(M_t \times E_t \times K_t\right),

where:

- ``M_t`` is iterator row count for term ``t``,
- ``E_t`` is emission count for term ``t``,
- ``K_t`` approximates per-branch expression/update work.

This is intentionally coarse, but it is usually enough to identify order-of-
magnitude regressions during code review:

- dense pair iterators can dominate ``M_t``,
- extra emissions increase ``E_t`` linearly,
- complex conditional update/amplitude trees increase ``K_t``.

Current implementation note
-----------------------------

The default pipeline computes fusion groups, but the stock
``jax_symbolic_v1`` lowerer currently lowers one runner per term.  In other
words, fusion metadata is available for introspection/extension, but not yet a
guaranteed runtime optimization in the built-in lowerer.

Architecture review checklist for contributors
================================================

When reviewing changes to symbolic internals, check the following explicitly:

1. Does the change preserve deterministic IR fingerprinting?
2. Does it preserve source vs emitted symbol semantics?
3. Does it preserve branch-multiset behavior (no accidental dedup)?
4. Does it preserve static output shape across all branches?
5. Does it preserve cache identity correctness for changed behavior?
6. Does it surface enough metadata for pass/lowering observability?

Treat this checklist as part of API-quality review, not optional polish.

Read next
===========

For field-level IR interpretation, continue with :doc:`ir_model`.

For user-facing DSL authoring and end-to-end operator examples, see
:ref:`symbolic_operators`.
