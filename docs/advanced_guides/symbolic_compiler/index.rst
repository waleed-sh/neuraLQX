.. _advanced_symbolic_compiler:

================================================
Declarative Symbolic IR and Compiler Internals
================================================

This track documents the implementation-level architecture behind
``neuralqx.experimental.operators.symbolic``.  It is written for users who
need to reason about invariants, not just call APIs.

Who this is for
=================

Use this track if you are:

- debugging subtle symbolic compile/runtime mismatches in research workloads,
- extending the compiler (passes, lowerers, cache policy),
- reviewing performance regressions where static fanout or iterator shape matters,
- building tooling that inspects or transforms symbolic IR.

If your goal is authoring operators in the DSL, start with
:ref:`symbolic_operators` first and return here when you need internals.

What this track covers
========================

The advanced pages focus on hard contracts and extension surfaces:

- the typed IR schema and what each field means operationally,
- stage-by-stage compiler execution (including cache key composition),
- runtime semantics of lowered ``get_conn_padded`` kernels,
- safe extension patterns and where current APIs are intentionally not pluggable.

The writing assumes familiarity with JAX and static-shape operator interfaces,
and intentionally favors precision over introductory pacing.

How to use these pages in practice
====================================

Two workflows tend to recur in real projects:

1. **Diagnosis workflow**
   Read :doc:`architecture` -> :doc:`ir_model` -> :doc:`compiler_pipeline`.
   This path is optimized for answering "what did the compiler believe?" and
   "at which stage did behavior diverge?".

2. **Extension workflow**
   Read :doc:`architecture` -> :doc:`extension_points` -> :doc:`ir_model`.
   This path is optimized for adding or modifying compiler components while
   preserving symbolic semantics and static-shape guarantees.

In both workflows, keep an IR dump (`print(op.to_ir())`) and pass reports close
at hand; they are the highest-signal debugging artifacts.

Reading order
===============

1. :doc:`architecture`
   System boundaries, lifecycle phases, and stage responsibilities.
2. :doc:`ir_model`
   Field-level IR contracts and line-by-line IR interpretation.
3. :doc:`compiler_pipeline`
   Cache signature semantics, pass execution model, and failure triage.
4. :doc:`extension_points`
   Concrete recipes for custom passes/lowerers and core-level extensions.

Experimental-status note
==========================

The symbolic subsystem is experimental.  Internal contracts documented here are
intended to help contributors and advanced users reason precisely, but they are
not yet governed by stable-API deprecation guarantees.

When you depend on these internals for production research pipelines, pin both
neuraLQX version and compiler options in experiment manifests.

From internals back to DSL authoring
======================================

For user-facing DSL construction patterns and end-to-end examples, see
:ref:`symbolic_operators`.


.. toctree::
   :hidden:
   :maxdepth: 2

   architecture
   ir_model
   compiler_pipeline
   extension_points
