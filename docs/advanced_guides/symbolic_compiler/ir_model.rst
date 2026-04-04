.. _advanced_symbolic_ir_model:

========================
Symbolic IR Data Model
========================

This page documents the typed IR consumed by the symbolic compiler and how to
interpret it in both programmatic and printed form.

All major IR containers are frozen dataclasses. Transformations create new
objects; in-place mutation is not part of the IR contract.


IR layers at a glance
=======================

.. code-block:: text

   SymbolicOperatorIR
      -> SymbolicIRTerm (one per iterator scope)
         -> KBodyIteratorSpec
         -> PredicateExpr
         -> EmissionSpec (one per emit)
            -> UpdateProgram (UpdateOp sequence)
            -> AmplitudeExpr

Read this as: operator-level container, then term-level scheduling/guarding,
then branch-level rewrite + matrix element.


Root container: ``SymbolicOperatorIR``
========================================

``SymbolicOperatorIR`` is the full declarative payload for one symbolic
operator.

Core fields:

- ``operator_name``: human-readable operator identity,
- ``mode``: ``"symbolic"`` for DSL-built operators,
- ``hilbert_size``: expected configuration width,
- ``dtype_str``: matrix-element dtype label,
- ``is_hermitian``: declared hermiticity flag,
- ``terms``: ordered tuple of ``SymbolicIRTerm``,
- ``metadata``: stable sorted key/value tuple.

High-value methods/properties:

- ``as_dict()``: deterministic JSON-serializable payload,
- ``static_fingerprint()``: SHA-256 digest used by cache signatures,
- ``free_symbols``: union of unresolved symbol names across terms,
- ``term_count``: number of terms.


Term container: ``SymbolicIRTerm``
====================================

A term corresponds to one iterator scope in ``DOperator``.

Core fields:

- ``name``: term identifier,
- ``iterator``: currently ``KBodyIteratorSpec`` in modern DSL usage,
- ``predicate``: ``PredicateExpr`` evaluated in source environment,
- ``emissions``: optional tuple of ``EmissionSpec`` (multi-emission path),
- ``fanout_hint``: optional static branch upper bound,
- ``metadata``: stable term metadata tuple.

Compatibility note:

``update_program`` / ``amplitude`` / ``branch_tag`` remain available for
single-emission compatibility. Runtime and tooling should primarily consume
``effective_emissions``.


Iterator model: ``KBodyIteratorSpec``
=======================================

``KBodyIteratorSpec`` fully materializes the iteration domain.

- ``labels``: tuple of iterator labels,
- ``index_sets``: tuple of index tuples, one per iteration.

Examples:

- global iterator: ``labels=()``, ``index_sets=((),)``,
- site iterator: ``labels=("i",)``, ``index_sets=((0,), (1,), ...)``,
- pair iterator: ``labels=("i", "j")``, full cartesian product.

Because index sets are explicit, compilation is deterministic and independent
of runtime graph traversal side effects.


Emission model: ``EmissionSpec``
==================================

Each emission is one branch candidate from one iterator evaluation.

- ``update_program``: ``x -> x'`` rewrite logic,
- ``amplitude``: matrix element expression,
- ``branch_tag``: optional diagnostics label.

Multiple ``emit(...)`` calls in one term create multiple emissions and therefore
multiple branch candidates per iterator row.


Expressions and predicate trees
=================================

``AmplitudeExpr``
-------------------

Expression tree for numeric matrix-element logic.

Main operation groups:

- constants/symbols: ``const``, ``symbol``,
- arithmetic: ``add``, ``sub``, ``mul``, ``div``, ``pow``, ``neg``,
- unary transforms: ``sqrt``, ``abs_``, ``conj``,
- state reads: ``static_index``, ``static_emitted_index``,
- Hilbert wrap helper: ``wrap_mod``.

``PredicateExpr``
-------------------

Boolean tree for branch activation.

Main operation groups:

- constants: ``const``,
- comparisons over amplitudes: ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``,
- boolean composition: ``and``, ``or``, ``not``.


Symbol namespace conventions
==============================

Compiler/lowerer resolve names by convention:

- ``site:<label>:value``
- ``site:<label>:index``
- ``emit:<label>:value``
- ``emit:<label>:index``

Symbols outside these bound namespaces are treated as free symbols.


Update IR
===========

``UpdateProgram`` is an ordered tuple of ``UpdateOp`` primitives.

Supported ``UpdateOp.kind`` values:

- ``write_site``
- ``shift_site``
- ``shift_mod_site``
- ``swap_sites``
- ``affine_site``
- ``permute_sites``
- ``scatter``
- ``cond_branch``
- ``invalidate_branch``

Execution semantics are sequential. A later op observes all prior mutations in
that same program.


Concrete printed IR example
=============================

The following is an actual ``print(op.to_ir())`` output from a two-emission
single-site operator:

.. code-block:: python

   import neuralqx.experimental.operators.symbolic as sym
   import netket as nk

   hi = nk.hilbert.Spin(s=0.5, N=3)

   op = (
       sym.DOperator(hi, "ladder_demo", hermitian=True)
       .for_each_site("i")
       .where(sym.site("i").value < 1)
       .emit(sym.shift("i", +1), amplitude=0.5, tag="raise")
       .emit(sym.shift("i", -1), amplitude=-0.5, tag="lower")
       .build()
   )

   print(op.to_ir())

.. code-block:: text

   symbolic.operator @"ladder_demo" [dtype=float64, hermitian=true, hilbert_size=3] {
     ; 1 term(s)

     term #0 "0" [kbody, n_iter=3, fanout=6] {
       iterate: for (i,) in [(0,), (1,), (2,)]
       where:   (x[i] < 1)
       emit #0 [tag='raise']:
         update:    x'[i] = (x[i] + 1)
         amplitude: 0.5
       emit #1 [tag='lower']:
         update:    x'[i] = (x[i] + -1)
         amplitude: -0.5
     }

   }


Annotated IR — token-by-token
==============================

The following annotates every token in the IR dump above. Cross-referencing
this against a ``print(op.to_ir())`` output is the fastest way to verify that
the compiler sees what you intended.

.. code-block:: text

   symbolic.operator @"ladder_demo" [dtype=float64, hermitian=true, hilbert_size=3] {

``symbolic.operator``
   Mode tag. The compiler's lowerer selection checks ``ir.mode == "symbolic"``
   before accepting this artifact. Only DSL-built operators carry this tag.

``@"ladder_demo"``
   ``ir.operator_name``: the string passed to ``DOperator(hi, "ladder_demo")``.
   Propagated to ``CompiledOperator.name``, the in-memory cache key, and all
   pass reports.  Pick names that are unique per experiment to avoid cache
   collisions.

``dtype=float64``
   ``ir.dtype_str``. Resolved from the ``dtype=`` argument to ``DOperator``,
   defaults to ``"float64"`` when omitted. The JAX lowerer casts all
   amplitude results to this dtype before assembling the padded output.

``hermitian=true``
   ``ir.is_hermitian``.  Set by ``DOperator(..., hermitian=True)``.  Propagated
   unchanged to ``CompiledOperator.is_hermitian``, does not alter the kernel
   structure. The VMC runtime uses this flag for expectation-value shortcuts
   (e.g. taking the real part rather than averaging over both directions).

``hilbert_size=3``
   ``ir.hilbert_size`` — ``hilbert.size`` captured at build time.  The lowerer
   uses this to size the configuration vector ``x ∈ ℤ^{hilbert_size}`` and to
   materialise the site iteration domain for ``for_each_site``.

.. code-block:: text

     ; 1 term(s)

A comment line encoding ``ir.term_count``. With one term there is one runner
function, the padded output shape is determined by that single term's fanout.
With N terms, N runners are composed and their outputs concatenated.

.. code-block:: text

     term #0 "0" [kbody, n_iter=3, fanout=6] {

``#0``
   Index of this term within ``ir.terms``.  The lowerer processes terms in this
   order, term ordering affects the column layout of the padded output but not
   correctness.

``"0"``
   ``term.name``, auto-assigned by ``DOperator`` from a sequential counter.
   Appears in pass reports and ``fusion_groups`` analysis output.  Can be used
   to reference specific terms when reading pass diagnostics.

``kbody``
   ``iterator.kind``.  Always ``"kbody"`` in modern DSL usage.  The fusion pass
   groups terms by this tag: two terms with the same ``kind`` are candidates for
   a shared kernel loop (when fusion is enabled).

``n_iter=3``
   ``len(iterator.index_sets)``.  The number of iterator evaluations per input
   configuration, for a ``for_each_site`` over ``hilbert.size = 3``, this is 3.
   For a ``for_each_pair``, it would be 9.  For a custom edge list with 47
   entries, it would be 47. This directly multiplies per-sample cost.

``fanout=6``
   ``term.fanout_hint``, the static per-term branch budget.  With ``n_iter=3``
   and 2 emissions per row, the budget is ``3 × 2 = 6``.  This value determines
   the padded output column count contributed by this term.  When
   ``fanout_hint`` is ``None`` (shown as ``?``), the analysis pass computes it
   automatically using the same formula.

.. code-block:: text

       iterate: for (i,) in [(0,), (1,), (2,)]

``for (i,) in [...]``
   The formatted form of ``iterator.labels`` and ``iterator.index_sets``.  The
   K-body iterator is a fully materialized, static list of index tuples, known
   at compile time.  The JAX lowerer converts this list into a constant
   ``jnp.array`` of shape ``[n_iter, K]`` and vmaps a per-row kernel over it.

``(i,)``
   Single-element label tuple: this is a 1-body (single-site) iterator with
   label ``"i"``.  A pair iterator would show ``(i, j)``; a plaquette iterator
   would show ``(a, b, c, d)``.

``[(0,), (1,), (2,)]``
   The explicit index set.  For ``for_each_site`` over a size-3 space, this is
   ``range(3)``.  For ``for_each(("src", "dst"), over=edge_list)``, this would
   be your edge list.  Because the set is static, compilation is deterministic
   and cache-stable regardless of runtime state.

.. code-block:: text

       where:   (x[i] < 1)

Textual representation of the ``PredicateExpr`` tree. ``x[i]`` means "the
quantum number at site ``i`` in the source configuration", resolved by the
lowerer as ``x[index_row[k_for_label_i]]`` inside the vmapped kernel.  Rows
that fail this predicate produce zero matrix elements and invalid branches;
the lowerer writes zeros into those output slots, not garbage values.

.. code-block:: text

       emit #0 [tag='raise']:
         update:    x'[i] = (x[i] + 1)
         amplitude: 0.5

``emit #0``
   Emission index within ``term.effective_emissions``.  Emission ordering
   determines which column group of the padded output receives these branches.

``[tag='raise']``
   ``EmissionSpec.branch_tag`` set via ``.emit(..., tag="raise")``.  Purely
   diagnostic: appears in IR dumps and pass reports but does not affect kernel
   behavior or output layout.

``update:    x'[i] = (x[i] + 1)``
   Textual form of the ``UpdateProgram`` for this emission.
   ``shift("i", +1)`` generates a single ``shift_site`` op with
   ``delta = const(1)``.  The lowerer applies update ops sequentially to
   produce the connected-state array.  The notation ``x'[i] = (x[i] + 1)``
   means "the quantum number at site ``i`` in the output equals the source
   value plus one"; all other sites are copied verbatim from ``x``.

``amplitude: 0.5``
   Textual form of the ``AmplitudeExpr`` for this emission, here a ``const``
   leaf.  The lowerer evaluates this expression in the environment:

   .. code-block:: text

      {
          "__x__":            x,               # source configuration
          "__x_prime__":      x',              # connected configuration (after update)
          "site:i:index":     index_row[0],    # integer site index
          "site:i:value":     x[index_row[0]], # quantum number at site i
          "emit:i:index":     index_row[0],    # same index, emitted side
          "emit:i:value":     x'[index_row[0]] # quantum number at site i in x'
      }

   Any ``AmplitudeExpr`` node can read any of these keys.  ``static_index(k)``
   reads ``__x__[k]``; ``static_emitted_index(k)`` reads ``__x_prime__[k]``.

.. code-block:: text

       emit #1 [tag='lower']:
         update:    x'[i] = (x[i] + -1)
         amplitude: -0.5

Second emission: analogous structure.  Both emissions are evaluated for every
iterator row that passes the ``where`` predicate, producing 2 branches per
active row.  Because ``n_iter=3`` sites and each active site produces 2
branches, the maximum (fanout) is ``3 × 2 = 6``.  Inactive rows contribute
two zero-mel/zero-xp padding entries each.

Runtime behavior summary
---------------------------

With this annotation in hand, runtime behavior is fully determined:

1. The lowerer vmaps a per-row kernel over the ``[3, 1]`` index array.
2. Each row evaluates the ``where`` predicate at its bound site index.
3. For active rows, both emissions run: ``x'`` is computed and ``mel`` is
   evaluated in the augmented environment.
4. Inactive rows produce two zero entries (both ``mel`` and the ``xp`` copy
   equal the source ``x``, though mel is zeroed).
5. All 6 entries (3 rows × 2 emissions) are concatenated and returned as the
   padded output.

Duplicate ``x'`` values across emissions are not merged.  If both ``raise``
and ``lower`` happened to produce the same ``x'`` (which they cannot here, but
can in more complex operators), both would appear as separate rows.


Programmatic inspection workflow
==================================

Use programmatic inspection when building custom passes/tooling:

.. code-block:: python

   ir = op.to_ir()
   print(ir.operator_name, ir.term_count)
   print(ir.static_fingerprint())
   print(ir.free_symbols)

   term = ir.terms[0]
   print(term.iterator.labels)
   print(term.iterator.index_sets)
   print(term.predicate)

   for emission in term.effective_emissions:
       print(emission.branch_tag)
       print(emission.update_program)
       print(emission.amplitude)


Validation contracts
======================

The validation layer enforces structural and scope constraints, including:

- terms must exist for ``mode="symbolic"``,
- update ops must satisfy required parameters by kind,
- site/emit symbol labels must be bound by the term iterator,
- global terms cannot reference unbound site labels.

Typical direct invocation path:

.. code-block:: python

   from neuralqx.experimental.operators.symbolic.ir.validate import validate_symbolic_ir

   ir = op.to_ir()
   summary = validate_symbolic_ir(ir)
   print(summary["term_count"], summary["term_symbols"])


Common IR debugging pitfalls
==============================

- assuming duplicate ``x'`` branches are auto-merged (they are not),
- forgetting that multiple ``emit`` calls multiply fanout,
- reading legacy single-emission fields instead of ``effective_emissions``,
- ignoring ``metadata`` when using ``shift_mod`` semantics.

DSL-to-IR mapping you can apply mechanically
==============================================

A reliable way to debug symbolic definitions is to map each builder fragment to
its IR field destination:

.. code-block:: text

   DOperator(hi, "name", dtype=..., hermitian=...)
     -> SymbolicOperatorIR.operator_name / dtype_str / is_hermitian / hilbert_size

   .for_each_* / .for_each(..., over=...)
     -> SymbolicIRTerm.iterator (KBodyIteratorSpec.labels + index_sets)

   .where(predicate_a).where(predicate_b)
     -> SymbolicIRTerm.predicate = and(predicate_a, predicate_b)

   .emit(update_u, amplitude=a, tag=t)
     -> append EmissionSpec(update_program=u, amplitude=a, branch_tag=t)

   .named("term_name")
     -> SymbolicIRTerm.name

   .fanout(k)
     -> SymbolicIRTerm.fanout_hint

When this mapping is clear, most semantic bugs become straightforward field
inspection tasks instead of runtime guesswork.

IR diff workflow for regression review
========================================

For non-trivial compiler or DSL changes, keep an IR diff in review artifacts.

Recommended workflow:

.. code-block:: python

   old_ir = old_op.to_ir().as_dict()
   new_ir = new_op.to_ir().as_dict()
   # serialize with sorted keys and compare with your preferred diff tool

What to compare first:

- iterator domains (``labels`` and ``index_sets`` cardinality),
- predicate operator trees (especially accidental ``and``/``or`` changes),
- emission count and ordering,
- update op order (sequential semantics),
- dtype/hermiticity and metadata payload.

Fingerprint changes are expected whenever semantically relevant IR fields
change.  If behavior changed but fingerprint did not, that is usually a bug in
serialization or signature construction.

Compatibility notes for tooling authors
=========================================

Tooling should prefer these stable access paths:

- ``SymbolicIRTerm.effective_emissions`` over legacy single-emission fields,
- ``SymbolicOperatorIR.as_dict()`` for serialization/export,
- ``SymbolicOperatorIR.static_fingerprint()`` for deterministic identity.

Avoid relying on formatting details of ``str(ir)`` for machine workflows, the
text dump is for diagnostics and may evolve independently of structural
IR contracts.

When introducing new IR fields, update all of:

1. dataclass schema,
2. serializer/fingerprint path,
3. validator,
4. lowerer consumer logic,
5. docs and migration notes.

Skipping any of these tends to create "looks valid, lowers incorrectly" bugs
that are costly to diagnose later.

Read next
===========

Continue with :doc:`compiler_pipeline` for staged execution, cache signatures,
and lowering behavior.

For user-facing DSL authoring examples, see :ref:`symbolic_operators`.
