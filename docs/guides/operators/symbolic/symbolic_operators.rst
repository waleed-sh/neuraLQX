.. _symbolic_operators:

=================================
Symbolic Operators
=================================

.. warning::

   **Experimental feature.**

   The symbolic operator DSL is part of neuraLQX's **experimental API**.  It
   is already useful for research, but it is **not** part of the stable public
   interface.  Signatures, semantics, and the IR format may change between
   releases without a deprecation period.  Pin your neuraLQX version if you
   depend on this feature.

   See the :ref:`experimental` section for the full experimental-API policy.


--------------------------------------------------------------------
What is a symbolic operator and why would you use one?
--------------------------------------------------------------------

neuraLQX offers two established ways to define an operator, based on NetKet's operator infrastructure:

* **Matrix-based** (the NetKet ``LocalOperator``): you provide a small local matrix
  acting on a few sites, and the NetKet handles everything else. Easy to
  write for standard spin or bosonic lattice models, but the underlying matrix
  representation grows with the number of local states involved, it cannot
  scale to large systems or to operators that act globally across the
  configuration.

* **Matrix-free** (neuraLQX's ``ComputationalJaxOperator``): based on NetKet's ``DiscreteOperatorJax`` you write a JAX kernel
  function that, given a configuration, directly returns connected
  configurations and matrix elements. Maximum control and performance, but you
  are writing low-level JAX code, array slicing, masking, ``lax.cond``,
  vmap, for every operator. It works well once you have experience, but it is
  tedious to write, hard to read back later, and easy to get the convention
  wrong.

The symbolic operator DSL is a **third way** that sits above both. Instead of
writing a matrix or a kernel, you write a concise *description* of the physics
in plain Python, using readable statements like:

   *"for each pair of sites (i, j) where site i is occupied, lower i by one
   and raise j by one, with matrix element equal to the square root of the
   occupation at i"*

and the framework **compiles** that description into an efficient JAX kernel
for you.

In practice, that description looks like this:

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import (
       DOperator,
       shift,
       site,
       AmplitudeExpr,
   )

   hop = (
       DOperator(hi, "hopping")
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(
           shift("i", -1).shift("j", +1),
           amplitude=AmplitudeExpr.sqrt(site("i").value),
       )
       .compile()
   )

   xp, mels = hop.get_conn_padded(x_batch)

The resulting ``hop`` object is an ordinary neuraLQX matrix-free ``ComputationalJaxOperator`` operator that
slots directly into variational Monte Carlo, ``vstate.expect(hop)`` and
``vstate.expect_and_grad(hop)`` work exactly as they would for any other
operator.

When does this approach pay off?
===================================

Symbolic operators shine when:

* You have **many terms**, sums over vertices, loop moves, plaquette actions.
  Writing one term symbolically takes a few lines, chaining ten of them in one
  expression stays readable where ten hand-written JAX kernels would not.

* The operator **structure follows the graph topology**. You can feed an
  adjacency list directly as the iterator, and the framework generates the
  correct kernel automatically.

* You want to **iterate fast** on the physics. Changing "raise by 1" to
  "raise by 2", adding a predicate, or splitting a term into two is a one-line
  change instead of touching low-level JAX code.

* You want to **avoid common mistakes**. The bra/row convention, correct
  amplitude evaluation in the source vs. connected state, static shapes for
  JIT, the compiler handles all of these for you.

If you are already writing well-tested JAX kernels and do not have many terms,
the matrix-free approach is equally valid. Both compile to the same underlying
interface and both work seamlessly with NetKet's expectation-value machinery.

Core mental model
===================

A symbolic operator can be read as a four-part sentence:

.. code-block:: text

   Operator(info)
     .on_what_to_act(...)          # iterator domain
     .under_what_conditions(...)   # predicate
     .how_to_act(...)              # update + amplitude
     .compile()

In concrete API form:

.. code-block:: text

   DOperator(hilbert, "name", dtype=..., hermitian=...)
     .for_each_*(labels, ...) or .globally()
     .where(predicate)
     .emit(update, amplitude=expr, tag=...)
     .compile()

This gives a stable way to reason about definitions before reading any compiler
internals:

1. ``DOperator(...)`` defines the operator identity and numerical contract.
2. Iterator methods define the visit schedule (the search space).
3. ``where(...)`` prunes the schedule.
4. ``emit(...)`` defines the branch dynamics and matrix elements.

That same decomposition is exactly how the compiler interprets your code.

**(1) Identity**, ``DOperator(hilbert, name)``
   Binds the definition to a Hilbert space and operator name. ``dtype`` and
   ``hermitian`` become part of the internal representation (IR) and compiled artifact metadata, and affect
   cache identity and runtime dtype casting.

**(2) Iterator**, ``.for_each_*`` / ``.globally``
   Declares the index domain that will be visited for every input state ``|σ⟩``.
   ``for_each_site`` is O(N), ``for_each_pair`` is O(N²), and
   ``for_each(labels, over=...)`` is O(len(over)).  Iterator choice is your
   primary per-sample complexity control.

**(3) Predicate**, ``.where(expr)``
   A Boolean filter evaluated in the source-state environment.  Failed rows
   produce no valid branch contribution.  Multiple ``where`` calls compose via
   logical AND.

**(4) Emission**, ``.emit(update, amplitude=expr)``
   Defines branch generation for each active iterator row.
   ``update`` constructs ``x'`` from ``x``, ``amplitude`` computes
   ``⟨x|O|x'⟩`` and may reference both source symbols (``site(...)``) and emitted
   symbols (``emitted(...)``).

Runtime expansion
--------------------

For one configuration ``x``, the lowered semantics are:

.. code-block:: text

   branches = []
   for visit in iterator.index_sets:
       if predicate(x, visit):
           for emission in emissions:
               x' = apply(emission.update, x, visit)
               mel = eval(emission.amplitude, x, x', visit)
               branches.append((x', mel))

   xp, mels = static_pad(branches, total_fanout)

``get_conn_padded`` is the batched, static-shape version of this logic.
Outputs are a **branch multiset**: duplicate ``x'`` rows are retained and not
implicitly merged. The real implementation uses a vectorised compiled JAX function.

Operational consequences:

- Iterator design dominates runtime work, choose sparse/static domains whenever
  the model allows.
- Multiple emissions in one term reuse the same iterator pass, which is often
  cleaner and cheaper than duplicating iterator blocks.
- Predicate debugging and amplitude debugging are separate concerns, isolate
  them independently during validation.

The remaining sections unpack each layer in implementation detail.

Performance model at a glance
--------------------------------

For one input sample, an upper-bound branch budget is:

.. math::

   \text{fanout}_\text{term} \le M \times E,

where ``M = len(iterator.index_sets)`` and ``E`` is the number of emissions in
that term. Total padded width is the sum across terms.

Practical guidance:

- ``globally()`` gives ``M = 1`` and is the cheapest iterator when it fits.
- ``for_each_site`` gives ``M = N``.
- ``for_each_pair`` gives ``M = N^2`` and should usually be paired with strict
  predicates or replaced by ``for_each(..., over=...)`` on sparse topologies.
- Multi-emission terms increase ``E`` linearly but avoid duplicating iterator
  traversal.

This cost model is simple enough to estimate by inspection and should be part
of every operator review, especially when targeting large Hilbert spaces. This fanout
corresponds to NetKet's ``n_conn``.

Import path
-----------

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import (
       DOperator,
       # update utilities
       identity, shift, shift_mod, write, swap, permute, affine, scatter,
       Update,
       # selectors and expressions
       site, emitted, symbol,
       # compiler (optional, .compile() on the operator is the usual entry point)
       SymbolicCompiler, SymbolicCompilerOptions,
   )


.. seealso::

   :ref:`advanced_symbolic_ir_model`, the typed IR structures and how to read
   them in practice.


From description to executable: what actually happens
--------------------------------------------------------

When you call ``.compile()`` on a symbolic operator as we will see later, neuraLQX runs it through
a **compilation pipeline**:

1. Your Python code (the ``DOperator(...).for_each_...().where(...).emit(...)``
   chain) is executed at **definition time** to produce an internal,
   structured **intermediate representation (IR)**, a plain data structure
   that captures the full meaning of the operator: which sites to iterate
   over, which condition must hold, how to update the configuration, and what
   matrix element to compute.  No JAX code exists at this point, the IR is
   purely a description.

2. The IR is **validated** to catch mistakes early (referencing a site label
   that was not declared, mismatched index lengths, and so on).

3. The compiler **analyses** the IR to determine, for instance, the maximum
   number of connected configurations that any input can produce.  This number
   sets the static array shapes required by JAX.

4. A **lowerer** translates the IR into a concrete, fully vectorised JAX
   function.  This is the step that turns your human-readable description into
   actual ``jax.lax.cond``, ``jax.lax.scan``, and array-indexing code.

5. The compiled function is **cached** in memory.  The next time you call
   ``.compile()`` with the same operator structure and options, the cached
   version is returned instantly, there is no re-compilation cost.

The end result is a ``CompiledOperator``: a standard neuraLQX matrix-free
operator backed by a pure JAX function, ready for JIT compilation, batching,
and automatic differentiation.

.. hint::

   Even if you are already comfortable writing JAX connectivity kernels by
   hand, symbolic operators are worth considering for operators with many
   terms (sums of vertex actions, multi-body plaquette terms, …).  The DSL
   can express complex logic in far fewer lines and lets you iterate faster on
   the physics.

.. contents:: On this page
   :local:
   :depth: 2


--------------------------------------------------------------------
Basic building blocks
--------------------------------------------------------------------

Before learning the DSL itself, you need to understand the four kinds of
objects that appear inside every operator definition.

.. _symbolic_site_selector:

``site(label)`` and ``emitted(label)``, symbolic site references
==================================================================

A :class:`~neuralqx.experimental.operators.symbolic.dsl.selectors.SiteSelector` is a
lightweight, **build-time** object that represents one named site in the
iterator environment. It carries no numeric value itself, it is a symbolic
handle that the compiler later resolves against the actual configuration array.

You obtain a selector by calling the free functions :func:`~neuralqx.experimental.operators.symbolic.dsl.selectors.site`
and :func:`~neuralqx.experimental.operators.symbolic.dsl.selectors.emitted`:

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import site, emitted

   s = site("i")    # refers to the source configuration x[i]
   e = emitted("i") # refers to the emitted/connected configuration x'[i]

The *label* must match the name you pass to the iterator (``for_each_site``,
``for_each_pair``, etc.).  If you call ``for_each_site("e")``, use
``site("e")``, if you call ``for_each_pair("i", "j")``, use ``site("i")`` and
``site("j")``.

Selector properties: ``.value`` and ``.index``
-----------------------------------------------

Every ``SiteSelector`` exposes two core properties:

``s.value``
   Returns an :class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr`
   that evaluates to the **quantum number** (the spin / flux / occupation number)
   stored at site *i* in the current configuration.

   * For a source selector: resolves to ``x[i]``.
   * For an emitted selector: resolves to ``x'[i]``.

``s.index``
   Returns an :class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr`
   that evaluates to the **integer site index** ``i`` itself, cast to a
   floating-point number.  This is useful when the matrix element depends on
   the *position* of a site, not its quantum number (e.g. a site-dependent
   hopping amplitude ``t[i]`` that you encode as a function of ``i``).

.. code-block:: python

   s = site("i")

   s.value          # AmplitudeExpr, x[i]
   s.index          # AmplitudeExpr, i  (float)

   e = emitted("i")
   e.value          # AmplitudeExpr, x'[i]   (value AFTER the update)
   e.index          # AmplitudeExpr, i        (same integer, both selectors share it)

A selector also supports comparison operators that return
:class:`~neuralqx.experimental.operators.symbolic.ir.predicates.PredicateExpr` nodes
directly without having to access ``.value`` explicitly:

.. code-block:: python

   site("i") > 0       # equivalent to site("i").value > 0
   site("i") == 1      # equivalent to site("i").value == 1
   site("i") < cutoff  # equivalent to site("i").value < cutoff

Finally, ``s.abs()`` returns ``|x[i]|`` as an amplitude expression.

``symbol(name)``, free named parameters
-----------------------------------------

:func:`~neuralqx.experimental.operators.symbolic.dsl.selectors.symbol` returns a
:class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr` that is not
bound to any site.  It represents a **free parameter**, something like a
hopping amplitude ``t``, a coupling constant ``lambda``, or any other
user-controlled numeric value.

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import symbol

   t = symbol("t")            # free parameter named "t"
   lam = symbol("lambda")     # free parameter named "lambda"

At the moment free symbols serve as symbolic labels in the expression tree, how
they are bound to concrete values at evaluation time is backend-dependent.  For
most use cases, folding the numeric value into the amplitude expression directly
(as a constant) is the most straightforward approach.

.. _symbolic_amplitude_expr:

``AmplitudeExpr``, numeric expressions
=========================================

:class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr` is the
typed IR node for **matrix-element expressions**.  Every amplitude you write in
the DSL is represented as an expression tree of these nodes.

You rarely construct ``AmplitudeExpr`` objects directly, instead, they arise
naturally from arithmetic on selectors:

.. code-block:: python

   s = site("i")

   s.value + 1                # AmplitudeExpr: x[i] + 1
   s.value * 2.0              # AmplitudeExpr: 2 * x[i]
   s.value ** 2               # AmplitudeExpr: x[i]^2
   -s.value                   # AmplitudeExpr: -x[i]
   s.value / site("j").value  # AmplitudeExpr: x[i] / x[j]

All standard Python arithmetic operators (``+``, ``-``, ``*``, ``/``, ``**``,
unary ``-``) are overloaded. The result of any such operation is a new
``AmplitudeExpr``, the objects are **immutable**.

Unary operations via class methods:

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import AmplitudeExpr

   AmplitudeExpr.sqrt(s.value)   # √x[i]
   AmplitudeExpr.conj(s.value)   # complex conjugate of x[i]
   AmplitudeExpr.abs_(s.value)   # |x[i]|   (same as s.abs())

Accessing the source configuration at a **fixed (compile-time constant) index**
without going through an iterator label:

.. code-block:: python

   AmplitudeExpr.static_index(5)           # reads x[5] from the source state
   AmplitudeExpr.static_emitted_index(10)  # reads x'[10] from the emitted state

This is useful for structured Hilbert spaces where the flat index encodes
multiple degrees of freedom (e.g. ``x[g * n_edges + e]`` in U(1) gauge
theories).

Hilbert-aware modulo wrapping:

.. code-block:: python

   AmplitudeExpr.wrap_mod(s.value + 1)  # (x[i] + 1) wrapped into local_states range

Comparison operators on ``AmplitudeExpr`` objects return
``PredicateExpr`` nodes and can be used directly in ``.where()``.

.. _symbolic_predicate_expr:

``PredicateExpr``, boolean expressions
=========================================

:class:`~neuralqx.experimental.operators.symbolic.ir.predicates.PredicateExpr` is the typed
IR node for **branch-filtering conditions**.  Predicates are created by
comparing amplitude expressions or combining existing predicates:

.. code-block:: python

   # comparison, returns PredicateExpr
   site("i").value > 0
   site("i").value <= m_max
   site("i").value == 1
   site("i").value != 0
   site("i").index < 5

   # logical composition
   pred_a = site("i").value > 0
   pred_b = site("j").value < 2
   pred_a & pred_b    # AND
   pred_a | pred_b    # OR
   ~pred_a            # NOT

Like ``AmplitudeExpr``, ``PredicateExpr`` nodes are **immutable** and fully
hashable, the compiler can deduplicate and cache them.


--------------------------------------------------------------------
Creating an operator
--------------------------------------------------------------------

All symbolic operators are created through :class:`~neuralqx.experimental.operators.symbolic.dsl.op.DOperator`,
the single entry point of the DSL:

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import DOperator

   op = DOperator(
       hilbert,            # netket DiscreteHilbert space
       "my_operator",      # human-readable name
       dtype="float64",    # matrix-element dtype (default: "float64")
       hermitian=False,    # declare as Hermitian (default: False)
   )

``DOperator`` is a **fluent builder**: every method call returns the same builder
object (``self``), so you chain calls together into one expression:

.. code-block:: python

   compiled = (
       DOperator(hi, "example")
       .for_each_site("i")            # which sites to visit
       .where(site("i") > 0)          # condition on site i
       .emit(shift("i", +1))          # connected state + matrix element
       .build()                       # seal into SymbolicOperator
       .compile()                     # lower to CompiledOperator
   )

The key rule is: **always call an iterator method first** (``globally``,
``for_each_site``, etc.) before calling ``.where()`` or ``.emit()``.  These
methods always operate on the *most recently opened term*.

Calling ``.build()`` finalises the operator and returns an
immutable :class:`~neuralqx.experimental.operators.symbolic.core.operator.SymbolicOperator`
that is not yet executable.  Calling ``.compile()`` on that (or directly on the
builder) triggers the compilation pipeline and returns an executable
:class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator`.

Term metadata controls: ``.named(...)`` and ``.fanout(...)``
=============================================================

Two builder methods are especially useful in large operator definitions where
you care about diagnostics and static-shape budgeting.

``.named("term_label")``
   Assigns a human-readable name to the **current open term** (the most recent
   iterator scope). This label appears in IR dumps and pass reports and makes
   debugging significantly faster than relying on numeric auto names only.

``.fanout(hint)``
   Sets an explicit upper bound on branches produced by the current term.
   If omitted, the DSL auto-infers a conservative bound:
   ``len(iterator.index_sets) * n_emissions``.

   When you know your predicate/update structure enforces a stricter bound, a
   tighter hint reduces unnecessary padded width in downstream kernels.

Example:

.. code-block:: python

   op = (
       DOperator(hi, "annotated")
       .for_each_pair("i", "j")
       .named("kinetic_hop")
       .fanout(hi.size * (hi.size - 1))  # excludes i == j by construction
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1), amplitude=-1.0, tag="hop")
       .build()
   )

Both methods must be called *after* an iterator is opened and *before* that
term is sealed by the next iterator or by ``build()``.


--------------------------------------------------------------------
Selecting what to act on, iterators
--------------------------------------------------------------------

The **iterator** of a term specifies how many evaluations the DSL engine
performs for each input configuration, and what site labels are in scope.
Every iterator method call **seals** the previous open term (if any) and
**opens a new one**. In the mental model above, this section is precisely the
"visit schedule" part.

.. _symbolic_globally:

``globally()``, one evaluation per configuration
===================================================

.. code-block:: python

   DOperator(hi, "N")
   .globally()
   .emit(identity(), amplitude=AmplitudeExpr.static_index(0))

``globally()`` iterates **once per configuration**: there is no loop over sites.
This is the right choice for:

* **Diagonal operators** whose value depends on the whole configuration (e.g. a
  total number operator that sums over all sites),
* Operators where all target sites are embedded directly in the update or
  amplitude expression via ``AmplitudeExpr.static_index(k)``.

In the K-body model, the global iterator is a 0-body iterator: there are no
site labels in scope (``labels = ()``). This means you must not reference
``site("i")`` labels unless they were explicitly introduced by a non-global
iterator.

``for_each_site(label)``, one evaluation per site
====================================================

.. code-block:: python

   DOperator(hi, "h+")
   .for_each_site("e")
   .where(site("e") < cutoff)
   .emit(shift("e", +1))

Iterates over every site index ``0, 1, ..., N−1`` where ``N = hilbert.size``.
The site index is bound to *label* in the evaluation environment:
``site(label).index`` evaluates to the current site index, and
``site(label).value`` evaluates to the quantum number stored there.

This is the correct iterator for **single-site operators**: raising/lowering,
diagonal number operators per site, single-body rewrite rules. In the mental
model: one visit per site, each visit may emit zero/one/many branches depending
on predicates and emissions.

``for_each_pair(label_a, label_b)``, all ordered site pairs
=============================================================

.. code-block:: python

   DOperator(hi, "hopping")
   .for_each_pair("i", "j")
   .where(site("i").index != site("j").index)  # exclude diagonal
   .where(site("i") > 0)
   .emit(shift("i", -1).shift("j", +1))

Iterates over the Cartesian product of all site indices:
``(0,0), (0,1), ..., (N−1,N−1)``. The two indices are bound to *label_a* and
*label_b* respectively.

The iteration includes the diagonal (``i == j``) by default. Use a
``.where(site("i").index != site("j").index)`` predicate to exclude it.

Use this for **two-body operators**: hopping terms, interaction terms, any
operator involving two distinct sites. In the mental model: each visit binds a
pair ``(i, j)`` and the rest of the term logic runs on that bound pair.

``for_each_triplet(label_a, label_b, label_c, *, over)``, a static list of 3-tuples
======================================================================================

.. code-block:: python

   vertex_triplets = [(0, 1, 2), (0, 3, 4), (1, 2, 3)]

   DOperator(hi, "vertex_amplitude")
   .for_each_triplet("e1", "e2", "e3", over=vertex_triplets)
   .emit(identity(), amplitude=site("e1").value * site("e2").value * site("e3").value)

Iterates over an **explicit, pre-computed list of 3-tuples**.  This is the
canonical iterator for graph-based operators where the triples come from graph
topology (e.g. triangles, vertex-adjacent triplets), not a dense ``N³`` sweep.

The ``over`` argument must be a sequence of ``(int, int, int)`` tuples.  All
three labels are in scope inside ``.where()`` and ``.emit()``. In the mental
model: you explicitly enumerate the visit schedule yourself instead of taking a
dense cartesian product.

``for_each_plaquette(label_a, label_b, label_c, label_d, *, over)``, 4-tuples
================================================================================

.. code-block:: python

   plaquettes = [(0, 1, 3, 2), (1, 4, 5, 3)]  # corners of each plaquette

   DOperator(hi, "plaquette_action")
   .for_each_plaquette("a", "b", "c", "d", over=plaquettes)
   .emit(
       shift_mod("a", +1).shift_mod("b", +1).shift_mod("c", -1).shift_mod("d", -1)
   )

Iterates over a static list of 4-tuples. Use for plaquette terms in lattice
gauge theories and similar 4-body interactions. This is often the best way to
encode geometric constraints while keeping operator intent readable.

``for_each(labels, *, over)``, arbitrary K-body static iterator
================================================================

.. code-block:: python

   edge_list = [(src, dst) for src, nbrs in adjacency.items() for dst in nbrs]

   DOperator(hi, "nbr_hopping")
   .for_each(("src", "dst"), over=edge_list)
   .where(site("src") > 0)
   .emit(shift("src", -1).shift("dst", +1))

The most general iterator.  Accepts:

* ``labels``: a tuple of K string labels,
* ``over``: a sequence of K-tuples of integer site indices.

``for_each_site``, ``for_each_pair``, ``for_each_triplet``, and
``for_each_plaquette`` are all convenience wrappers around ``for_each``.

**Important:** ``over`` must be **non-empty** and every tuple must have length
``len(labels)``.  The iterator body, the labels, predicate, and emissions,
applies to every tuple in ``over`` as a single batch.

How iterator calls seal terms
==============================

Every call to an iterator method seals the current in-progress term and starts
a new one. This means you can write multi-term operators in one continuous
chain:

.. code-block:: python

   op = (
       DOperator(hi, "hamiltonian")
       # term 0: kinetic
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1), amplitude=-1.0)
       # term 1: interaction (diagonal)
       .for_each_pair("a", "b")
       .where(site("a").index < site("b").index)
       .emit(identity(), amplitude=site("a").value * site("b").value)
       .build()
   )

The second ``for_each_pair`` seals the kinetic term and opens the interaction
term.  Each term is compiled independently and their outputs are concatenated in
the padded connectivity format.


--------------------------------------------------------------------
Predicates, filtering which visits produce output
--------------------------------------------------------------------

``.where(predicate)``
======================

.. code-block:: python

   .where(predicate)

Sets a **branch predicate** for the current term.  The predicate is evaluated
once per iterator slot (once per site, once per pair, etc.) in the source
configuration environment.  Only those slots where the predicate is ``True``
contribute connected states, the rest are silently skipped (zero matrix
element).

*predicate* can be a :class:`~neuralqx.experimental.operators.symbolic.ir.predicates.PredicateExpr`
or any expression that the compiler can coerce to one (most commonly a
comparison expression).

Chaining ``.where()`` calls
-----------------------------

Multiple ``.where()`` calls on the same term compose with **logical AND**:

.. code-block:: python

   .where(site("i") > 0)
   .where(site("j") < m_max)
   # equivalent to .where((site("i") > 0) & (site("j") < m_max))

Available comparison operators
--------------------------------

All six comparison operators are supported:

.. code-block:: python

   site("i").value > 0       # strict greater-than
   site("i").value >= 0      # greater-than-or-equal
   site("i").value < m_max   # strict less-than
   site("i").value <= m_max  # less-than-or-equal
   site("i").value == 1      # equality
   site("i").value != 0      # inequality

These can be applied to ``.value`` (quantum number), ``.index`` (site position),
or any ``AmplitudeExpr``, including composed expressions:

.. code-block:: python

   (site("i").value + site("j").value) > 0
   site("i").index < site("j").index   # enforce ordering

Available logical operators
-----------------------------

.. code-block:: python

   pred_a & pred_b   # AND   (also PredicateExpr.and_(pred_a, pred_b))
   pred_a | pred_b   # OR    (also PredicateExpr.or_(pred_a, pred_b))
   ~pred_a           # NOT   (also PredicateExpr.not_(pred_a))

These can be composed arbitrarily:

.. code-block:: python

   .where(
       (site("i") > 0) & ((site("j") < 2) | (site("k") == -1))
   )


--------------------------------------------------------------------
Emissions, connected states and matrix elements
--------------------------------------------------------------------

.. _symbolic_emit:

``.emit(update, *, amplitude, tag)``
=====================================

.. code-block:: python

   .emit(update, amplitude=1.0, tag=None)

Appends one **output branch** to the current term.  For each active iterator
slot (i.e. those passing the predicate), one connected state ``x'`` is produced
by applying *update* to the source configuration ``x``, paired with the given
*amplitude* as the matrix element.

You may call ``.emit()`` **multiple times** on the same term. Each call adds
an independent branch, this is how you model operators that connect one source
configuration to *multiple* targets per site (e.g. both raising and lowering
from a single site without iterating over the site twice):

.. code-block:: python

   DOperator(hi, "raising_and_lowering")
   .for_each_site("i")
   .emit(shift("i", +1), amplitude=+0.5)   # branch 1: raise
   .emit(shift("i", -1), amplitude=-0.5)   # branch 2: lower

Both branches are emitted for every active site in a single pass over the
sites.

Argument: ``update``, state transformation
--------------------------------------------

The *update* argument is a site-rewrite program that describes how to transform
the source configuration ``x`` into the connected/emitted configuration ``x'``.
It is constructed using the factory functions imported from
``neuralqx.experimental.operators.symbolic``.

**``identity()``**, no change (diagonal operators)

.. code-block:: python

   .emit(identity(), amplitude=site("i").value)

The connected state ``x'`` is identical to ``x``. Use this for diagonal
operators: the "connected component" is the configuration itself.  Passing
``None`` as the update argument is equivalent.

**``shift(site_ref, delta)``**, additive shift

.. code-block:: python

   shift("i", +1)              # x'[i] = x[i] + 1
   shift("j", -2)              # x'[j] = x[j] - 2
   shift(0, site("i").value)   # x'[0] = x[0] + x[i]  (expression delta)

Raises or lowers the quantum number at one site by a fixed or expression-valued
amount. The simplest and most common update operation.

**``shift_mod(site_ref, delta)``**, Hilbert-aware wrapped shift

.. code-block:: python

   shift_mod("i", +1)   # x'[i] = ((x[i] + 1 - state_min) % mod_span) + state_min

Like ``shift``, but wraps the result within the local Hilbert-space range using
modular arithmetic. This requires the Hilbert space to have
**contiguous integer local states** (e.g. ``[-m_max, ..., m_max]`` or
``[0, 1, 2, 3]``).  The wrap parameters are inferred automatically from
``hilbert.local_states`` at build time.

**``write(site_ref, value)``**, direct write

.. code-block:: python

   write("i", 0)               # x'[i] = 0
   write("k", site("j").value) # x'[k] = x[j]  (copy)

Overwrites the quantum number at *site_ref* with a constant or expression
*value*.

**``swap(site_a, site_b)``**, exchange two sites

.. code-block:: python

   swap("i", "j")    # x'[i] = x[j], x'[j] = x[i]
   swap(0, 5)        # exchange flat sites 0 and 5

Exchanges the quantum numbers at two sites atomically. Both old values are
captured before either write, so the operation is a true swap.

**``permute(*site_refs)``**, cyclic rotation over K sites

.. code-block:: python

   permute("i", "j", "k")
   # x'[i] <- x[j],  x'[j] <- x[k],  x'[k] <- x[i]

Performs a **cyclic rotation** over K ≥ 2 sites.  All source values are
captured simultaneously before any writes.  For ``K=2`` this is identical to
``swap``.

**``affine(site_ref, *, scale, bias)``**, affine transform

.. code-block:: python

   affine("i", scale=2, bias=-1)  # x'[i] = 2*x[i] - 1
   affine("i", scale=-1, bias=0)  # x'[i] = -x[i]

Applies the linear map ``x'[i] = scale * x[i] + bias``.  Both *scale* and
*bias* may be numeric constants or amplitude expressions.

**``scatter(flat_indices, values)``**, bulk writes to static indices

.. code-block:: python

   scatter([0, 5, 10], [1, -1, 0])
   scatter([0, 10], [site("i").value, 0])  # mixed expr / constant

Performs several writes simultaneously to compile-time-constant flat site
indices.  The ``flat_indices`` must be plain integers (known at build time).
The ``values`` may be amplitude expressions.

**Chaining update operations**

All factory functions return an :class:`~neuralqx.experimental.operators.symbolic.dsl.rewrite.Update`
object, and every method on ``Update`` returns a **new** ``Update`` with the
operation appended.  Operations are applied **sequentially**:

.. code-block:: python

   shift("i", -1).shift("j", +1)           # lower i, raise j
   swap("i", "j").write("k", 0)            # swap then zero k
   shift("i", +1).shift_mod("j", +1)       # mix shift and shift_mod
   affine("i", scale=2, bias=-1).swap("i", "j")  # affine then swap

The chain ``a.b.c`` applies ``a`` first, then ``b``, then ``c``.

**``Update.cond(predicate, *, if_true, if_false)``**, conditional update

.. code-block:: python

   Update.cond(
       site("i") > 0,
       if_true=shift("i", -1),
       if_false=write("i", 0),
   )

Produces a JAX-compatible conditional (``jax.lax.cond``).  The *predicate* is
a ``PredicateExpr`` or comparable value.  *if_true* and *if_false* are both
``Update`` objects.  *if_false* defaults to identity when not provided.

**``Update.invalidate(reason=None)``**, mark branch as invalid

.. code-block:: python

   Update().invalidate(reason="out of bounds")

Marks the branch as having zero matrix element at lowering time.  Useful for
boundary conditions where you emit a branch unconditionally but want the update
program itself to signal invalidity without needing a separate predicate.

Argument: ``amplitude``, matrix element expression
-----------------------------------------------------

The *amplitude* is the matrix element ``⟨x'|O|x⟩`` for the emitted branch.
It is evaluated in the **source** configuration environment, the expression can
refer to ``x`` values but cannot depend on ``x'`` values directly (use
``emitted("i").value`` for that).

The *amplitude* defaults to ``1.0`` when omitted.

**Numeric constants**

.. code-block:: python

   amplitude=1.0
   amplitude=-0.5
   amplitude=1+2j

Any Python ``int``, ``float``, or ``complex`` literal is accepted and
automatically coerced to an :class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr`
constant.

**Source-configuration expressions**

.. code-block:: python

   amplitude=site("i").value            # x[i]
   amplitude=site("i").value + 1        # x[i] + 1
   amplitude=site("i").value * site("j").value   # x[i] · x[j]
   amplitude=site("i").value ** 2       # x[i]²
   amplitude=site("i").abs()            # |x[i]|
   amplitude=AmplitudeExpr.sqrt(site("i").value)  # √x[i]
   amplitude=AmplitudeExpr.conj(site("i").value)  # x[i]*
   amplitude=site("i").index * 0.1      # 0.1 · i   (depends on site position)

**Emitted-configuration expressions**

The amplitude may also depend on the **emitted state** ``x'`` (the connected
configuration after the update has been applied):

.. code-block:: python

   amplitude=emitted("i").value         # x'[i]
   amplitude=emitted("i").value + site("i").value  # x'[i] + x[i]
   amplitude=AmplitudeExpr.sqrt(emitted("i").value) * AmplitudeExpr.sqrt(site("i").value)

This is useful for operators whose matrix element is naturally symmetric in
source and target (e.g. ``√(n+1) · √n`` for bosonic raising/lowering operators
where ``emitted("i")`` is the raised site).

**Free named symbols**

.. code-block:: python

   amplitude=symbol("t")                        # free parameter "t"
   amplitude=site("i").value * symbol("lambda") # coupling * quantum number

**Static flat-index reads**

.. code-block:: python

   amplitude=AmplitudeExpr.static_index(5)           # reads x[5] directly
   amplitude=AmplitudeExpr.static_emitted_index(10)  # reads x'[10] directly

**Hilbert-aware modulo wrapping in amplitudes**

.. code-block:: python

   amplitude=AmplitudeExpr.wrap_mod(site("i").value + 1)

**Callable amplitudes (computed at build time)**

If you pass a *callable*, it is called at build time with an
``ExpressionContext`` object that exposes ``site()``, ``emitted()``, and
``symbol()`` as methods.  This can be useful for programmatically generating
complex amplitude expressions:

.. code-block:: python

   def my_amplitude(ctx):
       return ctx.site("i").value * ctx.site("j").value + 0.5

   .emit(shift("i", +1), amplitude=my_amplitude)

Argument: ``tag``, diagnostic label
--------------------------------------

The optional *tag* string labels a specific emission branch for debugging.  It
appears in IR dumps and compiler diagnostics:

.. code-block:: python

   .emit(shift("i", +1), amplitude=+0.5, tag="raise_i")
   .emit(shift("i", -1), amplitude=-0.5, tag="lower_i")

Multi-emission semantics and the branch multiset
-------------------------------------------------

When you call ``.emit()`` multiple times on the same term, all emissions are
processed in a **single pass** over the iterator:

.. code-block:: python

   DOperator(hi, "two_branch")
   .for_each_site("i")
   .emit(shift("i", +1), amplitude=+0.5)
   .emit(shift("i", -1), amplitude=-0.5)

For each site ``i``, two connected states are produced.  Compare this to
writing two separate terms with ``for_each_site``:

.. code-block:: python

   # Semantically equivalent to the multi-emission above, but visits
   # every site twice instead of once:
   DOperator(hi, "two_branch_slow")
   .for_each_site("i")
   .emit(shift("i", +1), amplitude=+0.5)
   .for_each_site("i")
   .emit(shift("i", -1), amplitude=-0.5)

For a Hilbert space with ``N`` sites, the multi-emission version produces
``2N`` connected states from a single pass, the two-term version produces the
same ``2N`` connected states from two passes.  The final output is
**equivalent**, but the single-term multi-emission version can be more
efficient.

**Branch multiset:** if two emissions (within a term or across terms) produce
the *same* ``x'``, **both appear separately** in the padded output with their
individual matrix elements.  The output is a multiset of connected states, not
a deduplicated set.  If you are accumulating matrix elements, sum over
duplicates explicitly.


--------------------------------------------------------------------
Multi-term operators and operator algebra
--------------------------------------------------------------------

Adding terms via chaining
==========================

The most natural way to build multi-term operators is to chain multiple
iterator blocks in one ``DOperator`` builder expression.  Each iterator call
seals the current term:

.. code-block:: python

   H = (
       DOperator(hi, "H")
       # kinetic term
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1), amplitude=-1.0)
       # diagonal (on-site potential)
       .for_each_site("k")
       .emit(identity(), amplitude=site("k").value ** 2)
       .build()
   )

Adding ``SymbolicOperator`` instances
======================================

Two :class:`~neuralqx.experimental.operators.symbolic.core.operator.SymbolicOperator`
objects defined over the **same Hilbert space** can be combined with ``+``:

.. code-block:: python

   kinetic = (
       DOperator(hi, "kinetic")
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1), amplitude=-1.0)
       .build()
   )

   potential = (
       DOperator(hi, "potential")
       .for_each_site("k")
       .emit(identity(), amplitude=site("k").value ** 2)
       .build()
   )

   H = kinetic + potential   # SymbolicOperator whose IR is the union of both

The resulting ``SymbolicOperator`` has a combined name like
``"(kinetic + potential)"`` and contains all terms from both operators.  It can
be compiled as usual.

When terms come from different dtypes, the result uses the wider dtype
(complex128 > complex64 > float64 > float32).


--------------------------------------------------------------------
Compiling symbolic operators
--------------------------------------------------------------------

.. _symbolic_compilation_pipeline:

The compilation pipeline
===========================

Calling ``.compile()`` on a ``SymbolicOperator`` (or directly on the
``DOperator`` builder as a shortcut) triggers a multi-stage compiler pipeline.
Understanding each stage helps you debug unexpected failures and reason about
the output.

**Stage 1, IR extraction**

The ``SymbolicOperator`` is converted to a
:class:`~neuralqx.experimental.operators.symbolic.ir.program.SymbolicOperatorIR`, a
typed, hashable intermediate representation (IR) that contains the full term
list in a normalised form.  This IR is the input to all subsequent stages.

You can inspect the IR directly::

   ir = my_symbolic_op.to_ir()  # or print(my_symbolic_op.to_ir())

**Stage 2, Validation (pre-cache)**

The IR is checked for structural correctness:

* all labels referenced in predicates and emissions exist in the iterator,
* update programs are well-typed,
* amplitude expressions reference only known ops.

If ``strict_validation=True`` (the default), any error raises
``SymbolicCompilerError`` immediately.

**Stage 3, Normalisation (pre-cache)**

The normalization pass computes and records stable compile metadata used by later
stages, including IR fingerprint and resolved backend target
(``auto -> jax`` in the current backend set).  It also records a stable term
ordering summary for diagnostics.

**Stage 4, Cache lookup**

A **deterministic cache key** is computed from:

* the full IR (operator name, Hilbert size, dtype, all terms),
* the compiler options (backend, fusion flag).

If a previous compilation with the same key exists in the in-process artifact
store, it is returned immediately without re-running the heavy passes.  This
means that ``op.compile()`` is cheap to call repeatedly after the first call.

**Stage 5, Analysis (post-cache, on miss only)**

Static analysis estimates the **fan-out** of each term (how many connected
states per input configuration) and collects statistics used for padding
the output arrays.  The maximum fan-out determines the static shape of the
padded connectivity arrays.

**Stage 6, Fusion (post-cache, on miss only)**

If ``enable_fusion=True`` (the default), the fusion pass computes
``fusion_groups`` metadata in the compilation context.  This metadata is
available to lowerers that implement fused loop generation.

Current status of the built-in backend:

- the default lowerer records and preserves this planning
  metadata through pass reports,
- it still lowers terms independently (one term runner per IR term),
- so the pass is presently an analysis/planning stage rather than a runtime
  loop-merging transformation.

This is important for benchmarking: toggling ``enable_fusion`` currently
changes analysis metadata and cache identity, but does not yet guarantee a
runtime speedup with the stock lowerer.

**Stage 7, Lowering (JAX code generation)**

The current default JAX lowerer converts each IR term into a pure Python/JAX
runner and then composes all runners into one ``get_conn_padded`` kernel.
Conceptually, for a ``for_each_site`` term this looks like:

.. code-block:: python

   def _term_kernel(x):
       results = []
       for i in range(hilbert.size):
           if predicate(x, i):
               xp = apply_update(x, i)
               mel = evaluate_amplitude(x, i, xp)
               results.append((xp, mel))
       return pad(results, max_fanout)

In practice the code is fully vectorised and uses ``jax.lax.cond`` and
``jax.vmap``/``jax.lax.scan``, no Python loops appear in the final kernel.

All terms are composed into a single ``_get_conn_padded(x_batch)`` function
that concatenates their padded outputs along the connection axis.

**Stage 8, Artifact storage and return**

The compiled kernel is wrapped in a
:class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator` and stored
in the artifact cache.  The ``CompiledOperator`` is returned.

Compilation entry points
==========================

**Option 1, One-shot via the builder (simplest):**

.. code-block:: python

   compiled_op = (
       DOperator(hi, "hop")
       .for_each_pair("i", "j")
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1))
       .compile()
   )

**Option 2, Via ``SymbolicOperator.compile()``:**

.. code-block:: python

   sym_op = DOperator(hi, "hop").for_each_pair("i", "j").where(...).emit(...).build()
   compiled_op = sym_op.compile(backend="jax", cache=True)

**Option 3, Via an explicit ``SymbolicCompiler``:**

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import SymbolicCompiler, SymbolicCompilerOptions

   compiler = SymbolicCompiler(
       options=SymbolicCompilerOptions(
           backend_preference="jax",
           enable_fusion=True,
           strict_validation=True,
           cache_enabled=True,
       )
   )
   compiled_op = compiler.compile_operator(sym_op)

The explicit compiler is useful when you want to:

* disable caching for debugging (``cache_enabled=False``),
* disable fusion to see individual term output (``enable_fusion=False``),
* share a compiler with a custom artifact store across many operators.

**Option 4, Module-level convenience function:**

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import compile_symbolic_operator

   compiled_op = compile_symbolic_operator(sym_op)

Uses a module-level shared ``SymbolicCompiler`` (lazily created on first call).

``SymbolicCompilerOptions``
============================

.. code-block:: python

   SymbolicCompilerOptions(
       backend_preference="auto",   # "auto" resolves to "jax" (only backend)
       enable_fusion=True,          # merge compatible terms into one kernel
       strict_validation=True,      # raise on IR errors (vs. warn)
       cache_enabled=True,          # use in-process artifact cache
       cache_namespace="nqx_symbolic_v1",  # prefix for cache keys
   )

The options object is frozen (immutable) and hashable, so it participates in
cache-key generation.

Inspecting compilation artifacts and pass reports
=================================================

When you need compiler observability (cache key, pass timings, lowerer choice),
use ``SymbolicCompiler.compile(...)`` instead of ``compile_operator(...)``:

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import SymbolicCompiler

   compiler = SymbolicCompiler()
   artifact = compiler.compile(sym_op)

   print("backend:", artifact.backend)
   print("lowerer:", artifact.lowerer_name)
   print("cache token:", artifact.cache_token())
   print("metadata:", artifact.metadata_map())
   print("passes:", [r.pass_name for r in artifact.pass_reports])

   compiled_op = artifact.compiled_operator

Why this path matters in practice:

- it exposes whether you hit the cache or took a miss path,
- it shows exactly which pass sequence executed,
- it gives you a stable token for experiment-level reproducibility logs,
- it provides a structured place to carry custom lowerer metadata.

For production debugging and benchmarking, log artifact fields alongside
operator fingerprints, not just final runtime timings.


--------------------------------------------------------------------
Using the compiled operator
--------------------------------------------------------------------

``CompiledOperator`` and ``get_conn_padded``
==============================================

:class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator` is a
:class:`~neuralqx.operators.types.computational_operator.jax.ComputationalJaxOperator`, it implements the same padded connectivity interface used by all
neuraLQX matrix-free operators.

.. code-block:: python

   xp, mels = compiled_op.get_conn_padded(x_batch)

* ``x_batch``: shape ``(B, N)``, a batch of B input configurations, each of
  length N (the Hilbert space size).
* ``xp``: shape ``(B, n_conn, N)``, connected configurations for each input,
  padded to ``n_conn`` entries.
* ``mels``: shape ``(B, n_conn)``, corresponding matrix elements (complex).

Unused connection slots are padded with zeros (both ``xp`` and ``mels``).

The ``n_conn`` value is fixed at compile time and equals the maximum number of
connected states that any input configuration can produce under this operator.
It is determined automatically by the analysis pass (Stage 5).

Properties:

.. code-block:: python

   compiled_op.name          # operator name string
   compiled_op.hilbert       # the Hilbert space
   compiled_op.is_hermitian  # True / False
   compiled_op.dtype         # numpy dtype

JAX transformations
====================

The underlying kernel is a **pure JAX function**. It can be:

* ``jax.jit``-compiled (either directly or as part of a larger jit scope),
* ``jax.vmap``'d over an additional batch dimension,
* differentiated with ``jax.grad`` / ``jax.vjp`` if the amplitude expressions
  are differentiable.

It is also registered as a **JAX pytree** (with no array leaves), so it can
appear as a static argument in jit-compiled functions without triggering tracer
errors.

Integration with neuraLQX and NetKet
=====================================

Because ``CompiledOperator`` extends ``ComputationalJaxOperator``, it can be
passed anywhere in neuraLQX or NetKet that accepts a ``DiscreteOperator``.
In particular:

.. code-block:: python

   import netket as nk

   vstate = nk.vqs.MCState(sampler, model, n_samples=1024)
   E = vstate.expect(compiled_op)     # expectation value
   E, grads = vstate.expect_and_grad(compiled_op)  # gradients

You can also sum compiled operators with other operators (symbolic or
otherwise) using standard operator algebra.

Operator-level arithmetic before compilation
=============================================

If you want to combine operators **before** compilation, use ``SymbolicOperator``
addition as described in the multi-term section.  Adding two
``SymbolicOperator`` objects produces another ``SymbolicOperator`` with the
combined IR, which is then compiled in one pass:

.. code-block:: python

   H = (kinetic + potential).compile()   # one compiled artifact, combined kernel

Alternatively, you can compile each operator separately and let neuraLQX
combine their padded outputs at runtime via ``ComputationalJaxOperator`` sums:

.. code-block:: python

   H = kinetic.compile() + potential.compile()  # two separate kernels, summed at runtime

The first approach (combined compilation) is generally more efficient.

Debugging checklist for real operators
========================================

When a symbolic operator does not behave as expected, the fastest path is to
debug by layer rather than by symptom:

1. Builder / IR shape:
   check ``print(op.to_ir())`` first.  Verify iterator labels, index domain,
   term boundaries, and emission count.  Many issues are simply term-sealing
   surprises after a second iterator call.
2. Predicate logic:
   temporarily replace ``where(...)`` with ``where(True)`` or simplify it to a
   single comparison.  If branches reappear, the bug is in guard logic, not
   updates.
3. Update semantics:
   set ``amplitude=1.0`` and inspect ``xp`` only.  Confirm the state rewrite
   before diagnosing matrix-element math.
4. Amplitude semantics:
   compare source and emitted symbols explicitly
   (``site("i").value`` vs ``emitted("i").value``) to ensure you are reading the
   intended side of the transition.
5. Static fanout:
   inspect inferred/declared fanout.  Overly large fanout is a performance
   smell, overly tight fanout hints can truncate outputs if set incorrectly.

Useful triage pattern:

.. code-block:: python

   ir = op.to_ir()
   print(ir)
   compiled = op.compile()
   xp, mels = compiled.get_conn_padded(x_batch[:1])
   print(xp.shape, mels.shape)

For compiler internals, pass sequencing, and extension-level debugging, move to
the Advanced Guides linked below.


--------------------------------------------------------------------
Complete examples
--------------------------------------------------------------------

Diagonal number operator
==========================

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import (
       DOperator,
       identity,
       site,
       AmplitudeExpr,
   )

   # Acting on site k = 0

   N_op = (
       DOperator(hi, "total_number", hermitian=True)
       .globally()
       .emit(identity(), amplitude=AmplitudeExpr.static_index(k))
       .compile()
   )

Single-site raising operator
=============================

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import (
       DOperator,
       shift,
       site,
       AmplitudeExpr,
   )

   h_plus = (
       DOperator(hi, "h+")
       .for_each_site("e")  # iterate every element in a given basis configuration
       .where(site("e") < cutoff)          # only raise if the element is below cutoff
       .emit(shift("e", +1), amplitude=AmplitudeExpr.sqrt(site("e").value + 1))
       .build()
       .compile()
   )

Hopping operator
==================

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import (
       DOperator,
       shift,
       site,
       AmplitudeExpr,
   )

   hop = (
       DOperator(hi, "hopping")
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(
           shift("i", -1).shift("j", +1),
           amplitude=AmplitudeExpr.sqrt(site("i").value),
       )
       .build()
       .compile()
   )

Plaquette action with modular shifts
=======================================

.. code-block:: python

   plaquettes = [(0, 1, 3, 2), (1, 4, 5, 3), (2, 5, 6, 3)]

   plaquette_op = (
       DOperator(hi, "plaquette", hermitian=True)
       .for_each_plaquette("a", "b", "c", "d", over=plaquettes)
       .emit(
           shift_mod("a", +1).shift_mod("b", +1).shift_mod("c", -1).shift_mod("d", -1),
           amplitude=1.0,
           tag="forward",
       )
       .emit(
           shift_mod("a", -1).shift_mod("b", -1).shift_mod("c", +1).shift_mod("d", +1),
           amplitude=1.0,
           tag="backward",
       )
       .build()
       .compile()
   )

Multi-term Hamiltonian
========================

.. code-block:: python

   from neuralqx.experimental.operators.symbolic import (
       DOperator, shift, identity, site, AmplitudeExpr
   )

   H = (
       DOperator(hi, "bose_hubbard", hermitian=True)
       # hopping
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(
           shift("i", -1).shift("j", +1),
           amplitude=-1.0,
       )
       # on-site interaction
       .for_each_site("k")
       .emit(
           identity(),
           amplitude=0.5 * site("k").value * (site("k").value - 1),
       )
       .build()
       .compile()
   )


--------------------------------------------------------------------
Going deeper
--------------------------------------------------------------------

.. seealso::

   :ref:`advanced_symbolic_compiler`, advanced internals documentation,
   including architecture, IR contracts, compiler pipeline, and extension
   points.
