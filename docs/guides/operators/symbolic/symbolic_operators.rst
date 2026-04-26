.. _symbolic_operators:

=================================
Symbolic Operators
=================================

.. warning::

   The symbolic operator DSL has been extracted from neuraLQX into the
   standalone **nkDSL** package and **is deprecated in neuraLQX as of version 1.1.2**.
   It will be removed in a future release. **Please use nkDSL instead:**
   `nkDSL documentation <https://nkdsl.readthedocs.io/en/latest/>`_.

   nkDSL targets the broader NetKet ecosystem and provides the same DSL with more
   capabilities with an actively maintained API.


--------------------------------------------------------------------
What is a symbolic operator and why would you use one?
--------------------------------------------------------------------

neuraLQX offers two established ways to define an operator, based on NetKet's operator infrastructure:

* **Matrix-based** (the NetKet ``LocalOperator``): you provide a small local matrix
  acting on a few sites, and NetKet handles everything else. Easy to
  write for standard spin or bosonic lattice models, but the underlying matrix
  representation grows with the number of local states involved, it cannot
  scale to large systems or to operators that act globally across the
  configuration.

* **Matrix-free** (neuraLQX's ``ComputationalJaxOperator``): based on NetKet's ``DiscreteOperatorJax``
  you write a JAX kernel function that, given a configuration, directly returns connected
  configurations and matrix elements. Maximum control and performance, but you
  are writing low-level JAX code, array slicing, masking, ``lax.cond``,
  ``vmap``, for every operator. It works well once you have experience, but it is
  tedious to write, hard to read back later, and easy to get conventions wrong.

A **symbolic operator DSL** is a third way that sits above both. Instead of
writing a matrix or a kernel, you write a concise *description* of the physics
in plain Python, using readable statements like:

   *"for each pair of sites (i, j) where site i is occupied, lower i by one
   and raise j by one, with matrix element equal to the square root of the
   occupation at i"*

and the framework **compiles** that description into an efficient JAX kernel
for you.

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


--------------------------------------------------------------------
nkDSL
--------------------------------------------------------------------

nkDSL is a standalone package that implements the symbolic operator DSL for
NetKet. It was extracted from neuraLQX so that the broader NetKet community
can benefit from the DSL independently of the neuraLQX stack.

Install it with::

   pip install nkDSL

Full documentation: `nkdsl.readthedocs.io <https://nkdsl.readthedocs.io/en/latest/>`_


Quick start
============

The four-part mental model maps directly onto the nkDSL fluent API:

.. code-block:: text

   SymbolicDiscreteJaxOperator(hilbert, "name")
     .for_each_*(labels, ...)     # 1. where to act  - iterator domain
     .where(predicate)            # 2. when to act   - filter
     .emit(update, ...)           # 3. how to act    - update + amplitude
     .compile()                   # compile to a NetKet DiscreteJaxOperator

A hopping operator on a bosonic lattice:

.. code-block:: python

   import nkdsl

   hop = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "hopping")
       .for_each_distinct_pair("i", "j")
       .where(nkdsl.site("i") > 0)
       .emit(
           nkdsl.shift("i", -1).shift("j", +1),
           amplitude=(nkdsl.site("i").value).sqrt(),
       )
       .compile()
   )

   xp, mels = hop.get_conn_padded(x_batch)

The resulting object is an ordinary NetKet ``DiscreteJaxOperator`` and slots
directly into variational Monte Carlo: ``vstate.expect(hop)`` and
``vstate.expect_and_grad(hop)`` work exactly as they would for any other
operator.

Runtime semantics
------------------

For one configuration ``x``, the evaluation is:

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
Outputs are a branch multiset: duplicate ``x'`` rows are retained and not
implicitly merged.


Built-in iterators
===================

nkDSL ships several iterators covering the most common visit patterns:

* ``globally()``, a single visit with no site index, fanout ``M = 1``.
* ``for_each_site("i")``, one visit per site, fanout ``M = N``.
* ``for_each_pair("i", "j")``, all ordered pairs, fanout ``M = N²``.
* ``for_each_distinct_pair("i", "j")``, unordered pairs ``i < j``.
* Static K-body variants via ``for_each(labels, over=rows)`` where ``rows``
  is an explicit list of index tuples.

Iterator choice is your primary per-sample complexity control. Prefer
``for_each(..., over=edges)`` over ``for_each_pair`` on sparse graphs.


--------------------------------------------------------------------
Custom iterators
--------------------------------------------------------------------

Custom iterators let you control which site tuples are visited. They change
the domain *before* predicates run, unlike predicates, which filter rows
after iteration.

Implement ``nkdsl.AbstractIteratorClause`` and register it by subclassing:

.. code-block:: python

   import nkdsl

   class ForEachEvenSite(nkdsl.AbstractIteratorClause):
       clause_name = "for_each_even_site"

       def build_iterator(self, hilbert, label: str = "i"):
           n = int(hilbert.size)
           rows = tuple((k,) for k in range(n) if k % 2 == 0)
           if not rows:
               raise ValueError("No even sites available.")
           return (str(label),), rows

Once the class is defined it integrates into the fluent API automatically
under its ``clause_name``. You can then write:

.. code-block:: python

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "even-sites")
       .for_each_even_site("i")
       .emit(nkdsl.shift("i", +1), amplitude=1.0)
       .compile()
   )

For graph models, drive iteration from an explicit edge list:

.. code-block:: python

   class ForEachEdge(nkdsl.AbstractIteratorClause):
       clause_name = "for_each_edge"

       def build_iterator(self, hilbert, src: str = "i", dst: str = "j", *, edges):
           n = int(hilbert.size)
           rows = tuple((int(i), int(j)) for i, j in edges)
           if not rows:
               raise ValueError("edges must contain at least one pair")
           for i, j in rows:
               if i < 0 or j < 0 or i >= n or j >= n:
                   raise ValueError(f"edge ({i}, {j}) out of bounds")
           return (str(src), str(dst)), rows

.. tip::

   Keep each clause focused on a single selection rule and add validation
   inside ``build_iterator`` for early error detection. The IR lets you verify
   the iterator domain before examining predicate or emission logic.


--------------------------------------------------------------------
Predicates
--------------------------------------------------------------------

A predicate is a Boolean filter evaluated in the source-state environment.
Rows that fail the predicate produce no branch contribution. Multiple
``.where()`` calls compose via logical AND.

nkDSL ships comparison predicates and logical composition via ``where()``. For
domain-specific conditions, subclass ``nkdsl.AbstractPredicateClause``:

.. code-block:: python

   import nkdsl

   class AtLeastOccupancy(nkdsl.AbstractPredicateClause):
       clause_name = "at_least_occupancy"

       def build_predicate(self, ctx, label: str = "i", cutoff: int = 1):
           return ctx.site(label).value >= int(cutoff)

   nkdsl.register_predicate_clause(AtLeastOccupancy, replace=True)

After registration the clause is available on any operator definition:

.. code-block:: python

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "hop-threshold")
       .for_each_site("i")
       .at_least_occupancy("i", cutoff=2)
       .where(nkdsl.site("i") != 3)
       .emit(nkdsl.shift("i", -1), amplitude=1.0)
       .compile()
   )

The two ``.where``-style clauses above compose as
``(x[i] >= 2) AND (x[i] != 3)``.

Best practices:

* Give each clause a single logical responsibility.
* Validate parameters immediately inside ``build_predicate``.
* Inspect the IR ``where:`` field to verify predicate composition.


--------------------------------------------------------------------
Emissions
--------------------------------------------------------------------

An emission clause defines what branch to generate and what matrix element to
assign for each active iterator row. The standard ``.emit()`` method handles
the common case, for reusable domain-specific patterns, subclass
``nkdsl.AbstractEmissionClause``:

.. code-block:: python

   import nkdsl

   class EmitWhenAtLeast(nkdsl.AbstractEmissionClause):
       clause_name = "emit_when_at_least"

       def build_emission(self, ctx, label: str = "i", cutoff: int = 1):
           predicate = ctx.site(label).value >= int(cutoff)
           return nkdsl.EmissionClauseSpec(
               mode="emit_if",
               predicate=predicate,
               update=nkdsl.identity(),
               matrix_element=ctx.site(label).value,
               tag="emit-when-at-least",
           )

   nkdsl.register_emission_clause(EmitWhenAtLeast, replace=True)

Usage in an operator definition:

.. code-block:: python

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "custom-emission")
       .for_each_site("i")
       .emit_when_at_least("i", cutoff=2)
       .emit_else(nkdsl.identity(), matrix_element=0.0)
       .compile()
   )

The ``@nkdsl.register`` decorator offers a more compact registration syntax:

.. code-block:: python

   @nkdsl.register
   class EmitIfNonZero(nkdsl.AbstractEmissionClause):
       clause_name = "emit_if_nonzero"

       def build_emission(self, ctx, label: str = "i", *, mel: float = 1.0):
           return nkdsl.EmissionClauseSpec(
               mode="emit_if",
               predicate=ctx.site(label).value != 0,
               update=nkdsl.identity(),
               matrix_element=float(mel),
               tag="nonzero",
           )

The ``mode`` field accepts ``"emit"``, ``"emit_if"``, ``"emit_elseif"``, or
``"emit_else"``, which maps the clause into the conditional chain described in
the next section.


--------------------------------------------------------------------
Conditional emissions
--------------------------------------------------------------------

Many physical rules are naturally piecewise: the transition rule changes
depending on the local quantum number. Conditional emissions let you express
``if / elseif / else`` logic within a single term while keeping a static
output shape required by JAX.

Chain ``.emit_if()``, ``.emit_elseif()``, and ``.emit_else()``:

.. code-block:: python

   import nkdsl

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "piecewise")
       .for_each_site("i")
       .emit_if(
           nkdsl.site("i") == 0,
           nkdsl.write("i", 1),
           matrix_element=10.0,
       )
       .emit_elseif(
           nkdsl.site("i") == 1,
           nkdsl.write("i", 2),
           matrix_element=20.0,
       )
       .emit_else(
           nkdsl.write("i", 3),
           matrix_element=30.0,
       )
       .compile()
   )

The compiler automatically enforces mutual exclusivity: each subsequent branch
carries the implicit condition "all prior branches were false".  This means
you do not need to negate prior conditions manually.

Structural rules:

* ``.emit_elseif()`` and ``.emit_else()`` are only valid after an open
  ``.emit_if()`` chain, starting with either raises a ``ValueError``.
* A term-level ``.where()`` acts as a coarse gate applied before the branch
  chain, cleanly separating hard constraints from piecewise transitions.


--------------------------------------------------------------------
Reading the IR
--------------------------------------------------------------------

The symbolic IR is a textual representation of a compiled operator that
exposes iterator domains, predicates, and emitted updates. It is the primary
debugging tool for understanding what the framework will do with your
definition.

Print an operator's IR by calling ``str()`` on the built (pre-compiled) object
or by inspecting it in a REPL:

.. code-block:: text

   iterate: for (i,) in [(0,), (1,), (2,), ... +1 more]
   where:   (x[i] > 0)
   emit:
     update:    x'[i] = (x[i] + -1);  x'[j] = (x[j] + 1)
     amplitude: sqrt(x[i])

Reading order: **iterate -> where -> each emit block (update + amplitude)**.
This mirrors the compiler's interpretation order and lets you pinpoint issues
quickly.

Key fields:

``iterate``
   The iterator domain, which site tuples are visited. Verify this first
   when a term produces wrong connectivity.

``where``
   Composed predicate expression. Multiple ``.where()`` clauses appear here
   joined by AND.

``update``
   The state rewrite program applied to produce ``x'``.

``amplitude``
   The matrix-element expression. Symbols referencing the source state use
   ``x[...]``, symbols referencing the emitted state use ``x'[...]``.

For programmatic inspection use ``as_dict()``:

.. code-block:: python

   ir = op.ir()
   d  = ir.as_dict()

   print(d["free_symbols"])        # external parameters required
   print(ir.static_fingerprint())  # structural identity hash (cache key)

Conditional branches introduce per-emission ``where:`` fields that show the
effective branch predicate after mutual-exclusivity expansion, useful for
verifying that the ``if/elseif/else`` logic compiled to what you intended.

Debugging strategy:

1. Check the iterator domain.
2. Check predicate composition in ``where:``.
3. Check emission updates and amplitudes.

Tackle each layer independently rather than guessing at the combined behaviour.
