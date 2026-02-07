.. _gauge_groups:

==============
Gauge groups
==============

The :mod:`neuralqx.gauge_groups` module provides **gauge-group descriptors** that are tied to a
specific Hilbert space and expose a ready-to-use **Gauß constraint operator**. In neuraLQX, a
gauge group object is the place where you declare "what local redundancy exists" and obtain
the corresponding constraint operator in the operator backend you want to run with. 

This is intentionally separated from both the Hilbert-space construction and the physical-model layer. The idea is that:

* The **Hilbert space** tells you what configurations exist and how they are laid out (sites, gauge copies, indexing).
* The **gauge group** tells you what "gauge invariance" means on those configurations and provides a concrete operator that detects or penalizes violations.
* The **model** decides how (and whether) that constraint is used inside the optimisation objective (e.g. as a hard objective, a penalty term, a consistency check).

This separation is useful in practice because you can:

1) keep the same Hilbert space but swap gauge groups (when multiple are available),
2) benchmark constraint implementations independently of the dynamics,
3) choose the constraint backend (LocalOperator vs computational vs JAX computational) without rewriting the rest of your pipeline. 


Gauge invariance in neuraLQX: two complementary workflows
===========================================================

neuraLQX supports gauge invariance in two ways, and it is important to understand the difference.

**1) Constructive gauge invariance (Hilbert-space level)**  
You reduce the kinematical Hilbert space to a gauge-invariant subspace by parametrising states using
free variables and reconstructing the remaining degrees of freedom deterministically. This gives you
a smaller configuration space where every sampled configuration is already gauge-fixed.

**2) Active gauge invariance (operator level)**  
You work in the full kinematical space and include an operator representation of the Gauß constraint
in your optimisation target (for example, you minimise the expectation value of a positive
semi-definite constraint operator). 

The gauge-group objects in :mod:`neuralqx.gauge_groups` support the second workflow directly, and they
remain useful even if you use the first workflow. In a gauge-invariant Hilbert space, the same Gauß
operator becomes a strong diagnostic that your sampler/moves and operator logic behave consistently. 


The public abstraction: ``AbstractGaugeGroup``
================================================

The entry point of the module is the abstract interface :class:`neuralqx.gauge_groups.AbstractGaugeGroup`.
It defines what every gauge group implementation must provide, regardless of whether it is Abelian or
non-Abelian, and regardless of how the constraint operator is implemented. 

A gauge group instance has three responsibilities.

1. **Expose stable metadata**
   A gauge group object answers questions like:

   * what is the group name (e.g. ``"U(1)"``),
   * is it Abelian,
   * how many independent group factors (copies) exist when applicable.

   In neuraLQX, "how many copies" is tracked as *dimensions*, and for the U(1)\ :sup:`N` family this
   matches the Hilbert-space notion of ``gauge_dimensions``. The key point is that downstream code
   can inspect this without hard-coding U(1)-specific assumptions.

2. **Bind to a concrete Hilbert space**
   ``AbstractGaugeGroup`` stores a reference to the Hilbert interface the group acts on. That coupling
   guarantees that the constraint operator and all group-level conventions
   (indexing, local basis range, number of gauge copies) are consistent with the configuration space
   used by samplers and by NetKet estimators. 

3. **Provide the Gauß constraint operator through a stable property**
   The public attribute/property is:

   * ``group.constraint``

   Subclasses implement an internal ``_init_constraint()`` method that constructs this operator in
   one of several possible backends. The rest of the library treats ``constraint`` as "the thing you
   plug into expectation values / penalties / diagnostic checks".


Backend selection
--------------------

The gauge group constructor accepts the following three flags that control how the constraint is built:

``computational: bool``
  Prefer a **computational** (matrix-free) operator implementation, i.e. one that generates connected
  configurations algorithmically.

``jax: bool``
  Prefer a JAX-compatible computational implementation when available.

``lazy: bool``
  Delay constructing a ``LocalOperator`` implementation until it is actually needed (useful because
  building large ``LocalOperators`` can incur setup cost).

A typical workflow is:

* instantiate a gauge group once,
* start with a computational/JAX backend for performance,
* switch to a ``LocalOperator`` backend temporarily for debugging or for small-system comparisons,
* switch back without rebuilding the entire model. 


Concrete implementation: ``U1GaugeGroup``
=========================================

At the time of writing, the fully supported gauge group implementation is the Abelian U(1)\ :sup:`N`
family provided as :class:`neuralqx.gauge_groups.u1.U1GaugeGroup`. 

The object is constructed from a **U(1) Hilbert interface** (i.e. a ``neuralqx.hilbert.u1.HilbertU1`` class), and it infers the number of gauge copies
from that Hilbert space. This avoids a class of mismatches where a user accidentally asks for
U(1)\ :sup:`3` dynamics on a U(1)\ :sup:`1` Hilbert layout. 

Construction
------------

.. code-block:: python

   import neuralqx as nqx

   # graph + U(1) Hilbert space
   gtheta = nqx.graph.ThetaGraph()
   H = nqx.hilbert.u1.HilbertU1(gtheta, cutoff=2, gauge_dimensions=1)

   # gauge group descriptor, with computational JAX constraint backend
   G = nqx.gauge_groups.u1.U1GaugeGroup(H, lazy=True, computational=True, jax=True)

The gauge group exposes convenience accessors such as:

* ``G.dimensions`` (number of copies),
* ``G.name``,
* ``G.hilbert`` (the bound Hilbert interface). 


The U(1)\ :sup:`N` Gauß constraint: what is implemented
==========================================================

In the U(1)\ :sup:`N` setting, the Gauß constraint enforces **charge conservation at every vertex**
for each gauge copy.

Single-copy generator at a vertex
------------------------------------

For a single U(1) copy, the generator at vertex ``v`` is a signed sum over incident edges:

* incoming edges contribute with a ``+`` sign,
* outgoing edges contribute with a ``-`` sign,

and each edge contributes its diagonal flux/charge operator. 

In code terms, the implementation follows a direct graph-to-operator mapping where we

1. query the graph for the incident edges at a vertex,
2. classify each as incoming/outgoing relative to ``v``,
3. accumulate a signed sum of number-like (diagonal) edge operators.

Because the graph API stores oriented edges and maintains connectivity lists, the classification step
is implemented as a pure lookup rather than as geometric reasoning. 


Why neuraLQX exposes a squared constraint
--------------------------------------------

Instead of exposing the unsquared generators directly, neuraLQX provides a single global constraint
operator of the form "sum of squares of the local generators, across all vertices and all gauge
copies".

There are three practical reasons this is the default:

1. It is **positive semi-definite**, so its expectation value is a direct measure of constraint violation.
2. The global operator vanishes if and only if every local generator vanishes (in the finite basis setting).
3. In the truncated basis, it is **diagonal**, which makes it exceptionally cheap to evaluate. It has a trivial
   connection graph (each configuration only connects to itself).

That last point matters for performance. When you add the Gauß constraint as a penalty or as a primary
objective, you do not want it to dominate runtime by generating large numbers of connected states.
The diagonal structure keeps it lightweight.

Multiple copies: how U(1)\ :sup:`N` is built
--------------------------------------------

When ``dimensions > 1``, the full constraint is assembled by summing over gauge-copy contributions.
The copy bookkeeping follows the same **gauge-strided layout** used by the Hilbert space where each copy is
a contiguous block of edge sites, and shifting an operator from copy 0 to copy ``a`` is performed by
offsetting its acting-on indices accordingly. 


Constraint backends
===================

The same mathematical constraint can be represented in different operator backends. neuraLQX supports
both a ``LocalOperator`` implementation (matrix-based, explicit term storage) and computational
implementations (algorithmic connectivity kernels), including a JAX-flavoured computational variant. 

``LocalOperator`` backend
-----------------------------

In the ``LocalOperator`` representation, the constraint is built by explicitly creating (diagonal) number-like
edge operators and summing them into vertex generators, then squaring and summing those contributions
into a global operator. The implementation includes a per-copy builder for the single-copy constraint and
then reuses the gauge-strided shifting logic to cover all copies. 

This backend is attractive when:

* you want compatibility with NetKet's rich ``LocalOperator`` arithmetic and debugging tools,
* you are working at small enough sizes where constructing and storing the term list is cheap,
* you want a straightforward way to compare against explicit dense/sparse representations.

Computational backend (matrix-free)
-----------------------------------

The computational constraint operators implement the action directly on a configuration:

* gather the relevant edge slots around each vertex (and across all gauge copies),
* compute the signed vertex sums,
* square and sum them,
* return a diagonal "one-connection" result (connected state is the input state, matrix element is the computed violation value).

This backend is designed around fixed-shape lookups. Graph metadata is stored at operator instantiation
time so that evaluating the constraint on a batch of configurations becomes mostly indexing plus a small
amount of arithmetic. 

JAX computational backend
-------------------------

The JAX computational backend packages the same logic into a JAX-friendly kernel and exposes it as a
JAX-compatible computational operator type. This allows you to:

* JIT-compile constraint evaluation,
* vectorise across batch dimensions,
* integrate naturally with accelerator execution. 

The dedicated Gauß operator in the computational operator namespace can be instantiated directly; the
gauge group uses the same backend when you request ``computational=True`` and ``jax=True``. 

.. code-block:: python

   Gauss = nqx.operators.computational.Euclidean4d.GaussConstraintOperator(H, jax=True)

Switching backends at runtime
=============================

A common pattern in neuraLQX is to keep one gauge group instance but change the constraint backend
depending on what you are doing (profiling vs debugging vs production runs). This is supported through
the ``constraint`` setter.

The gauge group can be instantiated with one backend and later reinitialised into another backend
without creating a new gauge group object. 

.. code-block:: python

   import neuralqx as nqx

   H = nqx.hilbert.u1.HilbertU1(nqx.graph.K5Graph(), cutoff=2, gauge_dimensions=1)
   G = nqx.gauge_groups.u1.U1GaugeGroup(H, lazy=True, computational=True, jax=True)

   # switch to LocalOperator backend
   G.constraint = {
       "lazy": False,
       "computational": False,
       "jax": False,
       "reinit": True,
   }

The setter is intentionally explicit. **You** state whether you want computational or local, whether you want
JAX, and whether you want a forced rebuild. This keeps the "what backend is active?" state visible in a
single place. 


Modular arithmetic and truncated bases
========================================

Because neuraLQX works with truncated local label sets (finite ``Q``), repeated updates and reconstruction
steps (especially in constrained Hilbert-space workflows) can push intermediate values outside the
allowed range unless you apply a rule to bring them back.

The Hilbert-space gauge-fixing machinery uses **modular arithmetic** on the finite label set so that
reconstructed labels remain in-range under repeated updates. In that setting, the effective algebra is
no longer the plain integer version of U(1)\ :sup:`N`; it behaves like a "modded" variant that is closer
to a quantum-group style truncation. 

For consistency, :class:`~neuralqx.gauge_groups.u1.U1GaugeGroup` provides an optional *modded* Gauß
constraint interpretation where vertex sums are taken modulo the local basis range. This modded variant
is available through the **computational** backend (and **not** through the ``LocalOperator`` backend).

If you are mixing:

* constructive gauge invariance (gauge-fixed Hilbert space), and
* active constraint penalties (Gauß operator in the loss),

you should make sure that both sides interpret addition the same way. In most practical setups, that
means enabling the modded variant whenever your constrained reconstruction is modded.


Gauge group arithmetic
==============================

For U(1)\ :sup:`N`, neuraLQX exposes a convenient Python-level arithmetic that corresponds to the
direct product structure:

* multiplying groups corresponds to concatenating independent U(1) copies,
* powering corresponds to repeating the same factor multiple times. 

The operators are:

* ``G1 * G2``  (direct product)
* ``G1 ** k``  (k-fold power)

Operationally, this constructs a *new* ``U1GaugeGroup`` descriptor that:

1. preserves the backend-selection flags (computational / jax / lazy),
2. reinitialises its constraint operator using those same flags,
3. reports the new combined number of gauge dimensions. 

This is helpful when you want the gauge-copy structure to be visible explicitly at the level of the
gauge group object (and not only hidden inside Hilbert-space parameters), and when you want downstream
code to see a single coherent group instance with a correctly-instantiated constraint.

Example
-------

.. code-block:: python

   import neuralqx as nqx

   H = nqx.hilbert.u1.HilbertU1(nqx.graph.K5Graph(), cutoff=2, gauge_dimensions=4)

   # a U(1)^4 descriptor
   G1 = nqx.gauge_groups.u1.U1GaugeGroup(H, computational=True, jax=True)

   # U(1)^8 descriptor (same backend flags)
   G2 = G1 ** 2

   # U(1)^16 descriptor
   G3 = G1 * (G1 ** 3)

From the point of view of models and optimisers, this arithmetic does not introduce any special cases,
they just consume ``group.constraint`` and basic metadata (like ``group.dimensions`` and
``group.is_abelian``) and remain structurally compatible with future gauge groups added to the
framework. 


API reference
=============

.. currentmodule:: neuralqx.gauge_groups

.. autoclass:: AbstractGaugeGroup
   :members:
   :show-inheritance:

.. currentmodule:: neuralqx.gauge_groups.u1

.. autoclass:: U1GaugeGroup
   :members:
   :show-inheritance:
