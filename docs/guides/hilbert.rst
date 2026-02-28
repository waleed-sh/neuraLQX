.. _hilbert_spaces:

Hilbert spaces
==============

.. note::

   The current version of neuraLQX supports only Hilbert spaces with Abelian U(1)\ :sup:`N` degrees of freedom on edges.
   SU(2) Hilbert spaces are currently a feature under development.

The :mod:`neuralqx.hilbert` layer turns a :ref:`graph <graphs>` into an explicit configuration
space that you can:

* **sample** with Markov-chain Monte Carlo (MCMC),
* **index** deterministically, and
* **optionally restrict** to a **gauge-invariant subspace** in a *constructive* way.

In practice, this layer answers three questions for you:

* which local quantum numbers are allowed?
* how those degrees of freedom are laid out in memory (so NetKet/JAX can work efficiently)? and
* how gauge constraints are enforced when you choose to work in the physical subspace?

This page documents the public U(1)\ :sup:`N` Hilbert space implementation. The public entry point for the U(1) Hilbert
space is the ``neuralqx.hilbert.u1`` module. The SU(2) Hilbert space exists in an experimental namespace and follows the
same high-level design, but it is not considered part of the stable API yet.


Quick start
-----------

A Hilbert space is always constructed *on top of a graph*:

.. code-block:: python

   import neuralqx as nqx

   graph = nqx.graph.K5Graph()          # any Graph subclass works
   H = nqx.hilbert.u1.HilbertU1(graph, cutoff=3, gauge_dimensions=1)

   print(H)
   # HilbertU1(dimensions=2.82 x 10⁸, cutoff=3, dofs=StaticRange(start=-3, step=1, length=7, dtype=int64), is_gauge_invariant=False, is_finite=True, is_indexable=True)

From there you typically do one (or more) of the following:

* draw random configurations for initialisation or debugging,
* build operators that act on edge-local degrees of freedom,
* define MCMC proposal moves, and/or
* convert between configurations and integer labels.

The sections below go through the underlying conventions in detail so that operator code,
samplers, and indexing utilities compose cleanly.


Design overview: interfaces and cores
---------------------------------------

The Hilbert module is split into two layers:

**1) A thin user-facing interface:**
  The *public* object you instantiate is an *interface* class (conceptually,
  ``AbstractHilbertInterface``). It validates user inputs, exposes a stable attribute surface,
  and provides convenience methods that are useful in downstream workflows and lend themselves easily to physics
  applications.

**2) A concrete core implementation**
  The interface owns a *core* instance (derived from ``AbstractHilbertSpace``). The core is
  responsible for the computational details, so how the NetKet Hilbert object is constructed,
  how gauge fixing is stored, how states are reshaped, and how constraint-aware operations
  are performed. This core object is always exposed through the ``core`` attribute of any ``AbstractHilbertInterface``
  subclass.

This separation is there for a practical reason. Most users want a small, stable surface
(``HilbertU1(...).random_state(...)``, ``states_to_numbers(...)``, etc.), while the internal
implementation needs room to evolve (especially for constrained sampling and move sets)
without breaking call sites. In code you will occasionally access ``AbstractHilbertInterface.core`` when you need
lower-level operations (e.g. explicit gauge reimposition), but most workflows should stay at the interface level.


Local degrees of freedom (DOFs) and truncation
-------------------------------------------------

U(1)\ :sup:`N` edge labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the public U(1) implementation, each **graph edge** carries a discrete representation
label drawn from a finite set ``Q``. This set is implemented as a NetKet ``StaticRange`` and
is determined by three key pieces of metadata: a minimum, a maximum, and a step size.

The most common choice is the symmetric truncation controlled by:

* ``cutoff=qmax`` (default step size is 1)

which yields the label set ``{-qmax, ..., 0, ..., +qmax}``.

.. code-block:: python

   # Q = [-2, -1, 0, 1, 2]
   H = nqx.hilbert.u1.HilbertU1(graph, cutoff=2)

Positive-only labels
~~~~~~~~~~~~~~~~~~~~

Some applications prefer non-negative labels (e.g. a shifted convention for integer representations). You can request
that with:

* ``positive_qn=True`` to start at 0 instead of ``-qmax``,
* and optionally ``qn_start=<int>`` to choose the starting value.

.. code-block:: python

   # Q = [0, 1, 2, 3, 4]
   H0 = nqx.hilbert.u1.HilbertU1(graph, cutoff=2, positive_qn=True)

   # Q = [1, 2, 3, 4, 5]
   H1 = nqx.hilbert.u1.HilbertU1(graph, cutoff=2, positive_qn=True, qn_start=1)

The **cardinality** of ``Q`` stays the same for fixed ``cutoff`` and step size; you are only
choosing where the range begins.

Gauge copies per edge: ``gauge_dimensions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many LQG models, and in fact the standard 4-d Abelian model, treat an edge label as a **vector** (e.g. U(1)\ :sup:`3` charges per
edge). The parameter ``gauge_dimensions = G`` controls how many U(1) components live on each edge. The per-edge local space scales as
``|Q|**G`` and the full kinematical space scales as ``(|Q|**G)**|E|``.

In other words:

* ``gauge_dimensions=1`` means one integer label per edge (U(1) degrees of freedom)
* ``gauge_dimensions=3`` means a length-3 charge vector per edge (U(1)\ :sup:`3` degrees of freedom, stored in a flattened convention described next).

This "multi-copy" representation is central to how neuraLQX stays JAX-friendly: vector degrees of freedom remain simple
arrays, and operators can be vectorized over the copy index efficiently.


Configuration layout and edge indexing
--------------------------------------

The most important thing to internalise when writing operators or custom moves is **how a configuration is stored**. This
is *partially* provided through the Graph API in neuraLQX, but it does not accommodate to having vector degrees of freedom.
This job is delegated to the Hilbert spaces themselves, which need to specify a certain layout.

Edges map to sites
~~~~~~~~~~~~~~~~~~

neuraLQX uses the graph's **primal/dual mapping** convention where each *primal edge* corresponds to one *dual vertex*,
which becomes one "site" in the associated NetKet Hilbert space. Operationally, this means:

* configurations are arrays with one slot per edge (per gauge copy),
* and the graph API provides the stable mapping between an edge key and that slot.

If your graph supports multi-edges, edges are identified by keys like ``(u, v, k)`` (node-node
plus an edge index). When you interact with the Hilbert space at the level of *edge keys*,
you should always use the same key convention that the graph exposes.

Gauge-strided flattening
~~~~~~~~~~~~~~~~~~~~~~~~


Let

* ``E`` be the number of edges in one gauge copy
* ``G`` be ``gauge_dimensions``

Then a single flattened configuration is a 1D array of length ``N = G * E``. In the strided gauge-copy layout used by neuraLQX, the data is stored as ``G`` contiguous blocks of length ``E``,

``[copy 0 | copy 1 | ... | copy G-1]``.

The concrete implementation is :class:`StridedGaugeCopyLayout`, which is a subclass of the more general :class:`AbstractBasisLayout` abstraction. This layout defines the forward and backward maps between structured gauge coordinates and flattened site indices:

* ``encode(g, e) = g * E + e``
* ``decode(s) -> (g, e)``

where ``g`` is the gauge-copy index and ``e`` is the within-copy edge index.

For layout-agnostic code, the generic API is also available:

* ``site_of(GaugeCoord(g, e)) -> s``
* ``coord_of(s) -> GaugeCoord(g, e)``

.. note::

   The legacy helper ``site(e, g)`` is retained temporarily for backward compatibility, but it is deprecated and will be removed in a future release. Use ``encode(g, e)`` instead. The new API keeps the argument order consistent with ``decode(s) -> (g, e)``.

This convention appears throughout the codebase, including sampling, move proposals, constraint reimposition, and indexing utilities. Centralizing the layout logic avoids subtle off-by-block errors and keeps flattening semantics consistent across components.


Reshaping helpers: ``view`` and ``flatten``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because *most* humans prefer to think in blocks, the core exposes reshapes consistent with the gauge-strided storage. This
is accessible through

* ``view(sigma)`` reshapes a flat state from ``(N,)`` to ``(G, E)``,
  and a batch from ``(B, N)`` to ``(B, G, E)``.
* ``flatten(...)`` reverses the operation.

.. code-block:: python

   graph = nqx.graph.K5Graph()
   H = nqx.hilbert.u1.HilbertU1(graph, cutoff=3, gauge_dimensions=1)

   # generate a random state in the Hilbert space
   sigma = H.random_state(jax.random.PRNGKey(0), size=1)
   print(sigma)
   # [[ 1  0 -1  3  2  1  3  0  3  2  3  0  1  1  0 -2 -1  2  0  1 -1  1 -2  1 -3 -1 -2 -3  2  0]]

   # view each gauge copy in a dimension on its own
   print(H.core.view(sigma))
   # Array([[[ 1,  0, -1,  3,  2,  1,  3,  0,  3,  2],
   #         [ 3,  0,  1,  1,  0, -2, -1,  2,  0,  1],
   #         [-1,  1, -2,  1, -3, -1, -2, -3,  2,  0]]], dtype=int64)

   # back to the flat view
   sigma_flat  = H.core.flatten(sigma_block)
   print(sigma_flat)
   # Array([ 1,  0, -1,  3,  2,  1,  3,  0,  3,  2,  3,  0,  1,  1,  0, -2, -1, 2,  0,  1, -1,  1, -2,  1, -3, -1, -2, -3,  2,  0], dtype=int64)

Convenience accessors for “edge charge vectors”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``gauge_dimensions > 1``, you often want the vector of charges for a single edge. neuraLQX makes two convenience
options are available:

* Use ``H.core.view`` and pick the appropriate edge slot across copies.
* Use the higher-level helper ``H.edge_charges(sigma, edge_key)`` which returns a
  length-``G`` vector for that edge key.

.. code-block:: python

   # edge_key uses your graph's key convention, e.g. (u, v, k)
   edge_key = (0, 1, 0)

   # use the same sigma in the previous example
   q_vec = H.edge_charges(sigma, edge_key)
   print(q_vec)
   # Array([ 1,  3, -1], dtype=int64)

These helpers exist primarily to keep operator code readable (volume-like operators and
other multi-copy constructions become much easier to express).


Kinematical Hilbert space: ``HilbertU1`` (unconstrained)
--------------------------------------------------------

Constructing the full space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default construction gives you the full kinematical space, which is a direct product of local edge Hilbert spaces
over the entire edge set. You control its size through:

* ``cutoff`` (range of allowed labels),
* ``gauge_dimensions`` (number of U(1) copies per edge),
* and label convention options such as ``positive_qn`` mentioned above.

.. code-block:: python

   H = nqx.hilbert.u1.HilbertU1(graph, cutoff=2, gauge_dimensions=3)

Under the hood, the interface also exposes the underlying **NetKet Hilbert** object as
``H.hilbert`` (useful when constructing NetKet operators).

Accessing the local basis values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In operator construction you often need the allowed local values ``Q`` as an explicit array.
The allowed basis values can be obtained via the interface (for example through
``H.allowed_basis_states.all_states()``), which returns the list of values used on each site.

.. code-block:: python

   Q = H.allowed_basis_states.all_states()   # e.g. [-2, -1, 0, 1, 2]

This becomes helpful when building operators, where for example a typical pattern is to build a small local matrix
(dense or sparse) using ``Q`` and then embed it as a NetKet ``LocalOperator`` acting on one (or several) edge-sites
chosen via the graph's edge-to-index mapping.

Random states in the unconstrained space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random configuration generation is straightforward here. Each site is drawn independently and uniformly from ``Q``
(with appropriate batching). This is the behavior you get when you call ``H.random_state(key, size=...)`` on an
unconstrained space.

.. code-block:: python

   import jax

   samples = H.random_state(jax.random.PRNGKey(0), size=16)  # shape (16, G*E)


Gauge-invariant subspace: constructive gauge fixing
---------------------------------------------------

Many workflows want configurations that satisfy the Gauß constraint *by construction* (i.e. work in the gauge invariant
subspace from the start). In neuraLQX this is done by building a Hilbert space whose degrees of freedom are
parameterized by a set of **free** edges, and all **slave** edges are reconstructed deterministically
from the free ones using a gauge-fixing rule.

Compared to rejection sampling, this remains practical as graphs grow because you never
spend time proposing or sampling configurations that you immediately discard.

Enabling the gauge-invariant space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To construct the gauge-invariant subspace you set:

* ``is_gauge_invariant=True``

and then provide either:

* a manual ``gauge_fixing=...`` list, or
* ``auto_constraint=True`` to derive one from graph topology.

Manual gauge fixing format
~~~~~~~~~~~~~~~~~~~~~~~~~~

neuraLQX is particular about the format of the provided gauge fixing list for constructing gauge invariant subspaces.
A gauge fixing is provided as a list of relations of the form:

``[[LHS], [RHS_1, RHS_2, ...]]``

where the LHS names a slave edge and the RHS names edges it depends on. Signs are
encoded in one of two ways:

* prefix the RHS edge string with ``-`` to mark a negative contribution, or
* use a special token (e.g. ``"N"``) meaning "negate all subsequent RHS entries".

Edges are provided **as strings**, following the edge-key formatting used by the graph.

.. code-block:: python

   gauge_fixing = [
       [["(0, 1)"], ["(1, 2)"]],
       [["(0, 3)"], ["(3, 2)"]],
       [["(0, 2)"], ["-(0, 1)", "-(0, 3)"]],
   ]

   # using the 'N' token
   gauge_fixing = [
       [["(0, 1)"], ["(1, 2)"]],
       [["(0, 3)"], ["(3, 2)"]],
       [["(0, 2)"], ["N", "(0, 1)", "(0, 3)"]],
   ]

Validity requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A manual gauge fixing **must be cycle-free** in the dependency graph, that means the directed graph
"slave depends on RHS" must be *acyclic*. neuraLQX reconstructs slave edges in a
*topological order*, so cyclic dependencies are not resolvable.

When you design a gauge fixing by hand, it helps to think in these terms:

* You are choosing a set of free variables (often cycle edges),
* and writing every other edge as a deterministic function of those free variables.

If you find yourself creating mutual dependencies ("A depends on B" and "B depends on A"), you have introduced a cycle
and the reconstruction order breaks.

Constructing a gauge-invariant Hilbert space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have a gauge fixing list ready, constructing a gauge invariant Hilbert space in neuraLQX is relatively
straightforward. Just do

.. code-block:: python

   H_gi = nqx.hilbert.u1.HilbertU1(
       k5_graph,
       cutoff=2,
       gauge_dimensions=3,
       is_gauge_invariant=True,
       gauge_fixing=gauge_fixing,
   )
   print(H_gi)
   # HilbertU1(dimensions=3.81 x 10¹², cutoff=2, dofs=StaticRange(start=-2, step=1, length=5, dtype=int64), is_gauge_invariant=True, is_finite=True, is_indexable=False)

With gauge fixing enabled, the effective number of independent degrees of freedom drops to
the number of independent cycles per gauge copy (and the total space dimension shrinks
accordingly).

Automatic gauge fixing (recommended for large graphs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For nontrivial graphs, building a complete, cycle-free fixing by hand gets tedious *very* quickly. neuraLQX can take care
of this for you! Simply setting ``auto_constraint=True`` instructs neuraLQX to construct a suitable gauge fixing directly
from the graph topology.

.. code-block:: python

   H_auto = nqx.hilbert.u1.HilbertU1(
       k5_graph,
       cutoff=2,
       gauge_dimensions=3,
       is_gauge_invariant=True,
       auto_constraint=True,
   )
   print(H_auto)
   # HilbertU1(dimensions=3.81 x 10¹², cutoff=2, dofs=StaticRange(start=-2, step=1, length=5, dtype=int64), is_gauge_invariant=True, is_finite=True, is_indexable=False)

The resulting Hilbert space ``H_auto`` will be identical to the one with the manually specified gauge fixing ``H_gi``.
Conceptually, the algorithm follows a standard graph-theoretic pattern:

1. Build an oriented incidence matrix for the graph.
2. Choose a spanning tree (tree edges).
3. Treat all remaining edges as chords (cycle edges).
4. Solve conservation constraints to express tree-edge values in terms of chord-edge values.

The outcome is exactly what you want for sampling, the chord edges become the **free** parameters, and tree edges become
deterministic **slaves**.


Reimposing gauge fixing on configurations
-----------------------------------------

Even with constraint-aware moves, it is useful to have an explicit "repair" operation that takes an arbitrary
configuration and projects it back into the gauge-fixed manifold.

The gauge-invariant core stores:

* the set of free indices,
* the list of slave indices in a topological order,
* and for each slave, a signed dependency list.

Reimposition then:

1. reshapes the configuration into ``(B, G, E)`` using the gauge layout,
2. overwrites each slave edge in topological order using a signed modular sum of its
   dependencies,
3. flattens back to ``(B, G*E)``.

Two guarantees make this routine dependable in practice:

* **Free edges are left untouched**. Only slaves are overwritten.
* If a configuration already satisfies the gauge fixing, reimposition returns it unchanged
  (idempotent behavior on the constrained manifold).

In the current implementation, the relevant entry points are:

* ``H.core.reimpose_gauge_fixing(sigma)``
* ``H.core.is_gauge_invariant(sigma)``  (boolean per configuration)

.. code-block:: python

   ok_or_not = H_gi.core.is_gauge_invariant(samples)
   repaired = H_gi.core.reimpose_gauge_fixing(samples)

You will see this pattern in several places across the library: propose a change in the free
variables, then deterministically rebuild slaves. It keeps proposal kernels simple and avoids
spreading "constraint bookkeeping" throughout the codebase.


Dispatch-based operations: random states and proposal moves
-----------------------------------------------------------

One goal of the Hilbert API is that *higher-level code does not need to care* whether a Hilbert space is constrained.
neuraLQX achieves this by using multi-dispatch for key operations (in the style of NetKet) where the public call is stable,
while the implementation is selected based on the concrete core type.

This approach also plays nicely with JAX as it keeps Python-side branching out of traced inner loops by selecting the
correct implementation before tracing begins.

Random state generation
~~~~~~~~~~~~~~~~~~~~~~~

You call the same method either way:

.. code-block:: python

   import jax
   s1 = H.random_state(jax.random.PRNGKey(0), size=4)
   s2 = H_gi.random_state(jax.random.PRNGKey(0), size=4)

Internally:

* unconstrained spaces sample i.i.d. site values from ``Q``,
* gauge-invariant spaces sample only free edges and reconstruct slaves.

Proposal moves: ``flip_state`` and typed ``Move`` objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Markov chain Monte Carlo requires a proposal kernel ``sigma -> sigma'``. neuraLQX provides two layers from the Hilbert:

**1) A generic dispatch-based proposal function**
  ``flip_state(space, sigma, key)`` chooses some sites and updates them using either:

  * random replacement (uniform new value from ``Q``), or
  * adjacent steps (add/subtract a step size with wrap-around).

  In the unconstrained case this is always valid. In the constrained case the overload
  restricts updates to free-edge slots and then reimposes gauge fixing.

**2) A typed move system for constrained spaces**
  Constrained cores support an explicit ``Move`` API and a stable entry point:

  * ``propose(sigma, key, move)``

  where ``move`` is an immutable object such as:

  * ``FreeEdgeFlipSingleGauge``
  * ``PlaquetteFlipSingleGauge``
  * and "all-gauge" variants.

The typed-move approach is designed to be extensible without fragile string switches.
To add a new move type, you register a corresponding propose implementation rather than
editing a monolithic if/else chain.


Indexing configurations: ``states_to_numbers`` and ``numbers_to_states``
------------------------------------------------------------------------

Why indexing matters
~~~~~~~~~~~~~~~~~~~~

Indexing is useful in many places.

* caching (memoizing expensive local computations),
* deterministic labeling in tests,
* bridging between "configuration arrays" and compact integer IDs,
* small-system exact checks,
* plotting the entire state amplitudes.

NetKet can index many Hilbert spaces, but neuraLQX targets regimes where dimensions are
often astronomically large. For this reason it provides indexing utilities that mirror NetKet's
behavior where possible and remain available beyond NetKet's indexability limits. Note that these methods are fully
compatible to be used in jitted JAX routines and in JAX transformations.


Public API
~~~~~~~~~~

Every Hilbert interface exposes two methods which allow you to go from a concrete basis state (i.e. a JAX array
encoding a basis state) to a sequential labelling of the basis states, and back. These are

* ``H.states_to_numbers(state, backend="auto", ...)``
* ``H.numbers_to_states(i, backend="auto", ...)``

These accept both single states of shape ``(N,)`` and batches of shape ``(..., N)``, and return
numbers with the matching batch shape (and vice versa).

Backends
~~~~~~~~

In neuraLQX, there are three backends that are supported which make this functionality available. They are:

* ``backend="netket"``: delegate to NetKet's own indexing when available (i.e. space is NetKet indexable)
* ``backend="python"``: use arithmetic rank/unrank (works even when the space is not indexable, but integers may become very large)
* ``backend="auto"``: try NetKet first, then fall back to arithmetic ranking

Digitization and ordering conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing is performed on **digits** (0...L-1) corresponding to positions in the local basis set,
rather than directly on physical values in ``Q``. neuraLQX uses NetKet's local mapping to
ensure this digitization matches NetKet's internal basis ordering.

A subtle point is which site is treated as "most significant" in the base-``L`` representation. Different libraries
choose different conventions. When needed, neuraLQX infers NetKet's convention empirically from a small reference
construction rather than hard-coding an assumption. This allows us to be agnostic to the NetKet implementation.

Large integers and dtype choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use arithmetic ranking on a system with large ``N``, numbers scale like ``L**N``. Depending on your application you may prefer:

* fast 64-bit integers (with possible overflow),
* or Python big integers (arbitrary precision, slower).

The indexing utilities expose options such as ``return_dtype`` to control this behavior. You can choose between ``'int64'``
for 64-bit integers and ``'object'`` for Python big integers. Alternatively, the default option which is ``'auto'`` will
choose the best option depending on your Hilbert space.

Reduced indexing for gauge-invariant spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For gauge-fixed constrained spaces, indexing is performed on the free variables rather than the full edge list. Concretely,
the constrained configuration is parameterized by ``D = G*F`` free values (``F`` free edges per gauge copy), and
ranking/unranking proceeds in that reduced coordinate system before reconstructing slaves.

This keeps indexing aligned with the constructive definition of the space and avoids treating
deterministic slave slots as independent digits.

