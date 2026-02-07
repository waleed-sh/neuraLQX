.. _graphs:

=====================
Graphs in neuraLQX
=====================

The basic building block for any neuraLQX system is the graph. It is the computational framework that provides:

* a discrete support for basis configurations,
* a stable edge-to-array indexing convention suitable for NetKet's flat configuration layout,
* derived graph structures, such as dual graphs and minimal loops, used throughout operator kernels, and
* local orientation sign tables needed by geometric operators (e.g. volume) and constraint constructions.

Here, you will find the general documentation of the public graph API, the conventions it enforces, and the derived
structures you will rely on when implementing models, operators, and samplers.


Quick start
============

neuraLQX provides a flexible concrete graph object suitable for almost all graphs. The public entry point for it is the
:class:`neuralqx.graph.Graph` class. When constructing a graph, you will need to specify a list of *oriented* edges. A
simple example of a theta graph is as follows

.. code-block:: python

   import neuralqx as nqx

   # a θ-graph specified as oriented edges
   edges = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 2)]
   graph = nqx.graph.Graph(edges)


Once constructed, a :class:`~neuralqx.graph.Graph` exposes both raw information (edges,
vertex counts) and derived information (stable indexing, minimal loops, dual graphs, sign
tables) through a set of public attributes which we will describe below.


Graphs in neuraLQX
=============================================

All neuraLQX graphs start from the abstract base graph class :class:`~neuralqx.graph.core.AbstractGraph`. neuraLQX
provides a concrete implementation flexible enough for any graph through the :class:`~neuralqx.graph.Graph` class. This
object is intentionally kept lightweight. It acts as an interface with methods and properties that are designed to read
naturally in physics code.

The actual implementation work is delegated to an internal worker class, the :class:`~neuralqx.graph.core.GraphHandler`,
which is stored in the abstract interface and accessible through the ``handler`` attribute. This handler's job is to
collect and cache derived graph structures such as

* NetKet graph objects,
* NetworkX graph objects,
* minimal loop data,
* coordinate embeddings,
* sign tables.

.. hint::

   Treat :attr:`~neuralqx.graph.Graph.handler` as an *advanced* escape hatch. The stable public
   surface is the :class:`~neuralqx.graph.Graph` API documented here.


Public interface overview
----------------------------

The graph interface exposes (at least) the following commonly used attributes/methods:

* :attr:`~neuralqx.graph.Graph.edges`, which stores the oriented edges which represent the graph (and the number of edges in the graph conveniently stored in :attr:`~neuralqx.graph.Graph.n_edges`)

* :attr:`~neuralqx.graph.Graph.vertices` stores the vertices of the graph (and likewise the number of vertices is conveniently stored in :attr:`~neuralqx.graph.Graph.n_vertices`)

* :meth:`~neuralqx.graph.Graph.minimal_loops` exposes the minimal loops of the graph

* a set of indexing and mapping utilities can be found through :attr:`~neuralqx.graph.Graph.mapping`, :meth:`~neuralqx.graph.Graph.edge_to_index` and :meth:`~neuralqx.graph.Graph.index_to_edge`

* :attr:`~neuralqx.graph.Graph.signs` returns the signs table for edge pairs/triplets

* And some convenience helpers:

  * :meth:`~neuralqx.graph.Graph.edges_from` which gives the edges emanating from a specified vertex
  * :meth:`~neuralqx.graph.Graph.edges_at` which gives the edges incident at a specified vertex
  * :meth:`~neuralqx.graph.Graph.valence` gives the valence of a specified vertex

You will typically touch such methods and attributes frequently when implementing operators.




Core conventions
-----------------------


Keyed, oriented edges
~~~~~~~~~~~~~~~~~~~~~~~~~~

neuraLQX supports both planar and non-planar graphs. However, choosing between on or the other requires a different
representation of the oriented edges. neuraLQX defines edges as a triple

.. math::

   e \equiv (u, v, k),

where:

* ``u`` is the **tail** vertex (the vertex the edge emanates from),
* ``v`` is the **head** vertex (the vertex the edge is incident at),
* ``k`` is an integer **key** distinguishing *parallel edges* between the same unordered endpoints.

Why keys matter
^^^^^^^^^^^^^^^^^

Internally, graphs are treated as a NetworkX **MultiGraph** (not a simple graph). The key
``k`` is what makes multi-edges well-defined. This allows you to define graphs which have multiple edges between the
same set of vertices, for example, the following planar dipole graph

.. code-block:: python

   import neuralqx as nqx

   # a dipole graph specified as keyed oriented edges
   edges = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 3)]
   graph = nqx.graph.Graph(edges)


If you provide edges *without keys*, neuraLQX inserts default keys (the default is ``k = 0``) and validates that there
are no ambiguous duplicates.

.. important::

   If you truly have parallel edges connecting the same unordered endpoints, you must provide
   distinct keys (e.g. ``(0, 1, 0)``, ``(0, 1, 1)``) as shown in the dipole graph above so the edges are unambiguous.


Planar vs non-planar graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several LQG operators depend on discrete orientation information at vertices, for example the spatial volume operator
requires the sign between edges, depending on the spatial dimensions

.. math::

   \mathrm{sign}(e_I, e_J) \,\,(\text{2-d case}) \quad,\quad \mathrm{sign}(e_I, e_J, e_K) \,\,(\text{3-d case})

Such quantities depend on the planarity of the graph. Moreover, in the case of the spatial volume operator in 3-spatial
dimensions, this quantity further depends on how the graph is embedded. Therefore, neuraLQX supports both planar and
non-planar graphs and distinguishes the two cases based on how you label *vertices* as follows:

* **Planar graphs**: vertices are labeled by **integers**. neuraLQX constructs a 2D drawing
  (used both for visualisation and as a reference embedding) and computes sign factors from it.
* **Non-planar graphs**: vertices are given as coordinate triples ``(x, y, z)``. neuraLQX
  treats this as an intrinsically 3D embedding and computes orientation signs from local tangents.

The planarity choice affects what sign data is computed and what additional maps are exposed
(see :ref:`embeddings-and-signs`).



Edge indexing and configuration layout
--------------------------------------

.. _edge-indexing:

The central design choice (inherited from NetKet) is that a basis configuration is represented as a **flat array** whose
entries correspond to *degrees of freedom* located on edges (or more generally, edges and vertices). When constructing
operators, for instance, you will need to know which edge to act on, and hence you might require knowledge about where
the specific edge's degree of freedom sits in a given array-represented basis element.

To make this robust for multi-edges, neuraLQX constructs a bijection:

.. math::

   (u, v, k) \longleftrightarrow i \in \{0, \ldots, |E(\gamma)|-1\}.

Generally, this entire map is available as :attr:`~neuralqx.graph.Graph.mapping`.

Example: mapping on the θ-graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neuralqx as nqx

   theta_graph = nqx.graph.Graph([(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)])
   print(theta_graph.mapping)
   # {(0, 1, 0): 0, (0, 2, 0): 1, (0, 3, 0): 2, (1, 2, 0): 3, (2, 3, 0): 4}

Primal edge indexing: ``edge_to_index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessing this entire mapping and then searching for a specific edge is sometimes not the best strategy. For fast access,
neuraLQX implements the :meth:`~neuralqx.graph.Graph.edge_to_index` method, which you can query with a keyed oriented
edge and it returns for you its position in any given basis element.

For usability, the lookup is **orientation agnostic**: if ``(u, v, k)`` is not found, neuraLQX also tries ``(v, u, k)``.

.. code-block:: python

   theta_graph.edge_to_index((0, 1, 0))  # 0
   theta_graph.edge_to_index((1, 0, 0))  # 0 (reverse orientation)

This is the O(1) direction used throughout operator application code.

Reverse lookup: ``index_to_edge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Going the other way around (i.e. asking which edge corresponds to a specific degree of freedom located at a specific index in any given
basis element) can be done through the :meth:`~neuralqx.graph.Graph.index_to_edge` method, which returns the user-friendly edge triple
corresponding to an index:

.. code-block:: python

   theta_graph.index_to_edge(0)  # [0, 1, 0]

.. important::

   It is not always that the first edge in the list of oriented edges defining a graph sits in position 0 in a basis
   element. Therefore **users should not assume a layout and should use the indexing provided by neuraLQX.**




Dual graphs and NetKet interoperability
---------------------------------------

.. _dual-graphs:

neuraLQX runs on NetKet, and NetKet requires, generally, that degrees of freedom live on the vertices of the graph. This,
however, is not conceptually compatible with the current needs of neuraLQX (i.e. in the Abelian model of loop quantum gravity,
edges carry charge vectors, and vertices are "trivial"). To accommodate for this, neuraLQX users interact with neuraLQX
graphs, but neuraLQX also constructs both NetworkX and NetKet representations at instantiation time to ensure seamless
integration with NetKet.


Primal graph representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first set of internal graph objects created correspond directly to the graph specified by the user. This we call a
*primal graph*. The two internal representations are found under

* :attr:`~neuralqx.graph.Graph.nx_graph`: which gives a a NetworkX **MultiGraph** encoding **un-oriented** adjacency *including edge keys*
* :attr:`~neuralqx.graph.Graph.nk_graph`: which gives a :class:`netket.graph.Graph` copy used for NetKet interoperability.

.. code-block:: python

   theta_graph.nk_graph
   # Graph(n_nodes=4, n_edges=5)

Dual graph representations
~~~~~~~~~~~~~~~~~~~~~~~~~~

For the *dual* graph neuraLQX provides two similar objects which are found under

* :attr:`~neuralqx.graph.Graph.dual_nx_graph` for the NetworkX graph, and
* :attr:`~neuralqx.graph.Graph.dual_nk_graph` for the NetKet graph.

Specifically, the NetworkX dual construction corresponds to the **line graph** of the primal multi-graph:

* dual vertices correspond to **primal edges**
* dual edges connect pairs of primal edges meeting at a vertex

The existence of such dual graphs in the time being addresses the conceptual mismatch mentioned above, namely

* In current supported loop quantum gravity models, degrees of freedom are naturally associated with **edges**.
* In NetKet, degrees of freedom are associated with **vertices (sites)**.

The job of the dual graph is to bridge these representations and is also useful in models where operator
moves are naturally expressed on the dual lattice.


Dual vertex indexing
^^^^^^^^^^^^^^^^^^^^^^

Note that the same mapping is also interpreted as the identification of a **primal edge** with a
**dual vertex**. Concretely the integer label of a dual vertex is **exactly** the array index assigned to the
corresponding primal edge.

This is convenient whenever a model introduces dual degrees of freedom or dual loops: the
primal-dual relationship is already wired into the indexing layer.

.. seealso::

   The Hilbert layer uses this convention directly. Each primal edge corresponds to one dual
   vertex and hence to one NetKet "site" in the Hilbert space layout.





Minimal loops (plaquettes)
--------------------------

.. _minimal-loops:

Many LQG operators act on minimal cycles (plaquettes) and their orientations. neuraLQX therefore treats **minimal loops**
as a first-class derived structure of the graph. They are computed once during graph construction and therefore are readily
accessible for operators, samplers and models. The method :meth:`~neuralqx.graph.Graph.minimal_loops` returns a list of
loops, where each loop is itself a list of keyed, oriented edges.


How minimal loops are computed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internally, :meth:`GraphHandler.get_smallest_loops` combines two sources:

1. **Two-edge loops from parallel edges:**
   If two distinct keyed edges connect the same unordered pair of endpoints, they form a
   length-2 cycle. These are detected by grouping edges by unordered endpoints and taking
   combinations of distinct keys.

2. **Longer loops from a cycle basis:**
   The handler constructs a simple (keyless) graph by collapsing multi-edges to unordered
   endpoint pairs and uses NetworkX ``minimum_cycle_basis`` to obtain a set of short cycles
   spanning the cycle space. Each node-cycle is converted back into a keyed edge-cycle by
   selecting a representative key for each segment (currently the minimal key among matching edges).

.. important::

   Minimal loops in neuraLQX are the **smallest length** elements in the cycle basis of the cycle space of a given
   graph, and **not** the entire cycle basis (i.e. the returned neuraLQX minimal loops need not correspond to a proper cycle basis).


Loop normalisation and determinism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each loop is normalised by a canonical rotation so it has a stable representation independent of cyclic shifts. This
stability matters whenever loop lists are used as deterministic iterators (e.g. in sampling moves).

Additionally, when multiple loops exist, neuraLQX discards loops longer than the shortest loop
length found. This reflects the intended use: minimal loops represent the smallest plaquettes
available, not an arbitrary cycle basis.


Example: minimal loops on the θ-graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   theta_graph.n_minimal_loops
   # 2

   theta_graph.minimal_loops()
   # [[(0, 1, 0), (1, 2, 0), (2, 0, 0)],
   #  [(0, 3, 0), (3, 2, 0), (2, 0, 0)]]


Dressed minimal loops (orientation-aware plaquettes)
----------------------------------------------------

For operator definitions, a loop often must be augmented with orientation information. For example in Abelian models,
holonomy operators part of a minimal loop holonomy operator act as raising/lowering operators on edge degrees of
freedom, and the direction (raise vs lower) depends on the orientation of the loop edge relative to the underlying graph
edge.

The :meth:`~neuralqx.graph.Graph.dressed_minimal_loops` method returns the same minimal loops but *dressed*
edge-by-edge with metadata indicating whether the loop orientation matches the graph orientation.

Motivating example
~~~~~~~~~~~~~~~~~~

The minimal loops may contain edges oriented opposite to the corresponding graph edge.
For example, edge ``(0, 2)`` may appear as ``(2, 0)`` inside a loop.

.. code-block:: python

   theta_graph.edges
   # [(0, 1, 0), (0, 2, 0), (0, 3, 0), (3, 2, 0), (1, 2, 0)]

   theta_graph.minimal_loops()
   # [[(0, 1, 0), (1, 2, 0), (2, 0, 0)],
   #  [(0, 3, 0), (3, 2, 0), (2, 0, 0)]]

The dressed minimal loops expose the correct raising/lowering action per edge:

.. code-block:: python

   theta_graph.dressed_minimal_loops()
   # [[((0, 1, 0), {'type': 'creation', 'key': 0}),
   #   ((1, 2, 0), {'type': 'creation', 'key': 0}),
   #   ((2, 0, 0), {'type': 'annihilation', 'key': 0})],
   #  [((0, 3, 0), {'type': 'creation', 'key': 0}),
   #   ((3, 2, 0), {'type': 'creation', 'key': 0}),
   #   ((2, 0, 0), {'type': 'annihilation', 'key': 0})]]

In the example, in the first minimal loop ``[(0, 1, 0), (1, 2, 0), (2, 0, 0)]``, the edge ``(2, 0, 0)`` comes with an
``annihilation`` descriptor as it appears in opposite order relative to the edges which constitute the actual graph.

.. note::

   neuraLQX does not build extra edges when finding or constructing minimal loops. Minimal loops in neuraLQX are merely
   a bookkeeping process. This keeps the graph object static and becomes helpful for large graphs.

.. _embeddings-and-signs:

Planar/non-planar embeddings and sign tables
--------------------------------------------

Several canonical LQG operators, most notably the volume operator and Hamiltonian constraint constructions, depend on
discrete orientation data at vertices. In the continuum this is expressed via an :math:`\epsilon`-tensor. On a graph, it
becomes a discrete sign table that depends on how incident edges are embedded around each vertex.

neuraLQX computes and exposes this data through :attr:`~neuralqx.graph.Graph.signs`, and this is done differently based
on the planarity of the graph. Note that this is done automcatically at construction time, making the signs readily available after the
graph object has been created.

Planar case (integer-labeled vertices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If vertices are integers, neuraLQX treats the graph as planar and:

* constructs a 2D drawing (used for visualisation and as a reference embedding),
* computes sign factors based on **ordered pairs** of edges meeting at each vertex.

The resulting table is exposed via :attr:`~neuralqx.graph.Graph.signs`.

Example (θ-graph)
^^^^^^^^^^^^^^^^^

.. code-block:: python

   theta_graph.signs
   # {
   #  '0': { '((0, 1, 0), (0, 2, 0))': '1', ... },
   #  '1': { '((1, 0, 0), (1, 2, 0))': '-1', ... },
   #  ...
   # }

.. note::

   The sign table is returned as a nested dictionary keyed by vertices, with inner dictionaries
   keyed by ordered edge pairs (planar) and values being the corresponding sign.

Non-planar case (coordinate-labeled vertices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If vertices are provided as coordinate triples ``(x, y, z)``, neuraLQX treats the graph as non-planar. This enables an
intrinsically 3D orientation test.

Internally, neuraLQX converts coordinate-labeled vertices to sequential integer labels for
computation, but retains a bijective map between the two using the mapping and helper methods

* :attr:`~neuralqx.graph.Graph.nonplanar_vertex_mapping`
* :attr:`~neuralqx.graph.Graph.nonplanar_vertex_to_index`
* :attr:`~neuralqx.graph.Graph.nonplanar_edges` (recover original coordinate edges)


Sign computation in 3D
^^^^^^^^^^^^^^^^^^^^^^

For each incident edge at a vertex, neuraLQX forms the unit tangent vector leaving the vertex. For each ordered triple
of incident edges :math:`(e_I, e_J, e_K)` it assigns

.. math::

   \epsilon(e_I, e_J, e_K) = \mathrm{sgn}(\det[t_I\ t_J\ t_K]),

with a tolerance that yields :math:`\epsilon = 0` for (near-) coplanar tangents. This produces the
fully antisymmetric sign table needed by 3D geometric operators.

Parallel edges and degeneracy handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A subtle practical issue arises with parallel edges: if two edges leave a vertex in exactly the same
geometric direction, tangent vectors coincide and the determinant test becomes degenerate.

neuraLQX resolves this locally and deterministically:

* within each multi-edge family (same unordered endpoints), tangents are separated by rotating them by a small, symmetric set of angles around a (seeded) perpendicular axis.

This preserves the physical intent (a generic embedding in an :math:`\varepsilon`-ball around the vertex)
while making the sign table stable and well-defined.

Random embeddings (non-planar graphs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For non-planar graphs, the :class:`~neuralqx.graph.Graph` constructor optionally supports
``random_embedding=True``. This replaces provided vertex coordinates by draws from a Gaussian
distribution (mean, standard deviation, and seed are user-controlled with defaults).

This is primarily intended for:

* stress-testing sign computations,
* generating generic embeddings when only combinatorial structure matters.

.. code-block:: python

    import neuralqx as nqx

    # example K5 graph
    k5_graph = nqx.graph.K5Graph(non_planar=True)

    # same graph but with random embedding
    k5_random_graph = nqx.graph.K5Graph(non_planar=True, random_embedding=True)

    # random embedding with specified seed
    k5_seeded_graph = nqx.graph.K5Graph(non_planar=True, random_embedding=True, random_seed=42)

Graph visualisation and export
------------------------------

Graph objects can be plotted at construction time with ``plot=True`` and exported afterwards using
:meth:`~neuralqx.graph.Graph.export`.

For non-planar graphs, a 3D renderer is shown to help facilitate the visualisation of the embedding choice.

.. code-block:: python

   graph = nqx.graph.Graph(edges, plot=True)
   graph.export("my_graph_visualisation")  # choose your filename/path


Graph generators and subclasses
=================================

Beyond the base :class:`~neuralqx.graph.Graph`, neuraLQX provides a collection of commonly used graph generators as
subclasses (e.g. :class:`~neuralqx.graph.HexagonalLatticeGraph`, :class:`~neuralqx.graph.K5Graph`, :class:`~neuralqx.graph.LadderGraph` and more).

These constructors typically generate a standard lattice via NetworkX and then relabel vertices and edges into
neuraLQX's oriented keyed-edge convention.

Examples
~~~~~~~~

.. code-block:: python

   import neuralqx as nqx

   # Pre-implemented K5 graph
   k5_graph = nqx.graph.K5Graph(non_planar=True)

   # Non-planar triangular lattice with periodic boundary conditions
   triangular_graph = nqx.graph.TriangularLatticeGraph(3, 5, periodic=True, non_planar=True)

All such subclasses can also be constructed as non-planar graphs (when permissible). This ensures that
graphs built from high-level parameters (system size, periodicity, planarity choice) still enjoy the same
internal invariants as user-specified graphs.


Guaranteed contracts
-------------------------------------------------

The rest of neuraLQX (Hilbert spaces, operators, physical models, samplers) assumes the following
contracts are satisfied by any :class:`~neuralqx.graph.core.AbstractGraph` implementation:

* **Stable edge indexing** via :attr:`~neuralqx.graph.Graph.mapping` and
  :meth:`~neuralqx.graph.Graph.edge_to_index`.
* A consistent interpretation of the mapping as **primal-edge to dual-vertex** identification.
* **Precomputed derived structures** available at construction time:
  minimal loops (and dressed minimal loops), primal/dual NetKet graphs, and sign tables.
* Deterministic minimal loop ordering (canonical rotation) so loop iteration is reproducible.

If you implement custom graph subclasses, ensure you preserve these invariants to remain compatible
with the rest of the public API.


Common patterns and gotchas
==============================

I supplied an edge list without keys, and got an error
----------------------------------------------------------

If you accidentally provide ambiguous duplicate edges (same endpoints, no keys), neuraLQX cannot infer
a unique multi-edge structure. Supply explicit keys:

.. code-block:: python

   edges = [(0, 1, 0), (0, 1, 1), (1, 2, 0)]
   graph = nqx.graph.Graph(edges)


My operator needs orientation on a plaquette
----------------------------------------------------------

Use :meth:`~neuralqx.graph.Graph.dressed_minimal_loops` instead of :meth:`~neuralqx.graph.Graph.minimal_loops`. The
dressed loops carry per-edge metadata indicating whether the loop orientation matches the underlying graph edge.

I need a 3D sign table (volume operator / 3D Hamiltonian kernels)
---------------------------------------------------------------------------

Provide vertex coordinates ``(x, y, z)`` (non-planar graph) so neuraLQX computes triple-orientation
signs from local tangents. Use the non-planar mapping helpers to translate between coordinate and
internal index labels if needed.

.. note::

   If your embedding is (nearly) coplanar locally, the determinant test may yield zeros due to the
   tolerance used to detect coplanarity.
