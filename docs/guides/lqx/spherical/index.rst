.. _lqx-spherical:

Spherical Bojowald–Swiderski model
============================================

Spherically symmetric loop quantisations provide a concrete playground for reduced gravitational
dynamics. The Bojowald–Swiderski construction introduces a lattice-like discretisation along a radial
direction and builds the Hamiltonian from holonomy loops and commutators with volume.

In neuraLQX this model is exposed through the interface ``LqxBojowaldSwiderski`` in the module
``neuralqx.lqx`` (or the dedicated ``neuralqx.lqx.spherical`` module). The implementation is intentionally restrictive,
it enforces the specific geometry and dimensional assumptions required by the discretisation so that misuse fails loudly.


Construction requirements
-------------------------

The spherical interface requires:

* a ``HalfLadderGraph`` geometry (the discretisation used by this implementation),
* a **single Abelian gauge copy** (``gauge_dimensions = 1``),
* a 3-dimensional model descriptor (``spacetime_dimensions = 3``).

A typical construction pattern looks like:

.. code-block:: python

   import neuralqx as nqx
   from neuralqx.lqx.spherical import LqxBojowaldSwiderski

   graph = nqx.graphs.HalfLadderGraph(...)
   H = ...   # U(1) Hilbert space defined on the half-ladder graph
   G = ...   # matching U(1) gauge group

   lqx = LqxBojowaldSwiderski(H, G, spacetime_dimensions=3)
   lqx.initialize_constraint()

   Q = lqx.constraint

The returned ``Q`` is the effective objective.


Continuous parameters
---------------------------

Two continuous parameters are exposed and can be modified at runtime:

Immirzi
  The Immirzi parameter (appears in the standard quantisation of geometric and dynamical terms).

delta
  The polymerisation scale controlling the holonomy "step size" in the reduced variables.

Because these are runtime-tunable, you can sweep them without rebuilding the full model object
(when your surrounding workflow supports it).


Basic reduced operators
-------------------------

The interface provides direct access to the core reduced operators used in the quantisation.

``volume(vertex, shift=0.0)``
  Returns the spherical volume operator at a chosen vertex. The optional ``shift`` parameter is a
  pragmatic regularisation knob as it allows you to stabilise expressions by shifting all eigenvalues.

``flux(vertex)``
  Returns the reduced flux (triad) operator at a vertex. This is the diagonal ingredient that
  underlies the effective geometry in the reduced setting.

``holonomy(edge, adjoint=False)``
  Returns the holonomy operator acting on a chosen edge of the half-ladder graph. The ``adjoint``
  flag returns the inverse/adjoint action, useful when you want to flip orientation conventions.

``number(edge)``
  Returns a number-like operator that reads out the quantum number on an edge. In this model this
  operator is both diagnostically useful (monitoring charge distributions) and a building block for
  composing more involved expressions.


Hamiltonian constraint and default objective
----------------------------------------------

The Bojowald–Swiderski Hamiltonian is constructed from **local vertex contributions** built from
holonomies and commutators with the volume. The half-ladder discretisation naturally decomposes the
vertex contribution into three plaquette pieces (left / central / right), each built from the
appropriate local loops.

In the current implementation the core local object is exposed through the operator class

``SphericalVertexConstraintBojowaldSwiderskiOperator``,

and the default optimisation objective is a **sum of squared local contributions over inner vertices**:

.. math::

   \hat{Q} = \sum_{v \in V_{\mathrm{inner}}} \hat{H}(v)^2,
   \qquad
   \hat{H}(v) = \hat{H}_{L}(v) + \hat{H}_{C}(v) + \hat{H}_{R}(v).

The L/C/R decomposition corresponds directly to the left/central/right plaquettes on the half-ladder.

Lazy squaring and practical implications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internally, the default squaring is typically implemented using NetKet's lazy ``Squared`` wrapper.
This reduces the cost of explicitly forming operator products at the price of changing how local
estimators are computed (square of estimator vs estimator of product). If your workflow is sensitive
to estimator properties, you can rebuild the objective explicitly by replacing ``lqx.constraint`` with
a product-based construction (when available in the operator backend you choose).

Boundary penalties hook
~~~~~~~~~~~~~~~~~~~~~~~~~

The spherical interface includes a convenient mechanism for attaching **boundary penalties**.
This is useful because spherical discretisations often require boundary conditions or controlled
behaviour near the ends of the radial lattice. In optimisation workflows, boundary penalties are a
clean way to bias the state towards the intended sector without hard-coding constraints into the
sampler.

The exact penalty you choose depends on what you want to enforce (fixed boundary charges, decay,
mirror symmetry, etc.), but the key point is that the interface is designed for such runtime
compositions. In all cases, all penalties can be built into the ``.hamiltonian_boundaries()`` method of the core class.
