.. _lqx:

LQX models
=============================

The ``neuralqx.lqx`` module groups **physical models** that turn a chosen kinematics
(graph + Hilbert space + gauge group) into **operators you can actually optimise**.

If you are using ``neuralqx`` as a variational engine, an LQX object is the piece that answers:

* *Which operators represent the dynamics/constraints of my model?*
* *How do I build them on my concrete graph and Hilbert space?*
* *What is the "default objective" I should minimise (or drive to zero)?*
* *How do I toggle between explicit sparse/local operator materialisation and matrix-free
  computational backends?*

In practical workflows the most important attribute is **the effective constraint**
exposed through the interface's ``.constraint`` attribute. You typically feed that operator (or a controlled
composition of it) into a variational routine to define the loss.

This page documents the shared interface and the "model pattern" used across the
built-in models.


What an LQX object does
--------------------------

An LQX model sits between two worlds:

1. **Kinematics** (graph, Hilbert space, gauge group), which defines *what states exist*.
2. **Dynamics/constraints** (operators), which define *what states are physical*.

The LQX object is responsible for producing operators that are consistent with the chosen
kinematics:

* Edge/loop operators use the graph's indexing and orientation conventions.
* Vertex operators use the graph's incidence relations and (when available) embedding signs.
* Gauge terms, if requested, are produced by the gauge group object in a way that matches the
  Hilbert space (same number of gauge copies, same graph, same backend choices).

The output is designed to plug into NetKet-style expectation value pipelines, so your
variational state only needs to know: "here is an operator, estimate its expectation and gradients".


The two-layer design: interface vs core model
------------------------------------------------

All built-in models follow the same abstraction structure in neuraLQX where an interface wraps a concrete core. Namely,
LQX objects are composed of the two layers below:

**Interface layer**
  A user-facing class that validates inputs, stores configuration, and exposes a stable API
  (methods like ``holonomy(...)``, ``volume(...)``, ``curvature_constraint(...)``, etc.).
  Interfaces are the objects you instantiate.

**Core model**
  A concrete implementation that actually constructs operators (often several backends),
  and implements the physics-specific details.

Any concrete realisation of an interface must subclass the ``neuralqx.lqx.AbstractLqxInterface`` class, while concrete
cores subclass the ``neuralqx.lqx.AbstractLqxModel`` class. Interfaces require, at instantiation, a concrete Hilbert space
realisation as well as a concrete gauge group object. This split is deliberate as we try to keep separation of
concerns explicit.

* The interface stays stable even if internal operator implementations evolve.
* It becomes easy to swap the backend strategy (local vs computational) without changing user code.
* The same interface can wrap multiple concrete submodels (dispatch based on dimensions/gauge copies).

Concrete models must at minimum provide a *model-intrinsic* constraint (see below), but the interface
is the object you should pass around in user code.


Effective constraints and gauge terms
----------------------------------------

The central concept is the distinction between

* **model constraint**: the intrinsic constraint produced by the physical model, and
* **effective constraint**: the object you actually optimise against, after optionally adding gauge terms.

The interface constructs the effective constraint through an ``AbstractLqxInterface.initialize_constraint()`` following:

.. math::

   \hat{C} =
   \begin{cases}
   \hat{Q}_{\text{model}}, & \text{if the Hilbert space is gauge invariant}, \\
   \hat{Q}_{\text{model}} + \hat{G}, & \text{otherwise},
   \end{cases}

where :math:`\hat{G}` is the gauge constraint provided by the gauge group.

This choice matters in practice.

* If you work in a **gauge-invariant Hilbert space**, you typically want the model term only.
* If you work in an **unconstrained Hilbert space** (easier sampling, cheaper state preparation),
  adding :math:`\hat{G}` is a straightforward way to steer optimisation into the gauge-invariant sector.

A key design feature is that ``AbstractLqxInterface.constraint`` is a **settable property**. You can replace it at runtime to

* switch between different available constraints,
* add penalty terms (boundaries, symmetry breaking controls, soft projectors),
* build composite objectives (e.g. weighted sums).

This "replaceable constraint" design is used consistently across built-in models.


Operator backends and performance knobs
------------------------------------------

Most LQX operator constructors accept the backend flags

``computational``
  Request a matrix-free ``ComputationalOperator`` backend.

``jax``
  When ``computational=True``, request a JAX-compatible computational variant (where implemented).

The rule of thumb is:

* **Local/materialised operators** are convenient when you want explicit structure (inspection,
  debugging, algebraic composition), or when the Hilbert space is small enough.
* **Computational operators** are preferred when the Hilbert space is large and you want to avoid
  allocating matrices; they evaluate local estimators and connected configurations on the fly.

Interfaces store the default backend choice configured at construction time, and methods allow you
to override it per call. This keeps "global configuration" simple while still allowing fine control
in performance-critical paths.


Working with LQX constraints in variational code
---------------------------------------------------

Working with LQX objects is straightforward. A minimal pattern is

.. code-block:: python

   import neuralqx as nqx

   # build kinematics
   graph = ...                         # see the Graph API docs
   H = ...                             # see Hilbert space docs
   G = ...                             # see Gauge group docs

   # instantiate a physical model interface
   lqx = nqx.lqx.wcl.LqxWCL(H, G, spacetime_dimensions=4, computational=True)

   # build the effective constraint (adds gauge term if needed)
   lqx.initialize_constraint()

   # use lqx.constraint as your optimisation objective
   constraint_op = lqx.constraint

   # plug into a VMC pipeline (see the Solver API documentation)

The point is that your optimisation code stays almost completely model-agnostic. The model encapsulates
the operator construction details and the correct "default objective" for that model.


Built-in model families
-------------------------

``neuralqx.lqx`` currently exposes three main model families:

**Euclidean weak-coupling limit (WCL)**
  A family of Euclidean models built on Abelianised holonomy-flux algebras. One interface dispatches
  to submodels depending on spacetime dimension and the number of gauge copies.

**Single-vertex QRLG**
  A symmetry-reduced truncation in a cuboidal setting with three independent directions.

**Spherical Bojowald–Swiderski**
  A lattice discretisation along the radial direction with local plaquette contributions.

Each has its own dedicated page below.

.. toctree::
   :maxdepth: 1
   :caption: Model families

   euclidean/index
   qrlg/index
   spherical/index
