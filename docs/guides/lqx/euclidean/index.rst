.. _lqx-wcl:

Euclidean weak-coupling models
==========================================

The Euclidean weak-coupling limit (WCL) family in ``neuralqx`` provides a practical route from
graph-based kinematics to *optimisable constraints* in Abelianised LQG-style models.

The defining feature of this family is that all essential operators can be expressed in terms of:

* **edge holonomies** (shift/ladder operators on edge quantum numbers),
* **number-like operators** (diagonal operators reading out those quantum numbers),
* **loop holonomies** (products of edge holonomies along minimal loops),
* and, in higher-dimensional variants, **geometric operators** (area/volume) and
  **Thiemann-type regularised dynamics**.

The public entry point is the interface class ``neuralqx.lqx.LqxWCL``. It dispatches internally to the appropriate
submodel based on the spacetime dimension and the number of gauge copies carried by the Hilbert space.


Import and construction
-------------------------

.. code-block:: python

   import neuralqx as nqx
   from neuralqx.lqx.wcl import LqxWCL

   # H: a Hilbert space defined on a graph, with 3 gauge dimensions
   # G: a matching gauge group object
   # construct a 4-dimensional Euclidean model
   lqx = LqxWCL(H, G, spacetime_dimensions=4, computational=True)

After instantiation you typically call:

.. code-block:: python

   lqx.initialize_constraint()
   Q = lqx.constraint

``Q`` is the effective objective (model constraint plus gauge term when required).

Dispatch rules (which concrete submodel you get)
---------------------------------------------------

The interface chooses the concrete core implementation, all subclassing the ``AbstractLqxModel`` according to:

* ``spacetime_dimensions = 3`` and ``gauge_dimensions = 1``  → ``LqxWCL1D``
* ``spacetime_dimensions = 3`` and ``gauge_dimensions = 3``  → ``LqxWCL3D``
* ``spacetime_dimensions = 4`` and ``gauge_dimensions = 3``  → ``LqxWCL4D``

Internally, the ``LqxWCL1D`` implements a small U(1) BF model, the ``LqxWCL3D`` implements the 3d Euclidean Abelianised model,
and the ``LqxWCL4D`` 4d Abelianised model on non-planar graphs. This means user code can stay stable while you switch
between models by changing only the dimension/gauge configuration.


Common operator constructors
-------------------------------

All WCL submodels share a core set of default operator dispatchers. The interface forwards these to the
encapsulated concrete model.

Edge holonomy: ``holonomy(edge, adjoint=False, computational=None, jax=None)``
  Returns the holonomy operator for a single edge. In the computational basis this acts as a
  discrete shift on the quantum number(s) associated with that edge. The ``adjoint`` flag returns
  the inverse/adjoint action, which is the natural choice if you want the opposite orientation
  convention on that edge.

Number-like operator: ``number(edge, ...)``
  Returns a diagonal operator that reads out the quantum number associated with an edge. In Abelian
  models this is the basic flux-like ingredient, and higher operators (volume, prefactors, penalties)
  are typically built from combinations of these number operators.

Minimal loop holonomy: ``minimal_loop_holonomy(loop, ...)``
  Builds a holonomy operator for a loop as an ordered product of edge holonomies. The canonical input
  is a **minimal loop** as returned by the Graph API. The loop ordering/orientation matters: the
  Graph layer is responsible for returning loops in a consistent oriented form, and the holonomy
  builder uses that ordering directly.

Constraints: ``curvature_constraint(...)`` and (when supported) ``thiemann_quadratic_constraint(...)``
  These return model-specific constraints. The "default constraint" used by ``initialize_constraint()``
  depends on the submodel. You can override it by assigning to ``lqx.constraint``.

Backend flags and how to think about them
--------------------------------------------

Almost all constructors accept backend flags:

* If ``computational=True``, you request a matrix-free implementation.
* If ``jax=True`` *and* ``computational=True``, you request a JAX-compatible computational operator.

A single LQX object can mix strategies:

.. code-block:: python

   # default: computational operators (set at construction)
   h_e = lqx.holonomy((0, 1, 0))

   # override locally: materialise as a local operator
   n_e_local = lqx.number((0, 1, 0), computational=False)

This is useful when you want to debug or inspect some pieces while keeping the heavy operators
matrix-free.


Submodel 1: U(1) BF / flatness model (``LqxWCL1D``)
------------------------------------------------------

This is the simplest WCL submodel. It provides a nice testbed for numerics as for small enough graphs, it is possible to
use exact numerical methods to solve it. It provides one U(1) copy per edge, and dynamics expressed through a
flatness/curvature constraint built from minimal loop holonomies.

What you get
~~~~~~~~~~~~~~~~

You have the core edge operators (holonomy/number) and loop holonomies, and you can build the
flatness constraint as a sum over minimal loops.

The default curvature (flatness) constraint is:

.. math::

   \hat{F} = \sum_{\alpha \in L(\gamma)}
   (\hat{h}_{\alpha} - I)(\hat{h}_{\alpha}^{\dagger} - I),

where :math:`L(\gamma)` is the set of minimal loops of the graph.

This operator is positive by construction. Each loop contributes a non-negative term measuring
the deviation of the loop holonomy from the identity.

Typical usage
~~~~~~~~~~~~~~~

.. code-block:: python

   from neuralqx.lqx.wcl import LqxWCL

   # here H has gauge_dimensions=1
   lqx = LqxWCL(H, G, spacetime_dimensions=3, computational=True)
   lqx.initialize_constraint()

   # edge / loop operators
   e = (0, 1, 0)
   h_e = lqx.holonomy(e)
   n_e = lqx.number(e)

   loop = graph.minimal_loops()[0]
   h_loop = lqx.minimal_loop_holonomy(loop)

   # default objective (plus gauge term if depending on H)
   Q = lqx.constraint

Model scope and intentional omissions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this gauge-dimension setting the model does **not** provide geometric operators such as
area/volume, and it does not expose Thiemann-type Hamiltonian constraints. If you request those,
the corresponding methods raise ``NotImplementedError``. This is a deliberate guardrail as it keeps
the interface honest about what is defined in this reduced setting.


Submodel 2: 3d Euclidean Abelianised model (``LqxWCL3D``)
------------------------------------------------------------

This variant models a 3d Euclidean setting using **three independent U(1) copies per edge**.

Internal structure: lifted graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model internally "lifts" the base graph into three layers and extends its edge-to-index mapping
accordingly. As a user you still interact with the usual base graph edge keys. The model handles
the bookkeeping needed to build operators acting on the full U(1)^3 degree of freedom.

Holonomies and loop holonomies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an edge :math:`e` the holonomy operator is implemented as a sum of copy-wise holonomies:

.. math::

   \hat{h}_{e} = \sum_{I=1}^{N} \hat{h}^{(I)}_{e,I}, \qquad (N=3)

and the same pattern is used for minimal loop holonomies:

.. math::

   \hat{h}_{\alpha} = \sum_{I=1}^{N} \hat{h}^{(I)}_{\alpha,I}.

These summed operators are the default building blocks for constraints.

Curvature constraint (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default constraint is again a curvature/flatness-type operator, now expressed in terms of the
U(1)^3 loop holonomy shown above. Conceptually, the model still measures whether minimal loops
close “flatly”, but now across the three Abelian copies.

Thiemann-type Hamiltonian (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 3d WCL model exposes a Thiemann-regularised Hamiltonian construction through
``thiemann_quadratic_constraint(order=..., a=...)``. In the current implementation this is built
in the local-operator backend using explicit operator algebra. The method returns a quadratic constraint
rather than a literal quadratic form:

.. math::

   \hat{C}_{\mathrm{TRC}} = \sum_{v \in V(\gamma)} \hat{H}_v \hat{H}^{\dagger}_v.

The underlying regulated Hamiltonian is expressed (following the Thiemann prescription) as:

.. math::

   \hat{H}^{T(\gamma)}(N) f_{\gamma}
   := \lim_{\epsilon \to 0} \hat{H}^{T,\epsilon}(N) f_{\gamma}
   = \frac{2}{\hbar^2} \sum_{\Delta,\Delta' \in T, v}
   \epsilon_{ij}\,\epsilon_{kl}\,N(v)\,
   \mathrm{tr}\!\left(
      \hat{h}_{\alpha_{ij}}(\Delta')
      \hat{h}_{s_k}(\Delta)
      [\hat{h}^{-1}_{s_k}(\Delta), \sqrt{\hat{V}_v}]
      \hat{h}_{s_l}(\Delta)
      [\hat{h}^{-1}_{s_l}(\Delta), \sqrt{\hat{V}_v}]
   \right) f_{\gamma}.

To use it as optimisation target, you explicitly assign:

.. code-block:: python

   lqx.constraint = lqx.thiemann_quadratic_constraint(vertex, jax=False, adjoint=False, ...)

where:

* ``jax=True`` requests a JAX computational implementation when available,
* ``adjoint`` controls the vertex operator ordering used when forming quadratic combinations.

Note that here, setting ``adjoint=True`` returns the alternatively ordered :math:`\hat{H}_v^\dagger\hat{H}_v` while setting
it to ``False`` (the default) returns the standard ordering :math:`\hat{H}_v\hat{H}_v^\dagger`.

Volume operator
~~~~~~~~~~~~~~~~~~~~~~

The 3d model provides a spatial 2-volume operator at a vertex built from number operators.
Operationally the model constructs vector-component operators from oriented edge pairs and combines them to obtain

.. math::

   \hat{V}_v \sim \sum_{a=0}^{2} \sqrt{ \left(\hat{Q}^{(a)}_v\right)^2 },
   \qquad
   \hat{Q}^{(a)}_v =
   \sum_{(e_i,e_j)\ni v} \mathrm{sgn}_v(e_i,e_j)\,
   \bigl(\hat{m}_{e_i} \times \hat{m}_{e_j}\bigr)_a.

The required orientation signs are supplied by the Graph API (via oriented embedding data).

Because Thiemann-type expressions contain square roots of operators, the model also exposes a
simple Taylor regularisation helper:

.. math::

   f(\hat{X}) \approx \sum_{n=0}^{\text{order}}
   \frac{f^{(n)}(a)}{n!}(\hat{X}-a)^n,
   \qquad f(x)\in\{\sqrt{x},\sqrt[4]{x}\}.

This regularised volume is used internally when constructing the Thiemann operator.

.. important::

   Even though holonomy/number/loop operators exist in both local and computational backends, the
   3d Thiemann constraint is currently supported only as a computational operator.


Submodel 3: 4d Euclidean Abelianised model (``LqxWCL4D``)
-----------------------------------------------------------

The 4d WCL model targets non-planar graphs and extends the 3d operator set with

* a dedicated **area operator**,
* a 3-d spatial **volume** expression based on triple products,
* a computationally supported **Thiemann regularised vertex constraint**, and
* quadratic constraints built from vertex-local pieces.

Area operator
~~~~~~~~~~~~~~~

A surface :math:`S` is represented as a list of puncturing edges. The area operator acts diagonally:

.. math::

   \hat{A}(S)\,|\vec{m}\rangle
   =
   \sum_{e\cap S\neq\emptyset}
   \sqrt{(m_e^{(1)})^2+(m_e^{(2)})^2+(m_e^{(3)})^2}\,
   |\vec{m}\rangle.

From the API perspective this means:

* you pass a list of graph edge keys describing which edges puncture the surface,
* the computational backend returns a single matrix-free operator,
* the local backend may return structured contributions (per puncture) and applies non-linear functions
  through functional operator wrappers (see the experimental API).

Typical usage:

.. code-block:: python

   lqx = LqxWCL(H, G, spacetime_dimensions=4, computational=True)

   surface = [(0, 1, 0), (0, 2, 0)]

   # request JAX computational operator
   A = lqx.area(surface, squared=False, jax=True)

Area-related observables
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to plain area, the model provides "area comparison" style observables (for example,
differences between two edge sets representing two surfaces). These are useful when your optimisation
target encodes relative geometric information rather than absolute scale.

Volume operator
~~~~~~~~~~~~~~~~~~~~

The volume operator at a vertex is based on triple products over incident edges and is expressed as

.. math::

   \hat{V}_v \sim
   \sqrt[3]{\left|
      \sum_{(e_1,e_2,e_3)\ni v}
      \epsilon(e_1,e_2,e_3)\,
      \hat{m}_{e_1}\cdot(\hat{m}_{e_2}\times\hat{m}_{e_3})
   \right| }.

As in 3d, the required Levi-Civita / embedding signs are supplied by the graph embedding data, now computed from the
specific embedding choice for the non-planar case.

Thiemann regularised vertex operator and constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamical building block is the **vertex-local** Thiemann regularised operator :math:`\hat{H}_v`,
and the lapse-smeared sum:

.. math::

   \hat{H}^T(N) = \sum_{v\in V(\gamma)} N(v)\,\hat{H}_v
   =
   \sum_{v\in V(\gamma)} N(v)\sum_{v(\Delta\in T)=v}
   \epsilon_{ijk}\,
   \mathrm{Tr}\!\left(
      \hat{h}_{\alpha_{ij}}
      \hat{h}_{s_j}
      \left[\hat{h}^{-1}_{s_j},\hat{V}_v\right]
   \right).

In the implementation these objects are constructed algorithmically from precomputed admissible
triplets, minimal loops, and Levi-Civita signs collected at class instantiation.

You can access the vertex operator through:

``thiemann_regularized_constraint(vertex, fast=False, jax=False, adjoint=False, ...)``

where:

* ``jax=True`` requests a JAX computational implementation when available,
* ``fast=True`` selects an alternative Numba path optimised for speed,
* ``adjoint`` controls the vertex operator ordering used when forming quadratic combinations.

Quadratic constraints and the ``lazy`` option
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quadratic constraint constructors combine :math:`\hat{H}_v` into an objective.
By default, the method returns a *sum of squared vertex terms using NetKet's lazy squaring*:

.. math::

   \hat{Q} = \sum_{v\in V_{\text{admissible}}} \texttt{Squared}[\hat{H}_v].

This choice avoids explicitly forming products like :math:`\hat{H}_v^\dagger \hat{H}_v`, which can be
expensive. It also changes the estimator properties. Lazy squaring computes "square of the local
estimator" rather than the local estimator of the product. If you want explicit products, set
``lazy=False`` so the constraint is built using explicit product wrappers.

Practical usage pattern:

.. code-block:: python

   lqx = LqxWCL(H, G, spacetime_dimensions=4, computational=True)
   lqx.initialize_constraint()

   # replace the default objective by an explicit quadratic form
   Q_explicit = lqx.thiemann_quadratic_constraint(lazy=False, adjoint=False, jax=True)
   lqx.constraint = Q_explicit

This is the model where computational backends are typically most valuable. The graph sizes and
operator complexity grow quickly, and matrix-free evaluation becomes the default strategy for
serious runs.

Note that here, setting ``adjoint=True`` returns the alternatively ordered :math:`\hat{H}_v^\dagger\hat{H}_v` while setting
it to ``False`` (the default) returns the standard ordering :math:`\hat{H}_v\hat{H}_v^\dagger`.