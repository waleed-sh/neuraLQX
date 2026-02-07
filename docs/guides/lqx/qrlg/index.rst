.. _lqx-qrlg:

Single-vertex QRLG
=============================================

Quantum Reduced Loop Gravity (QRLG) is a symmetry-reduced setting tailored to diagonal/cuboidal
geometries. In neuraLQX the most compact realisation is the **single-vertex truncation** where the model is based on
a graph with one vertex and three independent edges aligned with fiducial directions.

In this truncation a computational basis state is labelled by three representation labels, which we
write schematically as:

.. math::

   |j_x, j_y, j_z\rangle.

The purpose of the QRLG interface is to give you a small, explicit operator set that matches the
standard single-vertex construction while staying compatible with variational optimisation to demonstrate the applicability
of neuraLQX to symmetry-reduced models in loop quantum gravity.


Construction and model pattern
---------------------------------

The public interface is ``LqxSingleVertexQR`` in ``neuralqx.lqx`` (or the dedicated ``neuralqx.lqx.qr`` module). It follows the same model
pattern as other LQX interfaces. It validates inputs, delegate operator construction to a wrapped core
model, and expose a ready-to-use ``.constraint`` after calling ``initialize_constraint()``.

.. code-block:: python

   import neuralqx as nqx
   from neuralqx.lqx.qr import LqxSingleVertexQR

   # H should describe the single-vertex Hilbert space (three directions)
   # G is the matching gauge object
   lqx = LqxSingleVertexQR(H, G, computational=True)
   lqx.initialize_constraint()

   Q = lqx.constraint


Edge indexing convention
--------------------------

In the single-vertex implementation the three edges are addressed by integers:

* ``edge = 0`` → :math:`x` direction
* ``edge = 1`` → :math:`y` direction
* ``edge = 2`` → :math:`z` direction

This convention is used consistently across all QRLG operators.


Operator set
--------------

The interface provides the standard single-vertex building blocks.

Discrete raising/lowering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``creation(edge, n)`` and ``annihilation(edge, n)``
  Implement discrete raising/lowering actions on one of the three edge quantum numbers.
  The parameter ``n`` selects the step size (useful if you want to build effective holonomies
  at different polymerisation scales).

Symmetric holonomy combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``s(edge)`` and ``c(edge)``
  Return symmetric holonomy combinations built directly from the creation/annihilation pair.
  They are the standard sine/cosine-like building blocks used in the single-vertex Hamiltonian.

Flux-like operators and inverses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``E(edge, power=1)`` and ``E_inv(edge, power=1)``
  Flux-like number operators and their inverse variants. The optional ``power`` parameter lets you
  request eigenvalues raised to a chosen power without having to explicitly wrap operators yourself.
  This is particularly important because the single-vertex constraint contains characteristic
  square-root prefactors that are most naturally expressed as fractional powers of flux eigenvalues.


Default dynamical constraint
-------------------------------

The currently implemented default dynamical constraint is the **Euclidean part of the scalar constraint** (the Lorentzian
contribution is planned but not exposed yet). The single-vertex Euclidean constraint is a sum of three cyclic terms coupling
pairs of directions with flux prefactors and a symmetric factor ordering:

.. math::

   \hat{C}^{E}(N)
   =
   -\,N(v)\sum_{(a,b,c)\in \mathrm{cyc}(x,y,z)}
   \hat{F}_{ab,c}\,\hat{s}_{a}\,\hat{s}_{b}\,\hat{F}_{ab,c},
   \qquad
   \hat{F}_{ab,c}
   :=
   \hat{E}_{a}^{1/4}\,\hat{E}_{b}^{1/4}\,\hat{E}_{c}^{-1/4}.

This ordering is chosen so that the operator stays manifestly symmetric while reproducing the
expected prefactor structure at the level of eigenvalues.

After ``initialize_constraint()``, ``lqx.constraint`` points to this Euclidean constraint.


Computational backend support
-----------------------------

QRLG is often used in regimes where the quantum numbers are large, which makes matrix-free evaluation
especially attractive. The interface supports a dedicated computational backend:

* set ``computational=True`` at construction time, or
* override per call when requesting individual operators.

Where available you can also request JAX variants (useful when your variational stack is JAX-based).

Example workflow:

.. code-block:: python

   lqx = LqxSingleVertexQR(H, G, computational=True)
   lqx.initialize_constraint()

   # build a building block operator
   sx = lqx.s(edge=0)

   # use the default Euclidean constraint as optimisation target
   Q = lqx.constraint

.. important::

   By default, in this model using the ``.initialize_constraint()`` returns the **non-squared** Euclidean constraint.
   To obtain the squared version simply use

   .. code-block:: python

      from netket.operator import Squared
      lqx.initialize_constraint()
      lqx.constraint = Squared(lqx.constraint)