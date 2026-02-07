.. _operators_overview:

======================
Operators in neuraLQX
======================


Operators are the objects that turn your **physics definitions** (observables, constraints, dynamics) into something that
can be evaluated on a Hilbert space and used inside variational algorithms.

In neuraLQX, operators sit at the boundary between:

* **Kinematics**: a Hilbert space over a graph (basis states are edge-labelled arrays).
* **Algorithms**: Monte Carlo estimation, gradient-based optimisation, and variational solvers.

All neuraLQX operators are NetKet operators sitting at different abstraction layers and serve different purposes. However,
irrespective of their type, at runtime, the single most important job of an operator is to provide **connected components**.

Connected components: what VMC actually needs
=============================================

Variational Monte Carlo does not need an explicit matrix for an operator on the full Hilbert space. Instead, for a sampled
basis configuration :math:`\sigma`, it needs the list of basis states :math:`\sigma'_k` that the operator connects to
:math:`\sigma`, together with the corresponding matrix elements. These are called the *connected components*.

Conceptually, the operator action looks like

.. math::

   \hat O\,|\sigma\rangle = \sum_{k=1}^{K(\sigma)} O_k(\sigma)\,|\sigma'_k\rangle.

The pair

* the connected configurations :math:`\sigma'_k`, and
* the matrix elements :math:`O_k(\sigma)`,

is what neuraLQX (via NetKet's operator interface) uses to assemble local estimators inside expectation values.

Why ``get_conn_padded()``?
--------------------------------

NetKet exposes a host of different functionalities ``get_conn_*()`` which return the connected components in different
layouts. However, most modern pipelines (especially when just-in-time compilation is involved) strongly prefer **static shapes**.
For that reason, NetKet exposes a "padded" connectivity interface, and neuraLQX leans on it heavily:

* ``get_conn_padded(sigma)`` returns a *fixed-size* array of connected states and a fixed-size array of matrix elements.
* If an operator has fewer than the maximum number of connections for a given input, the remainder is padded with zeros (and dummy states).

This is the interface you will see repeatedly in the operator APIs discussed below.

Operator families in neuraLQX
=============================

In practice, you will encounter two main ways of representing operators:

1. **Matrix-based local operators (aka NetKet ``LocalOperator``)**
   You provide a small matrix acting on a small subset of sites, and NetKet takes care of embedding it into the full Hilbert space efficiently. This is the default workhorse for many lattice-style models and for small-to-moderate local building blocks.

2. **Matrix-free algorithmic operators (neuraLQX ``ComputationalOperator``)**
   You provide the connectivity *algorithmically* (a kernel that returns connected states and matrix elements), and neuraLQX/NetKet uses that directly. This is the route you take when:
   * an operator is large and matrix bookkeeping becomes heavy,
   * the action is naturally expressed as a move on configurations (graph/loop moves, constraint moves),
   * you want full control over performance and JAX friendliness,
   * you deal with extremely large systems

Note that neuraLQX provides some implemented operators and custom operator types. The public entry point is the ``neuralqx.operators`` module.

Where to go next
================

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Matrix-based operators
      :link: local/local_operators.html
      :text-align: center

   .. grid-item-card:: Matrix-free operators
      :link: computational/computational_operators.html
      :text-align: center

.. toctree::
   :hidden:
   :maxdepth: 1

   local/local_operators
   computational/computational_operators
