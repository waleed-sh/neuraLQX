.. _computational_operator:

=========================================
Matrix-free Operators
=========================================

Operators built from the ``LocalOperator`` type from NetKet are extremely efficient, assuming you do not deal with large
system sizes. However, they suffer from two issues which are unavoidable. First, they store local terms as concrete matrices
which in turn means that their size grows exponentially with the system size. Second, you cannot take functions of operators as
one often needs to do in loop quantum gravity.

For this, neuraLQX introduces a new operator type, which is still a NetKet operator, just built from a more abstract operator type,
namely the ``DiscreteOperator`` type from NetKet. The neuraLQX :class:`neuralqx.operators.types.ComputationalOperator` is a
NetKet-compatible operator type where the action is provided **algorithmically**. Instead of giving a local matrix and asking
the framework to discover connectivity, you directly implement the connectivity-producing kernel:

* given a configuration (or batch),
* return connected configurations,
* return the corresponding matrix elements,

in the padded format expected by NetKet.

This operator family is designed for the cases where "write down a small matrix on a small support" is not the natural
description, or where matrix bookkeeping becomes a bottleneck.


---------------------------------------------------------------------------
Bra/row vs ket/column conventions: what it means for users
---------------------------------------------------------------------------

This is the single most important conceptual detail when writing or debugging computational kernels.

Why the **bra/row** format is required by the local estimator
======================================================================

NetKet evaluates expectation values by Monte Carlo sampling of configurations :math:`\sigma` distributed as
:math:`p(\sigma)\propto|\psi(\sigma)|^2`. Start from the normalized expectation value

.. math::

   \langle \hat O \rangle
   =
   \frac{\langle\psi|\hat O|\psi\rangle}{\langle\psi|\psi\rangle}.

Insert resolutions of the identity in the computational basis :math:`\{|\sigma\rangle\}`:

.. math::

   \langle\psi|\hat O|\psi\rangle
   =
   \sum_{\sigma,\sigma'}
   \psi^*(\sigma)\,
   \langle\sigma|\hat O|\sigma'\rangle\,
   \psi(\sigma').

Factor out :math:`|\psi(\sigma)|^2`:

.. math::

   \langle\psi|\hat O|\psi\rangle
   =
   \sum_{\sigma}
   |\psi(\sigma)|^2
   \sum_{\sigma'}
   \langle\sigma|\hat O|\sigma'\rangle
   \frac{\psi(\sigma')}{\psi(\sigma)}.

Dividing by :math:`\langle\psi|\psi\rangle=\sum_{\tilde\sigma}|\psi(\tilde\sigma)|^2` gives

.. math::

   \langle \hat O \rangle
   =
   \mathbb{E}_{\sigma\sim p(\sigma)}
   \left[
      O_{\mathrm{loc}}(\sigma)
   \right],

with the *local estimator*

.. math::

   O_{\mathrm{loc}}(\sigma)
   =
   \sum_{\sigma'}
   \langle\sigma|\hat O|\sigma'\rangle
   \frac{\psi(\sigma')}{\psi(\sigma)}.

This derivation shows why the connectivity interface is naturally a **bra/row** object:

* the sampled configuration :math:`\sigma` labels the **row** index,

* the kernel must provide matrix elements :math:`\langle\sigma|\hat O|\sigma'\rangle \equiv O_{\sigma\sigma'}`.

Equivalently, for a connectivity call with input configuration :math:`\sigma` NetKet interprets the returned data as

.. math::

   \text{given } \sigma,\ \text{return } \{\sigma'_k\}\ \text{and}\ \{O_{\sigma\sigma'_k}\},
   \qquad
   O_{\sigma\sigma'} = \langle \sigma | \hat O | \sigma' \rangle.

This matches the mental model "given a BRA, return connected KETs".

The common pitfall: thinking in ket-action builds the wrong connectivity
===========================================================================

When implementing a computational kernel, it is very common to think

   "Given a ket :math:`|\sigma\rangle`, what is :math:`\hat O|\sigma\rangle`."

That produces **column** entries

.. math::

   \hat O|\sigma\rangle = \sum_{\sigma'} \langle\sigma'|\hat O|\sigma\rangle\,|\sigma'\rangle
   \qquad\Rightarrow\qquad
   \text{coefficients } \langle\sigma'|\hat O|\sigma\rangle \equiv O_{\sigma'\sigma}.

But the local estimator requires **row** entries :math:`O_{\sigma\sigma'}`.
If you feed column connectivity into a row-based estimator, you effectively evaluate a transpose-related operator.
For real-valued matrix elements this is often experienced as "you accidentally built the adjoint".

The precise relationship is

.. math::

   \langle\sigma|\hat O|\sigma'\rangle
   =
   \left(\langle\sigma'|\hat O^\dagger|\sigma\rangle\right)^*.

So a ket-style rewrite rule most directly supplies what you need **only if it is the ket-action of the adjoint**
:math:`\hat O^\dagger` (up to complex conjugation of the coefficients).

.. note::

   For **diagonal** operators, none of this matters because the connectivity is trivial and only diagonal entries contribute.
   Rows and columns coincide on the diagonal, so the "bra vs ket" confusion does not show up.

What you should implement: bras or the adjoint on kets
==========================================================

There are two equivalent, correct ways to construct a computational kernel.

A) Implement the bra-action directly
---------------------------------------

Treat the input configuration as a bra :math:`\langle\sigma|` and return:

* connected configurations :math:`\{\sigma'_k\}`

* matrix elements :math:`m_k = \langle\sigma|\hat O|\sigma'_k\rangle`

This matches the local estimator exactly.

Implement the ket-action of the adjoint
------------------------------------------

If the operator is naturally written as a rewrite rule on kets, implement the ket-action of :math:`\hat O^\dagger` instead:

.. math::

   \hat O^\dagger|\sigma\rangle = \sum_k a_k(\sigma)\,|\sigma'_k\rangle,
   \qquad
   a_k(\sigma)=\langle\sigma'_k|\hat O^\dagger|\sigma\rangle.

Then you recover the required row elements by

.. math::

   \langle\sigma|\hat O|\sigma'_k\rangle = a_k(\sigma)^*.

In many LQG-style operators the coefficients are real, so the conjugation does nothing, but the adjoint still fixes the
connectivity direction.

What neuraLQX does for pre-implemented operators
===================================================

In neuraLQX, pre-implemented operators used in the package's own workflows are written so that NetKet's estimator receives the
correct **bra/row** matrix elements internally.

.. warning::

   Hamiltonian operators are often easiest to read when written as a ket-rewrite rule.
   In those cases, implementations frequently correspond to the **adjoint action** needed by the estimator
   **even if the operator class is named after the "mathematical" Hamiltonian.**
   This is intentional, and users should be aware of it when comparing to hand-derived ket-action formulas.

   Example:

   .. code-block::

      # this returns an operator implementing the ADJOINT action of the Thiemann Hamiltonian in the 4-d Euclidean model,
      # even though adjoint=False, and even though the returned object is named without an "adjoint" in it
      H = nqx.operators.computational.Euclidean4d.numba.ThiemannRegularisedVertexConstraint(
         vertex=0, adjoint=False
      )

      # this returns an operator implementing the NON-ADJOINT action of the Thiemann Hamiltonian in the 4-d Euclidean model,
      # even though adjoint=True, and even though the returned object is named with an "adjoint" in it
      H = nqx.operators.computational.Euclidean4d.numba.ThiemannRegularisedVertexConstraint(
         vertex=0, adjoint=True
      )

Sanity checks that catch convention bugs early
=================================================

* **Diagonal-only test:** verify that your diagonal operators return the input configuration and correct diagonal elements.

* **Dense check on a tiny system:** build the dense matrix from your kernel on a very small Hilbert space and compare against a
  reference ``LocalOperator`` implementation.

* **Hermitian expectation test:** for Hermitian operators, compare ``vstate.expect`` against a reference on small sizes. If the
  connectivity direction is wrong, off-diagonal contributions typically disagree first.

* **Product order test:** for non-commuting products, confirm that your wrapper logic reproduces the intended operator ordering.

Composite operators must preserve the estimator convention
=============================================================

neuraLQX implements sums/products/scaling for matrix-free operators using wrapper operators.
For correctness, wrappers must preserve the fact that connectivity must represent the **row elements**
:math:`\langle\sigma|\hat O|\sigma'\rangle` needed by the local estimator.

If you see symptoms such as "raising behaves like lowering", unexpected complex conjugation, or products that appear swapped,
the first debugging step is to verify whether a component kernel was implemented in ket-action form without taking the adjoint,
and whether wrapper operators preserved the intended convention.


---------------------------------------------------------------------------
The padded contract is the API
---------------------------------------------------------------------------

``ComputationalOperator`` is built around the same idea as the ``LocalOperator`` connectivity interface: **static-shape connectivity** via ``get_conn_padded()``.
However, unlike the ``LocalOperator`` type, **you** have to implement a kernel function (a ``_get_conn_padded_kernel()``) that this high-level function calls.
Your kernel must return something that behaves like:

* ``sigma_p``: shape ``(B, n_conn, ...)`` (connected configurations),
* ``mels``: shape ``(B, n_conn)`` (corresponding matrix elements),

with a fixed, operator-defined ``n_conn``.

**For the sake of simplicity, we will consider the ket action in what follows**. In that case, a good mental model is: "I am implementing the right-hand side of
:math:`\hat O|\sigma\rangle = \sum_k O_k(\sigma)\,|\sigma'_k\rangle`
directly."

Concrete example: the number operator on a θ-graph (same as LocalOperator)
=============================================================================

Let's go back to the number-like operator which we considered for the ``LocalOperator`` example. The setting is the same:

* we have a θ-graph,
* on which we build a U(1) Hilbert space,
* and pick edge `(0, 1)` using ``edge_to_index``,
* and the operator is diagonal: it returns exactly one connection.

Graph + Hilbert setup
------------------------------

.. code-block:: python

   import neuralqx as nqx
   import jax
   import jax.numpy as jnp

   # build a theta graph
   edges = [(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)]
   graph = nqx.graph.Graph(edges)

   # build a U(1) Hilbert space with cutoff 2
   H = nqx.hilbert.u1.HilbertU1(G, cutoff=2)

   # get the site on which the operator will act on
   edge_site = G.edge_to_index((0, 1))

Implementing a ComputationalOperator kernel
-------------------------------------------

The simplest diagonal kernel for :math:`\hat N` does two things:

1. Return the same configuration as the only connected configuration.
2. Return the value stored at the chosen site as the matrix element.

In pseudocode terms: "copy ``sigma`` into slot 0, and set ``mels[..., 0] = sigma[..., edge_site]``". neuraLQX's documentation
example shows this implemented in a padded form and stresses that it should handle both batched and single inputs.

In principle, you must create a new operator which subclasses the ``ComputationalOperator`` type and implement the
``_get_conn_padded_kernel()`` method for it. For the purposes of demonstration, we just show this kernel method. A minimal
JAX-friendly implementation looks like:

.. code-block:: python

   def _get_conn_padded_kernel(sigma: jnp.ndarray, edge_site: int):
       """

       :param sigma: Array of shape (B, Nsites) or (Nsites,).
       :param edge_site: Integer site index (from Graph.edge_to_index).

       :return sigma_p: Array of shape (B, 1, Nsites)
       :return mels: Array of shape (B, 1)
       """

       if sigma.ndim == 1:
           sigma = sigma[None, :]  # add batch to accommodate for single inputs

       vals = sigma[:, edge_site].astype(jnp.float64)
       sigma_p = sigma[:, None, :]
       mels = vals[:, None]
       return sigma_p, mels

You can sanity-check that this reproduces the same connectivity behaviour as the ``LocalOperator`` implementation:

.. code-block:: python

   sigma = H.random_state(jax.random.PRNGKey(42), size=1)
   sigma_p, mels = _get_conn_padded_kernel(sigma, edge_site)

The output interpretation is identical:

* one connected configuration (itself),
* one diagonal matrix element (the chosen edge's label).

---------------------------------------------------------------------------
Where does ``ComputationalJaxOperator`` fit?
---------------------------------------------------------------------------

For performance-sensitive workflows, neuraLQX also provides a JAX-jitted variant (the ``ComputationalJaxOperator``) where
the connectivity kernel is designed to live comfortably inside JAX transformations and distributed execution.

The key practical difference is that you structure your kernel as a pure JAX function that:

* uses JAX primitives (no Python-side loops over connections),
* returns arrays with shapes that are stable under ``jit`` and ``vmap``.

.. note::

   In JAX mode, the operator object itself is typically treated as a Python object (static) while the connectivity
   computation is jitted. If you attempt to trace the operator object through JAX transformations, you may encounter
   errors like "not a valid JAX type" or tracer attribute errors. The recommended approach is to keep the operator
   as a static argument and JIT only the numeric kernel.

---------------------------------------------------------------------------
Operator arithmetic for ComputationalOperators: wrappers
---------------------------------------------------------------------------

NetKet's ``DiscreteOperator`` base class (at the supported NetKet version) only enforces "can you produce connected components?".
It does not force any internal representation, so it cannot implement rich algebra automatically for every possible operator
without falling back to an explicit full matrix (which is infeasible for large Hilbert spaces).

neuraLQX restores operator arithmetic for matrix-free operators by returning **wrapper** ``ComputationalOperators`` that
implement algebra directly at the level of connectivity.

Scaled operators
----------------

Given a scalar :math:`\alpha` and an operator :math:`\hat O`, scaling is implemented by:

* reusing the same connected configurations,
* multiplying all matrix elements by :math:`\alpha`.

Sums
----

For :math:`\hat A \pm \hat B`, the wrapper:

1. calls ``get_conn_padded`` on both operands for the same input batch,
2. concatenates the connected configurations along the connection axis,
3. concatenates matrix elements (with the sign applied),
4. reorders/pads to satisfy the fixed-size padded contract.

This is the matrix-free analogue of "term list concatenation", but it is done at runtime in connectivity space.

Products
--------

For :math:`\hat C = \hat A \hat B`, neuraLQX composes connectivity:

1. apply :math:`\hat B` to :math:`|\sigma\rangle` to get intermediate branches :math:`|\tau_k\rangle`,
2. apply :math:`\hat A` to each :math:`|\tau_k\rangle` to get final branches :math:`|\kappa_{k,l}\rangle`,
3. multiply matrix elements along each path.

This approach avoids building local matrices and avoids forming union supports. Its cost is governed by the connectivity
fan-out (worst-case scaling like "connections of B times connections of A"), and neuraLQX includes pruning steps to skip
branches that are numerically zero.

Why this matters in practice
----------------------------

This wrapper approach is what lets you write complicated LQG-style operators as small algorithmic building blocks and
still assemble:

* sums over vertices,
* shifted constraints,
* quadratic penalties,
* products of constraint pieces,

without switching back to explicit matrices or indexable Hilbert spaces.

.. caution::::

   Computational operators in neuraLQX are **ket-action by default** (``is_ket_action = True``):
   your kernel should return matrix elements :math:`\langle \sigma'|\hat O|\sigma\rangle`
   (i.e. column entries :math:`O_{\sigma'\sigma}`).

   neuraLQX compensates for this at the **estimator level** so that ``MCState.expect`` still
   targets :math:`\langle\psi|\hat O|\psi\rangle / \langle\psi|\psi\rangle`.

   If you prefer NetKet's bra/row convention, set ``is_ket_action = False`` and return
   :math:`\langle \sigma|\hat O|\sigma'\rangle` instead.


---------------------------------------------------------------------------
Write kernels like you mean it
---------------------------------------------------------------------------

When you move to ``ComputationalOperators``, **you are taking responsibility for the part that LocalOperator normally hides**. Namely,

* keeping shapes static and batch-friendly,
* minimising Python control flow (especially for JAX variants),
* keeping ``n_conn`` as small as physics allows,
* writing kernels that vectorise over the batch dimension.

Done well, this is where you get large performance wins and where you can encode graph/loop move logic directly in the
operator, rather than reverse-engineering it from matrices.
