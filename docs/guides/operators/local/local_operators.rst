.. _netket_localoperator:

========================
Matrix-based operators
========================

The NetKet :class:`netket.operator.LocalOperator` is the standard way to build operators from **small matrices acting on small supports**.

You specify:

1. a NetKet Hilbert space (in neuraLQX, this is accessible through ``AbstractHilbertInterface.hilbert``),
2. a *local* operator matrix (dense or sparse), and
3. a list of sites (indices) on which that matrix acts,

and NetKet handles the rest: embedding, connectivity generation, and operator arithmetic. neuraLQX uses this operator type
because it maps cleanly onto "operators built from local building blocks on a graph".

Local operators as sum of supported terms
===========================================

Look again at the Hilbert space in the U(1) case. The edges all carry the same Hilbert space :math:`\mathscr{H}_e`, and thus the Hilbert space
over the graph is simply the tensor product space

.. math::

   \mathscr{H}_\mathrm{kin}^\gamma = \bigotimes_{e \in E(\gamma)} \mathscr{H}_e

Basis elements in each edge Hilbert space :math:`\mathscr{H}_e` are labeled, in this U(1) setting, by U(1) degrees of freedom (i.e. charges).
Consider the example of a maximal charge cutoff of 1. Then, for example, the basis elements in edge Hilbert spaces :math:`\mathscr{H}_e` are labelled as

.. math::

   \mid -1 \rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} \quad,\quad \mid 0 \rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} \quad,\quad \mid 1 \rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}

And once can, as we will do below, construct operators which act on these **local** basis elements, which we will call
**local terms**. A LocalOperator is best understood as a constant shift plus a sum of local terms:

* each term has a **support**: a small subset of sites,
* each term stores an explicit matrix on that support,
* NetKet keeps the global operator implicit.

In formulas, you can think of

.. math::

   \hat O = c\,\mathbb{I} + \sum_{t=1}^{T} \hat o_t,

where each :math:`\hat o_t` acts non-trivially only on a support set :math:`S_t`. 

This representation is what makes LocalOperator fast, as NetKet never materialises a full
:math:`\dim(\mathscr H)\times\dim(\mathscr H)` matrix, but only keeps the much smaller :math:`\hat{o}_t` matrices.


A concrete example in neuraLQX: the number operator on a θ-graph
===================================================================

We will build the diagonal "number"-like operator :math:`\hat N` that acts on one of the basis elements shown above and
returns the charge that represents it. This is a simple diagonal matrix

.. math::

   \hat{N} = \begin{pmatrix} -1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}

and it is easy to see that, for example, :math:`\hat{N} \mid -1 \rangle = -1 \cdot \mid -1 \rangle`. In this example, we will use:

* the θ-graph,
* a U(1) Hilbert space with cutoff,
* the edge `(0, 1)` mapped to a site index via the graph API.

The key practical rule in neuraLQX is:

**You pass site indices, not edge tuples, into NetKet operators.**
Use :meth:`neuralqx.graph.Graph.edge_to_index` for that translation. 

.. code-block:: python

   import neuralqx as nqx
   import netket as nk
   import numpy as np
   import scipy.sparse as sp

   # create a θ-graph
   edges = [(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)]
   graph = nqx.graph.Graph(edges)

   # create a Hilbert space on that graph (U(1) with cutoff qmax=2)
   H = nqx.hilbert.u1.HilbertU1(graph, cutoff=2)

   # get the local basis labels Q and the local diagonal matrix for N
   basis_labels = H.allowed_basis_states.all_states()
   number_matrix = np.diag(basis_labels, 0)

   # create a NetKet LocalOperator that acts on the edge (0, 1) by converting it to a site index
   edge_site = G.edge_to_index((0, 1))
   N = nk.operator.LocalOperator(H.hilbert, sp.coo_matrix(number_matrix), [edge_site])

What you have constructed is an operator on the *full* Hilbert space, but stored through a *tiny* local matrix plus bookkeeping.
This is the entire point of ``LocalOperator``, the stored object is small, and the global effect is produced on demand.

Connected components: ``get_conn_padded()``
===========================================

Every NetKet operator can generate connected components. For ``LocalOperator``, the most important method in neuraLQX workflows is:

``LocalOperator.get_conn_padded(sigma)``

because it returns static-shape arrays suitable for JIT-compiled estimators. 

Using the same θ-graph setting, if ``sigma`` is a batch of basis states, you obtain:

* ``sigma_p``: the connected configurations (padded),
* ``mels``: the corresponding matrix elements (padded).

For a diagonal operator like :math:`\hat N`, each configuration has exactly one connected state: itself.

.. code-block:: python

   import jax

   sigma = H.random_state(jax.random.PRNGKey(42), size=1)
   sigma_p, mels = N.get_conn_padded(sigma)

In the example above, the matrix element equals the value stored on the chosen edge, because that edge is mapped to the
first array entry in the chosen indexing convention.

Understanding the padded shapes
-------------------------------

Let:

* ``B`` be the batch size,
* ``Nsites`` be the number of sites (edges, in the default neuraLQX kinematical layout),
* ``n_conn`` be the maximum number of connections this operator can generate.

Then:

* ``sigma`` has shape ``(B, Nsites)``,
* ``sigma_p`` has shape ``(B, n_conn, Nsites)``,
* ``mels`` has shape ``(B, n_conn)``.

For diagonal operators, ``n_conn == 1``. For off-diagonal operators, ``n_conn`` is a fixed upper bound that NetKet deduces
and unused slots are padded with zeros.

This "static connectivity tensor" is what lets NetKet evaluate local estimators without ever constructing the full operator matrix.

How LocalOperator arithmetic works
==================================

Aside from being extremely efficiently designed for VMC operations, NetKet operators are also convenient objects. For example,
operator arithmetic is a first-class feature of ``LocalOperator``, and it is implemented by manipulating the internal "sum of terms" representation.

Scalar shifts and scaling
-------------------------

If you add a scalar shift ``alpha`` to an operator (``O + alpha``), NetKet updates the constant term. Scaling by a scalar
multiplies the constant and every local matrix.

These are cheap operations because they do not change supports, and they do not change how connectivity is generated-only
the returned matrix elements.

Sums
----

If you add two local operators, NetKet combines their term lists. Implementations often merge terms that have identical
supports by adding their matrices, which reduces term count and improves runtime.

Products
--------

The product is the operation that requires real care.

If you form ``C = A * B``, NetKet expands the product into:

* constant × constant,
* constant × terms,
* terms × constant,
* term × term products.

Term-by-term products remain local, but their support becomes the **union** of the supports of the multiplied terms. The
corresponding local matrix is produced by embedding each operand into that union support (tensoring with identities where
needed) and multiplying.

This is extremely effective when supports stay small. It is also the main reason ``LocalOperator`` can become heavy in
the LQG settings as repeated products can produce larger and larger union supports, and the local matrix dimension grows
*exponentially* in the support size.

Practical notes for neuraLQX users
==================================

* Prefer using :meth:`neuralqx.graph.Graph.edge_to_index` and pass indices into NetKet operators. 
* If you want a JAX-compiled local operator, NetKet provides a JAX variant, and many ``LocalOperators`` can be converted using ``to_jax_operator()``.
* ``LocalOperator`` exposes useful high-level helpers such as hermiticity checks and dense/sparse export methods (useful for debugging on small systems).
