.. _vqs_projectors:

============================
Projectors and Projections
============================

In neuraLQX, *projectors* are the mechanism for restricting a variational state to a
desired subspace **without changing the surrounding sampling/optimisation machinery**.

The key design choice is that projection is implemented at the **model (amplitude) level**.
A projector *wraps* your neural ansatz (a ``flax.linen.Module``) and post-processes its output
in **log-amplitude space**. Once wrapped, the projected state behaves like an ordinary
NetKet variational state, and you can sample it, measure expectations, and optimise it using the
same API.

This page covers:

* the general mathematical notion of projection used by neuraLQX,
* the **group projector** framework (including the discrete diffeomorphism/automorphism projector),
* the **no-vacuum projector**,
* and the generic ``MCState.project(...)`` hook for building your own projectors and composing them.


----------------------------------------
What "projection" means in neuraLQX
----------------------------------------

Let :math:`\mathcal{H}` be a Hilbert space with an orthonormal computational basis
:math:`\{|\sigma\rangle\}` labelled by configurations :math:`\sigma \in \mathcal{C}`.
A variational state is specified by amplitudes

.. math::

   \psi_\theta(\sigma) = \langle \sigma|\psi_\theta\rangle,
   \qquad
   \log \psi_\theta(\sigma)\in\mathbb{C}.

neuraLQX, as NetKet, typically parametrises **log-amplitudes** (complex-valued), and the amplitude is
recovered by exponentiation.

A projector (in the strict linear-algebra sense) is a linear map
:math:`\Pi:\mathcal{H}\rightarrow\mathcal{H}` with :math:`\Pi^2=\Pi`. Applied to the state,

.. math::

   |\psi_\theta^\Pi\rangle := \Pi|\psi_\theta\rangle,
   \qquad
   \psi_\theta^\Pi(\sigma)=\sum_{\sigma'\in\mathcal{C}}\Pi_{\sigma\sigma'}\,\psi_\theta(\sigma').

If :math:`\Pi` is Hermitian and idempotent, you can normalise the projected state by dividing by
:math:`\sqrt{\langle\psi_\theta|\Pi|\psi_\theta\rangle}`. In Monte Carlo optimisation, the global
normalisation is usually not needed explicitly because sampling/estimators handle it implicitly.

.. note::

    Two important practical notes:

    1. In many workflows you want a *projection effect* ("enforce this constraint on amplitudes") even if
       the implemented map is a numerically convenient hard/soft variant rather than a perfect
       Hermitian idempotent projector.
    2. Because neuraLQX implements projection by wrapping the ansatz, the projected state is still
       differentiable (where intended) and can be trained end-to-end.


----------------------------------------------------------
How projectors are implemented: model wrappers in Flax
----------------------------------------------------------

All neuraLQX projectors are implemented as ``flax.linen.Module`` **wrappers** that contain the original
ansatz as a submodule named ``base`` and return a modified log-amplitude.

Concretely, you start with a base model:

.. math::

   \sigma \mapsto \log\psi_\theta(\sigma)

and the projector produces a new model

.. math::

   \sigma \mapsto \log\psi_{\theta}^{(\Pi)}(\sigma).

This has two major consequences.

**A. You can project a trained state after the fact.**
The wrapper contains the original model in a dedicated subtree. neuraLQX can create the wrapper,
then transplant your trained parameters into ``params["base"]`` inside the wrapper. This means you
can train an unprojected model, then apply a projection without retraining (useful for diagnostics
and for quickly checking how a symmetry restriction changes observables).

**B. Projections are composable.**
Because "a projector returns a Flax model", you can stack multiple projectors by wrapping repeatedly
(e.g. first remove the vacuum, then group-average).

The generic entry point for custom wrappers is:

``MCState.project(wrap_model, **kwargs)``

where ``wrap_model`` is any function with signature:

.. math::

   \texttt{wrap\_model(base\_model, **kwargs)} \rightarrow \texttt{wrapped\_model}

and the wrapped model must follow the convention that the original model is stored as
a submodule named ``base``.


------------------------------------------------------------
Group projectors: the general framework (finite groups)
------------------------------------------------------------

A particularly robust and important class of projectors comes from a **finite group**
:math:`G` acting unitarily on :math:`\mathcal{H}` through a representation
:math:`U: G\rightarrow \mathrm{U}(\mathcal{H})`.

In neuraLQX, the action is typically induced by **index permutations** of the configuration vector.
If a group element :math:`g` permutes indices, then

.. math::

   U(g)|\sigma\rangle = |g\cdot\sigma\rangle,
   \qquad
   (g\cdot\sigma)_i = \sigma_{g^{-1}(i)}.

This is exactly what happens for graph automorphisms as they permute edge indices, so they act by
relabeling the configuration entries.

Trivial irrep (invariant subspace): uniform group averaging
---------------------------------------------------------------

The projector onto the invariant subspace (trivial representation) is the uniform group average:

.. math::

   \Pi^{(\mathrm{triv})}_{G}
   =
   \frac{1}{|G|}\sum_{g\in G} U(g),
   \qquad
   \psi_\theta^{(G)}(\sigma)
   =
   \frac{1}{|G|}\sum_{g\in G}\psi_\theta(g^{-1}\cdot\sigma).

This construction guarantees invariance:

.. math::

   \psi_\theta^{(G)}(h\cdot\sigma)=\psi_\theta^{(G)}(\sigma),
   \qquad \forall h\in G.

In words, the projected amplitude is the average of the original amplitude over the entire orbit
of the configuration under the group action.

Projecting onto a chosen irrep: character projectors
--------------------------------------------------------

More generally, if you want the component transforming in an irrep :math:`\rho` of dimension
:math:`d_\rho` with character :math:`\chi_\rho`, the standard character projector is:

.. math::

   \Pi^{(\rho)}_{G}
   =
   \frac{d_\rho}{|G|}
   \sum_{g\in G}
   \chi_\rho(g)^{*}\,U(g),

which gives

.. math::

   \psi_\theta^{(\rho)}(\sigma)
   =
   \frac{d_\rho}{|G|}
   \sum_{g\in G}
   \chi_\rho(g)^{*}\,\psi_\theta(g^{-1}\cdot\sigma).

This is useful when you want to enforce a symmetry type rather than full invariance, for example
picking a non-trivial one-dimensional representation of a discrete symmetry group.

Numerical stability: log-space "log-mean-exp" reduction
-----------------------------------------------------------

Directly evaluating the sums above in amplitude space is numerically fragile when
:math:`\log\psi_\theta(\sigma)` can vary strongly across a batch or when amplitudes are complex.

neuraLQX therefore implements group projectors in **log space**:

.. math::

   \log\psi_\theta^{(\rho)}(\sigma)
   =
   \log\left[
      \frac{d_\rho}{|G|}
      \sum_{g\in G}
      \chi_\rho(g)^{*}
      \exp\left(\log\psi_\theta(g^{-1}\cdot\sigma)\right)
   \right],

using a stable complex variant of the log-sum-exp / log-mean-exp reduction. This preserves
differentiability and makes it realistic to train projected models end-to-end.

API surface: ``neuralqx.nn.projectors.group_projector``
-------------------------------------------------------

neuraLQX exposes group averaging through the projector wrapper module
``neuralqx.nn.projectors.group_projector``. At the state level, you usually don't instantiate
the wrapper manually, instead you call a convenience method on the variational state
(see the diffeomorphism projector below).


-----------------------------------------------------------------
Diffeomorphism-invariant projector on a fixed graph (Aut(Γ))
-----------------------------------------------------------------

In canonical LQG, diffeomorphism invariance is implemented by solving the diffeomorphism constraint.
On a **fixed combinatorial graph** :math:`\Gamma`, the continuous diffeomorphism group is commonly
replaced by the discrete symmetry group of the graph, typically the **automorphism group** :math:`\mathrm{Aut}(\Gamma)`.

An automorphism is a relabeling of vertices/edges that preserves the incidence structure. In neuraLQX's
multigraph setting this also respects the "parallel-edge key" structure. Automorphisms act by permuting
edge indices, hence they act naturally on configurations by index permutation.

The discrete diffeomorphism projector
-----------------------------------------

The "diff-invariant" projector used here is the trivial-irrep group average over automorphisms

.. math::

   \Pi_{\mathrm{diff}}(\Gamma)
   :=
   \Pi^{(\mathrm{triv})}_{\mathrm{Aut}(\Gamma)}
   =
   \frac{1}{|\mathrm{Aut}(\Gamma)|}
   \sum_{\phi\in\mathrm{Aut}(\Gamma)} U(\phi),
   \qquad
   |\psi_\theta^{\mathrm{diff}}\rangle := \Pi_{\mathrm{diff}}(\Gamma)|\psi_\theta\rangle.

By construction, the projected state is invariant under all automorphisms.

How to construct it in code
-------------------------------

The workflow has two conceptual steps:

1) compute the relevant symmetry permutations of your graph, and
2) build a new projected ``MCState`` that evaluates the group-averaged amplitude.

neuraLQX provides a helper for step (1). Namely, the class

``neuralqx.utils.symmetries.GraphSymmetries(graph.edges)``

which can return automorphisms as a list of permutations.

A typical usage pattern is:

.. code-block:: python

   import neuralqx as nqx

   # graph: nqx.graph.Graph
   sym = nqx.utils.symmetries.GraphSymmetries(graph.edges)

   # automorphisms generate the "diff" symmetry on a fixed graph
   autos = sym.automorphisms

   # vstate: neuralqx.vqs.MCState
   vstate_diff = vstate.to_group_averaged(symmetries=autos, graph=graph)

The returned ``vstate_diff`` is a new ``MCState`` whose model is wrapped by the group-projector
module, with your trained parameters transplanted into the ``params["base"]`` subtree.
That transplant step is what makes projection cheap to apply post-hoc.

Projecting onto non-trivial symmetry types
----------------------------------------------

If you want to project onto a specific irrep rather than the trivial one, you can pass
a list of **characters** aligned with the provided symmetries/permutations. The resulting
projector is the character projector described earlier. The default is the trivial irrep
(i.e. uniform averaging).

Practical guidance:

* Use the trivial projector when you want strict invariance under all automorphisms.
* Use a non-trivial one-dimensional character when you want a "sign representation" type behaviour
  on a symmetry (for example, odd/even under a particular reflection), but still want a strict
  projection onto a symmetry sector.


-----------------------------------------
No-vacuum projector
-----------------------------------------

In LQG-like lattice models, the configuration

.. math::

   \sigma = 0

(i.e. all edge charges/labels equal to zero, for example in a U(1)\ :sup:`3` setting) plays the role
of a distinguished vacuum configuration. In expressive variational ansätze, a common failure mode is
a kind of mode collapse where the model learns to concentrate essentially all probability mass on this
single basis element, effectively producing the Ashtekar–Lewandowski vacuum. Once the Markov chain
mostly visits :math:`\sigma=0`, gradient estimates become dominated by the vacuum and exploration of
excited configurations becomes poor.

To prevent this, neuraLQX implements a hard "remove the vacuum basis vector" projector:

.. math::

   \Pi_{\neq 0} := I - |0\rangle\langle 0|,
   \qquad
   \psi_\theta^{(\neq 0)}(\sigma)
   =
   \begin{cases}
      0, & \sigma=0,\\
      \psi_\theta(\sigma), & \text{otherwise}.
   \end{cases}

In neuraLQX, this lives in ``neuralqx.nn.projectors.vacuum``.

Log-space implementation and differentiability
--------------------------------------------------

Since the neural network outputs :math:`\log\psi_\theta(\sigma)`, neuraLQX implements the hard zero
by mapping the vacuum to a large negative constant ``vacuum_value`` in log space:

.. math::

   \log\psi_\theta^{(\neq 0)}(\sigma)
   =
   \begin{cases}
      v_{\mathrm{vac}}, & \sigma=0,\\
      \log\psi_\theta(\sigma), & \text{otherwise},
   \end{cases}

with default :math:`v_{\mathrm{vac}}=-10^{30}` (so :math:`\exp(v_{\mathrm{vac}})\approx 0`). This keeps
the model differentiable on the non-vacuum sector while effectively enforcing
:math:`\psi^{(\neq 0)}_\theta(0)\approx 0`.

When to use it
------------------

Use the no-vacuum projector when:

* your sampler collapses to (or spends excessive time at) :math:`\sigma=0`,
* you observe near-zero variance because all samples are the vacuum,
* or your physics target explicitly requires support on non-trivial excitations.

.. note::

    If your physical problem genuinely expects the vacuum as the solution, you should not use this
    projector. It enforces a hard exclusion.


----------------------------------------------------
Constructing your own wrappers
----------------------------------------------------

Beyond built-in projectors, neuraLQX provides a generic mechanism to its variational state interface

``MCState.project(wrap_model, **kwargs)``

You supply a function that takes the base model and returns a wrapped model. The wrapped model must:

* store the base model as a submodule named ``base`` (this is required for parameter transplantation),
* accept the same inputs as your base model (commonly ``sigma`` plus optional keyword arguments),
* return the projected **log-amplitude**.

Minimal template
--------------------

A minimal template looks like:

.. code-block:: python

   import flax.linen as nn
   import jax.numpy as jnp

   class MyProjector(nn.Module):
       base: nn.Module  # REQUIRED: enables params["base"] transplantation

       @nn.compact
       def __call__(self, sigma, **kwargs):
           out = self.base(sigma, **kwargs)     # log ψθ(σ), shape (B,) typically
           # modify out in log-space here
           return out

   def wrap_model(base_model: nn.Module, **kwargs) -> nn.Module:
       return MyProjector(base=base_model, **kwargs)

   vstate_proj = vstate.project(wrap_model, **kwargs)

This pattern is exactly what neuraLQX uses internally for its built-ins. The distinction is that
your wrapper can encode any "physics-motivated amplitude rule" you want.

What counts as a "projector" in practice
--------------------------------------------

Not every useful wrapper is a strict linear projector :math:`\Pi^2=\Pi`.
In practice, you often want one of the following behaviours:

**Hard amplitude masks**
  Set certain forbidden configurations to (effectively) zero amplitude in log space
  (as in the no-vacuum projector).

**Symmetry averaging**
  Replace amplitudes by an average over an orbit (as in group projectors).

**Input reshaping / architecture adapters**
  Build a wrapper that changes how configurations are presented to the base model
  (for example, convert a flattened configuration into a "vector view" appropriate for U(1)\ :sup:`N`
  degrees of freedom). This can be valuable even if it is not a projector in the algebraic sense,
  because it lets the architecture align with physical structure.


---------------------------------------
Stacking projectors
---------------------------------------

Because projectors are wrappers that return new models, you can compose them by calling
``.project(...)`` repeatedly, or by wrapping one wrapper inside another.

A common pattern is:

1) remove the vacuum, then
2) enforce automorphism invariance.

Conceptually:

.. code-block:: python

   # step 1: remove vacuum
   v1 = vstate.project(wrap_no_vacuum, vacuum_value=-1e30)

   # step 2: group-average (automorphisms)
   v2 = v1.to_group_averaged(symmetries=autos, graph=graph)

The resulting state evaluates:

* first the base model,
* then the no-vacuum log-space override,
* then the group-averaged amplitude.

Because parameters are stored under ``params["base"]`` at each wrapper boundary, neuraLQX can keep
parameter trees consistent across composition steps.


-------------------------------------------------------
Training strategies: post-hoc projection vs end-to-end
-------------------------------------------------------

neuraLQX supports both of the following workflows:

**Post-hoc projection**
  Train an unprojected state, then apply a projector afterwards (cheap and good for quick tests
  and diagnostics).

**End-to-end projected training**
  Wrap the model first, then train the projected model directly. This is what you want if you
  need the *best* variational optimum within the projected manifold, because projection changes
  the functional class of the ansatz.

A practical heuristic:

* If the projection is a mild symmetry reduction (like automorphism invariance), training end-to-end
  often improves convergence and stability because the model never has to represent symmetry-breaking
  fluctuations.
* If the projection is a hard mask (like no-vacuum), applying it post-hoc is a quick sanity check,
  but end-to-end training is typically better if you truly want to exclude the masked sector during
  optimisation.
