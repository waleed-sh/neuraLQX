.. _experimental_nonhermitian_gradients:

=========================================
Non-Hermitian gradients: bi-covariance
=========================================

This page documents the experimental **exact bi-covariance gradient** for non-Hermitian operators in
neuraLQX. It explains what the formula is, when neuraLQX uses it, and what your operator must provide
for the path to activate.


----------------------------------------------
The problem with non-Hermitian gradients
----------------------------------------------

For a Hermitian operator :math:`\hat{H}`, the gradient of the expectation value

.. math::

   E(\theta) = \frac{\langle \psi_\theta | \hat{H} | \psi_\theta \rangle}
               {\langle \psi_\theta | \psi_\theta \rangle}

is given by the *covariance formula*:

.. math::

   \partial_\eta E = 2\,\Re\!\left[\operatorname{Cov}_{p_\theta}\!\left(\overline{O_\eta},\, L_H\right)\right],

where :math:`O_\eta = \partial_\eta \log\psi_\theta` is the log-derivative and
:math:`L_H(x) = (H\psi_\theta)(x)/\psi_\theta(x)` is the local estimator. This is exact, efficient,
and is implemented in neuraLQX via :func:`neuralqx.vqs.expect_and_forces`.

For a **non-Hermitian** operator :math:`A`, the covariance formula is no longer exact. The standard
fallback, used by both NetKet and neuraLQX by default, is to differentiate the Monte Carlo estimator
directly through reverse-mode autodiff (VJP through the local estimator). This is correct in exact
arithmetic but:

* requires back-propagating through the local estimator kernel, which can be slow,
* has estimator variance that grows with the norm of :math:`A` relative to a Hermitian operator,
* and is harder to reason about for sequence objectives.

The bi-covariance path offers an alternative: use the exact two-sided covariance formula by evaluating
forces for both :math:`A` and its adjoint :math:`A^\dagger` independently, then combine them. The trade-off
is one additional call to :func:`neuralqx.vqs.expect_and_forces`, but the gradient estimator is exact,
stays on the same code path as the Hermitian covariance formula, and benefits from the same chunking and
fused-kernel infrastructure.


----------------------------------------------
Enabling the bi-covariance path
----------------------------------------------

The bi-covariance gradient is gated behind two configuration flags: ``EXPERIMENTAL`` (the general
experimental namespace) and ``EXPERIMENTAL_GRAD`` (the specific gradient flag). Both must be set.

Enable via ``cfg.update`` (recommended for scripts and notebooks):

.. code-block:: python

   import neuralqx as nqx

   nqx.cfg.update("EXPERIMENTAL", True)
   nqx.cfg.update("EXPERIMENTAL_GRAD", True)

   # everything downstream is now wired to the bi-covariance path
   stats, grad = vstate.expect_and_grad(my_operator)

Enable via shell environment variables (recommended for batch jobs):

.. code-block:: bash

   export NQX_EXPERIMENTAL=1
   export NQX_EXPERIMENTAL_GRAD=1
   python your_script.py

Both flags must be set. Setting only ``NQX_EXPERIMENTAL=1`` enables the experimental namespace
(imports, drivers, states) but leaves gradient routing unchanged. Setting only ``NQX_EXPERIMENTAL_GRAD=1``
has no effect because the routing check in :mod:`neuralqx.vqs.mc.mc_state.expect_grad` tests both.

.. note::

   You can toggle the path at runtime without restarting Python by calling
   ``nqx.cfg.update("EXPERIMENTAL_GRAD", True/False)`` or using ``nqx.cfg.patch(...)``.
   Because the routing check runs on every call to ``expect_and_grad``, the change takes effect
   immediately for the next gradient step.


----------------------------------------------
Mathematical derivation
----------------------------------------------

Let

.. math::

   E_A(\theta) = \frac{\langle \psi_\theta | A | \psi_\theta \rangle}
                {\langle \psi_\theta | \psi_\theta \rangle}

be the variational expectation of a generic operator :math:`A`, with the Monte Carlo sampling distribution

.. math::

   p_\theta(x) = \frac{|\psi_\theta(x)|^2}{\sum_y |\psi_\theta(y)|^2}.

Define the local estimators

.. math::

   L_A(x) &= \frac{(A\psi_\theta)(x)}{\psi_\theta(x)}, \\
   L_{A^\dagger}(x) &= \frac{(A^\dagger\psi_\theta)(x)}{\psi_\theta(x)},

and the log-derivative with respect to a real coordinate :math:`\eta`,

.. math::

   O_\eta(x) = \partial_\eta \log \psi_\theta(x).

The *exact* gradient is then:

.. math::

   \partial_\eta E_A
   = \operatorname{Cov}_{p_\theta}\!\left(\overline{O_\eta},\, L_A\right)
   + \operatorname{Cov}_{p_\theta}\!\left(O_\eta,\, \overline{L_{A^\dagger}}\right).

Expanding each covariance:

.. math::

   \partial_\eta E_A
   &= \bigl\langle \overline{O_\eta}\, L_A \bigr\rangle
    - \bigl\langle \overline{O_\eta} \bigr\rangle \bigl\langle L_A \bigr\rangle \\
   &\quad + \bigl\langle O_\eta\, \overline{L_{A^\dagger}} \bigr\rangle
    - \bigl\langle O_\eta \bigr\rangle \bigl\langle \overline{L_{A^\dagger}} \bigr\rangle.

This is the bi-covariance formula. The name comes from the fact that both :math:`\overline{O_\eta}`
(conjugate log-derivative) and :math:`O_\eta` appear, one covariance term for the forward operator,
one for the adjoint.

In force notation (as used by :func:`neuralqx.vqs.expect_and_forces`):

.. math::

   F_A^{(j)} &= \operatorname{Cov}_{p_\theta}\!\left(\overline{O_j},\, L_A\right), \\
   F_{A^\dagger}^{(j)} &= \operatorname{Cov}_{p_\theta}\!\left(\overline{O_j},\, L_{A^\dagger}\right).

For real parameters, the gradient assembles as:

.. math::

   g_j = F_A^{(j)} + \overline{F_{A^\dagger}^{(j)}}.

For holomorphic models with complex parameters, the Wirtinger derivative with respect to
:math:`\bar{z}_j` reduces to :math:`g_j = F_A^{(j)}`, only the forward force term contributes
(see :ref:`holomorphic-branch` below).


----------------------------------------------
Implementation
----------------------------------------------

The bi-covariance path is implemented in
:mod:`neuralqx.experimental.vqs.mc.mc_state.expect_grad_biadjoint` and dispatched from the public
:meth:`neuralqx.vqs.MCState.expect_and_grad` entrypoint.

For a single operator
-----------------------

When ``expect_and_grad`` receives a single non-Hermitian operator and both flags are enabled, it
calls :func:`~neuralqx.experimental.vqs.mc.mc_state.expect_grad_biadjoint.expect_and_grad_biadjoint`.
The function:

1. Resolves :math:`A^\dagger` by calling ``operator.adjoint`` (or returns ``None`` to fall back if the
   operator does not expose it).
2. Calls ``expect_and_forces(vstate, A, ...)``, this is one call to the forces kernel, producing
   statistics :math:`\langle A \rangle` and the force pytree :math:`F_A`.
3. For non-holomorphic real-parameter models: calls ``expect_and_forces(vstate, A†, ...)`` on the
   same sample batch to produce :math:`F_{A^\dagger}`, then assembles
   :math:`g_j = F_A^{(j)} + \overline{F_{A^\dagger}^{(j)}}`.
4. For holomorphic all-complex-parameter models: skips step 3, returns :math:`g_j = F_A^{(j)}`
   cast to the parameter dtype.

For a sequence of operators
-----------------------------

When ``expect_and_grad`` receives a list of operators, it calls
:func:`~neuralqx.experimental.vqs.mc.mc_state.expect_grad_biadjoint.expect_and_grad_biadjoint_sequence`.
The function resolves adjoints for every operator upfront. If *any* operator in the list lacks
``adjoint``, the entire sequence falls back to the generic non-Hermitian VJP path and a warning is
emitted identifying the missing operator.

If all adjoints are available, the sequence is passed as a full list to ``expect_and_forces``. This
preserves the existing fused/unfused sequence behaviour controlled by ``NQX_FUSED_KERNELS``:

* ``NQX_FUSED_KERNELS=1``: one JIT-compiled pass over all operators in the sequence.
* ``NQX_FUSED_KERNELS=0``: separate kernel calls per operator, accumulated manually.

Both branches of the adjoint pass follow the same fused/unfused selection.

For ``Squared`` operators
---------------------------

:class:`netket.operator.Squared` wraps a parent operator :math:`P` as :math:`P^\dagger P` and is
Hermitian by construction. For routing purposes neuraLQX treats ``Squared(P)`` specially:

* Its adjoint resolves to itself (``Squared(P)`` is its own adjoint).
* The objective is real-valued, so the complex-dtype fallback is not triggered even when ``P.dtype``
  is complex.
* The gradient assembles as :math:`g_j = F_{\text{Sq}}^{(j)} + \overline{F_{\text{Sq}}^{(j)}}`,
  which simplifies to :math:`2\,\Re[F_{\text{Sq}}^{(j)}}]`, the standard Hermitian covariance form.

This means ``Squared`` objectives receive an exact gradient regardless of the parent operator's dtype,
as long as the bi-covariance flags are enabled.


.. _holomorphic-branch:

----------------------------------------------
Holomorphic models
----------------------------------------------

For variational states whose network is *holomorphic* (all parameter leaves have complex dtype and the
network satisfies the Cauchy-Riemann conditions), the Wirtinger calculus applies. The Wirtinger
derivative with respect to :math:`\bar{z}_j` is simply :math:`F_A^{(j)}`, and the adjoint correction
term vanishes identically.

neuraLQX detects this case by checking two conditions simultaneously:

1. **All parameter leaves are complex** (via a JAX pytree scan over ``vstate.parameters``).
2. **The model is probably holomorphic** (via NetKet's ``is_probably_holomorphic`` diagnostic, which
   evaluates the Cauchy-Riemann test on a small sample of configurations).

When both conditions hold, ``expect_and_forces`` is called only for the forward operator :math:`A`,
and the result is cast to the parameter leaf dtype. The adjoint-force pass is skipped entirely, so
holomorphic models pay exactly one ``expect_and_forces`` call regardless of whether the path is
bi-covariance or covariance.

When the model has complex parameters but fails the holomorphicity test (e.g. a non-holomorphic
complex ansatz, or a model using ``tanh`` with complex inputs), neuraLQX emits a warning and falls
back to the generic non-Hermitian VJP path. This is intentional: the Wirtinger shortcut is not valid
for non-holomorphic models, and neuraLQX does not attempt to apply it anyway.


----------------------------------------------
Routing decision tree
----------------------------------------------

When ``expect_and_grad`` is called on a non-Hermitian operator (or sequence), the dispatcher
evaluates the following checks in order:

.. code-block:: text

   cfg.EXPERIMENTAL and cfg.EXPERIMENTAL_GRAD?
       No  -> generic non-Hermitian VJP path
       Yes -> continue

   Is the objective complex-valued? (complex operator dtype, not Squared)
       Yes -> generic non-Hermitian VJP path  [+ warning]
       No  -> continue

   Does the model have complex parameter leaves?
       No  -> bi-covariance path (real model, two forces calls)
       Yes -> is_probably_holomorphic?
           No  -> generic non-Hermitian VJP path  [+ warning]
           Yes -> bi-covariance path (Wirtinger branch, one forces call)

   Does every operator expose `.adjoint`?
       No  -> generic non-Hermitian VJP path  [+ warning]
       Yes -> bi-covariance path proceeds

All fallbacks preserve the public API return shape ``(stats, grad)`` exactly. The caller never sees a
difference except in the gradient values and any warnings emitted via Python's ``warnings`` module.


----------------------------------------------
Fallback behaviour and warnings
----------------------------------------------

Any condition that prevents the bi-covariance path issues a ``UserWarning`` and returns ``None``
from the internal ``expect_and_grad_biadjoint`` function. The dispatcher in
:mod:`neuralqx.vqs.mc.mc_state.expect_grad` treats ``None`` as "fall through to the generic
non-Hermitian VJP path".

The warnings identify the reason precisely:

* *Missing adjoint*: ``"Operator FooBar does not implement adjoint. Falling back to the generic
  non-Hermitian gradient..."``.
* *Complex objective dtype*: ``"Detected a complex operator dtype; falling back to the generic
  non-Hermitian gradient..."``.
* *Non-holomorphic complex model*: ``"Detected a non-holomorphic model via
  is_probably_holomorphic; falling back to the generic non-Hermitian gradient..."``.
* *Flag not set*: ``"Experimental bi-covariance gradient is disabled because
  cfg.EXPERIMENTAL_GRAD is False."``.

In all cases the fallback is the same path that would have been used without the flags, so disabling
the feature is always safe.


----------------------------------------------
What your operator must implement
----------------------------------------------

To be compatible with the bi-covariance path, an operator must:

1. **Expose an** ``adjoint`` **property** that returns a fully constructed operator instance.
   The adjoint should be cheap to construct (do not trigger large kernel compilations inside
   ``adjoint`` getter calls).

2. **Match the Hilbert space** of the original operator. The local kernel for :math:`A^\dagger`
   will be evaluated on the same sampled configurations as :math:`A`, so the Hilbert space must
   agree exactly.

3. **Keep** ``is_hermitian`` **accurate**. The routing in ``expect_and_grad`` uses this flag to
   decide whether to attempt the covariance path at all. If ``is_hermitian`` is wrong, the operator
   will either fall through to the VJP path unnecessarily (if falsely ``False``) or attempt the
   covariance formula on a non-Hermitian objective (if falsely ``True``).

4. **Have a real-valued objective**, or be a ``Squared`` operator. The bi-covariance formula
   returns a real-valued gradient. Operators with a genuinely complex-valued ``dtype`` are routed
   to the VJP path.


----------------------------------------------
Profiling
----------------------------------------------

The bi-covariance path is instrumented with ``neuralqx.profile.section`` blocks under the
``vqs.grad`` category, matching the profiling conventions of the rest of the gradient stack.
The section names follow a hierarchical pattern:

* ``expect_and_grad.biadjoint.resolve_adjoint``, adjoint resolution and type check,
* ``expect_and_grad.biadjoint.objective_type``, complex-dtype guard,
* ``expect_and_grad.biadjoint.holomorphicity``, holomorphicity diagnostic,
* ``expect_and_grad.biadjoint.forces``, forward forces call,
* ``expect_and_grad.biadjoint.forces_adjoint``, adjoint forces call (skipped for holomorphic models),
* ``expect_and_grad.biadjoint.assemble``, gradient assembly,
* ``expect_and_grad.biadjoint.assemble_holomorphic_complex``, Wirtinger-branch assembly.

For sequences, the prefix is ``expect_and_grad.sequence.biadjoint.*``.

Setting ``NQX_PROFILE_SYNC=1`` enables device-synchronised timing across all sections (identical
policy to the existing single-state gradient paths). By default, profiling is non-blocking.

----------------------------------------------
Practical notes
----------------------------------------------

**Two forces calls, not one.** For real-parameter or non-holomorphic-complex models, the bi-covariance
path calls ``expect_and_forces`` twice per gradient step, once for :math:`A`, once for :math:`A^\dagger`.
Both calls share the same sample batch drawn at the start of the step, so no extra sampling is performed.
The total per-step cost is approximately double the single-operator forces cost. For holomorphic
all-complex models, only the forward call is made.

**Fused kernels interact with sequences.** The ``NQX_FUSED_KERNELS`` flag applies to both the forward
and adjoint sequence passes independently. If your operators support factored local kernels, fused mode
can meaningfully reduce compile times and kernel dispatch overhead for long sequences. If compile time
on first step is a concern, try ``NQX_FUSED_KERNELS=0`` first, then benchmark with it enabled.

**Runtime toggling with** ``cfg.patch``. You can switch the bi-covariance path on or off within a
context manager without affecting the rest of the run:

.. code-block:: python

   with nqx.cfg.patch("EXPERIMENTAL_GRAD", False):
       # this gradient step uses the VJP path
       stats, grad = vstate.expect_and_grad(A)

   # back to bi-covariance
   stats, grad = vstate.expect_and_grad(A)

This is useful for benchmarking or for isolating whether a numerical discrepancy originates in the
gradient path.

**Comparing both paths.** To numerically validate that the bi-covariance and VJP paths agree for your
operator, you can compute gradients with both flags on and off on the same sample batch:

.. code-block:: python

   nqx.cfg.update("EXPERIMENTAL_GRAD", True)
   _, grad_biadj = vstate.expect_and_grad(A)

   nqx.cfg.update("EXPERIMENTAL_GRAD", False)
   _, grad_vjp = vstate.expect_and_grad(A)

   import jax, jax.numpy as jnp
   for (k, g1), g2 in zip(
       jax.tree_util.tree_leaves_with_path(grad_biadj),
       jax.tree_util.tree_leaves(grad_vjp),
   ):
       print(k, jnp.max(jnp.abs(g1 - g2)))

Small discrepancies are expected from Monte Carlo noise. Large discrepancies indicate either that
the adjoint implementation is incorrect or that the operator dtype is being misclassified.
