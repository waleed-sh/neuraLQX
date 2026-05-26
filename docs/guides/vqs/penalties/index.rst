.. _vqs_penalties:

==============================
Penalty costs
==============================

neuraLQX supports a style of variational optimization that comes up constantly in canonical
LQG. You often optimize a **primary** operator (typically a constraint-derived quadratic form
or a Hamiltonian-like quantity) but you do *not* want the optimization to settle in parts of the
state space that are technically valid minima yet physically uninteresting (for example, degenerate
geometric sectors or large families of numerically equivalent solutions).

In practice, this is handled by **penalties**: additional terms added to the objective that
select, stabilize, or bias the optimization toward a desired sector while keeping the main
dynamical operator unchanged.

The key point in neuraLQX is that penalties are implemented as **wrapper objects** that
behave like operators at the API level (you pass them to ``expect`` / ``expect_and_grad``),
but their purpose is to represent **cost contributions**. This design is deliberate as it lets you
define objectives that

* combine operators from different backend families (e.g. NetKet ``LocalOperator`` together
  with neuraLQX ``ComputationalOperator``),
* change penalty strengths without rebuilding expensive composite operators, and
* express penalties that are **not** linear expectations (for example, penalties that depend
  nonlinearly on an expectation value).

This page documents the two penalty wrappers exposed through ``neuralqx.operators``:

* :class:`neuralqx.operators.PenaltyCost` for **linear** penalty contributions, and
* :class:`neuralqx.operators.InverseExpectationCost` for a common **nonlinear** penalty that
  suppresses small expectation values of a positive operator.

It also explains how to define your own expectation-level penalty by subclassing
``PenaltyCost`` or by registering Plum-dispatched penalty rules.

Throughout, we assume you are optimizing with a Monte-Carlo state such as
:class:`neuralqx.vqs.MCState`. The penalty objects are evaluated using the same estimator machinery as ordinary
operators.



Why penalties are wrappers, not sums
---------------------------------------

Suppose your primary optimization target is an operator :math:`\hat{Q}` and you want to add
a penalty based on another observable :math:`\hat{P}`. The "textbook" objective is

.. math::

   L(\theta) = \langle \hat{Q} \rangle_\theta + \lambda \, \langle \hat{P} \rangle_\theta,

where :math:`\theta` are the variational parameters and :math:`\lambda` is a penalty strength.

If *all* operators live in the same operator family and operator arithmetic is available, you
could build a composite operator :math:`\hat{Q} + \lambda \hat{P}` and measure it.

In neuraLQX this approach runs into three recurring issues:

1. **Backend mixing**. neuraLQX often mixes operator families intentionally (e.g. a
   matrix-free computational operator for an expensive constraint term plus a small
   NetKet local operator for a diagnostic penalty). Composite operator arithmetic is not
   guaranteed or even sometimes possible across families.

2. **Rebuilding cost**. Even *if* arithmetic exists, building composite local operators may
   be expensive. If you change :math:`\lambda` during training (annealing schedules are common),
   rebuilding the composite object is not what you want to pay for every few steps.

3. **Nonlinear penalties**. Many useful penalties in LQG are *functions of expectation values*
   rather than expectations of linear operators. Those cannot be represented as a single operator
   whose expectation equals the penalty.

neuraLQX therefore treats penalties as *objective contributions* and implements them by
modifying the **local estimator pipeline** while keeping the public ``MCState`` interface unchanged.



Recap: expectations are computed from local estimators
-------------------------------------------------------

neuraLQX inherits NetKet's estimator-based view of expectation values. For an operator
:math:`\hat{O}`, the local estimator is

.. math::

   O_{\mathrm{loc}}(\sigma) =
   \frac{\langle \sigma|\hat{O}|\psi_\theta\rangle}{\langle \sigma|\psi_\theta\rangle}
   = \sum_{\sigma'} O_{\sigma\sigma'} \, \frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.

Monte-Carlo sampling draws configurations :math:`\sigma` distributed as
:math:`p_\theta(\sigma)\propto|\psi_\theta(\sigma)|^2`, and the expectation is estimated by
averaging :math:`O_{\mathrm{loc}}(\sigma)` across samples.

Penalty wrappers work by transforming the *per-sample* values before aggregation. This is
crucial because it preserves:

* the single-pass Monte-Carlo workflow (shared samples),
* the correct statistics for sums of terms, and
* the gradient pipelines (covariance/forces or VJP depending on the operator).

You can think of penalties as producing their own local estimator array
:math:`L_{\mathrm{pen}}(\sigma)` that is compatible with the usual aggregation and gradient logic.


``PenaltyCost``: linear penalty wrapper
-------------------------------------------------------

:class:`neuralqx.operators.PenaltyCost` represents the simplest and most common case:
a penalty that is proportional to the expectation of some observable.

Mathematically, this wrapper targets objectives of the form

.. math::

   L(\theta) = \langle \hat{Q} \rangle_\theta + \lambda \, \langle \hat{P} \rangle_\theta,

where :math:`\hat{P}` is your penalty operator and :math:`\lambda` is a user-provided factor.

The wrapper itself is attached to the penalty operator :math:`\hat{P}`. In code it behaves like
an operator object, but when evaluated inside ``expect``/``expect_and_grad`` it contributes a
**scaled local estimator**.

Concretely, if :math:`P_{\mathrm{loc}}(\sigma)` is the usual local estimator of the wrapped
operator :math:`\hat{P}`, then the penalty wrapper contributes

.. math::

   L_{\mathrm{pen}}(\sigma) = \lambda \, P_{\mathrm{loc}}(\sigma),

and the Monte-Carlo average yields :math:`\lambda \langle \hat{P}\rangle_\theta`.

This looks trivial in math, but the implementation detail matters as scaling is applied
in the estimator pipeline rather than by constructing a composite operator object.


Creating a ``PenaltyCost``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The penalty wrapper is constructed from an operator and a factor:

.. code-block:: python

   import neuralqx as nqx

   P = number_operator  # any NetKet or neuraLQX operator
   PC = nqx.operators.PenaltyCost(P, factor=0.01)

The wrapper keeps a reference to the wrapped operator (commonly exposed as ``.parent`` in
the wrapper interface) so you can still access the original operator if needed for diagnostics.


Using ``PenaltyCost`` in objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The intended usage is to keep your primary operator :math:`\hat{Q}` separate and express the
objective as an explicit sum of expectations.

Because neuraLQX supports list-based expectation evaluation, the clean pattern is:

.. code-block:: python

   Q = constraint_operator
   P = number_operator
   PC = nqx.operators.PenaltyCost(P, factor=0.01)

   # one call, one sampling pass, one Stats object for the summed objective:
   stats = vstate.expect([Q, PC])

   # And if you are optimizing:
   stats, grad = vstate.expect_and_grad([Q, PC])

This is exactly the situation where penalty wrappers are most useful as it allows you to combine
operators even when they come from different abstraction levels.


Changing penalty strength during training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many workflows anneal the factor :math:`\lambda` as training progresses. The wrapper-based
design supports that naturally. In principle you can update the wrapper's factor between
steps without reconstructing the underlying operator.

Whether you treat the wrapper as immutable or update its attributes depends on how you
structure your training loop. The key conceptual point is that the factor lives with the wrapper
object, not in a composite operator that must be rebuilt.


What ``PenaltyCost`` changes internally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ordinary operators, the measurement pipeline needs two operator-dependent components:

* kernel arguments (typically connectivity and matrix elements), and
* a kernel that turns ``logpsi`` evaluations and those arguments into a local estimator array.

For :class:`~neuralqx.operators.PenaltyCost`, neuraLQX forwards connectivity requests to the
wrapped operator and then applies the scaling at the estimator/aggregation level.
From a user perspective the important consequences are:

* You can pass penalty wrappers anywhere you would pass an operator (including inside lists).
* The returned statistics correspond to the *full* objective you asked for.
* Gradients behave as if you were differentiating the summed objective
  :math:`\langle \hat{Q}\rangle + \lambda\langle \hat{P}\rangle`.


``InverseExpectationCost``: nonlinear penalty on an expectation
-------------------------------------------------------------------

Some penalties used in variational LQG are not linear expectations. A common example is
a penalty that discourages states where the expectation value of a positive operator is too
small. The motivating case is often the volume operator. If your constraint has a large kernel,
optimization can drift toward degenerate-volume solutions unless you actively bias away from them.

:class:`neuralqx.operators.InverseExpectationCost` (often abbreviated IEC) implements a
penalty contribution of the form

.. math::

   L_{\mathrm{IEC}}(\theta) =
   \lambda \, \frac{1}{\bigl(\alpha + \langle \hat{O} \rangle_\theta\bigr)^2},

where

* :math:`\hat{O}` is a (typically positive) operator whose expectation you want to keep away from zero,
* :math:`\lambda` is the penalty strength,
* :math:`\alpha>0` is a stabilizing shift that prevents divergences near :math:`\langle \hat{O} \rangle\approx 0`.

This is a scalar function of the expectation value. There is no linear operator
:math:`\hat{X}` such that :math:`\langle \hat{X}\rangle = L_{\mathrm{IEC}}(\theta)` for all states,
so the penalty must be implemented at the estimator level.


How IEC stays compatible with Monte-Carlo estimators
--------------------------------------------------------

The challenge is that NetKet-style measurement expects a per-sample local quantity to average.
IEC is a function of a global scalar :math:`\langle \hat{O}\rangle_\theta`, not of a single sample.

neuraLQX resolves this by constructing a per-sample quantity whose average equals the desired
nonlinear function *exactly* (up to Monte-Carlo error). Define

.. math::

   f(x) = \lambda (\alpha + x)^{-2}.

Compute :math:`\langle \hat{O}\rangle_\theta` from the same sample batch used for the main objective,
then define

.. math::

   L_{\mathrm{IEC}}(\sigma) = f'(\langle \hat{O}\rangle_\theta)\, O_{\mathrm{loc}}(\sigma) + g,

with

.. math::

   g = f(\langle \hat{O}\rangle_\theta) - f'(\langle \hat{O}\rangle_\theta)\, \langle \hat{O}\rangle_\theta.

This choice is engineered so that averaging over samples gives

.. math::

   \langle L_{\mathrm{IEC}}(\sigma)\rangle
   = f'(\langle \hat{O}\rangle)\, \langle O_{\mathrm{loc}}(\sigma)\rangle + g
   = f'(\langle \hat{O}\rangle)\, \langle \hat{O}\rangle + g
   = f(\langle \hat{O}\rangle).

So IEC becomes "just another per-sample local estimator" as far as the aggregation machinery is
concerned, which means it can participate in list-based objectives and in the usual Stats reporting.


Gradients of IEC
~~~~~~~~~~~~~~~~~~~~

Differentiate the IEC objective using the chain rule:

.. math::

   \nabla_\theta L_{\mathrm{IEC}}(\theta)
   = f'(\langle \hat{O}\rangle_\theta)\, \nabla_\theta \langle \hat{O}\rangle_\theta.

So IEC gradients reduce to gradients of the wrapped expectation value times a scalar prefactor
computed from the current samples.

In practice, neuraLQX evaluates :math:`\nabla_\theta \langle \hat{O}\rangle_\theta` using the same
Hermitian/covariance ("forces") pathway used for ordinary Hermitian operators, and then applies
the scalar :math:`f'(\langle \hat{O}\rangle)` factor.

.. important::
    Because the IEC uses the covariance method, this restricts its use to only Hermitian operators.


Creating and using IEC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conceptually, you construct IEC from a cost operator (the one you want to keep away from zero)
plus the penalty parameters. A typical pattern looks like:

.. code-block:: python

   import neuralqx as nqx

   # primary objective term
   Q = constraint_operator

   # some Hermitian operator
   V = volume_operator

   IEC = nqx.operators.InverseExpectationCost(
       V,
       factor=1.0,  # lambda
       alpha=1e-3,  # stabilizer shift
   )

   # one call measures the full objective <Q> + L_IEC:
   stats = vstate.expect([Q, IEC])

   # optimization uses the same interface
   stats, grad = vstate.expect_and_grad([Q, IEC])

The key behavioral guarantee is that both terms are evaluated on the same sample batch,
so the optimization "sees" a coherent objective rather than independent noisy measurements.


Defining custom penalty costs
--------------------------------

Custom nonlinear penalties usually have the form

.. math::

   L_{\mathrm{pen}}(\theta) = F(\langle \hat{O}\rangle_\theta),

where :math:`\hat{O}` is the wrapped operator and :math:`F` is a scalar objective function.
neuraLQX only needs two pieces of information:

* the value :math:`F(\langle \hat{O}\rangle)`, and
* the derivative :math:`F'(\langle \hat{O}\rangle)`.

The expectation, forces, gradient, sequence, and chunked fallback paths then construct the
appropriate affine local estimator automatically:

.. math::

   L_{\mathrm{pen}}(\sigma)
   =
   F'(\langle \hat{O}\rangle)\, O_{\mathrm{loc}}(\sigma)
   +
   F(\langle \hat{O}\rangle)
   -
   F'(\langle \hat{O}\rangle)\, \langle \hat{O}\rangle.

There are two supported ways to define the rule.


Option 1: subclass and override methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most users, the most direct approach is to subclass
:class:`neuralqx.operators.PenaltyCost` and override
``expectation_value`` and ``expectation_gradient``.

For example, this penalty implements
:math:`F(x)=\lambda\,(\mathrm{Re}\,x + s)^2`:

.. code-block:: python

   import jax.numpy as jnp
   import neuralqx as nqx


   class QuadraticExpectationCost(nqx.operators.PenaltyCost):
       def __init__(self, op, *, factor: float, shift: float = 0.0):
           super().__init__(op, factor=factor)
           self.shift = shift

       def expectation_value(self, parent_expectation):
           x = jnp.real(parent_expectation) + self.shift
           return self.factor * x**2

       def expectation_gradient(self, parent_expectation):
           x = jnp.real(parent_expectation) + self.shift
           return 2.0 * self.factor * x


   penalty = QuadraticExpectationCost(volume_operator, factor=0.5, shift=0.1)

   stats = vstate.expect(penalty)
   stats, grad = vstate.expect_and_grad([constraint_operator, penalty])

This is the recommended pattern when the mathematical definition belongs naturally to the
wrapper class.


Option 2: register Plum-dispatched rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to keep the objective rule outside the class, register methods on the public
Plum-dispatched functions instead. This is useful for extension packages or for families of
wrappers whose behavior is selected by dispatch.

For example, this penalty implements
:math:`F(x)=\lambda\,(\mathrm{Re}\,x + s)^3`:

.. code-block:: python

   import jax.numpy as jnp
   import neuralqx as nqx


   class CubicExpectationCost(nqx.operators.PenaltyCost):
       def __init__(self, op, *, factor: float, shift: float = 0.0):
           super().__init__(op, factor=factor)
           self.shift = shift


   @nqx.operators.penalty_expectation_value.dispatch
   def cubic_expectation_value(operator: CubicExpectationCost, parent_expectation):
       x = jnp.real(parent_expectation) + operator.shift
       return operator.factor * x**3


   @nqx.operators.penalty_expectation_gradient.dispatch
   def cubic_expectation_gradient(operator: CubicExpectationCost, parent_expectation):
       x = jnp.real(parent_expectation) + operator.shift
       return 3.0 * operator.factor * x**2


   penalty = CubicExpectationCost(volume_operator, factor=0.25, shift=0.2)
   stats, grad = vstate.expect_and_grad([constraint_operator, penalty])

.. important::
   Nonlinear expectation-level penalties compute gradients through the wrapped operator's
   expectation. For the covariance/forces route to be mathematically valid, the wrapped operator
   should be Hermitian and should report ``is_hermitian`` accurately.

If your custom wrapper is still exactly linear in the wrapped expectation and you want to keep
the constant-scale fast path, override ``is_linear_penalty`` and ``linear_scale`` (or register
``penalty_is_linear`` and ``penalty_linear_scale``) accordingly.


Reading results and logging components
-----------------------------------------

List-based expectations return a single Stats object corresponding to the summed objective.
If you also want to log the components separately, measure them explicitly:

.. code-block:: python

   stats_Q = vstate.expect(Q)
   stats_V = vstate.expect(V)      # to monitor <V>
   stats_tot = vstate.expect([Q, IEC])

This makes it easy to monitor whether the penalty is doing what you intended: in the volume case,
you typically want to see :math:`\langle V\rangle` rise away from zero while :math:`\langle Q\rangle`
decreases.



How penalties interact with multi-operator aggregation
---------------------------------------------------------

Penalty wrappers were designed to compose with neuraLQX’s "objective as a list of operators"
feature:

* ``expect([O1, O2, ...])`` forms the per-sample sum of local estimators and produces a single Stats.
* ``expect_and_grad([O1, O2, ...])`` computes the gradient of the summed objective consistently.

Penalty wrappers fit into this model because they produce well-defined per-sample quantities:

* ``PenaltyCost`` contributes :math:`\lambda P_{\mathrm{loc}}(\sigma)`.
* ``InverseExpectationCost`` contributes :math:`f'(\langle O\rangle) O_{\mathrm{loc}}(\sigma) + g`.

So the *total* local objective seen by the estimator is simply

.. math::

   L_\Sigma(\sigma) = Q_{\mathrm{loc}}(\sigma) + L_{\mathrm{pen}}(\sigma),

and everything downstream (statistics, gradient logic, MPI aggregation) proceeds exactly as for
ordinary operator sums.

This "penalties are estimator transforms" viewpoint is the cleanest way to reason about how to
build objectives in neuraLQX. You keep your physical operators as physical operators, and you
express optimization choices (degeneracy breaking, sector selection, stabilization) through wrappers
that speak the same measurement API.
